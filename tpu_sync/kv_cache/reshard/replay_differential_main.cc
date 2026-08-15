// Copyright 2026 Google LLC.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Gate G1: the offline differential harness. Feeds a recorded Python
// controller wire log (the TPU_RAIDEN_PLAN_DUMP_DIR jsonl from the plan-dump
// hook) into the C++ ReshardService and requires every outbound worker
// payload to match the Python controller's recorded payload byte-for-byte
// under canonical (deterministic) proto serialization — including entry and
// group order. Any mismatch is a port bug until proven otherwise.
//
// Usage:
//   replay_differential --log=<controller-*.jsonl> [--peer_metadata_log=...]
//
// --log is the controller whose behavior is replayed. GET_METADATA
// out-queries are answered from the same log's remote_metadata_in records
// (per-address FIFO). With --peer_metadata_log, this controller's OWN
// GET_METADATA responses are additionally checked against the peer
// controller's recorded remote_metadata_in payloads (directory replay
// proof for the destination side).

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <map>
#include <string>
#include <utility>
#include <vector>

#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/escaping.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/time/time.h"
#include "google/protobuf/io/coded_stream.h"
#include "google/protobuf/io/zero_copy_stream_impl_lite.h"
#include "google/protobuf/util/message_differencer.h"
#include "tpu_sync/core/transfer_program_reshard.h"
#include "tpu_sync/kv_cache/reshard/framed_rpc.h"
#include "tpu_sync/kv_cache/reshard/reshard_service.h"
#include "tpu_sync/rpc/raiden_service.pb.h"

ABSL_FLAG(std::string, log, "", "Recorded controller jsonl to replay.");
ABSL_FLAG(std::string, peer_metadata_log, "",
          "Peer controller jsonl whose remote_metadata_in records check "
          "this side's GET_METADATA responses.");
ABSL_FLAG(bool, program_roundtrip, true,
          "Also compile each outbound StartTransferRequest to a "
          "TransferProgram, lower it back, and assert compile o lower == "
          "identity.");

namespace {

using tpu_raiden::kv_cache::reshard::FramedTransport;
using tpu_raiden::kv_cache::reshard::ReshardService;

struct Record {
  std::string kind;
  int64_t seq = 0;
  std::string addr;
  std::string payload;
};

// Minimal parser for the dump hook's flat JSON lines (fixed key set,
// payload is base64; no nested objects).
bool ParseRecord(const std::string& line, Record* out) {
  auto find_string = [&](absl::string_view key, std::string* value) {
    const std::string needle = absl::StrCat("\"", key, "\": \"");
    size_t pos = line.find(needle);
    if (pos == std::string::npos) return false;
    pos += needle.size();
    size_t end = line.find('"', pos);
    if (end == std::string::npos) return false;
    *value = line.substr(pos, end - pos);
    return true;
  };
  auto find_int = [&](absl::string_view key, int64_t* value) {
    const std::string needle = absl::StrCat("\"", key, "\": ");
    size_t pos = line.find(needle);
    if (pos == std::string::npos) return false;
    pos += needle.size();
    *value = 0;
    bool any = false;
    while (pos < line.size() && line[pos] >= '0' && line[pos] <= '9') {
      *value = *value * 10 + (line[pos] - '0');
      ++pos;
      any = true;
    }
    return any;
  };
  std::string payload_b64;
  if (!find_string("kind", &out->kind)) return false;
  if (!find_int("seq", &out->seq)) return false;
  find_string("addr", &out->addr);
  if (!find_string("payload_b64", &payload_b64)) return false;
  return absl::Base64Unescape(payload_b64, &out->payload);
}

std::vector<Record> LoadLog(const std::string& path) {
  std::vector<Record> records;
  std::ifstream in(path);
  std::string line;
  while (std::getline(in, line)) {
    Record record;
    if (ParseRecord(line, &record)) {
      records.push_back(std::move(record));
    }
  }
  return records;
}

// Canonical serialization: deterministic proto bytes (sorted map entries),
// the comparison domain G1 defines.
template <typename Message>
std::string CanonicalBytes(const Message& message) {
  std::string out;
  {
    google::protobuf::io::StringOutputStream stream(&out);
    google::protobuf::io::CodedOutputStream coded(&stream);
    coded.SetSerializationDeterministic(true);
    message.SerializePartialToCodedStream(&coded);
  }
  return out;
}

absl::StatusOr<std::string> CanonicalControlRequest(
    const std::string& payload) {
  tpu_sync::rpc::ControlRequest req;
  if (!req.ParseFromString(payload)) {
    return absl::InvalidArgumentError("unparseable ControlRequest payload");
  }
  return CanonicalBytes(req);
}

int32_t ControlCommand(const std::string& payload) {
  tpu_sync::rpc::ControlRequest req;
  if (!req.ParseFromString(payload)) return -1;
  return req.command();
}

// Replay transport: answers GET_METADATA from the recorded
// remote_metadata_in FIFO; captures every other outbound call.
class ReplayTransport final : public FramedTransport {
 public:
  explicit ReplayTransport(
      std::map<std::string, std::vector<std::string>> metadata_fifo)
      : metadata_fifo_(std::move(metadata_fifo)) {}

  absl::StatusOr<std::string> Call(absl::string_view address,
                                   absl::string_view payload,
                                   absl::Duration /*timeout*/) override {
    const std::string addr(address);
    const std::string body(payload);
    if (ControlCommand(body) ==
        tpu_sync::rpc::ControlRequest::COMMAND_GET_METADATA) {
      auto& fifo = metadata_fifo_[addr];
      if (fifo.empty()) {
        return absl::NotFoundError(
            absl::StrCat("no recorded remote_metadata_in left for ", addr));
      }
      std::string response = fifo.front();
      fifo.erase(fifo.begin());
      return response;
    }
    captured_[addr].push_back(body);
    tpu_sync::rpc::ControlResponse ok;
    ok.set_success(true);
    ok.set_message("SUCCESS");
    return ok.SerializeAsString();
  }

  const std::map<std::string, std::vector<std::string>>& captured() const {
    return captured_;
  }

 private:
  std::map<std::string, std::vector<std::string>> metadata_fifo_;
  std::map<std::string, std::vector<std::string>> captured_;
};

// First-difference diagnostic between two canonical byte strings.
void ReportByteDiff(const std::string& expected, const std::string& actual) {
  size_t limit = std::min(expected.size(), actual.size());
  size_t offset = 0;
  while (offset < limit && expected[offset] == actual[offset]) ++offset;
  std::fprintf(stderr,
               "    canonical sizes: python=%zu cxx=%zu, first diff at "
               "byte %zu\n",
               expected.size(), actual.size(), offset);
  tpu_sync::rpc::ControlRequest expected_req, actual_req;
  if (expected_req.ParseFromString(expected) &&
      actual_req.ParseFromString(actual)) {
    std::string report;
    google::protobuf::util::MessageDifferencer differencer;
    differencer.ReportDifferencesToString(&report);
    differencer.Compare(expected_req, actual_req);
    if (report.size() > 4000) report.resize(4000);
    std::fprintf(stderr, "%s\n", report.c_str());
  }
}

}  // namespace

int main(int argc, char** argv) {
  absl::ParseCommandLine(argc, argv);
  const std::string log_path = absl::GetFlag(FLAGS_log);
  if (log_path.empty()) {
    std::fprintf(stderr, "--log is required\n");
    return 2;
  }
  std::vector<Record> records = LoadLog(log_path);
  if (records.empty()) {
    std::fprintf(stderr, "no records parsed from %s\n", log_path.c_str());
    return 2;
  }

  // Partition the log.
  std::map<std::string, std::vector<std::string>> metadata_fifo;
  std::vector<Record> rpc_in;
  std::map<std::string, std::vector<std::string>> recorded_out;
  int skipped_shutdown = 0;
  for (const Record& record : records) {
    if (record.kind == "remote_metadata_in") {
      metadata_fifo[record.addr].push_back(record.payload);
    } else if (record.kind == "rpc_in") {
      rpc_in.push_back(record);
    } else if (record.kind == "worker_out") {
      const int32_t command = ControlCommand(record.payload);
      if (command == tpu_sync::rpc::ControlRequest::COMMAND_GET_METADATA ||
          command == tpu_sync::rpc::ControlRequest::COMMAND_SHUTDOWN) {
        continue;  // transport-internal or teardown; not plan output
      }
      recorded_out[record.addr].push_back(record.payload);
    }
  }

  ReplayTransport transport(std::move(metadata_fifo));
  ReshardService::Options options;
  options.port = 0;
  options.transport = &transport;
  ReshardService service(options);

  // Optional peer-side metadata responses to check against.
  std::map<std::string, std::vector<std::string>> peer_metadata_fifo;
  const std::string peer_log = absl::GetFlag(FLAGS_peer_metadata_log);
  if (!peer_log.empty()) {
    for (const Record& record : LoadLog(peer_log)) {
      if (record.kind == "remote_metadata_in") {
        peer_metadata_fifo[record.addr].push_back(record.payload);
      }
    }
  }

  int failures = 0;
  int metadata_checked = 0;
  int frames = 0;
  for (const Record& record : rpc_in) {
    const int32_t command = ControlCommand(record.payload);
    if (command == tpu_sync::rpc::ControlRequest::COMMAND_SHUTDOWN) {
      ++skipped_shutdown;
      continue;
    }
    const std::string response = service.HandleFrame(record.payload);
    ++frames;
    if (command == tpu_sync::rpc::ControlRequest::COMMAND_GET_METADATA &&
        !peer_metadata_fifo.empty()) {
      // The peer recorded what the Python controller answered here.
      bool matched = false;
      for (auto& [addr, fifo] : peer_metadata_fifo) {
        if (fifo.empty()) continue;
        tpu_sync::rpc::ControlResponse expected_resp, actual_resp;
        if (!expected_resp.ParseFromString(fifo.front()) ||
            !actual_resp.ParseFromString(response)) {
          continue;
        }
        if (CanonicalBytes(expected_resp) == CanonicalBytes(actual_resp)) {
          fifo.erase(fifo.begin());
          matched = true;
          ++metadata_checked;
          break;
        }
      }
      if (!matched) {
        std::fprintf(stderr,
                     "FAIL: GET_METADATA response (rpc seq %ld) does not "
                     "match any peer-recorded metadata response\n",
                     static_cast<long>(record.seq));
        ++failures;
      }
    }
  }

  // Compare captured outbound payloads against the recording, per address,
  // in order.
  int compared = 0;
  const auto& captured = transport.captured();
  for (const auto& [addr, expected_list] : recorded_out) {
    auto it = captured.find(addr);
    const std::vector<std::string> empty;
    const std::vector<std::string>& actual_list =
        it == captured.end() ? empty : it->second;
    if (actual_list.size() != expected_list.size()) {
      std::fprintf(stderr,
                   "FAIL: %s: python sent %zu worker payloads, C++ sent "
                   "%zu\n",
                   addr.c_str(), expected_list.size(), actual_list.size());
      ++failures;
    }
    const size_t n = std::min(expected_list.size(), actual_list.size());
    for (size_t i = 0; i < n; ++i) {
      auto expected = CanonicalControlRequest(expected_list[i]);
      auto actual = CanonicalControlRequest(actual_list[i]);
      if (!expected.ok() || !actual.ok()) {
        std::fprintf(stderr, "FAIL: %s[%zu]: unparseable payload\n",
                     addr.c_str(), i);
        ++failures;
        continue;
      }
      ++compared;
      if (*expected != *actual) {
        std::fprintf(stderr, "FAIL: %s[%zu]: canonical bytes differ\n",
                     addr.c_str(), i);
        ReportByteDiff(*expected, *actual);
        ++failures;
      }
      if (absl::GetFlag(FLAGS_program_roundtrip)) {
        tpu_sync::rpc::ControlRequest req;
        if (req.ParseFromString(actual_list[i]) &&
            req.has_start_transfer_request()) {
          auto program =
              tpu_raiden::core::CompileStartTransfer(req.start_transfer_request());
          if (!program.ok()) {
            std::fprintf(stderr,
                         "FAIL: %s[%zu]: CompileStartTransfer failed: %s\n",
                         addr.c_str(), i,
                         program.status().ToString().c_str());
            ++failures;
          } else {
            auto lowered = tpu_raiden::core::LowerToStartTransfer(*program);
            if (!lowered.ok()) {
              std::fprintf(stderr,
                           "FAIL: %s[%zu]: LowerToStartTransfer failed: %s\n",
                           addr.c_str(), i,
                           lowered.status().ToString().c_str());
              ++failures;
            } else {
              tpu_sync::rpc::ControlRequest lowered_req;
              lowered_req.set_command(req.command());
              *lowered_req.mutable_peers() = req.peers();
              *lowered_req.mutable_start_transfer_request() = *lowered;
              auto lowered_canonical =
                  CanonicalControlRequest(lowered_req.SerializeAsString());
              if (!lowered_canonical.ok() || *lowered_canonical != *actual) {
                std::fprintf(stderr,
                             "FAIL: %s[%zu]: program roundtrip canonical bytes "
                             "differ\n",
                             addr.c_str(), i);
                ++failures;
              }
            }
          }
        }
      }
    }
  }
  for (const auto& [addr, actual_list] : captured) {
    if (recorded_out.find(addr) == recorded_out.end()) {
      std::fprintf(stderr,
                   "FAIL: C++ sent %zu payloads to %s; python sent none\n",
                   actual_list.size(), addr.c_str());
      ++failures;
    }
  }

  std::fprintf(stderr,
               "replayed %d frames (%d shutdown skipped), compared %d "
               "worker payloads, %d metadata responses checked, %d "
               "failures\n",
               frames, skipped_shutdown, compared, metadata_checked, failures);
  if (failures == 0) {
    std::fprintf(stdout, "G1 PASS: %d worker payloads byte-identical\n",
                 compared);
    return 0;
  }
  std::fprintf(stdout, "G1 FAIL: %d mismatches\n", failures);
  return 1;
}
