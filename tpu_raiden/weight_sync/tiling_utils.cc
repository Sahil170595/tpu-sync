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

#include "tpu_raiden/weight_sync/tiling_utils.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/types/span.h"
#include "xla/index_util.h"
#include "xla/layout.h"
#include "xla/layout_util.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/util.h"

namespace tpu_raiden::weight_sync {

namespace {

int64_t GetTiledBufferElements(const xla::Shape& shape) {
  const int num_dims = shape.dimensions().size();
  if (num_dims == 0) {
    return 1;
  }

  std::vector<int64_t> current_shape;
  current_shape.reserve(num_dims);
  for (int64_t i = num_dims - 1; i >= 0; --i) {
    int64_t logical_dim = shape.layout().minor_to_major(i);
    current_shape.push_back(shape.dimensions(logical_dim));
  }

  for (const xla::Tile& tile : shape.layout().tiles()) {
    const int64_t tile_rank = tile.dimensions().size();
    if (tile_rank > current_shape.size()) {
      int64_t pad_size = tile_rank - current_shape.size();
      current_shape.insert(current_shape.begin(), pad_size, 1);
    }

    const int64_t suffix_start = current_shape.size() - tile_rank;
    std::vector<int64_t> next_shape;
    next_shape.reserve(current_shape.size() + tile_rank);

    for (int i = 0; i < suffix_start; ++i) {
      next_shape.push_back(current_shape[i]);
    }

    for (int i = 0; i < tile_rank; ++i) {
      int64_t d = current_shape[suffix_start + i];
      int64_t t = tile.dimension(i);
      next_shape.push_back(xla::CeilOfRatio(d, t));
    }

    for (int i = 0; i < tile_rank; ++i) {
      int64_t t = tile.dimension(i);
      next_shape.push_back(t);
    }

    current_shape = std::move(next_shape);
  }

  int64_t total_elements = 1;
  for (int64_t dim_size : current_shape) {
    total_elements *= dim_size;
  }
  return total_elements;
}

bool IsStandardRowMajorTiled(const xla::Shape& shape,
                             const xla::Layout& layout) {
  if (layout.tiles().size() != 1) return false;
  if (layout.tiles(0).dimensions().size() != 2) return false;

  const int R = shape.dimensions().size();
  if (R < 2) return false;

  for (int i = 0; i < R; ++i) {
    if (layout.minor_to_major(i) != R - 1 - i) {
      return false;
    }
  }
  return true;
}

absl::Status TileBufferNDOptimized(const uint8_t* src_linear,
                                   uint8_t* dst_tiled, const xla::Shape& shape,
                                   const xla::Layout& layout) {
  const int R = shape.dimensions().size();
  int64_t H = shape.dimensions(layout.minor_to_major(1));
  int64_t W = shape.dimensions(layout.minor_to_major(0));
  int64_t itemsize =
      xla::ShapeUtil::ByteSizeOfPrimitiveType(shape.element_type());

  const xla::Tile& tile = layout.tiles(0);
  int64_t tile_H = tile.dimension(0);
  int64_t tile_W = tile.dimension(1);

  int64_t num_tiles_0 = xla::CeilOfRatio(H, tile_H);
  int64_t num_tiles_1 = xla::CeilOfRatio(W, tile_W);
  int64_t tile_size_bytes = tile_H * tile_W * itemsize;

  int64_t batch_size = 1;
  for (int i = 2; i < R; ++i) {
    batch_size *= shape.dimensions(layout.minor_to_major(i));
  }

  int64_t matrix_size_bytes = H * W * itemsize;
  int64_t tiled_matrix_size_bytes = num_tiles_0 * num_tiles_1 * tile_size_bytes;

  // Zero-initialize the destination buffer to handle padding automatically
  int64_t total_physical_elements = GetTiledBufferElements(shape);
  std::memset(dst_tiled, 0, total_physical_elements * itemsize);

  for (int64_t b = 0; b < batch_size; ++b) {
    const uint8_t* src_batch_ptr = src_linear + b * matrix_size_bytes;
    uint8_t* dst_batch_ptr = dst_tiled + b * tiled_matrix_size_bytes;

    for (int64_t tile_row = 0; tile_row < num_tiles_0; ++tile_row) {
      for (int64_t tile_col = 0; tile_col < num_tiles_1; ++tile_col) {
        int64_t tile_index = tile_row * num_tiles_1 + tile_col;
        uint8_t* dst_tile_ptr = dst_batch_ptr + tile_index * tile_size_bytes;

        for (int64_t r = 0; r < tile_H; ++r) {
          int64_t logical_row = tile_row * tile_H + r;
          if (logical_row >= H) {
            continue;
          }

          int64_t logical_col_start = tile_col * tile_W;
          int64_t valid_elements = std::min(tile_W, W - logical_col_start);
          if (valid_elements <= 0) {
            continue;
          }

          const uint8_t* src_row_ptr =
              src_batch_ptr + (logical_row * W + logical_col_start) * itemsize;
          uint8_t* dst_row_ptr = dst_tile_ptr + (r * tile_W) * itemsize;

          std::memcpy(dst_row_ptr, src_row_ptr, valid_elements * itemsize);
        }
      }
    }
  }
  return absl::OkStatus();
}

absl::Status DetileBufferNDOptimized(const uint8_t* src_tiled,
                                     uint8_t* dst_linear,
                                     const xla::Shape& shape,
                                     const xla::Layout& layout) {
  const int R = shape.dimensions().size();
  int64_t H = shape.dimensions(layout.minor_to_major(1));
  int64_t W = shape.dimensions(layout.minor_to_major(0));
  int64_t itemsize =
      xla::ShapeUtil::ByteSizeOfPrimitiveType(shape.element_type());

  const xla::Tile& tile = layout.tiles(0);
  int64_t tile_H = tile.dimension(0);
  int64_t tile_W = tile.dimension(1);

  int64_t num_tiles_0 = xla::CeilOfRatio(H, tile_H);
  int64_t num_tiles_1 = xla::CeilOfRatio(W, tile_W);
  int64_t tile_size_bytes = tile_H * tile_W * itemsize;

  int64_t batch_size = 1;
  for (int i = 2; i < R; ++i) {
    batch_size *= shape.dimensions(layout.minor_to_major(i));
  }

  int64_t matrix_size_bytes = H * W * itemsize;
  int64_t tiled_matrix_size_bytes = num_tiles_0 * num_tiles_1 * tile_size_bytes;

  for (int64_t b = 0; b < batch_size; ++b) {
    const uint8_t* src_batch_ptr = src_tiled + b * tiled_matrix_size_bytes;
    uint8_t* dst_batch_ptr = dst_linear + b * matrix_size_bytes;

    for (int64_t tile_row = 0; tile_row < num_tiles_0; ++tile_row) {
      for (int64_t tile_col = 0; tile_col < num_tiles_1; ++tile_col) {
        int64_t tile_index = tile_row * num_tiles_1 + tile_col;
        const uint8_t* src_tile_ptr =
            src_batch_ptr + tile_index * tile_size_bytes;

        for (int64_t r = 0; r < tile_H; ++r) {
          int64_t logical_row = tile_row * tile_H + r;
          if (logical_row >= H) {
            continue;
          }

          int64_t logical_col_start = tile_col * tile_W;
          int64_t valid_elements = std::min(tile_W, W - logical_col_start);
          if (valid_elements <= 0) {
            continue;
          }

          uint8_t* dst_row_ptr =
              dst_batch_ptr + (logical_row * W + logical_col_start) * itemsize;
          const uint8_t* src_row_ptr = src_tile_ptr + (r * tile_W) * itemsize;

          std::memcpy(dst_row_ptr, src_row_ptr, valid_elements * itemsize);
        }
      }
    }
  }
  return absl::OkStatus();
}

}  // namespace

absl::Status DetileBuffer(const uint8_t* src_tiled, uint8_t* dst_linear,
                          const xla::Shape& shape, const xla::Layout& layout) {
  if (layout.tiles().empty()) {
    return absl::InternalError("Buffer is not tiled");
  }

  if (IsStandardRowMajorTiled(shape, layout)) {
    return DetileBufferNDOptimized(src_tiled, dst_linear, shape, layout);
  }

  int64_t itemsize =
      xla::ShapeUtil::ByteSizeOfPrimitiveType(shape.element_type());

  xla::Shape standard_shape =
      xla::ShapeUtil::MakeShape(shape.element_type(), shape.dimensions());

  xla::ShapeUtil::ForEachIndexNoStatus(
      shape, [&](absl::Span<const int64_t> indices) -> bool {
        int64_t linear_offset =
            xla::IndexUtil::MultidimensionalIndexToLinearIndex(standard_shape,
                                                               indices) *
            itemsize;
        int64_t physical_offset =
            xla::LayoutUtil::LinearIndexForNestedTiling(shape, indices) *
            itemsize;
        std::memcpy(dst_linear + linear_offset, src_tiled + physical_offset,
                    itemsize);
        return true;
      });

  return absl::OkStatus();
}

absl::Status TileBuffer(const uint8_t* src_linear, uint8_t* dst_tiled,
                        const xla::Shape& shape, const xla::Layout& layout) {
  if (layout.tiles().empty()) {
    return absl::InternalError("Buffer is not tiled");
  }

  if (IsStandardRowMajorTiled(shape, layout)) {
    return TileBufferNDOptimized(src_linear, dst_tiled, shape, layout);
  }

  int64_t itemsize =
      xla::ShapeUtil::ByteSizeOfPrimitiveType(shape.element_type());

  int64_t total_physical_elements = GetTiledBufferElements(shape);
  std::memset(dst_tiled, 0, total_physical_elements * itemsize);

  xla::Shape standard_shape =
      xla::ShapeUtil::MakeShape(shape.element_type(), shape.dimensions());

  xla::ShapeUtil::ForEachIndexNoStatus(
      shape, [&](absl::Span<const int64_t> indices) -> bool {
        int64_t linear_offset =
            xla::IndexUtil::MultidimensionalIndexToLinearIndex(standard_shape,
                                                               indices) *
            itemsize;
        int64_t physical_offset =
            xla::LayoutUtil::LinearIndexForNestedTiling(shape, indices) *
            itemsize;
        std::memcpy(dst_tiled + physical_offset, src_linear + linear_offset,
                    itemsize);
        return true;
      });

  return absl::OkStatus();
}

}  // namespace tpu_raiden::weight_sync
