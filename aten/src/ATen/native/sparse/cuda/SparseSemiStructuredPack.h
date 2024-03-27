#pragma once

#include <ATen/native/sparse/cuda/StaticSort.h>
#include <cutlass/arch/memory.h>
#include <cutlass/array.h>
#include <cutlass/bfloat16.h>
#include <cutlass/fast_math.h>
#include <cutlass/half.h>
#include <cutlass/integer_subbyte.h>

namespace at::native {

using cutlass::uint1b_t;
using cutlass::uint2b_t;
using cutlass::uint4b_t;
using uint8b_t = cutlass::integer_subbyte<8, false>;
using ReorderedLayoutInputE = cutlass::layout::ColumnMajorInterleaved<2>;
using ElementInputE = uint16_t;
constexpr int kWarpX = 32;
constexpr int kWarpY = 64;
constexpr int kThreadX = 8;
constexpr int kThreadY = 8;

// bitmask of selected values, in col-major storage
// eg: indices & (1 << (col + 4 * row))
using Indices4x4 = uint16_t;

struct Tile8x8Masks {
  Indices4x4 a, b, c, d;
  CUTLASS_DEVICE Tile8x8Masks() {
    a = b = c = d = 0;
  }
};

static_assert(sizeof(Tile8x8Masks) == 8, "should be exactly uint64_t");

// Each thread has data for an 8x8 area of the input tensor
// Due to the very specific format of the metadata, 32 consecutive bits
// of the metadata tensor will live in 4 different threads.
// This functions does the required warp shuffling to send data to the
// right threads.
// This took some time to write (and get right), hopefully these slides
// can help
// https://docs.google.com/presentation/d/1DtmKThv8S5QAyBktuLRYzZhRzCvS1qSkBbrqNCjMPeA/edit#slide=id.g249eb2e2f2e_0_28
CUTLASS_DEVICE uint32_t
warp_shuffle_meta(uint32_t meta_ab, bool transposed = false) {
  // The required format is
  // (one line = 32 bits)
  // a[ 0,  0:16] a[ 8,  0:16] <- T0 [left]
  // a[ 0, 16:32] a[ 8, 16:32]
  // a[16,  0:16] a[24,  0:16]
  // a[16, 16:32] a[24, 16:32]
  // a[ 1,  0:16] a[ 9,  0:16] <- T4
  // a[ 1, 16:32] a[ 9, 16:32]
  // a[17,  0:16] a[25,  0:16]
  // a[17, 16:32] a[25, 16:32]
  // a[ 2,  0:16] a[10,  0:16] <- T1 [left, bottom]
  // a[ 2, 16:32] a[10, 16:32]
  // a[18,  0:16] a[26,  0:16]
  // a[18, 16:32] a[26, 16:32]
  // a[ 3,  0:16] a[11,  0:16] <- T5 [bottom]
  // a[ 3, 16:32] a[11, 16:32]
  // a[19,  0:16] a[27,  0:16]
  // a[19, 16:32] a[27, 16:32]
  // ...
  // Use warp-shuffles to send data around threads
  bool thread_left = (threadIdx.y % 2) == 0;
  bool thread_bottom = threadIdx.x % 2;

  if (transposed) {
    thread_left = (threadIdx.x % 2) == 0;
    thread_bottom = threadIdx.y % 2;
  }

  uint8b_t stage0_data[2] = {
      uint8b_t(meta_ab >> (8 * thread_left)),
      uint8b_t(meta_ab >> (8 * (thread_left + 2)))};
  // shfl t0-t4 / t1-t5
  stage0_data[0] =
      __shfl_xor_sync(0xffffffff, stage0_data[0], transposed ? 1 : 4);
  stage0_data[1] =
      __shfl_xor_sync(0xffffffff, stage0_data[1], transposed ? 1 : 4);

  uint16_t line0 = int(uint8b_t(meta_ab >> (8 * (1 - thread_left))))
      << ((1 - thread_left) * 8);
  line0 |= int(stage0_data[0]) << (thread_left * 8);
  uint16_t line1 = int(uint8b_t(meta_ab >> (8 * (1 - thread_left + 2))))
      << ((1 - thread_left) * 8);
  line1 |= int(stage0_data[1]) << (thread_left * 8);

  uint16_t stage1_data = thread_bottom ? line0 : line1;
  stage1_data = __shfl_xor_sync(0xffffffff, stage1_data, transposed ? 4 : 1);

  uint32_t final_metadata;
  if (thread_bottom) {
    final_metadata = uint32_t(stage1_data) | uint32_t(line1) << 16;
  } else {
    final_metadata = uint32_t(stage1_data) << 16 | uint32_t(line0);
  }
  return final_metadata;
}

CUTLASS_DEVICE void warp_shuffle_and_write_meta(
    ElementInputE* metadata_quad,
    uint32_t meta_ab,
    bool transposed = false) {
  bool thread_left = (threadIdx.y % 2) == 0;
  bool thread_bottom = threadIdx.x % 2;

  if (transposed) {
    thread_left = (threadIdx.x % 2) == 0;
    thread_bottom = threadIdx.y % 2;
  }

  uint32_t final_metadata = warp_shuffle_meta(meta_ab, transposed);

  int index = (!thread_left + 2 * thread_bottom) * 4;
  ((uint32_t*)metadata_quad)[index] = final_metadata;
}

template <typename Element_>
struct KernelTypes {
  using Element = Element_;
  using Fragment =
      cutlass::Array<Element, 8>; // always read from gmem in chunks of 128bits
  using Fragment4 = cutlass::Array<Element, 4>;
  using ValuesPacked = cutlass::Array<Element, 8>; // 4 first col, 4 second col

  struct Params {
    /// inputs
    Element const* input;
    int64_t input_s0;
    int64_t input_dim0;
    int64_t input_dim1;

    /// outputs
    Element* packed;
    int64_t packed_stride;

    Element* packed_trans;
    int64_t packed_trans_stride;

    uint64_t* threads_masks;

    __host__ dim3 getBlocksGrid() const {
      return dim3(
          cutlass::ceil_div(input_dim0, kWarpX),
          cutlass::ceil_div(input_dim1, kWarpY),
          1);
    }

    static CUTLASS_HOST_DEVICE dim3 getThreadsGrid() {
      return dim3(kWarpX / kThreadX, kWarpY / kThreadY, 1);
    }

    CUTLASS_DEVICE Tile8x8Masks* getCurrentThreadIndices() const {
      Tile8x8Masks* gmem_threads_masks = (Tile8x8Masks*)threads_masks;
      gmem_threads_masks += blockIdx.y * getThreadsGrid().y + threadIdx.y;
      int64_t strideX = gridDim.y * getThreadsGrid().y;
      gmem_threads_masks +=
          (blockIdx.x * getThreadsGrid().x + threadIdx.x) * strideX;
      return gmem_threads_masks;
    }
  };

  struct Tile4x4Accessor {
    using Element = Element_;

    Fragment (&_lines)[8];
    int _start_row;
    int _start_col;

    CUTLASS_DEVICE Tile4x4Accessor(
        Fragment (&lines)[8],
        int start_row,
        int start_col)
        : _lines(lines), _start_row(start_row), _start_col(start_col) {}

    CUTLASS_DEVICE typename Fragment::reference at(int r, int c) {
      return _lines[r + _start_row][c + _start_col];
    }
  };

  struct Tile4x4Packed {
    Fragment4 values[2];
    CUTLASS_DEVICE Tile4x4Packed() {
      values[0].clear();
      values[1].clear();
    }
  };

  // Returns a packed 4x4 tile (eg 2x4 values) which correspond to the values
  // that are in `indices`. Also fills the `meta` array in the right format
  // for consumption in the TensorCores.
  // Example:
  //  indices:  0011
  //            1001
  //            1001
  //            0100 (<- note, only 1 value on the last line)
  //  packed: values[0][2] values[1][0] values[2][0] values[3][1]
  //          values[0][3] values[1][3] values[2][3] Element(0)
  CUTLASS_DEVICE static Tile4x4Packed pack_4x4(
      Indices4x4 indices,
      Tile4x4Accessor tile,
      uint32_t& meta,
      int meta_pos,
      bool transpose = false) {
    Tile4x4Packed packed;
    CUTLASS_PRAGMA_UNROLL
    for (int row = 0; row < 4; ++row) {
      uint2b_t col0_from, col1_from;
      auto packValue = [&](uint2b_t col_to, uint2b_t col_from) {
        auto value = transpose ? tile.at(col_from, row).get()
                               : tile.at(row, col_from).get();
        packed.values[col_to][row] = value;
        if (col_to == uint2b_t(0)) {
          col0_from = col_from;
        } else {
          col1_from = col_from;
        }
      };
      auto isSelected = [&](int col) {
        if (transpose) {
          return indices & (1 << (row + 4 * col));
        }
        return indices & (1 << (col + 4 * row));
      };
      // Process cols 0/1
      // We know that col0 is always packed to position 0 if it's there
      // and col1 is packed to pos 0 or 1 (depending if col0 is selected)
      if (isSelected(1)) {
        packValue(0, 1);
      }
      if (isSelected(0)) {
        packValue(0, 0);
      }
      if (isSelected(0) && isSelected(1)) {
        packValue(1, 1);
      }
      // Process cols 2/3
      // same sort of heuristic
      if (isSelected(2)) {
        packValue(1, 2);
      }
      if (isSelected(3)) {
        packValue(1, 3);
      }
      if (isSelected(2) && isSelected(3)) {
        packValue(0, 2);
      }
      int add_mask = (col0_from | (col1_from << 2)) << (8 * row + meta_pos);
      meta |= add_mask;
    }
    return packed;
  }

  struct Tile8x8Meta {
    // meta_ab[row] |= (real_col << (8*row + 2*pos))
    uint32_t meta_ab;
    uint32_t meta_cd;

    // meta_ac_trans[col] |= (real_row << (8*col + 2*pos))
    uint32_t meta_ac_trans;
    uint32_t meta_bd_trans;

    CUTLASS_DEVICE Tile8x8Meta() {
      meta_ab = meta_cd = meta_ac_trans = meta_bd_trans = 0;
    }
  };

  CUTLASS_DEVICE static void writePacked(
      Element* ptr,
      Fragment4 packed0,
      Fragment4 packed1) {
    Fragment write;
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < 4; ++i) {
      write[i] = packed0[i].get();
      write[i + 4] = packed1[i].get();
    }
    cutlass::arch::global_store<Fragment, sizeof(Fragment)>(write, ptr, true);
  }

  CUTLASS_DEVICE static void writePackedT(
      Element* ptr,
      int64_t stride,
      Tile4x4Packed a,
      Tile4x4Packed b) {
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < 4; ++i) {
      Fragment4 write;
      write[0] = a.values[0][i].get();
      write[1] = a.values[1][i].get();
      write[2] = b.values[0][i].get();
      write[3] = b.values[1][i].get();
      cutlass::arch::global_store<Fragment4, sizeof(Fragment4)>(
          write, ptr + i * stride, true);
    }
  }

  template <typename Algorithm, typename MetadataStore>
  CUTLASS_DEVICE static void sparse_semi_structured_tile_kernel(
      Params p,
      MetadataStore metadata_gmem,
      Algorithm compute_tile_indices) {
    // Each thread is responsible for an 8x8 tile, which contains 4 4x4 tiles:
    // A, B, C and D, as displayed in the following schema:
    // +---+---+
    // | A | B |
    // +---+---+
    // | C | D |
    // +---+---+
    // Each warp (32 threads) will then be responsible for a 32x64 tile of the
    // input.
    // This configuration allows to read/write data in 128bits chunks. These
    // memory accesses are coalesced at the warp-level into 128bytes. See also:
    // https://docs.google.com/presentation/d/1DtmKThv8S5QAyBktuLRYzZhRzCvS1qSkBbrqNCjMPeA/edit#slide=id.g2494f30c7cf_0_0

    // Top-left of the 8x8 tile we own
    int warp_x = blockIdx.x * kWarpX;
    int warp_y = blockIdx.y * kWarpY;
    int x = warp_x + threadIdx.x * kThreadX;
    int y = warp_y + threadIdx.y * kThreadY;

    Element const* input = p.input + x * p.input_s0 + y;
    Element* packed = p.packed + x * p.packed_stride + (y / 2);
    Element* packed_trans =
        p.packed_trans + (x / 2) + y * p.packed_trans_stride;

    Fragment lines[8]; // Contains all values from the 8x8 tile

    Tile8x8Meta metadata;
    Tile8x8Masks indices;

    // Load/process tiles `A` and `B`
    Element fillValue = Algorithm::template outOfBoundsFillValue<Element>();
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < 4; ++i) {
      lines[i].fill(fillValue);
      cutlass::arch::global_load<Fragment, sizeof(Fragment)>(
          lines[i], input + i * p.input_s0, x + i < p.input_dim0);
    }
    indices.a = compute_tile_indices(Tile4x4Accessor(lines, 0, 0));
    indices.b = compute_tile_indices(Tile4x4Accessor(lines, 0, 4));

    // Compute packed tiles A & B
    {
      Tile4x4Packed packed_a = pack_4x4(
          indices.a, Tile4x4Accessor(lines, 0, 0), metadata.meta_ab, 0);
      Tile4x4Packed packed_b = pack_4x4(
          indices.b, Tile4x4Accessor(lines, 0, 4), metadata.meta_ab, 4);
      writePackedT(packed, p.packed_stride, packed_a, packed_b);
    }

    // Compute/store packed tiles A & B in transpose output
    Tile4x4Packed packed_trans_a = pack_4x4(
        indices.a,
        Tile4x4Accessor(lines, 0, 0),
        metadata.meta_ac_trans,
        0,
        true);
    Tile4x4Packed packed_trans_b = pack_4x4(
        indices.b,
        Tile4x4Accessor(lines, 0, 4),
        metadata.meta_bd_trans,
        0,
        true);
    // (NOTE) Now we no longer need A & B (`lines[0:4]`)

    // Load/process tiles `C` and `D`
    CUTLASS_PRAGMA_UNROLL
    for (int i = 4; i < 8; ++i) {
      lines[i].fill(fillValue);
      cutlass::arch::global_load<Fragment, sizeof(Fragment)>(
          lines[i], input + i * p.input_s0, x + i < p.input_dim0);
    }
    indices.c = compute_tile_indices(Tile4x4Accessor(lines, 4, 0));
    indices.d = compute_tile_indices(Tile4x4Accessor(lines, 4, 4));

    // Compute packed tiles C & D
    {
      Tile4x4Packed packed_c = pack_4x4(
          indices.c, Tile4x4Accessor(lines, 4, 0), metadata.meta_cd, 0);
      Tile4x4Packed packed_d = pack_4x4(
          indices.d, Tile4x4Accessor(lines, 4, 4), metadata.meta_cd, 4);
      writePackedT(
          packed + 4 * p.packed_stride, p.packed_stride, packed_c, packed_d);
    }

    // Compute/store packed tiles C & D in transpose output
    Tile4x4Packed packed_trans_c = pack_4x4(
        indices.c,
        Tile4x4Accessor(lines, 4, 0),
        metadata.meta_ac_trans,
        4,
        true);
    Tile4x4Packed packed_trans_d = pack_4x4(
        indices.d,
        Tile4x4Accessor(lines, 4, 4),
        metadata.meta_bd_trans,
        4,
        true);

    // Dump the metadata in a nice format
    *p.getCurrentThreadIndices() = indices;

    // Store packed A, B, C & D for transposed matrix
    writePackedT(
        packed_trans, p.packed_trans_stride, packed_trans_a, packed_trans_c);
    packed_trans += 4 * p.packed_trans_stride;
    writePackedT(
        packed_trans, p.packed_trans_stride, packed_trans_b, packed_trans_d);

    // Writing meta non-transposed
    {
      ElementInputE* packed_meta_reordered = metadata_gmem.get_metaN(
          warp_x, threadIdx.x * kThreadX, warp_y, threadIdx.y * kThreadY);
      warp_shuffle_and_write_meta(packed_meta_reordered, metadata.meta_ab);
      warp_shuffle_and_write_meta(packed_meta_reordered + 32, metadata.meta_cd);
    }

    // Writing meta transposed
    {
      ElementInputE* packed_trans_meta_reordered = metadata_gmem.get_metaT(
          warp_x, threadIdx.x * kThreadX, warp_y, threadIdx.y * kThreadY);
      warp_shuffle_and_write_meta(
          packed_trans_meta_reordered, metadata.meta_ac_trans, true);
      warp_shuffle_and_write_meta(
          packed_trans_meta_reordered + 32, metadata.meta_bd_trans, true);
    }
  }

  CUTLASS_DEVICE static void sparse_semi_structured_apply_kernel(Params p) {
    // See `sparse24_sparsify_both_ways_kernel`
    // It's basically the same, just that we skip
    // the part where compute the indices we keep

    // Top-left of the 8x8 tile we own
    int warp_x = blockIdx.x * kWarpX;
    int warp_y = blockIdx.y * kWarpY;
    int x = warp_x + threadIdx.x * kThreadX;
    int y = warp_y + threadIdx.y * kThreadY;

    Element const* input = p.input + x * p.input_s0 + y;
    Element* packed = p.packed + x * p.packed_stride + (y / 2);
    Element* packed_trans =
        p.packed_trans + (x / 2) + y * p.packed_trans_stride;

    Fragment lines[8]; // Contains all values from the 8x8 tile

    Tile8x8Meta metadata;
    Tile8x8Masks indices = *p.getCurrentThreadIndices();

    // Load/process tiles `A` and `B`
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < 8; ++i) {
      // NB: Values outside bounds is undefined, but shouldn't
      // be used anywhere
      cutlass::arch::global_load<Fragment, sizeof(Fragment)>(
          lines[i], input + i * p.input_s0, x + i < p.input_dim0);
    }

    // Compute packed tiles A & B
    {
      Tile4x4Packed packed_a = pack_4x4(
          indices.a, Tile4x4Accessor(lines, 0, 0), metadata.meta_ab, 0);
      Tile4x4Packed packed_b = pack_4x4(
          indices.b, Tile4x4Accessor(lines, 0, 4), metadata.meta_ab, 4);
      writePackedT(packed, p.packed_stride, packed_a, packed_b);
    }

    // Compute/store packed tiles A & B in transpose output
    Tile4x4Packed packed_trans_a = pack_4x4(
        indices.a,
        Tile4x4Accessor(lines, 0, 0),
        metadata.meta_ac_trans,
        0,
        true);
    Tile4x4Packed packed_trans_b = pack_4x4(
        indices.b,
        Tile4x4Accessor(lines, 0, 4),
        metadata.meta_bd_trans,
        0,
        true);
    // (NOTE) Now we no longer need A & B (`lines[0:4]`)

    // Compute packed tiles C & D
    {
      Tile4x4Packed packed_c = pack_4x4(
          indices.c, Tile4x4Accessor(lines, 4, 0), metadata.meta_cd, 0);
      Tile4x4Packed packed_d = pack_4x4(
          indices.d, Tile4x4Accessor(lines, 4, 4), metadata.meta_cd, 4);
      writePackedT(
          packed + 4 * p.packed_stride, p.packed_stride, packed_c, packed_d);
    }

    // Compute/store packed tiles C & D in transpose output
    Tile4x4Packed packed_trans_c = pack_4x4(
        indices.c,
        Tile4x4Accessor(lines, 4, 0),
        metadata.meta_ac_trans,
        4,
        true);
    Tile4x4Packed packed_trans_d = pack_4x4(
        indices.d,
        Tile4x4Accessor(lines, 4, 4),
        metadata.meta_bd_trans,
        4,
        true);

    // Store packed A, B, C & D for transposed matrix
    writePackedT(
        packed_trans, p.packed_trans_stride, packed_trans_a, packed_trans_c);
    packed_trans += 4 * p.packed_trans_stride;
    writePackedT(
        packed_trans, p.packed_trans_stride, packed_trans_b, packed_trans_d);
  }
};

} // namespace at::native
