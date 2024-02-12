#include <ATen/ScalarOps.h>
#include <ATen/Tensor.h>
#include <cutlass/array.h>
#include <cutlass/bfloat16.h>
#include <cutlass/coord.h>
#include <cutlass/gemm/gemm.h>
#include <cutlass/half.h>
#include <cutlass/layout/matrix.h>
#include <cutlass/tensor_ref.h>
#include <cutlass/tensor_view.h>
#include <torch/library.h>
#include "static_sort.h"
#include <ATen/Functions.h>

namespace {
// This is for 2:4 f16
using ElementInputE = uint16_t;
using LayoutInputE = cutlass::layout::RowMajor;
using ReorderedLayoutInputE = cutlass::layout::ColumnMajorInterleaved<2>;

using RefInp = typename cutlass::TensorRef<ElementInputE, LayoutInputE>;
using RefReordered =
    typename cutlass::TensorRef<ElementInputE, ReorderedLayoutInputE>;

at::Tensor _sparse24_pack_mask(const at::Tensor input) {
  TORCH_CHECK(input.is_contiguous(), "Expected contiguous tensor");
  TORCH_CHECK(input.dim() == 2, "Expected 2d tensor");
  TORCH_CHECK(
      input.size(0) % 32 == 0 && input.size(1) % 32 == 0,
      "Wrong dim, should be dividable by 32");
  TORCH_CHECK(
      input.scalar_type() == at::ScalarType::Bool, "Expected bool Tensor");

  at::Tensor packed = at::empty(
      {input.size(0), input.size(1) / 16},
      input.options().dtype(at::ScalarType::Short));
  auto input_a = input.accessor<bool, 2>();
  auto packed_a = packed.accessor<int16_t, 2>();
  for (int row = 0; row < input.size(0); ++row) {
    for (int col_s = 0; col_s < input.size(1); col_s += 16) {
      ElementInputE out = 0;
      for (int bit_shifts = 0; bit_shifts < 16; bit_shifts += 4) {
        int first_pos = -1;
        int second_pos = -1;
        for (int i = 0; i < 4; ++i) {
          if (input_a[row][col_s + bit_shifts + i]) {
            if (first_pos == -1) {
              first_pos = i;
            } else if (second_pos == -1) {
              second_pos = i;
            } else {
              TORCH_CHECK(
                  second_pos != -1,
                  "Invalid mask at (",
                  row,
                  ", ",
                  col_s + bit_shifts,
                  "): too many values");
            }
          }
        }
        TORCH_CHECK(
            second_pos != -1,
            "Invalid mask at (",
            row,
            ", ",
            col_s + bit_shifts,
            "): not enough values");
        out |= (first_pos | (second_pos * 4)) << bit_shifts;
      }
      packed_a[row][col_s / 16] = out;
    }
  }
  return packed;
}

// Taken from <cutlass/tools/util/include/cutlass/util/host_reorder.h>
// Can't include it directly as we have compilation errors...
template <typename Element, typename LayoutDest, typename LayoutSrc>
void reorder_meta(
    cutlass::TensorRef<Element, LayoutDest> dest,
    cutlass::TensorRef<Element, LayoutSrc> src,
    cutlass::gemm::GemmCoord problem_size) {
  for (int m = 0; m < problem_size.m(); m++) {
    for (int k = 0; k < problem_size.k(); k++) {
      // First reorder the rows.
      int group = (sizeof(Element) == 2) ? 32 : 16;
      int interweave = (sizeof(Element) == 2) ? 4 : 2;

      int dest_row = m / group * group + (m % 8) * interweave + (m % group) / 8;
      int dest_col = k;

      // Next swizzle the 2x2 blocks from Z to N.
      if (((dest_row % 2) == 0) && ((dest_col % 2) == 1)) {
        ++dest_row;
        --dest_col;
      } else if (((dest_row % 2) == 1) && ((dest_col % 2) == 0)) {
        --dest_row;
        ++dest_col;
      }

      dest.at({dest_row, dest_col}) = src.at({m, k});
    }
  }
}

at::Tensor _sparse24_reorder_meta(at::Tensor input) {
  TORCH_CHECK(input.dim() == 2, "Expected 2d tensor");
  TORCH_CHECK(input.size(0) % 32 == 0, "Wrong dim0");
  TORCH_CHECK(input.size(1) % 2 == 0, "Wrong dim1");
  TORCH_CHECK(
      input.scalar_type() == at::ScalarType::Short, "Expected int16 tensor");
  input = input.contiguous();
  cutlass::gemm::GemmCoord problem_size(input.size(0), 0, input.size(1));

  cutlass::MatrixCoord meta_dim{input.size(0), input.size(1)};
  auto reordered_layout = ReorderedLayoutInputE::packed(meta_dim);
  at::Tensor reordered =
      at::empty({reordered_layout.capacity(meta_dim)}, input.options());

  RefInp ref_inp{(uint16_t*)input.data_ptr(), LayoutInputE(input.stride(0))};
  RefReordered ref_reordered{(uint16_t*)reordered.data_ptr(), reordered_layout};

  reorder_meta(ref_reordered, ref_inp, problem_size);
  return reordered.view({input.size(1) / 2, input.size(0), 2})
      .permute({1, 2, 0});
}

at::Tensor _sparse24_pack_tensor_according_to_mask(
    at::Tensor a,
    at::Tensor meta_reordered) {
  TORCH_CHECK(a.dim() == 2, "Expected 2d tensor");
  TORCH_CHECK(a.size(0) % 32 == 0, "Wrong dim0");
  TORCH_CHECK(a.size(1) % 4 == 0, "Wrong dim1");
  TORCH_CHECK(
      meta_reordered.dim() == 3, "Expected meta to be reordered already");

  at::Tensor a_packed = at::empty({a.size(0), a.size(1) / 2}, a.options());
  cutlass::MatrixCoord meta_dim{
      meta_reordered.size(0), meta_reordered.size(1) * meta_reordered.size(2)};
  auto reordered_layout = ReorderedLayoutInputE::packed(meta_dim);
  at::Tensor reordered =
      at::empty({reordered_layout.capacity(meta_dim)}, a.options());
  RefReordered ref_meta_reordered{
      (uint16_t*)meta_reordered.data_ptr(), reordered_layout};
  RefInp ref_a{(uint16_t*)a.data_ptr(), LayoutInputE(a.stride(0))};
  RefInp ref_a_packed{
      (uint16_t*)a_packed.data_ptr(), LayoutInputE(a_packed.stride(0))};

  for (int m = 0; m < a.size(0); m++) {
    for (int k = 0; k < a.size(1) / 16; k++) {
      // First reorder the rows.
      int group = (sizeof(ElementInputE) == 2) ? 32 : 16;
      int interweave = (sizeof(ElementInputE) == 2) ? 4 : 2;

      int dest_row = m / group * group + (m % 8) * interweave + (m % group) / 8;
      int dest_col = k;

      // Next swizzle the 2x2 blocks from Z to N.
      if (((dest_row % 2) == 0) && ((dest_col % 2) == 1)) {
        ++dest_row;
        --dest_col;
      } else if (((dest_row % 2) == 1) && ((dest_col % 2) == 0)) {
        --dest_row;
        ++dest_col;
      }

      uint16_t pack_info = ref_meta_reordered.at({dest_row, dest_col});
      // For each group of 4, read the only 2 that are selected in the mask
      for (int group_shift = 0; group_shift < 16; group_shift += 4) {
        uint16_t element = 0;
        int pos0 = (pack_info >> group_shift) & 3;
        int pos1 = (pack_info >> (group_shift + 2)) & 3;
        ref_a_packed.at({m, 8 * k + group_shift / 2}) =
            ref_a.at({m, 16 * k + group_shift + pos0});
        ref_a_packed.at({m, 8 * k + group_shift / 2 + 1}) =
            ref_a.at({m, 16 * k + group_shift + pos1});
      }
    }
  }
  return a_packed;
}
} // namespace

TORCH_LIBRARY_IMPL(sparse, CPU, m) {
  m.impl(
      TORCH_SELECTIVE_NAME("sparse::_sparse24_pack_mask"),
      TORCH_FN(_sparse24_pack_mask));
  m.impl(
      TORCH_SELECTIVE_NAME("sparse::_sparse24_reorder_meta"),
      TORCH_FN(_sparse24_reorder_meta));
  m.impl(
      TORCH_SELECTIVE_NAME("sparse::_sparse24_pack_tensor_according_to_mask"),
      TORCH_FN(_sparse24_pack_tensor_according_to_mask));
}
