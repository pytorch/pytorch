#define TORCH_ASSERT_NO_OPERATORS
#include <ATen/native/sparse/SparseFactories.h>

#include <ATen/Dispatch.h>
#include <ATen/TensorIterator.h>
#include <ATen/core/TensorBase.h>
#include <ATen/native/cpu/Loops.h>
#include <c10/core/ScalarType.h>
#include <c10/util/Exception.h>

namespace at::native {

namespace {
void _spdiags_kernel_cpu(
    TensorIterator& iter,
    const TensorBase& diagonals,
    TensorBase& values,
    TensorBase& indices) {
  auto* row_index_write_ptr = indices.data_ptr<int64_t>();
  auto* col_index_write_ptr = row_index_write_ptr ? row_index_write_ptr + indices.stride(0) : nullptr;
  const int64_t diagonals_index_stride = diagonals.stride(0);
  const int64_t diagonals_read_stride = diagonals.stride(1);
  AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND4(
      at::ScalarType::BFloat16,
      at::ScalarType::Half,
      at::ScalarType::Bool,
      at::ScalarType::ComplexHalf,
      diagonals.scalar_type(),
      "spdiags_cpu",
      [&] {
        auto* const values_write_ptr = values.data_ptr<scalar_t>();
        const auto* const diagonals_ptr = diagonals.const_data_ptr<scalar_t>();

        cpu_kernel(
            iter,
            [&](int64_t diag_index,
                int64_t diag_offset,
                int64_t out_offset,
                int64_t n_out) -> int64_t {
              if (n_out > 0) {
                auto* rows_start = row_index_write_ptr + out_offset;
                auto* cols_start = col_index_write_ptr + out_offset;
                auto* vals_start = values_write_ptr + out_offset;
                const int64_t first_col = std::max<int64_t>(diag_offset, 0);
                const int64_t first_row = first_col - diag_offset;
                auto* data_read = (diagonals_ptr +
                                   diagonals_index_stride * diag_index +
                                   first_col * diagonals_read_stride);
                for (int64_t i = 0; i < n_out; ++i) {
                  rows_start[i] = first_row + i;
                  cols_start[i] = first_col + i;
                  vals_start[i] = data_read[i * diagonals_read_stride];
                }
              }
              // dummy return
              return 0;
            });
      });
}

} // namespace

REGISTER_DISPATCH(spdiags_kernel_stub, &_spdiags_kernel_cpu);

} // namespace at::native
