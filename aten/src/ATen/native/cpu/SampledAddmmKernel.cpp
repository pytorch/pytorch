#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/core/Tensor.h>
#include <ATen/ExpandUtils.h>
#include <ATen/Dispatch.h>
#include <ATen/Parallel.h>
#include <ATen/cpu/vec/functional.h>
#include <ATen/cpu/vec/vec.h>
#include <ATen/native/cpu/SampledAddmmKernel.h>
#include <ATen/native/cpu/utils.h>
#include <c10/util/irange.h>

namespace at::native {

namespace {

template <typename scalar_t, typename index_t>
void sampled_addmm_sparse_csr_kernel_impl(
    const Tensor& mat1,
    const Tensor& mat2,
    const Scalar& beta,
    const Scalar& alpha,
    const Tensor& result) {

  int64_t nnz = result._nnz();

  auto beta_ = beta.to<scalar_t>();
  auto alpha_ = alpha.to<scalar_t>();

  const scalar_t* mat1_data = mat1.const_data_ptr<scalar_t>();
  const scalar_t* mat2_data = mat2.const_data_ptr<scalar_t>();

  // mat1: {B, M, K}
  // mat2: {B, N, K}
  // crow: {B, M + 1}
  // col, values: {B, nnz}
  int64_t M = mat1.size(-2);
  int64_t K = mat1.size(-1);
  int64_t N = mat2.size(-2);
  int64_t B = mat1.numel() / M / K;

  auto values = result.values().reshape({-1, nnz});
  auto crow = result.crow_indices().reshape({-1, M + 1});
  auto col = result.col_indices().reshape({-1, nnz});

  auto values_acc = values.accessor<scalar_t, 2>();
  auto crow_acc = crow.accessor<const index_t, 2>();
  auto col_acc = col.accessor<const index_t, 2>();

  // usually, collapse B and M is a better option,
  // but for most commonly used case (mat1 and mat2 is 2d tensor), B = 1,
  // balance partition M by using parallel_sparse_csr.
  using Vec = vec::Vectorized<scalar_t>;
  for (const auto b : c10::irange(B)) {
    auto crow_slice = crow_acc[b];
    auto col_slice = col_acc[b];
    auto values_slice = values_acc[b];
    const scalar_t* mat1_ptr = mat1_data + b * M * K;
    const scalar_t* mat2_ptr = mat2_data + b * N * K;

    utils::parallel_sparse_csr(crow_slice, M, nnz, [&](int64_t begin, int64_t end) {
      for (const auto m : c10::irange(begin, end)) {
        int64_t row_start = crow_slice[m];
        int64_t row_end = crow_slice[m + 1];
        for (const auto e : c10::irange(row_start, row_end)) {
          int64_t n = col_slice[e];
          scalar_t val = values_slice[e];
          scalar_t dot = vec::map2_reduce_all<scalar_t>(
              [](Vec x, Vec y) { return x * y; },
              [](Vec x, Vec y) { return x + y; },
              mat1_ptr + m * K,
              mat2_ptr + n * K,
              K);
          val = alpha_ * dot + beta_ * val;
          values_slice[e] = val;
        }
      }
    });
  }
}

void sampled_addmm_sparse_csr_kernel(
    const Tensor& mat1,
    const Tensor& mat2,
    const Scalar& beta,
    const Scalar& alpha,
    const Tensor& result) {
  const auto index_type = result.crow_indices().scalar_type();
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(mat1.scalar_type(), "sampled_addmm_sparse_csr_kernel", [&]() {
    AT_DISPATCH_INDEX_TYPES(index_type, "sampled_addmm_sparse_csr_index", [&]() {
      sampled_addmm_sparse_csr_kernel_impl<scalar_t, index_t>(mat1, mat2, beta, alpha, result);
    });
  });
}

} // anonymous namespace

REGISTER_DISPATCH(sampled_addmm_sparse_csr_stub, &sampled_addmm_sparse_csr_kernel);

} // at::native
