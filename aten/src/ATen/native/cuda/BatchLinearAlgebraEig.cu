#define TORCH_ASSERT_ONLY_METHOD_OPERATORS

#include <ATen/core/Tensor.h>
#include <ATen/Dispatch_v2.h>
#include <ATen/native/BatchLinearAlgebra.h>
#include <ATen/native/LinearAlgebraUtils.h>
#include <c10/cuda/CUDAGuard.h>

namespace at::native {

namespace {

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ linalg_eig_make_complex_eigenvectors ~~~~~~~~~~~~~~~~~~~~~~~

// Processes all columns in parallel. For complex conjugate pairs, each thread
// reads from neighboring columns but writes only to its own column.
template <typename scalar_t>
__global__ void linalg_eig_make_complex_eigenvectors_kernel(
    c10::complex<scalar_t>* __restrict__ result,
    const c10::complex<scalar_t>* __restrict__ eigenvalues,
    const scalar_t* __restrict__ vectors,
    const int64_t batch_size,
    const int64_t n,
    const int64_t matrix_stride) {

  const int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int64_t total_elements = batch_size * n * n;

  if (idx >= total_elements) return;

  const int64_t batch_idx = idx / (n * n);
  const int64_t local_idx = idx % (n * n);
  const int64_t col = local_idx / n;
  const int64_t row = local_idx % n;

  const auto* batch_eigenvalues = eigenvalues + batch_idx * n;
  const auto* batch_vectors = vectors + batch_idx * matrix_stride;
  auto* batch_result = result + batch_idx * matrix_stride;

  const auto eigenvalue = batch_eigenvalues[col];

  if (eigenvalue.imag() == scalar_t(0)) {
    batch_result[col * n + row] = c10::complex<scalar_t>(
        batch_vectors[col * n + row],
        scalar_t(0));
  } else if (eigenvalue.imag() > scalar_t(0)) {
    batch_result[col * n + row] = c10::complex<scalar_t>(
        batch_vectors[col * n + row],
        batch_vectors[(col + 1) * n + row]);
  } else {
    batch_result[col * n + row] = c10::complex<scalar_t>(
        batch_vectors[(col - 1) * n + row],
        -batch_vectors[col * n + row]);
  }
}

template <typename scalar_t>
void linalg_eig_make_complex_eigenvectors_cuda_impl(
    const Tensor& complex_vectors,
    const Tensor& complex_values,
    const Tensor& real_vectors) {

  const auto n = real_vectors.size(-1);
  const auto matrix_stride = matrixStride(real_vectors);
  const auto batch_size = batchCount(real_vectors);

  if (batch_size == 0 || n == 0) return;

  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(complex_vectors.mT().is_contiguous());
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(complex_values.is_contiguous());
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(real_vectors.mT().is_contiguous());

  const int64_t total_elements = batch_size * n * n;

  const int threads = 256;
  const int blocks = (total_elements + threads - 1) / threads;

  auto* result_ptr = complex_vectors.data_ptr<c10::complex<scalar_t>>();
  const auto* eigenvalues_ptr = complex_values.const_data_ptr<c10::complex<scalar_t>>();
  const auto* vectors_ptr = real_vectors.const_data_ptr<scalar_t>();

  linalg_eig_make_complex_eigenvectors_kernel<scalar_t>
      <<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
          result_ptr,
          eigenvalues_ptr,
          vectors_ptr,
          batch_size,
          n,
          matrix_stride);

  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void linalg_eig_make_complex_eigenvectors_cuda(
    const Tensor& complex_vectors,
    const Tensor& complex_values,
    const Tensor& real_vectors) {

  TORCH_INTERNAL_ASSERT(complex_vectors.is_cuda());
  TORCH_INTERNAL_ASSERT(complex_values.is_cuda());
  TORCH_INTERNAL_ASSERT(real_vectors.is_cuda());

  c10::cuda::CUDAGuard device_guard(real_vectors.device());

  AT_DISPATCH_V2(
      real_vectors.scalar_type(),
      "linalg_eig_make_complex_eigenvectors_cuda",
      AT_WRAP([&] {
        linalg_eig_make_complex_eigenvectors_cuda_impl<scalar_t>(
            complex_vectors, complex_values, real_vectors);
      }),
      AT_EXPAND(AT_FLOATING_TYPES));
}

} // anonymous namespace

REGISTER_CUDA_DISPATCH(linalg_eig_make_complex_eigenvectors_stub, &linalg_eig_make_complex_eigenvectors_cuda)

} // namespace at::native
