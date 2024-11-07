#include <ATen/native/nested/NestedTensorBinaryOps.h>

#include <type_traits>

#include <ATen/ATen.h>
#include <ATen/Dispatch.h>

#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/detail/KernelUtils.h>
#include <ATen/cuda/detail/IndexUtils.cuh>
#include <ATen/native/cuda/Loops.cuh>
#include <ATen/native/cuda/MemoryAccess.cuh>

#include <c10/cuda/CUDAMathCompat.h>
#include <c10/cuda/CUDAStream.h>


#include <ATen/native/nested/NestedTensorUtils.h>

#define BLOCK_DIM 256

namespace at {
namespace native {


// only for nested [B, *, D], dense [B, 1, D]
template <typename T, typename func_t>
__global__ void op_dense_esuhm(
    const T* input,
    const T* dense,
    T* output,
    int64_t embedding_dim,
    const int64_t* offsets,
    const func_t& f)
{
  // each batch is handled by a block
  const int64_t batch_idx  = blockIdx.x;
  const int64_t grain_size = blockDim.x;
  const int64_t tid = threadIdx.x;
  const int64_t range = offsets[batch_idx + 1] - offsets[batch_idx];
  // each thread handles (embedding_dim // grain_size + (embedding_dim % grain_size <= tid)) elems
  // of the dense embedding
  for (int64_t idx = tid; idx < embedding_dim; idx += grain_size) {
    const T dense_elem = dense[batch_idx * embedding_dim + idx];
    for (int64_t nested_idx = idx; nested_idx < range; nested_idx += embedding_dim) {
      output[offsets[batch_idx] + nested_idx] = f(input[offsets[batch_idx] + nested_idx], dense_elem);
    }
  }
}

template <typename T, typename func_t>
void nested_op_dense_kernelLauncher(
    const T* input, // [sum(*) x embedding_dim]
    const T* dense, // [batch_size x embedding_dim]
    T* output, // [sum(*) x embedding_dim]
    int64_t batch_size,
    int64_t embedding_dim,
    const int64_t* input_offsets,  // [batch_size]
    func_t f)
{
  dim3 grid;
  grid.x = batch_size;
  const auto stream = at::cuda::getCurrentCUDAStream();

  op_dense_esuhm<<<grid, BLOCK_DIM, 0, stream>>>(
      input,
      dense,
      output,
      embedding_dim,
      input_offsets,
      f);
}

template <typename scalar_t, typename func_t>
void _nested_op_dense_esuhm_kernel(Tensor& result, const Tensor& self, const Tensor& other, func_t f) {
  auto self_ptr = get_nested_tensor_impl(self);
  auto result_ptr = get_nested_tensor_impl(result);

  const auto self_buffer = self_ptr->get_buffer();
  const auto offsets = self_ptr->get_storage_offsets();
  const auto batch_size = other.size(0);
  const auto embedding_size = other.size(2);

  auto result_buffer = result_ptr->get_buffer();
  auto result_offsets = at::cat({offsets, at::tensor(self_ptr->numel())});
  result_offsets = result_offsets.to(kCUDA);

  const scalar_t* self_data_ptr = self_buffer.const_data_ptr<scalar_t>();
  const scalar_t* other_data_ptr = other.const_data_ptr<scalar_t>();
  scalar_t* result_data_ptr = result_buffer.data_ptr<scalar_t>();
  int64_t* result_offsets_ptr = result_offsets.data_ptr<int64_t>();

  nested_op_dense_kernelLauncher(
    self_data_ptr,
    other_data_ptr,
    result_data_ptr,
    batch_size,
    embedding_size,
    result_offsets_ptr,
    f);
}

void _nested_op_dense_esuhm_cuda(Tensor& result, const Tensor& self, const Tensor& other, const NESTED_DENSE_OP& op) {
  AT_DISPATCH_ALL_TYPES_AND2(
    ScalarType::Half, ScalarType::BFloat16, self.scalar_type(), "_nested_op_dense_esuhm", [&]() {
    switch (op) {
      case NESTED_DENSE_OP::ADD :
        _nested_op_dense_esuhm_kernel<scalar_t>(result, self, other, [] __host__ __device__ (scalar_t a, scalar_t b) -> scalar_t { return a + b; });
        break;
      case NESTED_DENSE_OP::MUL :
        _nested_op_dense_esuhm_kernel<scalar_t>(result, self, other, [] __host__ __device__ (scalar_t a, scalar_t b) -> scalar_t { return a * b; });
        break;
    }
  });
}

REGISTER_CUDA_DISPATCH(nested_dense_elementwise_stub, &_nested_op_dense_esuhm_cuda)

} // namespace native
} // namespace at
