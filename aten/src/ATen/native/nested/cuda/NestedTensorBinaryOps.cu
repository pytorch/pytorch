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


#include <ATen/native/nested/NestedTensorMath.h>
#include <ATen/native/nested/NestedTensorUtils.h>

#define BLOCK_DIM 256

namespace at {
namespace native {


// only for nested [B, *, D], dense [B, 1, D]
template <typename T>
__global__ void add_dense_esuhm(
    const T* input,
    const T* dense,
    T* output,
    int64_t embedding_dim,
    const int64_t* offsets)
{
  // each batch is handled by a block
  const int64_t batch_idx  = blockIdx.x;
  const int64_t grain_size = blockDim.x;
  const int64_t tid = threadIdx.x;
  const int64_t range = offsets[batch_idx + 1] - offsets[batch_idx];
  // each thread handles (embedding_dim // grain_size + (embedding_dim % grain_size != 0)) elems
  // of the dense embedding
  for (int64_t idx = tid; idx < embedding_dim; idx += grain_size) {
    const T dense_elem = dense[batch_idx * embedding_dim + idx];
    for (int64_t nested_idx = idx; nested_idx < range; nested_idx += embedding_dim) {
      output[offsets[batch_idx] + nested_idx] = input[offsets[batch_idx] + nested_idx] + dense_elem;
    }
  }
}

template <typename T>
void nested_add_dense_kernelLauncher(
    const T* input, // [sum(*) x embedding_dim]
    const T* dense, // [batch_size x embedding_dim]
    T* output, // [sum(*) x embedding_dim]
    int64_t batch_size,
    int64_t embedding_dim,
    const int64_t* input_offsets /* [batch_size] */)
{
  dim3 grid;
  grid.x = batch_size;
  const auto stream = at::cuda::getDefaultCUDAStream();

  add_dense_esuhm<<<grid, BLOCK_DIM, 0, stream>>>(
      input,
      dense,
      output,
      embedding_dim,
      input_offsets);
}

Tensor _nested_add_dense_esuhm(const Tensor& self, const Tensor& other) {
  auto self_ptr = get_nested_tensor_impl(self);
  if (!nested_tensor_impl_is_contiguous(self_ptr)) {
    self_ptr = get_nested_tensor_impl(self.contiguous());
  }

  const auto self_buffer = self_ptr->get_buffer();
  const auto offsets = self_ptr->get_storage_offsets();
  const auto batch_size = other.size(0);
  const auto embedding_size = other.size(2);

  auto result_buffer = self_buffer.clone();
  auto result_offsets = at::cat({at::tensor(offsets), at::tensor(self_ptr->numel())});
  result_offsets = result_offsets.to(kCUDA);

  AT_DISPATCH_ALL_TYPES_AND2(
    ScalarType::Half, ScalarType::BFloat16, self_buffer.scalar_type(), "NestedTensor_elementwise_Tensor", [&]() {
    const scalar_t* self_data_ptr = self_buffer.data_ptr<scalar_t>();
    const scalar_t* other_data_ptr = other.data_ptr<scalar_t>();
    scalar_t* result_data_ptr = result_buffer.data_ptr<scalar_t>();
    int64_t* result_offsets_ptr = result_offsets.data_ptr<int64_t>();

    nested_add_dense_kernelLauncher(
      self_data_ptr,
      other_data_ptr,
      result_data_ptr,
      batch_size,
      embedding_size,
      result_offsets_ptr);
  });
  const auto self_sizes = self_ptr->get_nested_size_tensor();
  return wrap_buffer(result_buffer, self_sizes);
}

} // namespace native
} // namespace at