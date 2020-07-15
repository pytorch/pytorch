#include <ATen/ATen.h>
#include <ATen/ExpandUtils.h>
#include <ATen/NativeFunctions.h>
#include <ATen/SparseTensorUtils.h>
#include <ATen/WrapDimUtilsMulti.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAUtils.h>
#include <ATen/native/sparse/SparseTensorMath.h>
#include <ATen/cuda/CUDAApplyUtils.cuh>
#include <ATen/cuda/detail/IndexUtils.cuh>
#include <ATen/native/sparse/cuda/SparseCUDAApplyUtils.cuh>
#include <ATen/native/sparse/cuda/SparseCUDABlas.cuh>

#include <THC/THCTensorMathPointwise.cuh>
#include <THC/THCThrustAllocator.cuh>

#include <thrust/binary_search.h>
#include <thrust/device_ptr.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <thrust/system/cuda/execution_policy.h>

#include <cuda_runtime_api.h>
#include <cusparse.h>
#include <bitset>

#include <c10/cuda/CUDAMathCompat.h>
#include <ATen/cuda/CUDAApplyUtils.cuh>
#include <ATen/cuda/detail/IndexUtils.cuh>
#include <ATen/cuda/detail/OffsetCalculator.cuh>
#include <ATen/native/cuda/Loops.cuh>

namespace at {
namespace native {
namespace {

void sparse_hardshrink_kernel(TensorIterator& iter, Scalar value) {
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      iter.dtype(),
      "sparse_hardshrink_cuda",
      [&]() {
        AT_SKIP_BFLOAT16_IF_NOT_ROCM(scalar_t, "sparse_hardshrink_cuda", [&] {
          auto lambd = value.to<scalar_t>();
          gpu_kernel(iter, [lambd] GPU_LAMBDA(scalar_t a) -> scalar_t {
            return (a >= -lambd && a <= lambd) ? scalar_t(0) : a;
          });
        });
      });
}

void sparse_hardshrink_backward_kernel(TensorIterator& iter, Scalar value) {
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      iter.dtype(),
      "sparse_hardshrink_backward_cuda",
      [&]() {
        AT_SKIP_BFLOAT16_IF_NOT_ROCM(
            scalar_t, "sparse_hardshrink_backward_cuda", [&] {
              auto lambd = value.to<scalar_t>();
              gpu_kernel(
                  iter,
                  [lambd] GPU_LAMBDA(
                      scalar_t grad_val, scalar_t self_val) -> scalar_t {
                    return (self_val >= -lambd && self_val <= lambd)
                        ? scalar_t(0)
                        : grad_val;
                  });
            });
      });
}

} // end anonymous namespace

Tensor sparse_hardshrink_cuda(const Tensor& input_, Scalar lambd) {
  TORCH_INTERNAL_ASSERT(input_.is_sparse());
  auto input = input_.coalesce();
  Tensor result = at::empty_like(input);
  if (input.numel() == 0) {
    return result;
  }
  auto indices = input._indices().contiguous();
  auto values = input._values().contiguous();

  int64_t nnz_result = values.size(0);

  auto result_indices = result._indices().contiguous();
  auto result_values = result._values().contiguous();

  std::vector<int64_t> values_size = {nnz_result};

  result_indices.resize_as_(indices);
  result_indices.copy_(indices);

  result_values.resize_(values_size);

  auto iter = TensorIterator::unary_op(result_values, values);
  sparse_hardshrink_kernel(iter, lambd);
  return result;
}

Tensor sparse_hardshrink_backward_cuda(
    const Tensor& grad,
    const Tensor& input_,
    Scalar lambd) {
  TORCH_INTERNAL_ASSERT(input_.is_sparse());
  auto input = input_.coalesce();
  Tensor result = at::empty_like(input);
  if (input.numel() == 0) {
    return result;
  }
  auto indices = input._indices().contiguous();
  auto values = input._values().contiguous();

  int64_t nnz_result = values.size(0);

  auto result_indices = result._indices().contiguous();
  auto result_values = result._values().contiguous();

  std::vector<int64_t> values_size = {nnz_result};

  result_indices.resize_as_(indices);
  result_indices.copy_(indices);

  result_values.resize_(values_size);

  auto grad_values = grad._values().contiguous();

  auto iter = TensorIterator::binary_op(result_values, grad_values, values);
  sparse_hardshrink_backward_kernel(iter, lambd);
  return result;
}

} // namespace native
} // namespace at
