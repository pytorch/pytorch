#include <ATen/ATen.h>
#include <ATen/Config.h>
#include <ATen/NamedTensorUtils.h>
#include <ATen/NativeFunctions.h>
#include <ATen/Parallel.h>
#include <ATen/SparseTensorUtils.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/cpu/Loops.h>
#include <map>

namespace at {
namespace native {
namespace {

void sparse_hardshrink_kernel(TensorIterator& iter, Scalar lambd) {
  AT_DISPATCH_FLOATING_TYPES(iter.dtype(), "sparse_hardshrink_cpu", [&] {
    auto lambd_val = lambd.to<scalar_t>();
    cpu_kernel_vec(
        iter,
        [=](scalar_t self_val) {
          return (self_val >= -lambd_val && self_val <= lambd_val) ? scalar_t(0)
                                                                   : self_val;
        },
        [=](Vec256<scalar_t> self_val) {
          return ((self_val < -lambd_val) | (self_val > lambd_val)) & self_val;
        });
  });
}

void sparse_hardshrink_backward_kernel(TensorIterator& iter, Scalar lambd) {
  AT_DISPATCH_FLOATING_TYPES(
      iter.dtype(), "sparse_hardshrink_backward_cpu", [&] {
        auto lambd_val = lambd.to<scalar_t>();
        cpu_kernel_vec(
            iter,
            [=](scalar_t grad_val, scalar_t self_val) {
              return (self_val >= -lambd_val && self_val <= lambd_val)
                  ? scalar_t(0)
                  : grad_val;
            },
            [=](Vec256<scalar_t> grad_val, Vec256<scalar_t> self_val) {
              return ((self_val < -lambd_val) | (self_val > lambd_val)) &
                  grad_val;
            });
      });
}

} // end anonymous namespace

Tensor sparse_hardshrink_cpu(const Tensor& input_, Scalar lambd) {
  TORCH_INTERNAL_ASSERT(input_.is_sparse());
  auto input = input_.coalesce();
  Tensor result = at::native::empty_like(input);
  if (input.numel() == 0) {
    return result;
  }
  auto indices = input._indices().contiguous();
  auto input_values = input._values().contiguous();

  int64_t nnz_result = input_values.size(0);

  auto result_indices = result._indices().contiguous();
  auto result_values = result._values().contiguous();

  std::vector<int64_t> values_size = {nnz_result};

  result_indices.resize_as_(indices);
  result_indices.copy_(indices);

  result_values.resize_(values_size);

  auto iter = TensorIterator::unary_op(result_values, input_values);
  sparse_hardshrink_kernel(iter, lambd);
  return result;
}

Tensor sparse_hardshrink_backward_cpu(
    const Tensor& grad,
    const Tensor& input_,
    Scalar lambd) {
  TORCH_INTERNAL_ASSERT(input_.is_sparse());
  auto input = input_.coalesce();
  Tensor result = at::native::empty_like(input);
  if (input.numel() == 0) {
    return result;
  }

  auto indices = input._indices().contiguous();
  auto input_values = input._values().contiguous();

  int64_t nnz_result = input_values.size(0);

  auto result_indices = result._indices().contiguous();
  auto result_values = result._values().contiguous();

  std::vector<int64_t> values_size = {nnz_result};

  result_indices.resize_as_(indices);
  result_indices.copy_(indices);

  result_values.resize_(values_size);

  auto grad_values = grad._values().contiguous();

  auto iter =
      TensorIterator::binary_op(result_values, grad_values, input_values);
  sparse_hardshrink_backward_kernel(iter, lambd);
  return result;
}

} // namespace native
} // namespace at
