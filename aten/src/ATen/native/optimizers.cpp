#include <ATen/native/optimizers.h>

#include <tuple>

#include <ATen/ATen.h>
#include <ATen/AccumulateType.h>
#include <ATen/CPUApplyUtils.h>
#include <ATen/Config.h>
#include <ATen/Parallel.h>
#include <ATen/SparseTensorImpl.h>
#include <ATen/SparseTensorUtils.h>

namespace at {
namespace native {

namespace {

sparse::SparseTensor MakeSparseTensor(
    const Tensor& grad,
    const Tensor& values) {
  const auto sizes = grad.sizes();
  const int64_t sparse_dim = grad.sparse_dim();
  const int64_t dense_dim = grad.dense_dim();
  return _sparse_coo_tensor_with_dims_and_tensors(
      sparse_dim, dense_dim, sizes, grad._indices(), values, grad.options());
}

} // namespace

std::tuple<sparse::SparseTensor, sparse::SparseTensor, sparse::SparseTensor>
sparse_adam_step_cpu(
    const sparse::SparseTensor& grad,
    const Tensor& moment1,
    const Tensor& moment2,
    double alpha,
    double beta1,
    double beta2,
    double eps,
    int64_t step) {
  const auto grad_coalesced = grad.coalesce();
  const auto grad_val = grad_coalesced._values();
  const auto moment1_val = moment1.sparse_mask(grad_coalesced)._values();
  const auto moment2_val = moment2.sparse_mask(grad_coalesced)._values();
  Tensor adam_step = at::native::empty_like(grad_val);
  Tensor moment1_step = at::native::empty_like(moment1_val);
  Tensor moment2_step = at::native::empty_like(moment2_val);
  TensorIterator it = TensorIterator();
  it.add_output(adam_step);
  it.add_output(moment1_step);
  it.add_output(moment2_step);
  it.add_input(grad_val);
  it.add_input(moment1_val);
  it.add_input(moment2_val);
  it.build();
  sparse_adam_step_stub(kCPU, alpha, beta1, beta2, eps, step, &it);
  return std::make_tuple(
      MakeSparseTensor(grad_coalesced, adam_step),
      MakeSparseTensor(grad_coalesced, moment1_step),
      MakeSparseTensor(grad_coalesced, moment2_step));
}

std::tuple<sparse::SparseTensor, sparse::SparseTensor, sparse::SparseTensor>
sparse_adam_step_cuda(
    const sparse::SparseTensor& grad,
    const Tensor& moment1,
    const Tensor& moment2,
    double alpha,
    double beta1,
    double beta2,
    double eps,
    int64_t step) {
  const auto grad_coalesced = grad.coalesce();
  const auto grad_val = grad_coalesced._values();
  const auto moment1_val = moment1.sparse_mask(grad_coalesced)._values();
  const auto moment2_val = moment2.sparse_mask(grad_coalesced)._values();
  Tensor adam_step = at::native::empty_like(grad_val);
  Tensor moment1_step = at::native::empty_like(moment1_val);
  Tensor moment2_step = at::native::empty_like(moment2_val);
  TensorIterator it = TensorIterator();
  it.add_output(adam_step);
  it.add_output(moment1_step);
  it.add_output(moment2_step);
  it.add_input(grad_val);
  it.add_input(moment1_val);
  it.add_input(moment2_val);
  it.build();
  sparse_adam_step_stub(kCUDA, alpha, beta1, beta2, eps, step, &it);
  return std::make_tuple(
      MakeSparseTensor(grad_coalesced, adam_step),
      MakeSparseTensor(grad_coalesced, moment1_step),
      MakeSparseTensor(grad_coalesced, moment2_step));
}

DEFINE_DISPATCH(sparse_adam_step_stub);

} // namespace native
} // namespace at
