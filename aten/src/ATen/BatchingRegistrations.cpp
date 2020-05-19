#include <ATen/BatchedTensorImpl.h>
#include <ATen/BatchingRules.h>
#include <torch/library.h>

namespace at {

/*
 * Operator Registrations for BatchedTensor.
 * Contains some glue to hook up the batching rules to BatchedTensorImpl.
 */

Tensor BatchedTensor_sum(const Tensor& self, IntArrayRef dim, bool keepdim, c10::optional<ScalarType> dtype) {
  Tensor self_, result_;
  BatchDimsRef self_bdims;
  BatchDims result_bdims;

  std::tie(self_, self_bdims) = unpackBatched(self);
  std::tie(result_, result_bdims) = sum_batching_rule(self_, self_bdims, dim, keepdim, dtype);
  return detail::make_tensor<BatchedTensorImpl>(result_, result_bdims);
}

void batchedTensorFallback(const c10::OperatorHandle& op, torch::jit::Stack* stack) {
  TORCH_CHECK(false, "NYI: Calling ", op.schema().name(), " inside of vmap");
}

TORCH_LIBRARY_IMPL(_, BatchedTensorKey, m) {
  m.fallback(torch::CppFunction::makeFromBoxedFunction<&batchedTensorFallback>());
}

TORCH_LIBRARY_IMPL(aten, BatchedTensorKey, m) {
  m.impl_UNBOXED("sum.dim_IntList", BatchedTensor_sum);
}

} // namespace at
