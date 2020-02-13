#include <ATen/core/op_registration/op_registration.h>
#include "caffe2/core/export_c10_op_to_caffe2.h"
#include "caffe2/core/tensor.h"
#include "caffe2/utils/math.h"

using caffe2::CPUContext;
using caffe2::Tensor;

namespace caffe2 {
namespace {
template <class DataType>
void enforce_finite_op_impl_cpu(const at::Tensor& input_) {
  Tensor input(input_);
  const DataType* input_data = input.template data<DataType>();
  auto size = input.numel();

  for (auto i = 0; i < size; i++) {
    CAFFE_ENFORCE(
        std::isfinite(input_data[i]),
        "Index ",
        i,
        " is not finite (e.g., NaN, Inf): ",
        input_data[i]);
  }
}

static auto registry = c10::RegisterOperators().op(
    "_c10_experimental::EnforceFinite",
    c10::RegisterOperators::options()
      .kernel<
        decltype(enforce_finite_op_impl_cpu<float>),
        &enforce_finite_op_impl_cpu<float>>(DispatchKey::CPUTensorId));

} // namespace

C10_EXPORT_C10_OP_TO_CAFFE2_CPU(
    "_c10_experimental::EnforceFinite",
    C10EnforceFinite_DontUseThisOpYet)

} // namespace caffe2
