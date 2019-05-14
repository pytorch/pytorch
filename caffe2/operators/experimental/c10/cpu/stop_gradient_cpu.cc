#include <ATen/core/op_registration/op_registration.h>
#include "caffe2/core/operator_c10wrapper.h"
#include "caffe2/core/tensor.h"
#include "caffe2/utils/math.h"

using caffe2::BaseContext;
using caffe2::Tensor;

namespace caffe2 {
namespace {
template <class DataType>
void stop_gradient_op_cpu_impl(
    const at::Tensor& input_,
    const at::Tensor& output_) {
  Tensor input(input_);
  Tensor output(output_);
  if (!output.is_same(input)) {
    output.CopyFrom(input);
  }
}

static auto registry = c10::RegisterOperators().op(
    FunctionSchema(
        "_c10_experimental::StopGradient",
        "",
        (std::vector<c10::Argument>{c10::Argument("input"),
                                    c10::Argument("output")}),
        (std::vector<c10::Argument>{})),
    c10::kernel<
        decltype(stop_gradient_op_cpu_impl<float>),
        &stop_gradient_op_cpu_impl<float>>(),
    c10::dispatchKey(CPUTensorId()));

} // namespace

REGISTER_C10_OPERATOR_FOR_CAFFE2_DISPATCH_CPU(
    "_c10_experimental::StopGradient",
    C10StopGradient_DontUseThisOpYet)

} // namespace caffe2
