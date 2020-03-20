#include <ATen/core/op_registration/op_registration.h>
#include "caffe2/core/export_c10_op_to_caffe2.h"
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
    "_c10_experimental::StopGradient",
    c10::RegisterOperators::options()
      .kernel<
        decltype(stop_gradient_op_cpu_impl<float>),
        &stop_gradient_op_cpu_impl<float>>(DispatchKey::CPUTensorId));

} // namespace

C10_EXPORT_C10_OP_TO_CAFFE2_CPU(
    "_c10_experimental::StopGradient",
    C10StopGradient_DontUseThisOpYet)

} // namespace caffe2
