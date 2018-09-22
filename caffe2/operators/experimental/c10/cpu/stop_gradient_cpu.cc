#include "caffe2/core/dispatch/KernelRegistration.h"
#include "caffe2/operators/experimental/c10/schemas/stop_gradient.h"
#include "caffe2/utils/math.h"

using caffe2::BaseContext;
using caffe2::Tensor;

namespace caffe2 {
namespace {
template <class DataType>
void stop_gradient_op_cpu_impl(
    const Tensor& input,
    Tensor* output,
    BaseContext* context) {
  if (output != &input) {
    output->CopyFrom(input, context);
  }
}
} // namespace
} // namespace caffe2

namespace c10 {
C10_REGISTER_KERNEL(caffe2::ops::StopGradient)
    .kernel(&caffe2::stop_gradient_op_cpu_impl<float>)
    .dispatchKey({DeviceTypeId::CPU,
                  LayoutId(0),
                  caffe2::TypeMeta::Id<float>()});
} // namespace c10
