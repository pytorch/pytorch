#include <ATen/core/dispatch/KernelRegistration.h>
#include "caffe2/operators/experimental/c10/schemas/stop_gradient.h"
#include "caffe2/utils/math.h"
#include "caffe2/core/tensor.h"

using caffe2::BaseContext;
using caffe2::Tensor;

namespace caffe2 {
namespace {
template <class DataType>
void stop_gradient_op_cpu_impl(
    const at::Tensor& input_,
    const at::Tensor& output_) {
  Tensor input{C10Tensor(input_)};
  Tensor output{C10Tensor(output_)};
  if (!output.is_same(input)) {
    output.CopyFrom(input);
  }
}
} // namespace
} // namespace caffe2

namespace c10 {
C10_REGISTER_KERNEL(caffe2::ops::StopGradient)
    .kernel<&caffe2::stop_gradient_op_cpu_impl<float>>()
    .dispatchKey({DeviceTypeId::CPU,
                  LayoutId(0),
                  caffe2::TypeMeta::Id<float>()});
} // namespace c10
