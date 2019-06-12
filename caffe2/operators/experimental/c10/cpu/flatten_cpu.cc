#include "caffe2/core/dispatch/KernelRegistration.h"
#include "caffe2/operators/experimental/c10/schemas/flatten.h"
#include "caffe2/utils/math.h"

using caffe2::BaseContext;
using caffe2::Tensor;

namespace caffe2 {
namespace {
template <class DataType, class Context>
void flatten_op_cpu_impl(
    const Tensor& input,
    Tensor* output,
    int axis,
    BaseContext* context) {
  CAFFE_ENFORCE_GE(
      input.sizes().size(), axis, "The rank of the tensor must be >= axis.");
  output->Resize(input.size_to_dim(axis), input.size_from_dim(axis));
  context->CopyItemsSameDevice(
      input.dtype(),
      input.numel(),
      input.raw_data(),
      output->raw_mutable_data(input.dtype()));
}
} // namespace
} // namespace caffe2

namespace c10 {
C10_REGISTER_KERNEL(caffe2::ops::Flatten)
    .kernel(&caffe2::flatten_op_cpu_impl<float, caffe2::CPUContext>)
    .dispatchKey({DeviceTypeId::CPU,
                  LayoutId(0),
                  caffe2::TypeMeta::Id<float>()});
} // namespace c10
