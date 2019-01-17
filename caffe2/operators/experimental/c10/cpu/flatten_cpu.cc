#include <c10/core/dispatch/KernelRegistration.h>
#include "caffe2/operators/experimental/c10/schemas/flatten.h"
#include "caffe2/utils/math.h"
#include "caffe2/core/tensor.h"

using caffe2::BaseContext;
using caffe2::Tensor;

namespace caffe2 {
namespace {
template <class DataType, class Context>
void flatten_op_cpu_impl(
    const C10Tensor& input_,
    const C10Tensor& output_,
    int axis) {
  Tensor input(input_);
  Tensor output(output_);
  CPUContext context;
  CAFFE_ENFORCE_GE(
      input.sizes().size(), axis, "The rank of the tensor must be >= axis.");
  output.Resize(input.size_to_dim(axis), input.size_from_dim(axis));
  context.CopyItemsSameDevice(
      input.dtype(),
      input.numel(),
      input.raw_data(),
      output.raw_mutable_data(input.dtype()));
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
