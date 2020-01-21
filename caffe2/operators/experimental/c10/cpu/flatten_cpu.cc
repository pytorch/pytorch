#include <ATen/core/op_registration/op_registration.h>
#include "caffe2/core/export_c10_op_to_caffe2.h"
#include "caffe2/core/tensor.h"
#include "caffe2/utils/math.h"

using caffe2::BaseContext;
using caffe2::Tensor;

namespace caffe2 {
namespace {
template <class DataType, class Context>
void flatten_op_cpu_impl(
    const at::Tensor& input_,
    const at::Tensor& output_,
    int64_t axis) {
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

static auto registry = c10::RegisterOperators().op(
    "_c10_experimental::Flatten",
    c10::RegisterOperators::options()
      .kernel<
        decltype(flatten_op_cpu_impl<float, CPUContext>),
        &flatten_op_cpu_impl<float, CPUContext>>(DispatchKey::CPUTensorId));

} // namespace

C10_EXPORT_C10_OP_TO_CAFFE2_CPU(
    "_c10_experimental::Flatten",
    C10Flatten_DontUseThisOpYet)

} // namespace caffe2
