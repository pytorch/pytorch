#include <ATen/core/op_registration/op_registration.h>
#include "caffe2/core/operator_c10wrapper.h"
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
  Tensor input{C10Tensor(input_)};
  Tensor output{C10Tensor(output_)};
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
    FunctionSchema(
        "_c10_experimental::Flatten",
        "",
        (std::vector<c10::Argument>{c10::Argument("input"),
                                    c10::Argument("output"),
                                    c10::Argument("axis", IntType::get())}),
        (std::vector<c10::Argument>{})),
    c10::kernel<
        decltype(flatten_op_cpu_impl<float, CPUContext>),
        &flatten_op_cpu_impl<float, CPUContext>>(),
    c10::dispatchKey(CPUTensorId()));

} // namespace

REGISTER_C10_OPERATOR_FOR_CAFFE2_DISPATCH_CPU(
    "_c10_experimental::Flatten",
    C10Flatten_DontUseThisOpYet)

} // namespace caffe2
