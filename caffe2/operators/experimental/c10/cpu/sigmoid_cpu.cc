#include <ATen/core/op_registration/op_registration.h>
#include "caffe2/core/operator_c10wrapper.h"
#include "caffe2/core/tensor.h"
#include "caffe2/utils/eigen_utils.h"
#include "caffe2/utils/math.h"

using caffe2::Tensor;

namespace caffe2 {
namespace {
template <class DataType>
void sigmoid_op_cpu_impl(
    const at::Tensor& input_,
    const at::Tensor& output_) {
  Tensor input(input_);
  Tensor output(output_);
  output.ResizeLike(input);

  caffe2::ConstEigenVectorArrayMap<DataType> xM(
      input.data<DataType>(), input.numel());
  caffe2::EigenVectorArrayMap<DataType>(
      output.mutable_data<DataType>(), input.numel()) =
      1. / (1. + (-xM).exp());
}

static auto registry = c10::RegisterOperators().op(
    FunctionSchema(
        "_c10_experimental::Sigmoid",
        "",
        (std::vector<c10::Argument>{c10::Argument("input"),
                                    c10::Argument("output")}),
        (std::vector<c10::Argument>{})),
    c10::kernel<
        decltype(sigmoid_op_cpu_impl<float>),
        &sigmoid_op_cpu_impl<float>>(),
    c10::dispatchKey(CPUTensorId()));

} // namespace

REGISTER_C10_OPERATOR_FOR_CAFFE2_DISPATCH_CPU(
    "_c10_experimental::Sigmoid",
    C10Sigmoid_DontUseThisOpYet)

} // namespace caffe2
