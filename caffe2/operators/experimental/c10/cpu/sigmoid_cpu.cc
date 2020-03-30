#include <ATen/core/op_registration/op_registration.h>
#include "caffe2/core/export_c10_op_to_caffe2.h"
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
    "_c10_experimental::Sigmoid",
    c10::RegisterOperators::options()
      .kernel<
        decltype(sigmoid_op_cpu_impl<float>),
        &sigmoid_op_cpu_impl<float>>(DispatchKey::CPUTensorId));

} // namespace

C10_EXPORT_C10_OP_TO_CAFFE2_CPU(
    "_c10_experimental::Sigmoid",
    C10Sigmoid_DontUseThisOpYet)

} // namespace caffe2
