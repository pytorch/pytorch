#include <ATen/core/op_registration/op_registration.h>
#include "caffe2/core/operator_c10wrapper.h"
#include "caffe2/core/tensor.h"
#include "caffe2/utils/eigen_utils.h"
#include "caffe2/utils/math.h"

using caffe2::Tensor;

namespace caffe2 {
namespace {
template <class DataType>
void relu_op_cpu_impl(
    const at::Tensor& input_,
    const at::Tensor& output_) {
  Tensor input(input_);
  Tensor output(output_);

  output.ResizeLike(input);

#ifdef CAFFE2_USE_ACCELERATE
  const float zero = 0.0f;
  vDSP_vthres(
      input.data<float>(),
      1,
      &zero,
      output.mutable_data<float>(),
      1,
      input.size());
#else
  caffe2::EigenVectorMap<float>(output.mutable_data<float>(), input.numel()) =
      caffe2::ConstEigenVectorMap<float>(input.data<float>(), input.numel())
          .cwiseMax(0.f);
#endif
  /* Naive implementation
  const float* input_data = input.data<float>();
  float* output_data = output.mutable_data<float>();
  for (int i = 0; i < input.size(); ++i) {
    output_data[i] = std::max(input_data[i], 0.f);
  }
  */
}

static auto registry = c10::RegisterOperators().op(
    "_c10_experimental::Relu",
    c10::kernel<decltype(relu_op_cpu_impl<float>), &relu_op_cpu_impl<float>>(),
    c10::dispatchKey(CPUTensorId()));

} // namespace

REGISTER_C10_OPERATOR_FOR_CAFFE2_DISPATCH_CPU(
    "_c10_experimental::Relu",
    C10Relu_DontUseThisOpYet)

} // namespace caffe2
