#include <ATen/core/dispatch/KernelRegistration.h>
#include "caffe2/operators/experimental/c10/schemas/sigmoid.h"
#include "caffe2/utils/eigen_utils.h"
#include "caffe2/utils/math.h"
#include "caffe2/core/tensor.h"

using caffe2::Tensor;

namespace caffe2 {
namespace {
template <class DataType>
void sigmoid_op_cpu_impl(
    const at::Tensor& input_,
    const at::Tensor& output_) {
  Tensor input{C10Tensor(input_)};
  Tensor output{C10Tensor(output_)};
  output.ResizeLike(input);

  caffe2::ConstEigenVectorArrayMap<DataType> xM(
      input.data<DataType>(), input.numel());
  caffe2::EigenVectorArrayMap<DataType>(
      output.mutable_data<DataType>(), input.numel()) =
      1. / (1. + (-xM).exp());
}
} // namespace
} // namespace caffe2

namespace c10 {
C10_REGISTER_KERNEL(caffe2::ops::Sigmoid)
    .kernel<&caffe2::sigmoid_op_cpu_impl<float>>()
    .dispatchKey({DeviceTypeId::CPU,
                  LayoutId(0),
                  caffe2::TypeMeta::Id<float>()});
} // namespace c10
