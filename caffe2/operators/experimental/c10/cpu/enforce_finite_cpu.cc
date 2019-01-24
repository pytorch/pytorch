#include <ATen/core/dispatch/KernelRegistration.h>
#include "caffe2/operators/experimental/c10/schemas/enforce_finite.h"
#include "caffe2/utils/math.h"
#include "caffe2/core/tensor.h"

using caffe2::CPUContext;
using caffe2::Tensor;

namespace caffe2 {
namespace {
template <class DataType>
void enforce_finite_op_impl_cpu(const at::Tensor& input_) {
  Tensor input{C10Tensor(input_)};
  const DataType* input_data = input.template data<DataType>();
  auto size = input.numel();

  for (auto i = 0; i < size; i++) {
    CAFFE_ENFORCE(
        std::isfinite(input_data[i]),
        "Index ",
        i,
        " is not finite (e.g., NaN, Inf): ",
        input_data[i]);
  }
}
} // namespace
} // namespace caffe2

namespace c10 {
C10_REGISTER_KERNEL(caffe2::ops::EnforceFinite)
    .kernel<&caffe2::enforce_finite_op_impl_cpu<float>>()
    .dispatchKey({DeviceTypeId::CPU,
                  LayoutId(0),
                  caffe2::TypeMeta::Id<float>()});
} // namespace c10
