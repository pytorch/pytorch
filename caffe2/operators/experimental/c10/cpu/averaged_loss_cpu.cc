#include <c10/core/dispatch/KernelRegistration.h>
#include "caffe2/operators/experimental/c10/schemas/averaged_loss.h"
#include "caffe2/utils/math.h"
#include "caffe2/core/tensor.h"

using caffe2::BaseContext;
using caffe2::Tensor;
using std::vector;

namespace caffe2 {
namespace {

template <class T, class Context>
void averaged_loss_op_cpu_impl(
    const C10Tensor& X_,
    const C10Tensor& sum_,
    caffe2::ops::AveragedLoss::State* state) {
  Tensor X(X_);
  Tensor sum(sum_);
  CPUContext context;

  sum.Resize(vector<int64_t>());

  T* data = sum.template mutable_data<T>();

  Tensor scratch(state->scratch);
  caffe2::math::Sum<T, Context>(
      X.numel(),
      X.template data<T>(),
      data,
      static_cast<Context*>(&context),
      &scratch);
  if (X.numel() > 0) {
    caffe2::math::Scale<T, T, Context>(
        1,
        static_cast<T>(1.) / X.numel(),
        sum.template data<T>(),
        data,
        static_cast<Context*>(&context));
  }
}
} // namespace
} // namespace caffe2

namespace c10 {
C10_REGISTER_KERNEL(caffe2::ops::AveragedLoss)
    .kernel(&caffe2::averaged_loss_op_cpu_impl<float, caffe2::CPUContext>)
    .dispatchKey(c10::DispatchKey<1>{c10::details::TensorParameterDispatchKey{
        DeviceTypeId::CPU,
        LayoutId(0),
        caffe2::TypeMeta::Id<float>()}});
} // namespace c10
