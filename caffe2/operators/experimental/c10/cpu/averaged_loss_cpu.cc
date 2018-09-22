#include "caffe2/core/dispatch/KernelRegistration.h"
#include "caffe2/operators/experimental/c10/schemas/averaged_loss.h"
#include "caffe2/utils/math.h"

using caffe2::BaseContext;
using caffe2::Tensor;
using caffe2::TIndex;
using std::vector;

namespace caffe2 {
namespace {

template <class T, class Context>
void averaged_loss_op_cpu_impl(
    const Tensor& X,
    Tensor* sum,
    caffe2::ops::AveragedLoss::State* state,
    BaseContext* context) {
  sum->Resize(vector<TIndex>());

  T* data = sum->template mutable_data<T>();

  caffe2::math::Sum<T, Context>(
      X.size(), X.template data<T>(), data, static_cast<Context*>(context), &state->scratch);
  if (X.size() > 0) {
    caffe2::math::Scale<T, T, Context>(
        1,
        static_cast<T>(1.) / X.size(),
        sum->template data<T>(),
        data,
        static_cast<Context*>(context));
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
