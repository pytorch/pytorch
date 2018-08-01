#include "caffe2/core/dispatch/KernelRegistration.h"
#include "caffe2/operators/experimental/c10/schemas/averaged_loss.h"
#include "caffe2/utils/math.h"

using caffe2::CPUContext;
using caffe2::Tensor;
using caffe2::TIndex;
using std::vector;

namespace caffe2 {
namespace {

template <class T>
void averaged_loss_op_cpu_impl(
    const Tensor<CPUContext>& X,
    Tensor<CPUContext>* sum,
    caffe2::ops::AveragedLoss::State* state,
    CPUContext* context) {
  sum->Resize(vector<TIndex>());

  T* data = sum->template mutable_data<T>();

  caffe2::math::Sum<T, CPUContext>(
      X.size(), X.template data<T>(), data, context, &state->scratch);
  if (X.size() > 0) {
    caffe2::math::Scale<T, CPUContext>(
        1,
        static_cast<T>(1.) / X.size(),
        sum->template data<T>(),
        data,
        context);
  }
}
} // namespace
} // namespace caffe2

namespace c10 {
C10_REGISTER_KERNEL(caffe2::ops::AveragedLoss)
    .kernel(&caffe2::averaged_loss_op_cpu_impl<float>)
    .dispatchKey(c10::DispatchKey<1>{c10::details::TensorParameterDispatchKey{
        DeviceTypeId::CPU,
        LayoutId(0),
        caffe2::TypeMeta::Id<float>()}});
} // namespace c10
