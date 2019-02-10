#include <ATen/core/dispatch/KernelRegistration.h>
#include "caffe2/operators/experimental/c10/schemas/averaged_loss.h"
#include "caffe2/utils/math.h"
#include "caffe2/core/tensor.h"

using caffe2::BaseContext;
using caffe2::Tensor;
using std::vector;

namespace caffe2 {
namespace {

struct Cache final : public c10::KernelCache {
  at::Tensor scratch = at::Tensor(C10Tensor(empty({}, CPU)));
};

template <class T, class Context>
void averaged_loss_op_cpu_impl(
    const at::Tensor& X_,
    const at::Tensor& sum_,
    Cache* state) {
  Tensor X{C10Tensor(X_)};
  Tensor sum{C10Tensor(sum_)};
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
    .withCache<caffe2::Cache>()
    .kernel<decltype(caffe2::averaged_loss_op_cpu_impl<float, caffe2::CPUContext>), &caffe2::averaged_loss_op_cpu_impl<float, caffe2::CPUContext>>()
    .dispatchKey(CPUTensorId());
} // namespace c10
