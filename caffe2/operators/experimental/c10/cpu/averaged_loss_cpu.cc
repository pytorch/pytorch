#include <ATen/core/dispatch/OperatorRegistration.h>
#include "caffe2/core/operator_c10wrapper.h"
#include "caffe2/core/tensor.h"
#include "caffe2/utils/math.h"

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

static auto registry = c10::RegisterOperators().op(
    FunctionSchema(
        "_c10_experimental::AveragedLoss",
        "",
        (std::vector<c10::Argument>{c10::Argument("input"),
                                    c10::Argument("output")}),
        (std::vector<c10::Argument>{})),
    c10::kernel<
        decltype(averaged_loss_op_cpu_impl<float, CPUContext>),
        &averaged_loss_op_cpu_impl<float, CPUContext>,
        Cache>(),
    c10::dispatchKey(CPUTensorId()));

} // namespace

REGISTER_C10_OPERATOR_FOR_CAFFE2_DISPATCH_CPU(
    "_c10_experimental::AveragedLoss",
    "",
    C10AveragedLoss_DontUseThisOpYet)

} // namespace caffe2
