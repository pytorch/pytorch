#ifndef CAFFE2_OPERATORS_SQUARE_ROOT_DIVIDE_OP_H_
#define CAFFE2_OPERATORS_SQUARE_ROOT_DIVIDE_OP_H_

#include "caffe2/core/context.h"
#include "caffe2/core/operator.h"
#include "caffe2/utils/math.h"
#include "c10/util/irange.h"

namespace caffe2 {

template <class Context>
class SquareRootDivideOp final : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  USE_DISPATCH_HELPER;

  template <class... Args>
  explicit SquareRootDivideOp(Args&&... args)
      : Operator<Context>(std::forward<Args>(args)...) {}

  bool RunOnDevice() override {
    return DispatchHelper<TensorTypes<float>>::call(this, Input(DATA));
  }

 private:
  template <typename TData>
  bool DoRunWithType() {
    return DispatchHelper<TensorTypes2<float, int32_t, int64_t>, TData>::call(
        this, Input(SCALE));
  }

  template <typename TData, typename TScale>
  bool DoRunWithType2() {
    auto& data = Input(DATA);
    auto& scale = Input(SCALE);

    auto* Y = Output(0, data.sizes(), at::dtype<TData>());
    size_t batchSize = data.size(0);
    size_t exampleSize = data.size_from_dim(1);
    // NOLINTNEXTLINE(clang-diagnostic-sign-compare)
    CAFFE_ENFORCE(batchSize == scale.size(0), batchSize, " != ", scale.size(0));
    auto* scalePtr = scale.template data<TScale>();
    auto* dataPtr = data.template data<TData>();
    auto* yPtr = Y->template mutable_data<TData>();
    for (const auto i : c10::irange(0U, batchSize)) {
      auto scale = scalePtr[i];
      CAFFE_ENFORCE(scale >= 0, scale, " < 0");
      auto multiplier = scale == 0 ? 1.0 : 1 / std::sqrt(scale);
      math::Scale<float, TData, Context>(
          exampleSize,
          multiplier,
          dataPtr + i * exampleSize,
          yPtr + i * exampleSize,
          &context_);
    }
    return true;
  }

  INPUT_TAGS(DATA, SCALE);
};

} // namespace caffe2

#endif // CAFFE2_OPERATORS_SQUARE_ROOT_DIVIDE_OP_H_
