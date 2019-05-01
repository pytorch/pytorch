#pragma once

#include "caffe2/core/context.h"
#include "caffe2/core/operator.h"
#include "caffe2/utils/eigen_utils.h"
#include "caffe2/utils/math.h"

namespace caffe2 {
namespace detail {

template <typename T, typename Context>
void VariableLengthSequencePadding(
    int N,
    int B,
    int M,
    T* X,
    const int32_t* seqLengths,
    const T padValue,
    Context* /*context*/) {
  for (int j = 0; j < B; j++) {
    for (int i = seqLengths[j]; i < N; i++) {
      EigenVectorArrayMap<T>(X + B * M * i + M * j, M).setConstant(padValue);
    }
  }
}

} // namespace detail

template <typename T, typename Context>
class VariableLengthSequencePaddingOp : public Operator<Context> {
 public:
  template <class... Args>
  explicit VariableLengthSequencePaddingOp(Args&&... args)
      : Operator<Context>(std::forward<Args>(args)...) {}
  USE_OPERATOR_CONTEXT_FUNCTIONS;

  bool RunOnDevice() override {
    const auto N = Input(INPUT).size(0);
    const auto B = Input(INPUT).size(1);
    const auto M = Input(INPUT).size(2);

    auto X = Output(OUTPUT)->template mutable_data<T>();

    auto seqLengths = Input(SEQ_LENGTHS).template data<int32_t>();

    detail::VariableLengthSequencePadding<T, Context>(
        N, B, M, X, seqLengths, 0, &context_);
    return true;
  }

 protected:
  INPUT_TAGS(INPUT, SEQ_LENGTHS);
  OUTPUT_TAGS(OUTPUT);
};

} // namespace caffe2
