#ifndef CAFFE2_OPERATORS_CTC_GREEDY_DECODER_OP_H_
#define CAFFE2_OPERATORS_CTC_GREEDY_DECODER_OP_H_

#include "caffe2/core/context.h"
#include "caffe2/core/operator.h"

namespace caffe2 {

template <class Context>
class CTCGreedyDecoderOp : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  template <class... Args>
  explicit CTCGreedyDecoderOp(Args&&... args)
      : Operator<Context>(std::forward<Args>(args)...) {
    merge_repeated_ =
        this->template GetSingleArgument<bool>("merge_repeated", true);
  }

  bool RunOnDevice() override;

 protected:
  bool merge_repeated_;
  INPUT_TAGS(INPUTS, SEQ_LEN);
  OUTPUT_TAGS(OUTPUT_LEN, VALUES);
  // Input: X, 3D tensor; L, 1D tensor. Output: Y sparse tensor
};

} // namespace caffe2

#endif // CAFFE2_OPERATORS_CTC_GREEDY_DECODER_OP_H_
