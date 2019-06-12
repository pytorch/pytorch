#ifndef CAFFE2_OPERATORS_CTC_BEAM_SEARCH_OP_H_
#define CAFFE2_OPERATORS_CTC_BEAM_SEARCH_OP_H_

#include "caffe2/core/context.h"
#include "caffe2/core/operator.h"

namespace caffe2 {

template <class Context>
class CTCBeamSearchDecoderOp : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  template <class... Args>
  explicit CTCBeamSearchDecoderOp(Args&&... args)
      : Operator<Context>(std::forward<Args>(args)...) {
    beam_width_ = this->template GetSingleArgument<int32_t>("beam_width", 10);
    prune_threshold_ =
        this->template GetSingleArgument<float>("prune_threshold", 0.001);
  }

  bool RunOnDevice() override;

 protected:
  int32_t beam_width_;
  float prune_threshold_;
  INPUT_TAGS(INPUTS, SEQ_LEN);
  OUTPUT_TAGS(OUTPUT_LEN, VALUES);
  // Input: X, 3D tensor; L, 1D tensor. Output: Y sparse tensor
};

} // namespace caffe2

#endif // CAFFE2_OPERATORS_CTC_BEAM_SEARCH_OP_H_
