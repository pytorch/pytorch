#ifndef CAFFE2_OPERATORS_COSINE_EMBEDDING_CRITERION_OP_H_
#define CAFFE2_OPERATORS_COSINE_EMBEDDING_CRITERION_OP_H_

#include "caffe2/core/context.h"
#include "caffe2/core/logging.h"
#include "caffe2/core/operator.h"

namespace caffe2 {

template <class Context>
class CosineEmbeddingCriterionOp final : public Operator<Context> {
 public:
  template <class... Args>
  explicit CosineEmbeddingCriterionOp(Args&&... args)
      : Operator<Context>(std::forward<Args>(args)...),
        OP_SINGLE_ARG(float, "margin", margin_, 0.0) {}
  USE_OPERATOR_CONTEXT_FUNCTIONS;

  bool RunOnDevice() override;

 protected:
  float margin_;
};

template <class Context>
class CosineEmbeddingCriterionGradientOp final : public Operator<Context> {
 public:
  template <class... Args>
  explicit CosineEmbeddingCriterionGradientOp(Args&&... args)
      : Operator<Context>(std::forward<Args>(args)...),
        OP_SINGLE_ARG(float, "margin", margin_, 0.0) {}
  USE_OPERATOR_CONTEXT_FUNCTIONS;

  bool RunOnDevice() override;

 protected:
  float margin_;
};

} // namespace caffe2

#endif // CAFFE2_OPERATORS_COSINE_EMBEDDING_CRITERION_OP_H_
