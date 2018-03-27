#ifndef CAFFE2_OPERATORS_CROSS_ENTROPY_OP_H_
#define CAFFE2_OPERATORS_CROSS_ENTROPY_OP_H_

#include "caffe2/core/context.h"
#include "caffe2/core/logging.h"
#include "caffe2/core/operator.h"
#include "caffe2/utils/math.h"

namespace caffe2 {

template <typename T, class Context>
class LabelCrossEntropyOp final : public Operator<Context> {
 public:
  USE_SIMPLE_CTOR_DTOR(LabelCrossEntropyOp);
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  bool RunOnDevice() override;

 protected:
  static constexpr T kLOG_THRESHOLD() {
    return static_cast<T>(1e-20);
  }
  // Input: X, label
  // Output: Y
};

template <typename T, class Context>
class LabelCrossEntropyGradientOp final : public Operator<Context> {
 public:
  USE_SIMPLE_CTOR_DTOR(LabelCrossEntropyGradientOp);
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  bool RunOnDevice() override;

 protected:
  // Input: X, label, dY
  // Ouptut: dX. There is no gradient with respect to the label.
  static constexpr T kLOG_THRESHOLD() {
    return static_cast<T>(1e-20);
  }
};

// Hacky: turns a vector of probabilities into a 2-column matrix with
// complimentary probabilities for binary classification
template <typename T, class Context>
class MakeTwoClassOp final : public Operator<Context> {
 public:
  USE_SIMPLE_CTOR_DTOR(MakeTwoClassOp);
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  bool RunOnDevice() override;

 protected:
  // Input: X
  // Output: Y = vstack(1-X, X)
};

template <typename T, class Context>
class MakeTwoClassGradientOp final : public Operator<Context> {
 public:
  USE_SIMPLE_CTOR_DTOR(MakeTwoClassGradientOp);
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  bool RunOnDevice() override;

 protected:
  // Input: dY
  // Ouptut: dX
};

template <typename T, class Context>
class SigmoidCrossEntropyWithLogitsOp final : public Operator<Context> {
 public:
  USE_SIMPLE_CTOR_DTOR(SigmoidCrossEntropyWithLogitsOp);
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  bool RunOnDevice() override;
};

template <typename T, class Context>
class SigmoidCrossEntropyWithLogitsGradientOp final : public Operator<Context> {
 public:
  USE_SIMPLE_CTOR_DTOR(SigmoidCrossEntropyWithLogitsGradientOp);
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  bool RunOnDevice() override;
};

template <typename T, class Context>
class WeightedSigmoidCrossEntropyWithLogitsOp final : public Operator<Context> {
 public:
  USE_SIMPLE_CTOR_DTOR(WeightedSigmoidCrossEntropyWithLogitsOp);
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  bool RunOnDevice() override;
};

template <typename T, class Context>
class WeightedSigmoidCrossEntropyWithLogitsGradientOp final
    : public Operator<Context> {
 public:
  USE_SIMPLE_CTOR_DTOR(WeightedSigmoidCrossEntropyWithLogitsGradientOp);
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  bool RunOnDevice() override;
};

template <typename T, class Context>
class CrossEntropyOp final : public Operator<Context> {
 public:
  USE_SIMPLE_CTOR_DTOR(CrossEntropyOp);
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  bool RunOnDevice() override;

 protected:
  // Input: X, label
  // Output: Y
  static constexpr T kLOG_THRESHOLD() {
    return static_cast<T>(1e-20);
  }
};

template <typename T, class Context>
class CrossEntropyGradientOp final : public Operator<Context> {
 public:
  USE_SIMPLE_CTOR_DTOR(CrossEntropyGradientOp);
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  bool RunOnDevice() override;

 protected:
  // Input: X, label, dY
  // Ouptut: dX. There is no gradient with respect to the label.
  static constexpr T kLOG_THRESHOLD() {
    return static_cast<T>(1e-20);
  }
};

} // namespace caffe2

#endif // CAFFE2_OPERATORS_CROSS_ENTROPY_OP_H_
