#ifndef CAFFE2_OPERATORS_ELEMENTWISE_OPS_H_
#define CAFFE2_OPERATORS_ELEMENTWISE_OPS_H_

#include <iterator>
#include <string>
#include <tuple>
#include <vector>

#include "caffe2/core/common_omp.h"
#include "caffe2/core/context.h"
#include "caffe2/core/logging.h"
#include "caffe2/core/operator.h"
#include "caffe2/core/tensor.h"
#include "caffe2/operators/elementwise_ops_utils.h"
#include "caffe2/utils/eigen_utils.h"
#include "caffe2/utils/math.h"

namespace caffe2 {

using NumericTypes = TensorTypes<int32_t, int64_t, float, double>;
using IntTypes = TensorTypes<int32_t, int64_t>;
using BoolTypes = TensorTypes<bool>;
using IntBoolTypes = TensorTypes<int32_t, int64_t, bool>; // discrete types

struct SameTypeAsInput {
  template <typename T>
  using type = T;
};

template <typename R>
struct FixedType {
  template <typename T>
  using type = R;
};

template <
    typename InputTypes,
    class Context,
    class Functor,
    class OutputTypeMap = SameTypeAsInput>
class UnaryElementwiseWithArgsOp final : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;

  template <class... Args>
  explicit UnaryElementwiseWithArgsOp(Args&&... args)
      : Operator<Context>(std::forward<Args>(args)...), functor_(*this) {}

  bool RunOnDevice() override {
    return DispatchHelper<InputTypes>::call(this, Input(0));
  }

  template <typename T>
  bool DoRunWithType() {
    const auto& X = Input(0);

    auto* Y = Output(
        0, X.sizes(), at::dtype<typename OutputTypeMap::template type<T>>());
    return functor_(
        X.numel(),
        X.template data<T>(),
        Y->template mutable_data<typename OutputTypeMap::template type<T>>(),
        &context_);
  }

 private:
  Functor functor_;
};

// UnaryFunctorWithDefaultCtor is a functor that can be used as the functor of
// an UnaryElementwiseWithArgsOp. It simply forwards the operator() call into
// another functor that doesn't accept arguments in its constructor.
template <class Functor>
struct UnaryFunctorWithDefaultCtor {
  explicit UnaryFunctorWithDefaultCtor(OperatorBase& /* op */) {}

  template <typename TIn, typename TOut, class Context>
  bool operator()(const int size, const TIn* X, TOut* Y, Context* context)
      const {
    return functor(size, X, Y, context);
  }

  Functor functor{};
};

// UnaryElementwiseOp is a wrapper around UnaryElementwiseWithArgsOp, with the
// difference that it takes a functor with default constructor, e.g. that does
// not need to take into consideration any arguments during operator creation.
template <
    typename InputTypes,
    class Context,
    class Functor,
    class OutputTypeMap = SameTypeAsInput>
using UnaryElementwiseOp = UnaryElementwiseWithArgsOp<
    InputTypes,
    Context,
    UnaryFunctorWithDefaultCtor<Functor>,
    OutputTypeMap>;

template <
    typename InputTypes,
    class Context,
    class Functor,
    class OutputTypeMap = SameTypeAsInput>
class BinaryElementwiseWithArgsOp final : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;

  template <class... Args>
  explicit BinaryElementwiseWithArgsOp(Args&&... args)
      : Operator<Context>(std::forward<Args>(args)...),
        OP_SINGLE_ARG(bool, "broadcast", legacy_broadcast_, false),
        OP_SINGLE_ARG(int, "axis", axis_, -1),
        OP_SINGLE_ARG(string, "axis_str", axis_str_, string("")),
        OP_SINGLE_ARG(string, "order", order_, "NCHW"),
        functor_(*this) {
    if (legacy_broadcast_) {
      if (axis_ != -1) {
        // Get axis from an explicit axis argument.
        CAFFE_ENFORCE_EQ(
            axis_str_.size(),
            0U,
            "Args axis and axis_str cannot be used simultaneously.");
      } else if (axis_str_.size()) {
        // Get the axis index semantically.
        CAFFE_ENFORCE_EQ(
            axis_str_.size(), 1U, "Unsupported axis string", axis_str_);
        const size_t semantic_axis_ = order_.find(axis_str_);
        CAFFE_ENFORCE_NE(
            semantic_axis_,
            string::npos,
            "Unrecognizable axis string ",
            axis_str_,
            " from order string ",
            order_);
        axis_ = semantic_axis_;
      } else {
        CAFFE_ENFORCE(
            axis_ == -1 && axis_str_.empty(),
            "Do not specify axis or axis_str if broadcast is not enabled.");
      }
    }
  }

  bool RunOnDevice() override {
    return DispatchHelper<InputTypes>::call(this, Input(0));
  }

  template <typename T>
  bool DoRunWithType() {
    const auto& A = Input(0);
    const auto& B = Input(1);

    const T* A_data = A.template data<T>();
    const T* B_data = B.template data<T>();
    std::vector<int> A_dims;
    std::vector<int> B_dims;
    std::vector<int64_t> C_dims;

    if (legacy_broadcast_) {
      CAFFE_ENFORCE(
          !IsInputOutputAlias(1, 0),
          "In-place is allowed only with the first tensor when "
          "legacy-broadcasting");
      C_dims = A.sizes().vec();
      if (B.numel() == 1) {
        A_dims = {static_cast<int>(A.numel())};
        B_dims = {1};
      } else {
        size_t pre, n, post;
        std::tie(pre, n, post) =
            elementwise_ops_utils::ComputeLegacyBroadcastSizes(A, B, axis_);
        A_dims = {
            static_cast<int>(pre), static_cast<int>(n), static_cast<int>(post)};
        B_dims = {static_cast<int>(n), 1};
      }
    } else {
      A_dims.reserve(A.sizes().size());
      B_dims.reserve(B.sizes().size());

      std::copy(
          A.sizes().cbegin(), A.sizes().cend(), std::back_inserter(A_dims));
      std::copy(
          B.sizes().cbegin(), B.sizes().cend(), std::back_inserter(B_dims));
      // TODO: change the types to vector<int64_t>
      auto C_dims_int =
          elementwise_ops_utils::ComputeBinaryBroadcastForwardDims(
              A_dims, B_dims);
      C_dims.reserve(C_dims_int.size());
      std::copy(
          C_dims_int.cbegin(), C_dims_int.cend(), std::back_inserter(C_dims));
      if (IsInputOutputAlias(0, 0)) {
        CAFFE_ENFORCE_EQ(C_dims_int, A_dims);
      } else if (IsInputOutputAlias(1, 0)) {
        CAFFE_ENFORCE_EQ(C_dims_int, B_dims);
      }
    }

    auto* C = Output(
        0, C_dims, at::dtype<typename OutputTypeMap::template type<T>>());
    auto* C_data =
        C->template mutable_data<typename OutputTypeMap::template type<T>>();
    return functor_.Forward(A_dims, B_dims, A_data, B_data, C_data, &context_);
  }

 private:
  const bool legacy_broadcast_;
  int axis_;
  const std::string axis_str_;
  const std::string order_;

  Functor functor_;
};

template <
    typename InputTypes,
    class Context,
    class Functor,
    class OutputTypeMap = SameTypeAsInput,
    class GradientTypeMap = SameTypeAsInput>
class BinaryElementwiseWithArgsGradientOp final : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;

  template <class... Args>
  explicit BinaryElementwiseWithArgsGradientOp(Args&&... args)
      : Operator<Context>(std::forward<Args>(args)...),
        OP_SINGLE_ARG(bool, "broadcast", legacy_broadcast_, false),
        OP_SINGLE_ARG(int, "axis", axis_, -1),
        OP_SINGLE_ARG(string, "axis_str", axis_str_, ""),
        OP_SINGLE_ARG(string, "order", order_, "NCHW"),
        functor_(*this) {
    if (legacy_broadcast_) {
      if (axis_ != -1) {
        // Get axis from an explicit axis argument.
        CAFFE_ENFORCE_EQ(
            axis_str_.size(),
            0U,
            "Args axis and axis_str cannot be used simultaneously.");
      } else if (axis_str_.size()) {
        // Get the axis index semantically.
        CAFFE_ENFORCE_EQ(
            axis_str_.size(), 1U, "Unsupported axis string", axis_str_);
        const size_t semantic_axis_ = order_.find(axis_str_);
        CAFFE_ENFORCE_NE(
            semantic_axis_,
            string::npos,
            "Unrecognizable axis string ",
            axis_str_,
            " from order string ",
            order_);
        axis_ = semantic_axis_;
      } else {
        CAFFE_ENFORCE(
            axis_ == -1 && axis_str_.empty(),
            "Do not specify axis or axis_str if broadcast is not enabled.");
      }
    }
  }

  bool RunOnDevice() override {
    return DispatchHelper<InputTypes>::call(this, Input(1));
  }

  template <typename T>
  bool DoRunWithType() {
    const auto& dC = Input(0);
    const auto& A = Input(1);
    const auto& B = Input(2);

    vector<int> A_dims;
    vector<int> B_dims;
    if (legacy_broadcast_) {
      if (B.numel() == 1) {
        A_dims = {static_cast<int>(A.numel())};
        B_dims = {1};
      } else {
        size_t pre, n, post;
        std::tie(pre, n, post) =
            elementwise_ops_utils::ComputeLegacyBroadcastSizes(A, B, axis_);
        A_dims = {
            static_cast<int>(pre), static_cast<int>(n), static_cast<int>(post)};
        B_dims = {static_cast<int>(n), 1};
      }
    } else {
      std::copy(
          A.sizes().cbegin(), A.sizes().cend(), std::back_inserter(A_dims));
      std::copy(
          B.sizes().cbegin(), B.sizes().cend(), std::back_inserter(B_dims));
    }
    const typename OutputTypeMap::template type<T>* C_data = nullptr;
    if (InputSize() == 4) {
      const auto& C = Input(3);
      C_data = C.template data<typename OutputTypeMap::template type<T>>();
    }
    const auto* dC_data =
        dC.template data<typename GradientTypeMap::template type<T>>();
    const T* A_data = A.template data<T>();
    const T* B_data = B.template data<T>();
    auto* dA = Output(
        0, A.sizes(), at::dtype<typename GradientTypeMap::template type<T>>());
    auto* dB = Output(
        1, B.sizes(), at::dtype<typename GradientTypeMap::template type<T>>());
    auto* dA_data =
        dA->template mutable_data<typename GradientTypeMap::template type<T>>();
    auto* dB_data =
        dB->template mutable_data<typename GradientTypeMap::template type<T>>();
    return functor_.Backward(
        A_dims,
        B_dims,
        dC_data,
        A_data,
        B_data,
        C_data,
        dA_data,
        dB_data,
        &context_);
  }

 private:
  const bool legacy_broadcast_;
  int axis_;
  const std::string axis_str_;
  const std::string order_;

  Functor functor_;
};

template <class Functor>
struct BinaryFunctorWithDefaultCtor {
  explicit BinaryFunctorWithDefaultCtor(OperatorBase& /* op */) {}

  template <typename TIn, typename TOut, class Context>
  bool Forward(
      const std::vector<int>& A_dims,
      const std::vector<int>& B_dims,
      const TIn* A_data,
      const TIn* B_data,
      TOut* C_data,
      Context* context) const {
    return functor.Forward(A_dims, B_dims, A_data, B_data, C_data, context);
  }

  template <typename TGrad, typename TIn, typename TOut, class Context>
  bool Backward(
      const std::vector<int>& A_dims,
      const std::vector<int>& B_dims,
      const TGrad* dC_data,
      const TIn* A_data,
      const TIn* B_data,
      const TOut* C_data,
      TGrad* dA_data,
      TGrad* dB_data,
      Context* context) const {
    return functor.Backward(
        A_dims,
        B_dims,
        dC_data,
        A_data,
        B_data,
        C_data,
        dA_data,
        dB_data,
        context);
  }

  Functor functor{};
};

template <class Functor>
struct BinaryFunctorWithBroadcastOptionsCtor {
  explicit BinaryFunctorWithBroadcastOptionsCtor(OperatorBase& op)
      : functor{op.GetSingleArgument<bool>("allow_broadcast_fastpath", false)} {}

  template <typename TIn, typename TOut, class Context>
  bool Forward(
      const std::vector<int>& A_dims,
      const std::vector<int>& B_dims,
      const TIn* A_data,
      const TIn* B_data,
      TOut* C_data,
      Context* context) const {
    return functor.Forward(A_dims, B_dims, A_data, B_data, C_data, context);
  }

  template <typename TGrad, typename TIn, typename TOut, class Context>
  bool Backward(
      const std::vector<int>& A_dims,
      const std::vector<int>& B_dims,
      const TGrad* dC_data,
      const TIn* A_data,
      const TIn* B_data,
      const TOut* C_data,
      TGrad* dA_data,
      TGrad* dB_data,
      Context* context) const {
    return functor.Backward(
        A_dims,
        B_dims,
        dC_data,
        A_data,
        B_data,
        C_data,
        dA_data,
        dB_data,
        context);
  }

  Functor functor;
};

// BinaryElementwiseOp is a wrapper around BinaryElementwiseWithArgsOp, with the
// difference that it takes a functor with default constructor, e.g. that does
// not need to take into consideration any arguments during operator creation.
template <
    typename InputTypes,
    class Context,
    class Functor,
    class TypeMap = SameTypeAsInput>
using BinaryElementwiseOp = BinaryElementwiseWithArgsOp<
    InputTypes,
    Context,
    BinaryFunctorWithDefaultCtor<Functor>,
    TypeMap>;

// BinaryElementwiseGradientOp is a wrapper around
// BinaryElementwiseGradientWithArgsOp, with the difference that it takes a
// functor with default constructor, e.g. that does not need to take into
// consideration any arguments during operator creation.
template <
    typename InputTypes,
    class Context,
    class Functor,
    class OutputTypeMap = SameTypeAsInput,
    class GradientTypeMap = SameTypeAsInput>
using BinaryElementwiseGradientOp = BinaryElementwiseWithArgsGradientOp<
    InputTypes,
    Context,
    BinaryFunctorWithDefaultCtor<Functor>,
    OutputTypeMap,
    GradientTypeMap>;

// BinaryElementwiseBroadcastOp is a wrapper around BinaryElementwiseWithArgsOp,
// with the difference that it takes a functor with a constructor that accepts
// broadcast-related arguments (just a single boolean for whether broadcast
// fastpaths are allowed at the time this comment was written).
template <
    typename InputTypes,
    class Context,
    class Functor,
    class TypeMap = SameTypeAsInput>
using BinaryElementwiseBroadcastOp = BinaryElementwiseWithArgsOp<
    InputTypes,
    Context,
    BinaryFunctorWithBroadcastOptionsCtor<Functor>,
    TypeMap>;

// BinaryElementwiseGradientBroadcastOp is a wrapper around
// BinaryElementwiseWithArgsGradientOp, with the difference that it takes a
// functor with a constructor that accepts broadcast-related arguments (just a
// single boolean for whether broadcast fastpaths are allowed at the time this
// comment was written).
template <
    typename InputTypes,
    class Context,
    class Functor,
    class OutputTypeMap = SameTypeAsInput,
    class GradientTypeMap = SameTypeAsInput>
using BinaryElementwiseGradientBroadcastOp = BinaryElementwiseWithArgsGradientOp<
    InputTypes,
    Context,
    BinaryFunctorWithBroadcastOptionsCtor<Functor>,
    OutputTypeMap,
    GradientTypeMap>;

// Forward-only Unary Functors.
template <class Context>
struct NotFunctor {
  bool operator()(const int N, const bool* X, bool* Y, Context* context) const {
    math::Not(N, X, Y, context);
    return true;
  }
};

template <class Context>
struct SignFunctor {
  template <typename T>
  bool operator()(const int N, const T* X, T* Y, Context* context) const {
    math::Sign(N, X, Y, context);
    return true;
  }
};

// Forward-only Binary Functors.
#define C10_DECLARE_FORWARD_ONLY_BINARY_FUNCTOR(FunctorName) \
  template <class Context>                                   \
  struct FunctorName##Functor {                              \
    template <typename TIn, typename TOut>                   \
    bool Forward(                                            \
        const std::vector<int>& A_dims,                      \
        const std::vector<int>& B_dims,                      \
        const TIn* A,                                        \
        const TIn* B,                                        \
        TOut* C,                                             \
        Context* context) const {                            \
      math::FunctorName(                                     \
          A_dims.size(),                                     \
          A_dims.data(),                                     \
          B_dims.size(),                                     \
          B_dims.data(),                                     \
          A,                                                 \
          B,                                                 \
          C,                                                 \
          context);                                          \
      return true;                                           \
    }                                                        \
  };

// Compare functors.
C10_DECLARE_FORWARD_ONLY_BINARY_FUNCTOR(EQ);
C10_DECLARE_FORWARD_ONLY_BINARY_FUNCTOR(NE);
C10_DECLARE_FORWARD_ONLY_BINARY_FUNCTOR(LT);
C10_DECLARE_FORWARD_ONLY_BINARY_FUNCTOR(LE);
C10_DECLARE_FORWARD_ONLY_BINARY_FUNCTOR(GT);
C10_DECLARE_FORWARD_ONLY_BINARY_FUNCTOR(GE);

// Logical functors.
C10_DECLARE_FORWARD_ONLY_BINARY_FUNCTOR(And);
C10_DECLARE_FORWARD_ONLY_BINARY_FUNCTOR(Or);
C10_DECLARE_FORWARD_ONLY_BINARY_FUNCTOR(Xor);

// Bitwise functors.
C10_DECLARE_FORWARD_ONLY_BINARY_FUNCTOR(BitwiseAnd);
C10_DECLARE_FORWARD_ONLY_BINARY_FUNCTOR(BitwiseOr);
C10_DECLARE_FORWARD_ONLY_BINARY_FUNCTOR(BitwiseXor);

#undef C10_DECLARE_FORWARD_ONLY_BINARY_FUNCTOR

namespace SRLHelper {

template <typename T>
void sum2one(const T* a, T* y, size_t n);

template <typename T>
void RunWithBroadcastFront(const T* a, T* y, size_t pre, size_t n, CPUContext*);

template <typename T>
void RunWithBroadcastBack(const T* a, T* y, size_t post, size_t n, CPUContext*);

template <typename T>
void RunWithBroadcast2(
    const T* a,
    T* y,
    size_t pre,
    size_t n,
    size_t post,
    CPUContext*);

} // namespace SRLHelper

// Sum reduction operator that is used for computing the gradient in cases
// where the forward op is in broadcast mode.
template <class Context>
class SumReduceLikeOp final : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  template <class... Args>
  explicit SumReduceLikeOp(Args&&... args)
      : Operator<Context>(std::forward<Args>(args)...),
        OP_SINGLE_ARG(int, "axis", axis_, -1),
        OP_SINGLE_ARG(string, "axis_str", axis_str_, ""),
        OP_SINGLE_ARG(string, "order", order_, "NCHW") {
    if (axis_ != -1) {
      // Get axis from an explicit axis argument.
      CAFFE_ENFORCE_EQ(
          axis_str_.size(),
          0U,
          "Args axis and axis_str cannot be used simultaneously.");
    } else if (axis_str_.size()) {
      // Get the axis index semantically.
      CAFFE_ENFORCE_EQ(
          axis_str_.size(), 1U, "Unsupported axis string", axis_str_);
      size_t semantic_axis = order_.find(axis_str_);
      CAFFE_ENFORCE_NE(
          semantic_axis,
          string::npos,
          "Unrecognizable axis string ",
          axis_str_,
          " from order string ",
          order_);
      axis_ = semantic_axis;
    }
  }

  bool RunOnDevice() override {
    return DispatchHelper<TensorTypes<float, double>>::call(this, Input(0));
  }

  template <typename T>
  bool DoRunWithType();

 private:
  int axis_;
  string axis_str_;
  string order_;
  Tensor ones_{Context::GetDeviceType()};
  Tensor sum_buffer_{Context::GetDeviceType()};
};

} // namespace caffe2

#endif // CAFFE2_OPERATORS_ELEMENTWISE_OPS_H_
