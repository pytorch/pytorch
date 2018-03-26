#ifndef CAFFE2_OPERATORS_ELEMENTWISE_OP_H_
#define CAFFE2_OPERATORS_ELEMENTWISE_OP_H_

#include "caffe2/core/common_omp.h"
#include "caffe2/core/context.h"
#include "caffe2/core/logging.h"
#include "caffe2/core/operator.h"
#include "caffe2/core/tensor.h"
#include "caffe2/utils/math.h"

#include <tuple>

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
    class TypeMap = SameTypeAsInput>
class UnaryElementwiseWithArgsOp : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  UnaryElementwiseWithArgsOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<Context>(operator_def, ws), functor_(*this) {}

  bool RunOnDevice() override {
    return DispatchHelper<InputTypes>::call(this, Input(0));
  }

  template <typename T>
  bool DoRunWithType() {
    auto& input = Input(0);
    auto* output = Output(0);
    output->ResizeLike(input);
    using R = typename TypeMap::template type<T>;
    functor_(
        input.size(),
        input.template data<T>(),
        output->template mutable_data<R>(),
        &context_);
    return true;
  }

 private:
  Functor functor_;
};

/**
 * WithDefaultConstructor is a functor that can be used as the functor of an
 * UnaryElementwiseWithArgsOp. It simply forwards the operator() call into
 * another functor that doesn't accept arguments in its constructor.
 */
template <typename Functor>
struct WithDefaultConstructor {
  explicit WithDefaultConstructor(OperatorBase& /*op*/) {}

  template <typename In, typename Out, typename Context>
  void operator()(int n, const In* in, Out* out, Context* c) {
    Functor()(n, in, out, c);
  }
};

/**
 * UnaryElementwiseOp is a wrapper around UnaryElementwiseWithArgsOp, with the
 * difference that it takes a functor with default constructor, e.g. that does
 * not need to take into consideration any arguments during operator creation.
 */
template <
    typename InputTypes,
    class Context,
    class Functor,
    class OutputType = SameTypeAsInput>
using UnaryElementwiseOp = UnaryElementwiseWithArgsOp<
    InputTypes,
    Context,
    WithDefaultConstructor<Functor>,
    OutputType>;

template <typename Context>
std::tuple<size_t, size_t, size_t> calculate_broadcast_sizes(
    const Tensor<Context>& A,
    const Tensor<Context>& B,
    int axis) {
  CAFFE_ENFORCE_GE(
      A.ndim(),
      B.ndim(),
      "If you are doing broadcasting, input1 should have "
      "a smaller or equal number of dimensions.");
  if (axis == -1) {
    axis = A.ndim() - B.ndim();
  }
  CAFFE_ENFORCE(
      axis >= 0 && axis <= A.ndim() - B.ndim(),
      "Broadcast axis should be in the range of"
      "[0, A.ndim() - B.ndim()], but axis = ",
      axis);

  int b_dim_start = 0;
  while (b_dim_start < B.ndim() && B.dim(b_dim_start) == 1) {
    ++b_dim_start;
  }
  int b_dim_end = B.ndim() - 1;
  while (b_dim_end >= b_dim_start && B.dim(b_dim_end) == 1) {
    --b_dim_end;
  }
  size_t pre = 1, n = 1, post = 1;
  for (int i = 0; i < axis + b_dim_start; ++i) {
    pre *= A.dim(i);
  }
  for (int i = b_dim_start; i <= b_dim_end; ++i) {
    CAFFE_ENFORCE_EQ(
        A.dim(i + axis), B.dim(i), "Broadcast dimension mismatch.");
    n *= B.dim(i);
  }
  for (int i = axis + b_dim_end + 1; i < A.ndim(); ++i) {
    post *= A.dim(i);
  }
  return std::make_tuple(pre, n, post);
}

/**
 * Performs a binary operation (e.g. +, - or /) with optional broadcast support.
 *
 * Functor specifies actual operation to be performed.
 *
 * If AllowBroadcast=false tensors has to be of exactly the same shape.
 *
 * If AllowBroadcast=true it support limited broadcasting of the right-hand-side
 * argument to match the shape of left-hand-side argument. Only suffix matching
 * is supported for now (1-dim expansion is allowed on both ends). E.g. this
 * will be accepted:
 * A dims: 2 3 4 5 6
 * B dims:   1 4 1
 *           ^
 *           |
 *          axis = 1
 */
template <
    typename InputTypes,
    class Context,
    class Functor,
    class TypeMap = SameTypeAsInput>
class BinaryElementwiseOp : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;

  BinaryElementwiseOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<Context>(operator_def, ws),
        OP_SINGLE_ARG(bool, "broadcast", enable_broadcast_, 0),
        OP_SINGLE_ARG(int, "axis", axis_, -1),
        OP_SINGLE_ARG(string, "axis_str", axis_str_, ""),
        OP_SINGLE_ARG(string, "order", order_, "NCHW"),
        functor_() {
    // Figure out the correct axis to use.
    if (enable_broadcast_) {
      if (axis_ != -1) {
        // Get axis from an explicit axis argument.
        CAFFE_ENFORCE_EQ(
            axis_str_.size(),
            0,
            "Args axis and axis_str cannot be used simultaneously.");
      } else if (axis_str_.size()) {
        // Get the axis index semantically.
        CAFFE_ENFORCE_EQ(
            axis_str_.size(), 1, "Unsupported axis string", axis_str_);
        size_t semantic_axis_ = order_.find(axis_str_);
        CAFFE_ENFORCE_NE(
            semantic_axis_,
            string::npos,
            "Unrecognizable axis string ",
            axis_str_,
            " from order string ",
            order_);
        axis_ = semantic_axis_;
      }
    } else {
      CAFFE_ENFORCE(
          axis_ == -1 && axis_str_.size() == 0,
          "Do not specify axis or axis_str if broadcast is not enabled.");
    }
  }

  bool RunOnDevice() override {
    return DispatchHelper<InputTypes>::call(this, Input(0));
  }

  template <typename T>
  bool DoRunWithType() {
    const auto& A = Input(0);
    const auto& B = Input(1);
    auto* C = Output(0);
    CAFFE_ENFORCE(
        &B != C || !enable_broadcast_,
        "In-place is allowed only with the first tensor when broadcasting");
    C->ResizeLike(A);
    const T* Adata = A.template data<T>();
    const T* Bdata = B.template data<T>();
    auto* Cdata =
        C->template mutable_data<typename TypeMap::template type<T>>();
    if (!enable_broadcast_) {
      CAFFE_ENFORCE_EQ(
          A.dims(),
          B.dims(),
          "Dimension mismatch - did you forget to set broadcast=1?");
      functor_.template Run<false>(A.size(), Adata, Bdata, Cdata, &context_);
    } else if (B.size() == 1) {
      functor_.template Run<true>(A.size(), Adata, Bdata, Cdata, &context_);
    } else {
      size_t pre, n, post;
      std::tie(pre, n, post) = calculate_broadcast_sizes(A, B, axis_);
      if (post == 1) {
        functor_.RunWithBroadcast(Adata, Bdata, Cdata, pre, n, &context_);
      } else {
        functor_.RunWithBroadcast2(
            Adata, Bdata, Cdata, pre, n, post, &context_);
      }
    }
    return true;
  }

 private:
  bool enable_broadcast_;
  int axis_;
  string axis_str_;
  string order_;
  Functor functor_;
};

template <typename Functor>
struct WithoutBroadcast {
  template <bool b_is_scalar, typename T, typename R, typename Context>
  inline void Run(size_t n, const T* a, const T* b, R* out, Context* c) {
    if (b_is_scalar) {
      CAFFE_THROW("Broadcast not supported.");
    } else {
      Functor().Run(n, a, b, out, c);
    }
  }
  template <typename T, typename R, typename Context>
  inline void RunWithBroadcast(
      const T* /*a*/,
      const T* /*b*/,
      R* /*out*/,
      size_t /*pre*/,
      size_t /*n*/,
      Context*) {
    CAFFE_NOT_IMPLEMENTED;
  }
  template <typename T, typename R, typename Context>
  inline void RunWithBroadcast2(
      const T* /*a*/,
      const T* /*b*/,
      R* /*out*/,
      size_t /*pre*/,
      size_t /*n*/,
      size_t /*post*/,
      Context*) {
    CAFFE_NOT_IMPLEMENTED;
  }
};

// Gradient operator for elementwise division.
template <class Context>
class DivGradientOp final : public Operator<Context> {
 public:
  USE_SIMPLE_CTOR_DTOR(DivGradientOp);
  USE_OPERATOR_CONTEXT_FUNCTIONS;

  bool RunOnDevice() override;
};

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
  SumReduceLikeOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<Context>(operator_def, ws),
        OP_SINGLE_ARG(int, "axis", axis_, -1),
        OP_SINGLE_ARG(string, "axis_str", axis_str_, ""),
        OP_SINGLE_ARG(string, "order", order_, "NCHW") {
    if (axis_ != -1) {
      // Get axis from an explicit axis argument.
      CAFFE_ENFORCE_EQ(
          axis_str_.size(),
          0,
          "Args axis and axis_str cannot be used simultaneously.");
    } else if (axis_str_.size()) {
      // Get the axis index semantically.
      CAFFE_ENFORCE_EQ(
          axis_str_.size(), 1, "Unsupported axis string", axis_str_);
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
  Tensor<Context> ones_;
  Tensor<Context> sum_buffer_;
};

template <class Context>
bool DivGradientOp<Context>::RunOnDevice() {
  auto& Y = Input(0);
  auto& Z = Input(1);
  auto& dZ = Input(2);
  auto* dX = Output(0);
  auto* dY = Output(1);

  dX->ResizeLike(Y);
  dY->ResizeLike(Y);

  const float* Ydata = Y.template data<float>();
  const float* Zdata = Z.template data<float>();
  const float* dZdata = dZ.template data<float>();
  float* dXdata = dX->template mutable_data<float>();
  float* dYdata = dY->template mutable_data<float>();

  ElementWiseDivide(context_, Y.size(), dXdata, dYdata, dZdata, Ydata, Zdata);
  return true;
}

// For arithmetic operators, Eigen provides a good way to vectorize even
// when broadcasting.
#define EIGEN_FUNCTOR(name, eigen_op, input_type, output_type)               \
  struct Eigen##name##Functor {                                              \
    template <int b_is_scalar, typename T, typename R>                       \
    inline void Run(size_t n, const T* a, const T* b, R* out, CPUContext*) { \
      if (b_is_scalar) {                                                     \
        EigenVectorArrayMap<R>(out, n) =                                     \
            eigen_op((ConstEigenVectorArrayMap<T>(a, n)), (b[0]));           \
      } else {                                                               \
        EigenVectorArrayMap<R>(out, n) = eigen_op(                           \
            (ConstEigenVectorArrayMap<T>(a, n)),                             \
            (ConstEigenVectorArrayMap<T>(b, n)));                            \
      }                                                                      \
    }                                                                        \
    template <typename T, typename R>                                        \
    void RunWithBroadcast(                                                   \
        const T* a,                                                          \
        const T* b,                                                          \
        R* out,                                                              \
        size_t pre,                                                          \
        size_t n,                                                            \
        CPUContext*) {                                                       \
      EigenArrayMap<R>(out, n, pre) = eigen_op(                              \
          (ConstEigenArrayMap<T>(a, n, pre).colwise()),                      \
          (ConstEigenVectorArrayMap<T>(b, n)));                              \
    }                                                                        \
    template <typename T, typename R>                                        \
    void RunWithBroadcast2(                                                  \
        const T* a,                                                          \
        const T* b,                                                          \
        R* out,                                                              \
        size_t pre,                                                          \
        size_t n,                                                            \
        size_t post,                                                         \
        CPUContext*) {                                                       \
      for (int i = 0; i < pre; ++i) {                                        \
        EigenArrayMap<R>(out + i * n * post, post, n) = eigen_op(            \
            (ConstEigenArrayMap<T>(a + i * n * post, post, n).rowwise()),    \
            (Eigen::Map<const Eigen::Array<T, 1, Eigen::Dynamic>>(b, n)));   \
      }                                                                      \
    }                                                                        \
  };                                                                         \
  REGISTER_CPU_OPERATOR(                                                     \
      name,                                                                  \
      BinaryElementwiseOp<                                                   \
          input_type,                                                        \
          CPUContext,                                                        \
          Eigen##name##Functor,                                              \
          output_type>)

} // namespace caffe2

#endif // CAFFE2_OPERATORS_ELEMENTWISE_OP_H_
