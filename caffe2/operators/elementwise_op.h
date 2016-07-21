#ifndef CAFFE2_OPERATORS_ELEMENTWISE_OP_H_
#define CAFFE2_OPERATORS_ELEMENTWISE_OP_H_

#include "caffe2/core/context.h"
#include "caffe2/core/operator.h"
#include "caffe2/utils/math.h"
#include "caffe2/core/logging.h"

namespace caffe2 {

using NumericTypes = TensorTypes<int32_t, int64_t, float, double>;
class SameTypeAsInput {};

template<typename OutputTemplate, typename InputType>
struct TypeForOutput {
  using value = OutputTemplate;
};

template <typename InputType>
struct TypeForOutput<SameTypeAsInput, InputType> {
  using value = InputType;
};

template <typename InputTypes, class Context, class Functor,
          class OutputType = SameTypeAsInput>
class UnaryElementwiseOp : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  USE_SIMPLE_CTOR_DTOR(UnaryElementwiseOp);

  bool RunOnDevice() override {
    return DispatchHelper<InputTypes>::call(this, Input(0));
  }

  template <typename T>
  bool DoRunWithType() {
    auto& input = Input(0);
    auto* output = Output(0);
    output->ResizeLike(input);
    using R = typename TypeForOutput<OutputType, T>::value;
    Functor()(input.size(), input.template data<T>(),
              output->template mutable_data<R>(), &context_);
    return true;
  }

};

/**
 * Performs a binary operation (e.g. +, - or /) with optional broadcast support.
 *
 * Functor specifies actual operation to be performed.
 *
 * If AllowBroadcast=false tensors has to be of exactly the same shape.
 *
 * If AllowBroadcast=true it support limited broadcasting of the right-hand-side
 * argument to match the shape of left-hand-side argument. Only suffix matching
 * is supported for now, 1-dim expansion doesn't work yet. More precisely
 * tensors A and B can be operated on iff
 *   `shape(A)[-len(shape(B)):] == * shape(B)`
 */
template <
    typename InputTypes,
    class Context,
    class Functor,
    class OutputType = SameTypeAsInput,
    bool AllowBroadcast = false>
class BinaryElementwiseOp : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;

  BinaryElementwiseOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<Context>(operator_def, ws),
        OP_SINGLE_ARG(int, "broadcast", enable_broadcast_, 0) {}

  bool RunOnDevice() override {
    return DispatchHelper<InputTypes>::call(this, Input(0));
  }

  template <typename T>
  bool DoRunWithType() {
    auto& input0 = Input(0);
    auto& input1 = Input(1);
    // Currently we don't check shapes. Shall we enforce shape matching?
    auto N0 = input0.size();
    auto N1 = input1.size();
    DCHECK_GT(N0, 0);
    DCHECK_GT(N1, 0);
    if (N0 != N1 || input0.ndim() != input1.ndim()) {
      RunBroadcast<T, AllowBroadcast>();
      return true;
    }
    DCHECK(input0.dims() == input1.dims())
        << "Tensor should have exactly the same shape"
#ifndef __CUDACC__
        << ", have " << input0.dims() << " and " << input1.dims()
#endif
        ;
    auto* output = Output(0);
    output->ResizeLike(input0);
    using R = typename TypeForOutput<OutputType, T>::value;
    Functor()(
        N0,
        input0.template data<T>(),
        input1.template data<T>(),
        output->template mutable_data<R>(),
        &context_);
    return true;
  }

 private:
  static bool isShapeSuffix(const vector<TIndex>& a, const vector<TIndex>& b) {
    if (a.size() < b.size()) {
      return false;
    }
    for (int i = 0; i < b.size(); ++i) {
      if (a[a.size()-1-i] != b[b.size()-1-i]) {
        return false;
      }
    }
    return true;
  }

  template <typename T, bool Broadcast>
  typename std::enable_if<Broadcast, void>::type RunBroadcast() {
    CHECK(enable_broadcast_)
        << "Tensor have different shape, pass `broadcast=1` to enable "
        << "broadcasting of the second argument";
    // We enforce that the second argument gets broadcasted. For addition it's
    // not strictly necessary, but makes code easier
    auto& input0 = Input(0);
    auto& input1 = Input(1);
    auto* output = Output(0);
    auto N0 = input0.size();
    auto N1 = input1.size();
    CHECK_EQ(N0 % N1, 0)
        << "Sizes of tensors don't match for broadcasting to work: " << N0
        << " and " << N1;
    DCHECK(isShapeSuffix(input0.dims(), input1.dims()))
        << "Bad shapes for broadcasting: " << input0.dims() << " and "
        << input1.dims();
    CHECK_NE(&input1, output)
        << "In-place is allowed only with the first tensor when broadcasting";
    output->ResizeLike(input0);
    using R = typename TypeForOutput<OutputType, T>::value;
    Functor().WithBroadcast(
        N0 / N1,
        N1,
        input0.template data<T>(),
        input1.template data<T>(),
        output->template mutable_data<R>(),
        &context_);
  }

  template <typename T, bool Broadcast>
  typename std::enable_if<!Broadcast, void>::type RunBroadcast() {
    CHECK(false) << "Broadcasting is not supported for this op, "
                 << "inputs should be of the same size";
  }

  bool enable_broadcast_;
};

template <typename T, class Context>
class DivGradientOp final : public Operator<Context> {
 public:
  USE_SIMPLE_CTOR_DTOR(DivGradientOp);
  USE_OPERATOR_CONTEXT_FUNCTIONS;

  bool RunOnDevice() override;
};

#define CAFFE2_BINARY_FUNCTOR_WRAPPER(name)                               \
  struct name##Functor {                                                  \
    template <typename T, class Context>                                  \
    inline void operator()(                                               \
        const int n,                                                      \
        const T* x,                                                       \
        const T* y,                                                       \
        T* output,                                                        \
        Context* device_context) {                                        \
      math::name<T, Context>(n, x, y, output, device_context);            \
    }                                                                     \
    template <typename T, class Context>                                  \
    inline void WithBroadcast(                                            \
        const int m,                                                      \
        const int n,                                                      \
        const T* a,                                                       \
        const T* b,                                                       \
        T* y,                                                             \
        Context* device_context) {                                        \
      math::name##ToRow<T, Context>(m, n, a, b, y, device_context);       \
    }                                                                     \
  };                                                                      \
  template <class DC>                                                     \
  using name##Op = BinaryElementwiseOp<NumericTypes, DC,                  \
    name##Functor, SameTypeAsInput, true>

CAFFE2_BINARY_FUNCTOR_WRAPPER(Add);
CAFFE2_BINARY_FUNCTOR_WRAPPER(Sub);
CAFFE2_BINARY_FUNCTOR_WRAPPER(Mul);
CAFFE2_BINARY_FUNCTOR_WRAPPER(Div);

#undef CAFFE2_BINARY_FUNCTOR_WRAPPER

#define CAFFE2_BINARY_FUNCTOR_BINARY_RESULT_WRAPPER(name)                     \
  struct name##Functor {                                                      \
    template <typename T, class Context>                                      \
    inline void operator()(                                                   \
        const int n,                                                          \
        const T* x,                                                           \
        const T* y,                                                           \
        bool* output,                                                         \
        Context* device_context) {                                            \
      math::name<T, Context>(n, x, y, output, device_context);                \
    }                                                                         \
    template <typename T, typename Context>                                   \
    inline void WithBroadcast(                                                \
        const int m,                                                          \
        const int n,                                                          \
        const T* a,                                                           \
        const T* b,                                                           \
        bool* y,                                                              \
        Context* device_context) {                                            \
      math::name##ToRow<T, Context>(m, n, a, b, y, device_context);           \
    }                                                                         \
  };                                                                          \
  template <class DC>                                                         \
  using name##Op = BinaryElementwiseOp<                                       \
      NumericTypes, DC, name##Functor, bool, true>

CAFFE2_BINARY_FUNCTOR_BINARY_RESULT_WRAPPER(LT);
CAFFE2_BINARY_FUNCTOR_BINARY_RESULT_WRAPPER(LE);
CAFFE2_BINARY_FUNCTOR_BINARY_RESULT_WRAPPER(GT);
CAFFE2_BINARY_FUNCTOR_BINARY_RESULT_WRAPPER(GE);

#undef CAFFE2_BINARY_FUNCTOR_BINARY_RESULT_WRAPPER
}  // namespace caffe2

#endif  // CAFFE2_OPERATORS_ELEMENTWISE_OP_H_
