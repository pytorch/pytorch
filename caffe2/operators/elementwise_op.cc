#include "caffe2/operators/elementwise_op.h"

namespace caffe2 {

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

// For some comparison and logical operators, eigen does not have vectorized
// math so we need to improvise.
#define NAIVE_FUNCTOR(name, op, input_type, output_type)                       \
  struct Naive##name##Functor {                                                \
    template <int b_is_scalar, typename T, typename R>                         \
    inline void Run(size_t n, const T* a, const T* b, R* out, CPUContext*) {   \
      for (int i = 0; i < n; ++i) {                                            \
        out[i] = op(a[i], b[b_is_scalar ? 0 : i]);                             \
      }                                                                        \
    }                                                                          \
    template <typename T, typename R>                                          \
    void RunWithBroadcast(                                                     \
        const T* a,                                                            \
        const T* b,                                                            \
        R* out,                                                                \
        size_t pre,                                                            \
        size_t n,                                                              \
        CPUContext*) {                                                         \
      for (int i = 0; i < pre; ++i) {                                          \
        for (int j = 0; j < n; ++j) {                                          \
          out[i * n + j] = op(a[i * n + j], b[j]);                             \
        }                                                                      \
      }                                                                        \
    }                                                                          \
    template <typename T, typename R>                                          \
    void RunWithBroadcast2(                                                    \
        const T* a,                                                            \
        const T* b,                                                            \
        R* out,                                                                \
        size_t pre,                                                            \
        size_t n,                                                              \
        size_t post,                                                           \
        CPUContext*) {                                                         \
      for (int i = 0; i < pre; ++i) {                                          \
        for (int j = 0; j < n; ++j) {                                          \
          for (int k = 0; k < post; ++k) {                                     \
            out[(i * n + j) * post + k] = op(a[(i * n + j) * post + k], b[j]); \
          }                                                                    \
        }                                                                      \
      }                                                                        \
    }                                                                          \
  };                                                                           \
  REGISTER_CPU_OPERATOR(                                                       \
      name,                                                                    \
      BinaryElementwiseOp<                                                     \
          input_type,                                                          \
          CPUContext,                                                          \
          Naive##name##Functor,                                                \
          output_type>)

// See the operations supported here:
// https://eigen.tuxfamily.org/dox-devel/group__QuickRefPage.html
#define EIGEN_ADD(x, y) ((x) + (y))
EIGEN_FUNCTOR(Add, EIGEN_ADD, NumericTypes, SameTypeAsInput);
#undef EIGEN_ADD
#define EIGEN_SUB(x, y) ((x) - (y))
EIGEN_FUNCTOR(Sub, EIGEN_SUB, NumericTypes, SameTypeAsInput);
#undef EIGEN_SUB
#define EIGEN_MUL(x, y) ((x) * (y))
EIGEN_FUNCTOR(Mul, EIGEN_MUL, NumericTypes, SameTypeAsInput);
#undef EIGEN_MUL
#define EIGEN_DIV(x, y) ((x) / (y))
EIGEN_FUNCTOR(Div, EIGEN_DIV, NumericTypes, SameTypeAsInput);
#undef EIGEN_DIV

#define NAIVE_LT(x, y) ((x) < (y))
NAIVE_FUNCTOR(LT, NAIVE_LT, NumericTypes, FixedType<bool>);
#undef NAIVE_LT
#define NAIVE_LE(x, y) ((x) <= (y))
NAIVE_FUNCTOR(LE, NAIVE_LE, NumericTypes, FixedType<bool>);
#undef NAIVE_LE
#define NAIVE_GT(x, y) ((x) > (y))
NAIVE_FUNCTOR(GT, NAIVE_GT, NumericTypes, FixedType<bool>);
#undef NAIVE_GT
#define NAIVE_GE(x, y) ((x) >= (y))
NAIVE_FUNCTOR(GE, NAIVE_GE, NumericTypes, FixedType<bool>);
#undef NAIVE_GE
#define NAIVE_EQ(x, y) ((x) == (y))
NAIVE_FUNCTOR(EQ, NAIVE_EQ, IntTypes, FixedType<bool>);
#undef NAIVE_EQ
#define NAIVE_AND(x, y) ((x) & (y))
NAIVE_FUNCTOR(And, NAIVE_AND, BoolTypes, FixedType<bool>);
#undef NAIVE_AND
#define NAIVE_OR(x, y) ((x) | (y))
NAIVE_FUNCTOR(Or, NAIVE_OR, BoolTypes, FixedType<bool>);
#undef NAIVE_OR
#define NAIVE_XOR(x, y) ((x) ^ (y))
NAIVE_FUNCTOR(Xor, NAIVE_XOR, BoolTypes, FixedType<bool>);
#undef NAIVE_XOR

struct NotFunctor {
  inline void operator()(const int n, const bool* x, bool* y, CPUContext*) {
    for (int i = 0; i < n; ++i) {
      y[i] = !x[i];
    }
  }
};
REGISTER_CPU_OPERATOR(
    Not,
    UnaryElementwiseOp<BoolTypes, CPUContext, NotFunctor>);

template <>
bool DivGradientOp<float, CPUContext>::RunOnDevice() {
  auto& Y = Input(0);
  auto& Z = Input(1);
  auto& dZ = Input(2);
  auto* dX = Output(0);
  auto* dY = Output(1);
  DCHECK_GT(Y.size(), 0);
  DCHECK_GT(Z.size(), 0);
  dX->ResizeLike(Y);
  dY->ResizeLike(Y);

  const float* Ydata = Y.data<float>();
  const float* Zdata = Z.data<float>();
  const float* dZdata = dZ.data<float>();
  float* dXdata = dX->mutable_data<float>();
  float* dYdata = dY->mutable_data<float>();
  #pragma omp parallel for
  for (int i = 0; i < Y.size(); ++i) {
    dXdata[i] = dZdata[i] / Ydata[i];
    dYdata[i] = - (dZdata[i] * Zdata[i]) / Ydata[i];
  }
  return true;
}

REGISTER_CPU_OPERATOR(DivGradient, DivGradientOp<float, CPUContext>);

}  // namespace caffe2
