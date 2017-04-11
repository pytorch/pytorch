#include "caffe2/operators/elementwise_op.h"

namespace caffe2 {

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

#define EIGEN_SUB(x, y) ((x) - (y))
EIGEN_FUNCTOR(Sub, EIGEN_SUB, NumericTypes, SameTypeAsInput);
#undef EIGEN_SUB
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

void ElementWiseDivide(
    CPUContext& context,
    const int n,
    float* dXdata,
    float* dYdata,
    const float* dZdata,
    const float* Ydata,
    const float* Zdata) {
  ConstEigenVectorArrayMap<float> dZdataVec(dZdata, n);
  ConstEigenVectorArrayMap<float> YdataVec(Ydata, n);
  ConstEigenVectorArrayMap<float> ZdataVec(Zdata, n);
  EigenVectorArrayMap<float>(dXdata, n) = dZdataVec / YdataVec;
  EigenVectorArrayMap<float>(dYdata, n) = - (dZdataVec * ZdataVec) / YdataVec;
}

REGISTER_CPU_OPERATOR(DivGradient, DivGradientOp<CPUContext>);

}  // namespace caffe2
