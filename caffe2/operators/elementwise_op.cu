#include "caffe2/core/common_gpu.h"
#include "caffe2/core/context_gpu.h"
#include "caffe2/operators/elementwise_op.h"

namespace caffe2 {

#define CUDA_FUNCTOR(name, op, input_type, output_type) \
template <int b_is_scalar, typename T, typename R> \
__global__ void name##Kernel(const T* a, const T* b, R* out, int n) { \
  CUDA_1D_KERNEL_LOOP(i, n) { \
    out[i] = op(a[i], b[b_is_scalar ? 0 : i]); \
  } \
} \
template <typename T, typename R> \
__global__ void name##BroadcastKernel( \
    const T* a, const T* b, R* out, int pre, int n) { \
  CUDA_1D_KERNEL_LOOP(i, pre * n) { \
    out[i] = op(a[i], b[i % n]); \
  } \
} \
template <typename T, typename R> \
__global__ void name##Broadcast2Kernel( \
    const T* a, const T* b, R* out, int pre, int n, int post) { \
  CUDA_1D_KERNEL_LOOP(i, pre * n * post) { \
    out[i] = op(a[i], b[(i / post) % n]); \
  } \
} \
 \
struct Cuda##name##Functor { \
  template <bool b_is_scalar, typename T, typename R> \
  inline void Run( \
      size_t n, const T* a, const T* b, R* out, CUDAContext* context) { \
    name##Kernel<b_is_scalar, T, R><<<CAFFE_GET_BLOCKS(n), \
                                      CAFFE_CUDA_NUM_THREADS, \
                                      0, context->cuda_stream()>>>( \
        a, b, out, n); \
  } \
  template <typename T, typename R> \
  void RunWithBroadcast( \
      const T* a, const T* b, R* out, size_t pre, size_t n, \
      CUDAContext* context) { \
    name##BroadcastKernel<T, R><<<CAFFE_GET_BLOCKS(pre * n), \
                                  CAFFE_CUDA_NUM_THREADS, \
                                  0, context->cuda_stream()>>>( \
        a, b, out, pre, n); \
  } \
  template <typename T, typename R> \
  void RunWithBroadcast2( \
      const T* a, const T* b, R* out, size_t pre, size_t n, size_t post, \
      CUDAContext* context) { \
    name##Broadcast2Kernel<T, R><<<CAFFE_GET_BLOCKS(pre * n * post), \
                                   CAFFE_CUDA_NUM_THREADS, \
                                   0, context->cuda_stream()>>>( \
        a, b, out, pre, n, post); \
  } \
}; \
REGISTER_CUDA_OPERATOR( \
    name, BinaryElementwiseOp< \
        input_type, CUDAContext, Cuda##name##Functor, output_type>)

#define CUDA_ADD(x, y) ((x) + (y))
CUDA_FUNCTOR(Add, CUDA_ADD, NumericTypes, SameTypeAsInput);
#undef CUDA_ADD
#define CUDA_SUB(x, y) ((x) - (y))
CUDA_FUNCTOR(Sub, CUDA_SUB, NumericTypes, SameTypeAsInput);
#undef CUDA_SUB
#define CUDA_MUL(x, y) ((x) * (y))
CUDA_FUNCTOR(Mul, CUDA_MUL, NumericTypes, SameTypeAsInput);
#undef CUDA_MUL
#define CUDA_DIV(x, y) ((x) / (y))
CUDA_FUNCTOR(Div, CUDA_DIV, NumericTypes, SameTypeAsInput);
#undef CUDA_DIV
#define CUDA_LT(x, y) ((x) < (y))
CUDA_FUNCTOR(LT, CUDA_LT, NumericTypes, FixedType<bool>);
#undef CUDA_LT
#define CUDA_LE(x, y) ((x) <= (y))
CUDA_FUNCTOR(LE, CUDA_LE, NumericTypes, FixedType<bool>);
#undef CUDA_LE
#define CUDA_GT(x, y) ((x) > (y))
CUDA_FUNCTOR(GT, CUDA_GT, NumericTypes, FixedType<bool>);
#undef CUDA_GT
#define CUDA_GE(x, y) ((x) >= (y))
CUDA_FUNCTOR(GE, CUDA_GE, NumericTypes, FixedType<bool>);
#undef CUDA_GE
#define CUDA_EQ(x, y) ((x) == (y))
CUDA_FUNCTOR(EQ, CUDA_EQ, IntTypes, FixedType<bool>);
#undef CUDA_EQ
#define CUDA_AND(x, y) ((x) & (y))
CUDA_FUNCTOR(And, CUDA_AND, BoolTypes, FixedType<bool>);
#undef CUDA_AND
#define CUDA_OR(x, y) ((x) | (y))
CUDA_FUNCTOR(Or, CUDA_OR, BoolTypes, FixedType<bool>);
#undef CUDA_OR
#define CUDA_XOR(x, y) ((x) ^ (y))
CUDA_FUNCTOR(Xor, CUDA_XOR, BoolTypes, FixedType<bool>);
#undef CUDA_XOR

__global__ void NotKernel(const int n, const bool* x, bool* y) {
  CUDA_1D_KERNEL_LOOP(i, n) {
    y[i] = !x[i];
  }
}
struct CudaNotFunctor {
  inline void operator()(
      const int n, const bool* x, bool* y, CUDAContext* context) {
    NotKernel<<<CAFFE_GET_BLOCKS(n), CAFFE_CUDA_NUM_THREADS, 0,
                context->cuda_stream()>>>(n, x, y);
  }
};
REGISTER_CUDA_OPERATOR(Not, UnaryElementwiseOp<BoolTypes, CUDAContext, CudaNotFunctor>);


__global__ void DivKernel(const int n, float *dXdata, float *dYdata,
                          const float *dZdata, const float *Ydata,
                          const float *Zdata) {
  CUDA_1D_KERNEL_LOOP(i, n) {
    dXdata[i] = dZdata[i] / Ydata[i];
    dYdata[i] = - (dZdata[i] * Zdata[i]) / Ydata[i];
  }
}


void ElementWiseDivide(CUDAContext &context, const int n, float *dXdata, float *dYdata,
                            const float *dZdata, const float *Ydata,
                            const float *Zdata) {
  DivKernel<<<CAFFE_GET_BLOCKS(n), CAFFE_CUDA_NUM_THREADS, 0,
              context.cuda_stream()>>>(n, dXdata, dYdata, dZdata, Ydata, Zdata);
}

REGISTER_CUDA_OPERATOR(DivGradient, DivGradientOp<CUDAContext>);

}  // namespace caffe2
