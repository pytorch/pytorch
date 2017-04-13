#define CUB_STDERR
#include <cub/block/block_load.cuh>
#include <cub/block/block_reduce.cuh>
#include <cub/device/device_reduce.cuh>
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
REGISTER_CUDA_OPERATOR(
    Not,
    UnaryElementwiseOp<BoolTypes, CUDAContext, CudaNotFunctor>);

__global__ void DivKernel(const int n, float *dXdata, float *dYdata,
                          const float *dZdata, const float *Ydata,
                          const float *Zdata) {
  CUDA_1D_KERNEL_LOOP(i, n) {
    dXdata[i] = dZdata[i] / Ydata[i];
    dYdata[i] = - (dZdata[i] * Zdata[i]) / Ydata[i];
  }
}

void ElementWiseDivide(
    CUDAContext& context,
    const int n,
    float* dXdata,
    float* dYdata,
    const float* dZdata,
    const float* Ydata,
    const float* Zdata) {
  DivKernel<<<CAFFE_GET_BLOCKS(n), CAFFE_CUDA_NUM_THREADS, 0,
              context.cuda_stream()>>>(n, dXdata, dYdata, dZdata, Ydata, Zdata);
}

REGISTER_CUDA_OPERATOR(DivGradient, DivGradientOp<CUDAContext>);

namespace {

template <typename T>
__global__ void
reduce_sum_like_post1(const T* g_idata, T* g_odata, int pre, int N) {
  int n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n >= N) {
    return;
  }

  T sum = (T)0;
  for (int i = 0; i < pre; ++i) {
    sum += g_idata[i * N + n];
  }

  g_odata[n] = sum;
}

template <typename T>
void device_reduce(
    const T* d_in,
    T* d_out,
    int N,
    Tensor<CUDAContext>* buffer,
    CUDAContext* context) {
  // Determine temporary device storage requirements
  size_t temp_storage_bytes = 0;
  cub::DeviceReduce::Sum(NULL, temp_storage_bytes, d_in, d_out, N);

  auto buffer_size = temp_storage_bytes / sizeof(T);
  buffer_size += temp_storage_bytes % sizeof(T) != 0 ? 1 : 0;
  buffer->Resize(buffer_size);
  void* d_temp_storage = static_cast<void*>(buffer->template mutable_data<T>());
  // Run sum-reduction
  cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, d_in, d_out, N);
}

template <typename T, int BLOCK_THREADS>
__global__ void
reduce_sum_like(const T* g_idata, T* g_odata, int pre, int N, int post) {
  int n = blockIdx.x;
  T sum = (T)0;
  int limit = pre * post;
  for (int i = threadIdx.x; i < limit; i += blockDim.x) {
    int curPre = i / post;
    int curPost = i % post;

    sum += g_idata[curPre * N * post + n * post + curPost];
  }
  // uses a shared memory reduction within block
  typedef cub::BlockReduce<T, BLOCK_THREADS> BlockReduceT;
  // Shared memory
  __shared__ typename BlockReduceT::TempStorage temp_storage;
  T aggregate = BlockReduceT(temp_storage).Sum(sum);
  if (threadIdx.x == 0) {
    g_odata[n] = aggregate;
  }
}
} // namespace

template <>
template <typename T>
bool SumReduceLikeOp<CUDAContext>::DoRunWithType() {
  const auto& A = Input(0);
  const auto& B = Input(1);
  auto* C = Output(0);
  auto count = A.size();
  CAFFE_ENFORCE(&B != C, "In-place is not allowed.");
  C->ResizeLike(B);
  const T* Adata = A.template data<T>();
  auto* Cdata = C->template mutable_data<T>();
  if (B.size() == 1) {
    device_reduce<T>(Adata, Cdata, count, &sum_buffer_, &context_);
  } else {
    CAFFE_ENFORCE_GT(
        A.ndim(),
        B.ndim(),
        "If you are doing ReduceSumLike, input1 should have "
        "a smaller number of dimensions.");
    const int axis = (axis_ == -1 ? A.ndim() - B.ndim() : axis_);
    CAFFE_ENFORCE(
        axis >= 0 && axis < A.ndim(),
        "ReduceSum axis should be in the range of the number "
        "of dimensions of the first input.");
    size_t pre = 1, n = 1, post = 1;
    for (int i = 0; i < axis; ++i) {
      pre *= A.dim(i);
    }
    for (int i = 0; i < B.ndim(); ++i) {
      CAFFE_ENFORCE_EQ(
          A.dim(i + axis), B.dim(i), "Broadcast dimension mismatch.");
      n *= B.dim(i);
    }
    for (int i = axis + B.ndim(); i < A.ndim(); ++i) {
      post *= A.dim(i);
    }

    // because we check shape(B) \in shape(A) before,
    // post and pre cannot be 1 at same time
    if (post == 1) {
      reduce_sum_like_post1<T>
          <<<CAFFE_GET_BLOCKS(n),
             CAFFE_CUDA_NUM_THREADS,
             0,
             context_.cuda_stream()>>>(Adata, Cdata, pre, n);
    } else {
      if (post >= 128) {
        reduce_sum_like<T, 512>
            <<<n, 512, 0, context_.cuda_stream()>>>(Adata, Cdata, pre, n, post);
      } else if (post >= 64) {
        reduce_sum_like<T, 128>
            <<<n, 128, 0, context_.cuda_stream()>>>(Adata, Cdata, pre, n, post);
      } else if (post >= 32) {
        reduce_sum_like<T, 64>
            <<<n, 64, 0, context_.cuda_stream()>>>(Adata, Cdata, pre, n, post);
      } else {
        reduce_sum_like<T, 32>
            <<<n, 32, 0, context_.cuda_stream()>>>(Adata, Cdata, pre, n, post);
      }
    }
  }
  return true;
}

REGISTER_CUDA_OPERATOR(SumReduceLike, SumReduceLikeOp<CUDAContext>);

}  // namespace caffe2
