#include "caffe2/operators/elementwise_ops.h"

#include <cub/block/block_load.cuh>
#include <cub/block/block_reduce.cuh>
#include <cub/device/device_reduce.cuh>

#include "caffe2/core/common_gpu.h"
#include "caffe2/core/context_gpu.h"
#include "caffe2/utils/conversions.h"

#ifdef __HIPCC__
// rocblas doesn't fully support fp16 yet
#define ROCBLAS_FP16 0
#endif

namespace caffe2 {

REGISTER_CUDA_OPERATOR(
    Not,
    UnaryElementwiseOp<BoolTypes, CUDAContext, NotFunctor<CUDAContext>>);
REGISTER_CUDA_OPERATOR(
    Sign,
    UnaryElementwiseOp<NumericTypes, CUDAContext, SignFunctor<CUDAContext>>);

#define REGISTER_CUDA_COMPARE_OPERATOR(Op)                    \
  REGISTER_CUDA_OPERATOR(                                     \
      Op,                                                     \
      BinaryElementwiseOp<                                    \
          TensorTypes<bool, int32_t, int64_t, float, double>, \
          CUDAContext,                                        \
          Op##Functor<CUDAContext>,                           \
          FixedType<bool>>)

REGISTER_CUDA_COMPARE_OPERATOR(EQ);
REGISTER_CUDA_COMPARE_OPERATOR(NE);
REGISTER_CUDA_COMPARE_OPERATOR(LT);
REGISTER_CUDA_COMPARE_OPERATOR(LE);
REGISTER_CUDA_COMPARE_OPERATOR(GT);
REGISTER_CUDA_COMPARE_OPERATOR(GE);

#undef REGISTER_CUDA_COMPARE_OPERATOR

#define REGISTER_CUDA_LOGICAL_BINARY_OPERATOR(Op) \
  REGISTER_CUDA_OPERATOR(                         \
      Op,                                         \
      BinaryElementwiseOp<BoolTypes, CUDAContext, Op##Functor<CUDAContext>>)

REGISTER_CUDA_LOGICAL_BINARY_OPERATOR(And);
REGISTER_CUDA_LOGICAL_BINARY_OPERATOR(Or);
REGISTER_CUDA_LOGICAL_BINARY_OPERATOR(Xor);

#undef REGISTER_CUDA_LOGICAL_BINARY_OPERATOR

#define REGISTER_CUDA_BITWISE_BINARY_OPERATOR(Op) \
  REGISTER_CUDA_OPERATOR(                         \
      Op,                                         \
      BinaryElementwiseOp<                        \
          IntBoolTypes,                           \
          CUDAContext,                            \
          Op##Functor<CUDAContext>>)

REGISTER_CUDA_BITWISE_BINARY_OPERATOR(BitwiseAnd);
REGISTER_CUDA_BITWISE_BINARY_OPERATOR(BitwiseOr);
REGISTER_CUDA_BITWISE_BINARY_OPERATOR(BitwiseXor);

#undef REGISTER_CUDA_BITWISE_BINARY_OPERATOR

namespace {

template <typename T>
__global__ void
reduce_sum_like_post1(const T* g_idata, T* g_odata, int pre, int N) {
  int n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n >= N) {
    return;
  }

  float sum = 0.0;
  for (int i = 0; i < pre; ++i) {
    sum += convert::To<T, float>(g_idata[i * N + n]);
  }

  g_odata[n] = convert::To<float, T>(sum);
}

template <typename T>
void device_reduce(
    const T* d_in,
    T* d_out,
    int N,
    Tensor* buffer,
    CUDAContext* context) {
  // Determine temporary device storage requirements
  size_t temp_storage_bytes = 0;
  cub::DeviceReduce::Sum(
      NULL, temp_storage_bytes, d_in, d_out, N, context->cuda_stream());

  auto buffer_size = temp_storage_bytes / sizeof(T);
  buffer_size += temp_storage_bytes % sizeof(T) != 0 ? 1 : 0;
  buffer->Resize(buffer_size);
  void* d_temp_storage = static_cast<void*>(buffer->template mutable_data<T>());
  // Run sum-reduction
  cub::DeviceReduce::Sum(
      d_temp_storage,
      temp_storage_bytes,
      d_in,
      d_out,
      N,
      context->cuda_stream());
}

template <>
void device_reduce<at::Half>(
    const at::Half* in,
    at::Half* out,
    int N,
    Tensor* buffer,
    CUDAContext* context) {
#if defined(__HIPCC__) && !ROCBLAS_FP16
  CAFFE_THROW("HIP rocblas doesn't fully support fp16 device_reduce yet.");
#else
  auto buffer_size = 1;

  if (buffer->numel() != buffer_size) {
    buffer->Resize(buffer_size);

    math::Set<at::Half, CUDAContext>(
        N,
        convert::To<float, at::Half>(1.),
        buffer->template mutable_data<at::Half>(),
        context);
  }

  CUBLAS_ENFORCE(cublasDotEx(
      context->cublas_handle(),
      N,
      in,
      CUDA_R_16F,
      1,
      buffer->data<at::Half>(),
      CUDA_R_16F,
      0,
      out,
      CUDA_R_16F,
      CUDA_R_32F));
#endif
}

template <typename T, int BLOCK_THREADS>
__global__ void
reduce_sum_like(const T* g_idata, T* g_odata, int pre, int N, int post) {
  int n = blockIdx.x;
  float sum = 0.0;
  int limit = pre * post;
  for (int i = threadIdx.x; i < limit; i += blockDim.x) {
    int curPre = i / post;
    int curPost = i % post;

    sum +=
        convert::To<T, float>(g_idata[curPre * N * post + n * post + curPost]);
  }
  // uses a shared memory reduction within block
  typedef cub::BlockReduce<float, BLOCK_THREADS> BlockReduceT;
  // Shared memory
  __shared__ typename BlockReduceT::TempStorage temp_storage;
  float aggregate = BlockReduceT(temp_storage).Sum(sum);
  if (threadIdx.x == 0) {
    g_odata[n] = convert::To<float, T>(aggregate);
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

  if (C->size() == 0) {
    // output is empty, nothing to do, not even launching the CUDA kernel
    return true;
  }

  if (B.size() == 1) {
    device_reduce<T>(Adata, Cdata, count, &sum_buffer_, &context_);
  } else {
    size_t pre, n, post;
    std::tie(pre, n, post) =
        elementwise_ops_utils::ComputeLegacyBroadcastSizes(A, B, axis_);
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

template <>
bool SumReduceLikeOp<CUDAContext>::RunOnDevice() {
  return DispatchHelper<TensorTypes<float, at::Half>>::call(this, Input(0));
}

REGISTER_CUDA_OPERATOR(SumReduceLike, SumReduceLikeOp<CUDAContext>);

} // namespace caffe2
