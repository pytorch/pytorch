#include "caffe2/operators/elementwise_ops.h"

#include <cub/block/block_load.cuh>
#include <cub/block/block_reduce.cuh>
#include <cub/device/device_reduce.cuh>

#include "caffe2/core/hip/common_hip.h"
#include "caffe2/core/hip/context_hip.h"
#include "caffe2/utils/conversions.h"

#define ROCBLAS_FP16 0

namespace caffe2 {

REGISTER_HIP_OPERATOR(
    Not,
    UnaryElementwiseOp<BoolTypes, HIPContext, NotFunctor<HIPContext>>);
REGISTER_HIP_OPERATOR(
    Sign,
    UnaryElementwiseOp<NumericTypes, HIPContext, SignFunctor<HIPContext>>);

#define REGISTER_HIP_COMPARE_OPERATOR(Op)                    \
  REGISTER_HIP_OPERATOR(                                     \
      Op,                                                     \
      BinaryElementwiseOp<                                    \
          TensorTypes<bool, int32_t, int64_t, float, double>, \
          HIPContext,                                        \
          Op##Functor<HIPContext>,                           \
          FixedType<bool>>)

REGISTER_HIP_COMPARE_OPERATOR(EQ);
REGISTER_HIP_COMPARE_OPERATOR(NE);
REGISTER_HIP_COMPARE_OPERATOR(LT);
REGISTER_HIP_COMPARE_OPERATOR(LE);
REGISTER_HIP_COMPARE_OPERATOR(GT);
REGISTER_HIP_COMPARE_OPERATOR(GE);

#undef REGISTER_HIP_COMPARE_OPERATOR

#define REGISTER_HIP_LOGICAL_BINARY_OPERATOR(Op) \
  REGISTER_HIP_OPERATOR(                         \
      Op,                                         \
      BinaryElementwiseOp<BoolTypes, HIPContext, Op##Functor<HIPContext>>)

REGISTER_HIP_LOGICAL_BINARY_OPERATOR(And);
REGISTER_HIP_LOGICAL_BINARY_OPERATOR(Or);
REGISTER_HIP_LOGICAL_BINARY_OPERATOR(Xor);

#undef REGISTER_HIP_LOGICAL_BINARY_OPERATOR

#define REGISTER_HIP_BITWISE_BINARY_OPERATOR(Op) \
  REGISTER_HIP_OPERATOR(                         \
      Op,                                         \
      BinaryElementwiseOp<                        \
          IntBoolTypes,                           \
          HIPContext,                            \
          Op##Functor<HIPContext>>)

REGISTER_HIP_BITWISE_BINARY_OPERATOR(BitwiseAnd);
REGISTER_HIP_BITWISE_BINARY_OPERATOR(BitwiseOr);
REGISTER_HIP_BITWISE_BINARY_OPERATOR(BitwiseXor);

#undef REGISTER_HIP_BITWISE_BINARY_OPERATOR

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
    Tensor<HIPContext>* buffer,
    HIPContext* context) {
  // Determine temporary device storage requirements
  size_t temp_storage_bytes = 0;
  cub::DeviceReduce::Sum(
      NULL, temp_storage_bytes, d_in, d_out, N, context->hip_stream());

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
      context->hip_stream());
}

template <>
void device_reduce<float16>(
    const float16* in,
    float16* out,
    int N,
    Tensor<HIPContext>* buffer,
    HIPContext* context) {
#if ROCBLAS_FP16
  auto buffer_size = 1;

  if (buffer->size() != buffer_size) {
    buffer->Resize(buffer_size);

    math::Set<float16, HIPContext>(
        N,
        convert::To<float, float16>(1.),
        buffer->mutable_data<float16>(),
        context);
  }

  ROCBLAS_ENFORCE(hipblasDotEx(
      context->rocblas_handle(),
      N,
      in,
      hipR16F,
      1,
      buffer->data<float16>(),
      hipR16F,
      0,
      out,
      hipR16F,
      hipR32F));
#else
  CAFFE_THROW("Unsupported math type");
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
bool SumReduceLikeOp<HIPContext>::DoRunWithType() {
  const auto& A = Input(0);
  const auto& B = Input(1);
  auto* C = Output(0);
  auto count = A.size();
  CAFFE_ENFORCE(&B != C, "In-place is not allowed.");
  C->ResizeLike(B);
  const T* Adata = A.template data<T>();
  auto* Cdata = C->template mutable_data<T>();

  if (C->size() == 0) {
    // output is empty, nothing to do, not even launching the HIP kernel
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
      hipLaunchKernelGGL(reduce_sum_like_post1<T>, dim3(CAFFE_GET_BLOCKS(static_cast<int>(n))), dim3(CAFFE_HIP_NUM_THREADS), 0, context_.hip_stream(), Adata, Cdata, static_cast<int>(pre), static_cast<int>(n));
    } else {
      if (post >= 128) {
        hipLaunchKernelGGL(reduce_sum_like<T, 512>, dim3(static_cast<int>(n)), dim3(512), 0, context_.hip_stream(), Adata, Cdata, static_cast<int>(pre), static_cast<int>(n), static_cast<int>(post));
      } else if (post >= 64) {
        hipLaunchKernelGGL(reduce_sum_like<T, 128>, dim3(static_cast<int>(n)), dim3(128), 0, context_.hip_stream(), Adata, Cdata, static_cast<int>(pre), static_cast<int>(n), static_cast<int>(post));
      } else if (post >= 32) {
        hipLaunchKernelGGL(reduce_sum_like<T, 64>, dim3(static_cast<int>(n)), dim3(64), 0, context_.hip_stream(), Adata, Cdata, static_cast<int>(pre), static_cast<int>(n), static_cast<int>(post));
      } else {
        hipLaunchKernelGGL(reduce_sum_like<T, 32>, dim3(static_cast<int>(n)), dim3(32), 0, context_.hip_stream(), Adata, Cdata, static_cast<int>(pre), static_cast<int>(n), static_cast<int>(post));
      }
    }
  }
  return true;
}

template <>
bool SumReduceLikeOp<HIPContext>::RunOnDevice() {
  return DispatchHelper<TensorTypes<float, float16>>::call(this, Input(0));
}

REGISTER_HIP_OPERATOR(SumReduceLike, SumReduceLikeOp<HIPContext>);

} // namespace caffe2
