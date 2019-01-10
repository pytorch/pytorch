#define CUB_STDERR
#include <cub/block/block_load.cuh>
#include <cub/block/block_reduce.cuh>
#include <cub/device/device_reduce.cuh>
#include "caffe2/core/common_gpu.h"
#include "caffe2/core/context_gpu.h"
#include "caffe2/operators/elementwise_op.h"
#include "caffe2/utils/conversions.h"

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
CUDA_FUNCTOR(EQ, CUDA_EQ, IntBoolTypes, FixedType<bool>);
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
    Tensor<CUDAContext>* buffer,
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
void device_reduce<float16>(
    const float16* in,
    float16* out,
    int N,
    Tensor<CUDAContext>* buffer,
    CUDAContext* context) {
  auto buffer_size = 1;

  if (buffer->size() != buffer_size) {
    buffer->Resize(buffer_size);

    math::Set<float16, CUDAContext>(
        N,
        convert::To<float,float16>(1.),
        buffer->mutable_data<float16>(),
        context);
  }

  CUBLAS_ENFORCE(cublasDotEx(
              context->cublas_handle(),
              N,
              in,
              CUDA_R_16F,
              1,
              buffer->data<float16>(),
              CUDA_R_16F,
              0,
              out,
              CUDA_R_16F,
              CUDA_R_32F));
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

    sum += convert::To<T, float>(g_idata[curPre * N * post + n * post + curPost]);
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
  if (B.size() == 1) {
    device_reduce<T>(Adata, Cdata, count, &sum_buffer_, &context_);
  } else {
    size_t pre, n, post;
    std::tie(pre, n, post) = calculate_broadcast_sizes(A, B, axis_);
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
  return DispatchHelper<TensorTypes<float, float16>>::call(this, Input(0));
}

REGISTER_CUDA_OPERATOR(SumReduceLike, SumReduceLikeOp<CUDAContext>);

namespace {

template <bool is_scaler, typename T, typename M>
__global__ void binary_add_kernel(const int N, const T* a, const T* b, T* r) {
  CUDA_1D_KERNEL_LOOP(idx, N) {
    r[idx] = convert::To<M, T>(
        convert::To<T, M>(a[idx]) +
        convert::To<T, M>(is_scaler ? b[0] : b[idx]));
  }
}

template <bool no_post, typename T, typename M>
__global__ void binary_add_kernel_broadcast(
    const T* a,
    const T* b,
    T* r,
    const int pre,
    const int post,
    const int n) {
  CUDA_1D_KERNEL_LOOP(idx, no_post ? pre * n : pre * post * n) {
    r[idx] = convert::To<M, T>(
        convert::To<T, M>(a[idx]) +
        convert::To<T, M>(no_post ? b[idx % n] : b[(idx / post) % n]));
  }
}
} // namespace

// Actual Add operator, because the above macros are read-only.
class CUDAAddOp final : public Operator<CUDAContext> {
 public:
  CUDAAddOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<CUDAContext>(operator_def, ws),
        OP_SINGLE_ARG(bool, "broadcast", enable_broadcast_, 0),
        OP_SINGLE_ARG(int, "axis", axis_, -1),
        OP_SINGLE_ARG(string, "axis_str", axis_str_, ""),
        OP_SINGLE_ARG(string, "order", order_, "NCHW") {
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

  ~CUDAAddOp() {}

  template <typename T, typename M>
  bool DoRunWithType() {
    auto& X0 = Input(0);
    auto& X1 = Input(1);
    auto* output = Output(0);

    output->ResizeLike(X0);

    const T* X0data = X0.template data<T>();
    const T* X1data = X1.template data<T>();
    T* outputData = output->template mutable_data<T>();

    if (!enable_broadcast_) {
      CAFFE_ENFORCE_EQ(
          X0.dims(),
          X1.dims(),
          "Dimension mismatch - did you forget to set broadcast=1?");
      binary_add_kernel<false, T, M><<<
          CAFFE_GET_BLOCKS(X0.size()),
          CAFFE_CUDA_NUM_THREADS,
          0,
          context_.cuda_stream()>>>(X0.size(), X0data, X1data, outputData);
    } else if (X1.size() == 1) {
      binary_add_kernel<true, T, M><<<
          CAFFE_GET_BLOCKS(X0.size()),
          CAFFE_CUDA_NUM_THREADS,
          0,
          context_.cuda_stream()>>>(X0.size(), X0data, X1data, outputData);
    } else {
      size_t pre, n, post;
      std::tie(pre, n, post) = calculate_broadcast_sizes(X0, X1, axis_);
      if (post == 1) {
        binary_add_kernel_broadcast<true, T, M><<<
            CAFFE_GET_BLOCKS(pre * n),
            CAFFE_CUDA_NUM_THREADS,
            0,
            context_.cuda_stream()>>>(X0data, X1data, outputData, pre, post, n);
      } else {
        binary_add_kernel_broadcast<false, T, M><<<
            CAFFE_GET_BLOCKS(pre * post * n),
            CAFFE_CUDA_NUM_THREADS,
            0,
            context_.cuda_stream()>>>(X0data, X1data, outputData, pre, post, n);
      }
    }
    return true;
  }

  bool RunOnDevice() override {
    if (Input(0).IsType<float>()) {
      return DoRunWithType<float, float>();
    } else if (Input(0).IsType<float16>()) {
      return DoRunWithType<float16, float>();
    } else if (Input(0).IsType<int32_t>()) {
      return DoRunWithType<int32_t, int32_t>();
    } else if (Input(0).IsType<int64_t>()) {
      return DoRunWithType<int64_t, int64_t>();
    } else {
      return false;
    }
  }

 private:
  bool enable_broadcast_;
  int axis_;
  string axis_str_;
  string order_;
};

REGISTER_CUDA_OPERATOR(Add, CUDAAddOp);

}  // namespace caffe2
