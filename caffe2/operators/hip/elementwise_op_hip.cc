#include "hip/hip_runtime.h"

#define CUB_STDERR
#include <cub/block/block_load.cuh>
#include <cub/block/block_reduce.cuh>
#include <cub/device/device_reduce.cuh>
#include "caffe2/core/hip/common_hip.h"
#include "caffe2/core/hip/context_hip.h"
#include "caffe2/operators/elementwise_op.h"
#include "caffe2/utils/conversions.h"

namespace caffe2 {

#define HIP_FUNCTOR(name, op, input_type, output_type)                                    \
    template <int b_is_scalar, typename T, typename R>                                    \
    __global__ void name##Kernel(const T* a, const T* b, R* out, int n)                   \
    {                                                                                     \
        HIP_1D_KERNEL_LOOP(i, n) { out[i] = op(a[i], b[b_is_scalar ? 0 : i]); }           \
    }                                                                                     \
    template <typename T, typename R>                                                     \
    __global__ void name##BroadcastKernel(const T* a, const T* b, R* out, int pre, int n) \
    {                                                                                     \
        HIP_1D_KERNEL_LOOP(i, pre* n) { out[i] = op(a[i], b[i % n]); }                    \
    }                                                                                     \
    template <typename T, typename R>                                                     \
    __global__ void name##Broadcast2Kernel(                                               \
        const T* a, const T* b, R* out, int pre, int n, int post)                         \
    {                                                                                     \
        HIP_1D_KERNEL_LOOP(i, pre* n* post) { out[i] = op(a[i], b[(i / post) % n]); }     \
    }                                                                                     \
                                                                                          \
    struct Hip##name##Functor                                                             \
    {                                                                                     \
        template <bool b_is_scalar, typename T, typename R>                               \
        inline void Run(size_t n, const T* a, const T* b, R* out, HIPContext* context)    \
        {                                                                                 \
            hipLaunchKernelGGL((name##Kernel<b_is_scalar, T, R>),                         \
                               CAFFE_GET_BLOCKS(n),                                       \
                               CAFFE_HIP_NUM_THREADS,                                     \
                               0,                                                         \
                               context->hip_stream(),                                     \
                               a,                                                         \
                               b,                                                         \
                               out,                                                       \
                               static_cast<int>(n));                                      \
        }                                                                                 \
        template <typename T, typename R>                                                 \
        void RunWithBroadcast(                                                            \
            const T* a, const T* b, R* out, size_t pre, size_t n, HIPContext* context)    \
        {                                                                                 \
            hipLaunchKernelGGL((name##BroadcastKernel<T, R>),                             \
                               CAFFE_GET_BLOCKS(pre* n),                                  \
                               CAFFE_HIP_NUM_THREADS,                                     \
                               0,                                                         \
                               context->hip_stream(),                                     \
                               a,                                                         \
                               b,                                                         \
                               out,                                                       \
                               static_cast<int>(pre),                                     \
                               static_cast<int>(n));                                      \
        }                                                                                 \
        template <typename T, typename R>                                                 \
        void RunWithBroadcast2(const T* a,                                                \
                               const T* b,                                                \
                               R* out,                                                    \
                               size_t pre,                                                \
                               size_t n,                                                  \
                               size_t post,                                               \
                               HIPContext* context)                                       \
        {                                                                                 \
            hipLaunchKernelGGL((name##Broadcast2Kernel<T, R>),                            \
                               CAFFE_GET_BLOCKS(pre* n* post),                            \
                               CAFFE_HIP_NUM_THREADS,                                     \
                               0,                                                         \
                               context->hip_stream(),                                     \
                               a,                                                         \
                               b,                                                         \
                               out,                                                       \
                               static_cast<int>(pre),                                     \
                               static_cast<int>(n),                                       \
                               static_cast<int>(post));                                   \
        }                                                                                 \
    };                                                                                    \
    REGISTER_HIP_OPERATOR(                                                                \
        name, BinaryElementwiseOp<input_type, HIPContext, Hip##name##Functor, output_type>)

#define HIP_SUB(x, y) ((x) - (y))
HIP_FUNCTOR(Sub, HIP_SUB, NumericTypes, SameTypeAsInput);
#undef HIP_SUB
#define HIP_MUL(x, y) ((x) * (y))
HIP_FUNCTOR(Mul, HIP_MUL, NumericTypes, SameTypeAsInput);
#undef HIP_MUL
#define HIP_DIV(x, y) ((x) / (y))
HIP_FUNCTOR(Div, HIP_DIV, NumericTypes, SameTypeAsInput);
#undef HIP_DIV
#define HIP_LT(x, y) ((x) < (y))
HIP_FUNCTOR(LT, HIP_LT, NumericTypes, FixedType<bool>);
#undef HIP_LT
#define HIP_LE(x, y) ((x) <= (y))
HIP_FUNCTOR(LE, HIP_LE, NumericTypes, FixedType<bool>);
#undef HIP_LE
#define HIP_GT(x, y) ((x) > (y))
HIP_FUNCTOR(GT, HIP_GT, NumericTypes, FixedType<bool>);
#undef HIP_GT
#define HIP_GE(x, y) ((x) >= (y))
HIP_FUNCTOR(GE, HIP_GE, NumericTypes, FixedType<bool>);
#undef HIP_GE
#define HIP_EQ(x, y) ((x) == (y))
HIP_FUNCTOR(EQ, HIP_EQ, IntTypes, FixedType<bool>);
#undef HIP_EQ
#define HIP_AND(x, y) ((x) & (y))
HIP_FUNCTOR(And, HIP_AND, BoolTypes, FixedType<bool>);
#undef HIP_AND
#define HIP_OR(x, y) ((x) | (y))
HIP_FUNCTOR(Or, HIP_OR, BoolTypes, FixedType<bool>);
#undef HIP_OR
#define HIP_XOR(x, y) ((x) ^ (y))
HIP_FUNCTOR(Xor, HIP_XOR, BoolTypes, FixedType<bool>);
#undef HIP_XOR

__global__ void NotKernel(const int n, const bool* x, bool* y)
{
    HIP_1D_KERNEL_LOOP(i, n) { y[i] = !x[i]; }
}
struct HipNotFunctor
{
    inline void operator()(const int n, const bool* x, bool* y, HIPContext* context)
    {
        hipLaunchKernelGGL((NotKernel),
                           dim3(CAFFE_GET_BLOCKS(n)),
                           dim3(CAFFE_HIP_NUM_THREADS),
                           0,
                           context->hip_stream(),
                           n,
                           x,
                           y);
    }
};
REGISTER_HIP_OPERATOR(Not, UnaryElementwiseOp<BoolTypes, HIPContext, HipNotFunctor>);

__global__ void DivKernel(const int n,
                          float* dXdata,
                          float* dYdata,
                          const float* dZdata,
                          const float* Ydata,
                          const float* Zdata)
{
    HIP_1D_KERNEL_LOOP(i, n)
    {
        dXdata[i] = dZdata[i] / Ydata[i];
        dYdata[i] = -(dZdata[i] * Zdata[i]) / Ydata[i];
    }
}

void ElementWiseDivide(HIPContext& context,
                       const int n,
                       float* dXdata,
                       float* dYdata,
                       const float* dZdata,
                       const float* Ydata,
                       const float* Zdata)
{
    hipLaunchKernelGGL((DivKernel),
                       dim3(CAFFE_GET_BLOCKS(n)),
                       dim3(CAFFE_HIP_NUM_THREADS),
                       0,
                       context.hip_stream(),
                       n,
                       dXdata,
                       dYdata,
                       dZdata,
                       Ydata,
                       Zdata);
}

REGISTER_HIP_OPERATOR(DivGradient, DivGradientOp<HIPContext>);

namespace {

template <typename T>
__global__ void reduce_sum_like_post1(const T* g_idata, T* g_odata, int pre, int N)
{
    int n = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    if(n >= N)
    {
        return;
    }

    float sum = 0.0;
    for(int i = 0; i < pre; ++i)
    {
        sum += convert::To<T, float>(g_idata[i * N + n]);
    }

    g_odata[n] = convert::To<float, T>(sum);
}

template <typename T>
void device_reduce(const T* d_in, T* d_out, int N, Tensor<HIPContext>* buffer, HIPContext* context)
{
    // Determine temporary device storage requirements
    size_t temp_storage_bytes = 0;
    cub::DeviceReduce::Sum(NULL, temp_storage_bytes, d_in, d_out, N, context->hip_stream());

    auto buffer_size = temp_storage_bytes / sizeof(T);
    buffer_size += temp_storage_bytes % sizeof(T) != 0 ? 1 : 0;
    buffer->Resize(buffer_size);
    void* d_temp_storage = static_cast<void*>(buffer->template mutable_data<T>());
    // Run sum-reduction
    cub::DeviceReduce::Sum(
        d_temp_storage, temp_storage_bytes, d_in, d_out, N, context->hip_stream());
}

template <>
void device_reduce<float16>(
    const float16* in, float16* out, int N, Tensor<HIPContext>* buffer, HIPContext* context)
{
#if 0 // Ashish TBD: rocblas with fp16
  auto buffer_size = 1;

  if (buffer->size() != buffer_size) {
    buffer->Resize(buffer_size);

    math::Set<float16, HIPContext>(
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
#endif
}

template <typename T, int BLOCK_THREADS>
__global__ void reduce_sum_like(const T* g_idata, T* g_odata, int pre, int N, int post)
{
    int n     = hipBlockIdx_x;
    float sum = 0.0;
    int limit = pre * post;
    for(int i = hipThreadIdx_x; i < limit; i += hipBlockDim_x)
    {
        int curPre  = i / post;
        int curPost = i % post;

        sum += convert::To<T, float>(g_idata[curPre * N * post + n * post + curPost]);
    }
    // uses a shared memory reduction within block
    typedef cub::BlockReduce<float, BLOCK_THREADS> BlockReduceT;
    // Shared memory
    __shared__ typename BlockReduceT::TempStorage temp_storage;
    float aggregate = BlockReduceT(temp_storage).Sum(sum);
    if(hipThreadIdx_x == 0)
    {
        g_odata[n] = convert::To<float, T>(aggregate);
    }
}
} // namespace

template <>
template <typename T>
bool SumReduceLikeOp<HIPContext>::DoRunWithType()
{
    const auto& A = Input(0);
    const auto& B = Input(1);
    auto* C       = Output(0);
    auto count    = A.size();
    CAFFE_ENFORCE(&B != C, "In-place is not allowed.");
    C->ResizeLike(B);
    const T* Adata = A.template data<T>();
    auto* Cdata    = C->template mutable_data<T>();
    if(B.size() == 1)
    {
        device_reduce<T>(Adata, Cdata, count, &sum_buffer_, &context_);
    }
    else
    {
        size_t pre, n, post;
        std::tie(pre, n, post) = calculate_broadcast_sizes(A, B, axis_);
        // because we check shape(B) \in shape(A) before,
        // post and pre cannot be 1 at same time
        if(post == 1)
        {
            hipLaunchKernelGGL((reduce_sum_like_post1<T>),
                               dim3(CAFFE_GET_BLOCKS(n)),
                               dim3(CAFFE_HIP_NUM_THREADS),
                               0,
                               context_.hip_stream(),
                               Adata,
                               Cdata,
                               static_cast<int>(pre),
                               static_cast<int>(n));
        }
        else
        {
            if(post >= 128)
            {
                hipLaunchKernelGGL((reduce_sum_like<T, 512>),
                                   dim3(n),
                                   dim3(512),
                                   0,
                                   context_.hip_stream(),
                                   Adata,
                                   Cdata,
                                   static_cast<int>(pre),
                                   static_cast<int>(n),
                                   static_cast<int>(post));
            }
            else if(post >= 64)
            {
                hipLaunchKernelGGL((reduce_sum_like<T, 128>),
                                   dim3(n),
                                   dim3(128),
                                   0,
                                   context_.hip_stream(),
                                   Adata,
                                   Cdata,
                                   static_cast<int>(pre),
                                   static_cast<int>(n),
                                   static_cast<int>(post));
            }
            else if(post >= 32)
            {
                hipLaunchKernelGGL((reduce_sum_like<T, 64>),
                                   dim3(n),
                                   dim3(64),
                                   0,
                                   context_.hip_stream(),
                                   Adata,
                                   Cdata,
                                   static_cast<int>(pre),
                                   static_cast<int>(n),
                                   static_cast<int>(post));
            }
            else
            {
                hipLaunchKernelGGL((reduce_sum_like<T, 32>),
                                   dim3(n),
                                   dim3(32),
                                   0,
                                   context_.hip_stream(),
                                   Adata,
                                   Cdata,
                                   static_cast<int>(pre),
                                   static_cast<int>(n),
                                   static_cast<int>(post));
            }
        }
    }
    return true;
}

template <>
bool SumReduceLikeOp<HIPContext>::RunOnDevice()
{
    return DispatchHelper<TensorTypes<float, float16>>::call(this, Input(0));
}

REGISTER_HIP_OPERATOR(SumReduceLike, SumReduceLikeOp<HIPContext>);

namespace {

template <bool is_scaler, typename T, typename M>
__global__ void binary_add_kernel(const int N, const T* a, const T* b, T* r)
{
    HIP_1D_KERNEL_LOOP(idx, N)
    {
        r[idx] = convert::To<M, T>(convert::To<T, M>(a[idx]) +
                                   convert::To<T, M>(is_scaler ? b[0] : b[idx]));
    }
}

template <bool no_post, typename T, typename M>
__global__ void binary_add_kernel_broadcast(
    const T* a, const T* b, T* r, const int pre, const int post, const int n)
{
    HIP_1D_KERNEL_LOOP(idx, no_post ? pre * n : pre * post * n)
    {
        r[idx] = convert::To<M, T>(convert::To<T, M>(a[idx]) +
                                   convert::To<T, M>(no_post ? b[idx % n] : b[(idx / post) % n]));
    }
}
} // namespace

// Actual Add operator, because the above macros are read-only.
class HIPAddOp final : public Operator<HIPContext>
{
    public:
    HIPAddOp(const OperatorDef& operator_def, Workspace* ws)
        : Operator<HIPContext>(operator_def, ws),
          OP_SINGLE_ARG(bool, "broadcast", enable_broadcast_, 0),
          OP_SINGLE_ARG(int, "axis", axis_, -1),
          OP_SINGLE_ARG(string, "axis_str", axis_str_, ""),
          OP_SINGLE_ARG(string, "order", order_, "NCHW")
    {
        // Figure out the correct axis to use.
        if(enable_broadcast_)
        {
            if(axis_ != -1)
            {
                // Get axis from an explicit axis argument.
                CAFFE_ENFORCE_EQ(
                    axis_str_.size(), 0, "Args axis and axis_str cannot be used simultaneously.");
            }
            else if(axis_str_.size())
            {
                // Get the axis index semantically.
                CAFFE_ENFORCE_EQ(axis_str_.size(), 1, "Unsupported axis string", axis_str_);
                size_t semantic_axis_ = order_.find(axis_str_);
                CAFFE_ENFORCE_NE(semantic_axis_,
                                 string::npos,
                                 "Unrecognizable axis string ",
                                 axis_str_,
                                 " from order string ",
                                 order_);
                axis_ = semantic_axis_;
            }
        }
        else
        {
            CAFFE_ENFORCE(axis_ == -1 && axis_str_.size() == 0,
                          "Do not specify axis or axis_str if broadcast is not enabled.");
        }
    }

    ~HIPAddOp() {}

    template <typename T, typename M>
    bool DoRunWithType()
    {
        auto& X0     = Input(0);
        auto& X1     = Input(1);
        auto* output = Output(0);

        output->ResizeLike(X0);

        const T* X0data = X0.template data<T>();
        const T* X1data = X1.template data<T>();
        T* outputData   = output->template mutable_data<T>();

        if(!enable_broadcast_)
        {
            CAFFE_ENFORCE_EQ(
                X0.dims(), X1.dims(), "Dimension mismatch - did you forget to set broadcast=1?");
            hipLaunchKernelGGL((binary_add_kernel<false, T, M>),
                               dim3(CAFFE_GET_BLOCKS(X0.size())),
                               dim3(CAFFE_HIP_NUM_THREADS),
                               0,
                               context_.hip_stream(),
                               static_cast<const int>(X0.size()),
                               X0data,
                               X1data,
                               outputData);
        }
        else if(X1.size() == 1)
        {
            hipLaunchKernelGGL((binary_add_kernel<true, T, M>),
                               dim3(CAFFE_GET_BLOCKS(X0.size())),
                               dim3(CAFFE_HIP_NUM_THREADS),
                               0,
                               context_.hip_stream(),
                               static_cast<const int>(X0.size()),
                               X0data,
                               X1data,
                               outputData);
        }
        else
        {
            size_t pre, n, post;
            std::tie(pre, n, post) = calculate_broadcast_sizes(X0, X1, axis_);
            if(post == 1)
            {
                hipLaunchKernelGGL((binary_add_kernel_broadcast<true, T, M>),
                                   dim3(CAFFE_GET_BLOCKS(pre * n)),
                                   dim3(CAFFE_HIP_NUM_THREADS),
                                   0,
                                   context_.hip_stream(),
                                   X0data,
                                   X1data,
                                   outputData,
                                   static_cast<const int>(pre),
                                   static_cast<const int>(post),
                                   static_cast<const int>(n));
            }
            else
            {
                hipLaunchKernelGGL((binary_add_kernel_broadcast<false, T, M>),
                                   dim3(CAFFE_GET_BLOCKS(pre * post * n)),
                                   dim3(CAFFE_HIP_NUM_THREADS),
                                   0,
                                   context_.hip_stream(),
                                   X0data,
                                   X1data,
                                   outputData,
                                   static_cast<const int>(pre),
                                   static_cast<const int>(post),
                                   static_cast<const int>(n));
            }
        }
        return true;
    }

    bool RunOnDevice() override
    {
        if(Input(0).IsType<float>())
        {
            return DoRunWithType<float, float>();
        }
        else if(Input(0).IsType<float16>())
        {
            return DoRunWithType<float16, float>();
        }
        else if(Input(0).IsType<int32_t>())
        {
            return DoRunWithType<int32_t, int32_t>();
        }
        else if(Input(0).IsType<int64_t>())
        {
            return DoRunWithType<int64_t, int64_t>();
        }
        else
        {
            return false;
        }
    }

    private:
    bool enable_broadcast_;
    int axis_;
    string axis_str_;
    string order_;
};

REGISTER_HIP_OPERATOR(Add, HIPAddOp);

} // namespace caffe2
