#include "caffe2/core/common_cudnn.h"
#include "caffe2/core/context_gpu.h"
#include "caffe2/core/cudnn_wrappers.h"
#include "caffe2/operators/conv_op.h"
#include "caffe2/operators/conv_op_cache_cudnn.h"
#include "caffe2/operators/conv_pool_op_base.h"

namespace caffe2 {

// Modified from TensorFlow,
// https://github.com/tensorflow/tensorflow/blob/4cb482dc3e0424c3d658ba373a6354dded6a32df/tensorflow/core/kernels/depthwise_conv_op_gpu.cu.cc

// A Cuda kernel to compute the depthwise convolution forward pass
// in NCHW format.

struct DepthwiseArgs {
  // Input layer dimensions
  int batch{0};
  int in_rows{0};
  int in_cols{0};
  int in_depth{0};
  int filter_rows{0};
  int filter_cols{0};
  int stride{0};
  int pad_rows{0};
  int pad_cols{0};

  // Output layer dimensions
  int out_rows{0};
  int out_cols{0};
  int out_depth{0};
};

template <typename T, int kKnownFilterWidth, int kKnownFilterHeight>
__global__ void DepthwiseConv2dGPUKernelNCHW(
    const DepthwiseArgs args,
    const T* input,
    const T* filter,
    T* output,
    int num_outputs) {
  const int in_rows = args.in_rows;
  const int in_cols = args.in_cols;
  const int in_depth = args.in_depth;
  const int filter_rows = kKnownFilterHeight;
  const int filter_cols = kKnownFilterWidth;
  const int stride = args.stride;
  const int pad_rows = args.pad_rows;
  const int pad_cols = args.pad_cols;
  const int out_rows = args.out_rows;
  const int out_cols = args.out_cols;
  const int out_depth = args.out_depth;

  CUDA_1D_KERNEL_LOOP(thread_id, num_outputs) {
    const int OW = thread_id % out_cols;
    const int OH = (thread_id / out_cols) % out_rows;
    const int OC = (thread_id / out_cols / out_rows) % out_depth;
    const int OB = thread_id / out_cols / out_rows / out_depth;
    const int in_d = OC;

    const int input_offset_temp = (OB * in_depth + OC) * (in_rows * in_cols);
    const int input_row_start = OH * stride - pad_rows;
    const int input_col_start = OW * stride - pad_cols;
    const int input_row_end = input_row_start + filter_rows;
    const int input_col_end = input_col_start + filter_cols;
    const float* filter_start = filter + in_d * filter_rows * filter_cols;
    T sum = 0;
    if (input_row_start >= 0 && input_col_start >= 0 &&
        input_row_end < in_rows && input_col_end < in_cols) {
// Loop that doesn't need to check for boundary conditions.
#pragma unroll
      for (int f_r = 0; f_r < filter_rows; ++f_r) {
        const int in_r = input_row_start + f_r;
        const float* filter_offset = filter_start + filter_cols * f_r;
#pragma unroll
        for (int f_c = 0; f_c < filter_cols; ++f_c) {
          const int in_c = input_col_start + f_c;

          const int input_offset =
              (input_offset_temp) + (in_r * in_cols) + in_c;
#if __CUDA_ARCH__ >= 350
          sum += __ldg(input + input_offset) * __ldg(filter_offset + f_c);
#else
          sum += input[input_offset] * filter_offset[f_c];
#endif
        }
      }
    } else {
// Loop that needs to check for boundary conditions.
#pragma unroll
      for (int f_r = 0; f_r < filter_rows; ++f_r) {
        const int in_r = input_row_start + f_r;
        const float* filter_offset = filter_start + filter_cols * f_r;
#pragma unroll
        for (int f_c = 0; f_c < filter_cols; ++f_c) {
          const int in_c = input_col_start + f_c;
          if (in_r >= 0 && in_r < in_rows && in_c >= 0 && in_c < in_cols) {
            const int in_c = input_col_start + f_c;
            const int input_offset =
                (input_offset_temp) + (in_r * in_cols) + in_c;
#if __CUDA_ARCH__ >= 350
            sum += __ldg(input + input_offset) * __ldg(filter_offset + f_c);
#else
            sum += input[input_offset] * filter_offset[f_c];
#endif
          }
        }
      }
    }

    output[thread_id] = sum;
  }
}

// A Cuda kernel to compute the depthwise convolution backprop w.r.t. filter.
template <typename T, int kKnownFilterWidth, int kKnownFilterHeight>
__global__ void DepthwiseConv2dBackpropFilterGPUKernelNCHW(
    const DepthwiseArgs args,
    const T* out_backprop,
    const T* input,
    T* filter_backprop,
    int num_out_backprop) {
  const int in_rows = args.in_rows;
  const int in_cols = args.in_cols;
  const int in_depth = args.in_depth;
  const int filter_rows = kKnownFilterHeight;
  const int filter_cols = kKnownFilterWidth;
  const int stride = args.stride;
  const int pad_rows = args.pad_rows;
  const int pad_cols = args.pad_cols;
  const int out_rows = args.out_rows;
  const int out_cols = args.out_cols;
  const int out_depth = args.out_depth;

  CUDA_1D_KERNEL_LOOP(thread_id, num_out_backprop) {
    // Compute the indexes of this thread in the output.
    const int OW = thread_id % out_cols;
    const int OH = (thread_id / out_cols) % out_rows;
    const int OC = (thread_id / out_cols / out_rows) % out_depth;
    const int OB = thread_id / out_cols / out_rows / out_depth;

    // Compute the input depth and the index of depth multiplier.
    const int in_d = OC;

    // Decide if all input is valid, if yes, we can skip the boundary checks
    // for each input.
    const int in_r_start = OH * stride - pad_rows;
    const int in_c_start = OW * stride - pad_cols;
    const int in_r_end = in_r_start + filter_rows;
    const int in_c_end = in_c_start + filter_cols;

    const int out_backprop_offset = (OB * out_depth * out_rows * out_cols) +
        (OC * out_rows * out_cols) + (OH * out_cols) + (OW);

#if __CUDA_ARCH__ >= 350
    const T out_bp = __ldg(out_backprop + out_backprop_offset);
#else
    const T out_bp = out_backprop[out_backprop_offset];
#endif
    if (in_r_start >= 0 && in_c_start >= 0 && in_r_end < in_rows &&
        in_c_end < in_cols) {
#pragma unroll
      for (int f_r = 0; f_r < filter_rows; ++f_r) {
        const int in_r = in_r_start + f_r;
        // Avoid repeated computation.
        const int input_offset_temp = (OB * in_depth * in_rows * in_cols) +
            (OC * in_rows * in_cols) + (in_r * in_cols);

#pragma unroll
        for (int f_c = 0; f_c < filter_cols; ++f_c) {
          const int in_c = in_c_start + f_c;
          const int input_offset = input_offset_temp + in_c;
#if __CUDA_ARCH__ >= 350
          T partial_sum = __ldg(input + input_offset) * out_bp;
#else
          T partial_sum = input[input_offset] * out_bp;
#endif
          T* addr = filter_backprop + (in_d * filter_rows * filter_cols) +
              (f_c + filter_cols * f_r);
          atomicAdd(addr, partial_sum);
        }
      }
    } else {
#pragma unroll
      for (int f_r = 0; f_r < filter_rows; ++f_r) {
        const int in_r = in_r_start + f_r;
        // Avoid repeated computation.
        const int input_offset_temp = (OB * in_depth * in_rows * in_cols) +
            (OC * in_rows * in_cols) + (in_r * in_cols);
#pragma unroll
        for (int f_c = 0; f_c < filter_cols; ++f_c) {
          const int in_c = in_c_start + f_c;

          if (in_r >= 0 && in_r < in_rows && in_c >= 0 && in_c < in_cols) {
            const int input_offset = input_offset_temp + in_c;
#if __CUDA_ARCH__ >= 350
            T partial_sum = __ldg(input + input_offset) * out_bp;
#else
            T partial_sum = input[input_offset] * out_bp;
#endif
            T* addr = filter_backprop + (in_d * filter_rows * filter_cols) +
                (f_c + filter_cols * f_r);
            atomicAdd(addr, partial_sum);
          }
        }
      }
    }
  }
}

template <typename T, int kKnownFilterWidth, int kKnownFilterHeight>
__global__ void DepthwiseConv2dBackpropInputGPUKernelNCHW(
    const DepthwiseArgs args,
    const T* out_backprop,
    const T* filter,
    T* in_backprop,
    int num_in_backprop) {
  const int in_rows = args.in_rows;
  const int in_cols = args.in_cols;
  const int in_depth = args.in_depth;
  const int filter_rows = kKnownFilterHeight;
  const int filter_cols = kKnownFilterWidth;
  const int stride = args.stride;
  const int pad_rows = args.pad_rows;
  const int pad_cols = args.pad_cols;
  const int out_rows = args.out_rows;
  const int out_cols = args.out_cols;
  const int out_depth = args.out_depth;

  // TODO(vrv): Consider assigning threads to output and using
  // atomics for accumulation, similar to the filter case.
  CUDA_1D_KERNEL_LOOP(thread_id, num_in_backprop) {
    const int IW = thread_id % in_cols;
    const int IH = (thread_id / in_cols) % in_rows;
    const int IC = (thread_id / in_cols / in_rows) % in_depth;
    const int IB = thread_id / in_cols / in_rows / in_depth;

    T sum = 0;

    const int out_r_start =
        max(0, (IH - filter_rows + pad_rows + stride) / stride);
    const int out_r_end = min(out_rows - 1, (IH + pad_rows) / stride);
    const int out_c_start =
        max(0, (IW - filter_cols + pad_cols + stride) / stride);
    const int out_c_end = min(out_cols - 1, (IW + pad_cols) / stride);

#pragma unroll
    for (int out_r = out_r_start; out_r <= out_r_end; ++out_r) {
      const int f_r = IH + pad_rows - out_r * stride;
      for (int out_c = out_c_start; out_c <= out_c_end; ++out_c) {
        const int f_c = IW + pad_cols - out_c * stride;
        const int filter_offset =
            IC * filter_rows * filter_cols + f_r * filter_cols + f_c;
        const int out_backprop_offset = (IB * out_depth * out_rows * out_cols) +
            (IC * out_rows * out_cols) + (out_r * out_cols) + (out_c);

#if __CUDA_ARCH__ >= 350
        sum += __ldg(out_backprop + out_backprop_offset) *
            __ldg(filter + filter_offset);
#else
        sum += out_backprop[out_backprop_offset] * filter[filter_offset];
#endif
      }
    }
    const int in_backprop_offset = (IB * in_rows * in_cols * in_depth) +
        (IC * in_rows * in_cols) + (IH * in_cols) + (IW);
    in_backprop[in_backprop_offset] = sum;
  }
}

class Depthwise3x3ConvOp final : public ConvPoolOpBase<CUDAContext> {
 public:
  USE_CONV_POOL_BASE_FUNCTIONS(CUDAContext);
  Depthwise3x3ConvOp(const OperatorDef& operator_def, Workspace* ws)
      : ConvPoolOpBase<CUDAContext>(operator_def, ws),
        cudnn_wrapper_(&context_) {
    OPERATOR_NEEDS_FEATURE(
        this->order_ == StorageOrder::NCHW,
        "Depthwise3x3ConvOp only supports NCHW order");
    CUDNN_ENFORCE(cudnnCreateTensorDescriptor(&bias_desc_));
    CUDNN_ENFORCE(cudnnCreateTensorDescriptor(&top_desc_for_bias_));
  }

  ~Depthwise3x3ConvOp() {
    CUDNN_ENFORCE(cudnnDestroyTensorDescriptor(bias_desc_));
    CUDNN_ENFORCE(cudnnDestroyTensorDescriptor(top_desc_for_bias_));
  }

  bool RunOnDeviceWithOrderNCHW() override {
    const Tensor& X = Input(0);
    auto& filter = Input(1);
    const int N = X.dim32(0), C = X.dim32(1);
    CAFFE_ENFORCE_EQ(X.dim(), filter.dim());
    const int M = filter.dim32(0);

    CAFFE_ENFORCE_EQ(M, X.dim32(1));
    CAFFE_ENFORCE_EQ(C, X.dim32(1));
    CAFFE_ENFORCE_EQ(C, this->group_);
    CAFFE_ENFORCE_GT(this->group_, 1);
    CAFFE_ENFORCE_EQ(this->kernel_w(), 3);
    CAFFE_ENFORCE_EQ(this->kernel_h(), 3);
    CAFFE_ENFORCE_EQ(this->stride_h(), this->stride_w());
    auto sizes = ConvPoolOpBase<CUDAContext>::GetOutputSize(X, filter.dim32(0));
    Tensor* Y = Output(0, sizes, at::dtype<float>());
    DepthwiseArgs args;
    args.batch = X.dim32(0);
    args.in_rows = X.dim32(2);
    args.in_cols = X.dim32(3);
    args.in_depth = X.dim32(1);
    args.filter_cols = 3;
    args.filter_rows = 3;
    args.stride = this->stride_w();
    args.pad_rows = this->pad_t();
    args.pad_cols = this->pad_l();
    args.out_rows = Y->dim32(2);
    args.out_cols = Y->dim32(3);
    args.out_depth = Y->dim32(1);
    DepthwiseConv2dGPUKernelNCHW<float, 3, 3>
        <<<CAFFE_GET_BLOCKS(Y->size()),
           CAFFE_CUDA_NUM_THREADS,
           0,
           context_.cuda_stream()>>>(
            args,
            X.data<float>(),
            filter.data<float>(),
            Y->mutable_data<float>(),
            Y->size());
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    if (InputSize() == 3) {
      CUDNN_ENFORCE(cudnnSetTensor4dDescriptor(
          bias_desc_,
          GetCudnnTensorFormat(order_),
          cudnnTypeWrapper<float>::type,
          1,
          M,
          1,
          1));
      CUDNN_ENFORCE(cudnnSetTensor4dDescriptor(
          top_desc_for_bias_,
          GetCudnnTensorFormat(order_),
          cudnnTypeWrapper<float>::type,
          Y->dim32(0),
          M,
          Y->dim32(2),
          Y->dim32(3)));
      auto& bias = Input(2);
      CAFFE_ENFORCE_EQ(bias.dim(), 1);
      CAFFE_ENFORCE_EQ(bias.dim32(0), M);
      CUDNN_ENFORCE(cudnnAddTensor(
          cudnn_wrapper_.inline_cudnn_handle(),
          cudnnTypeWrapper<float>::kOne(),
          bias_desc_,
          bias.data<float>(),
          cudnnTypeWrapper<float>::kOne(),
          top_desc_for_bias_,
          Y->mutable_data<float>()));
    }

    return true;
  }

 private:
  CuDNNWrapper cudnn_wrapper_;
  cudnnTensorDescriptor_t bias_desc_;
  cudnnTensorDescriptor_t top_desc_for_bias_;
};

class Depthwise3x3ConvGradientOp final : public ConvPoolOpBase<CUDAContext> {
 public:
  USE_CONV_POOL_BASE_FUNCTIONS(CUDAContext);
  Depthwise3x3ConvGradientOp(const OperatorDef& operator_def, Workspace* ws)
      : ConvPoolOpBase<CUDAContext>(operator_def, ws),
        cudnn_wrapper_(&context_),
        no_bias_(OperatorBase::GetSingleArgument<int>("no_bias", 0)) {
    CAFFE_ENFORCE(
        !(no_bias_ && OutputSize() == 3),
        "If bias is not present, you should not have 3 grad output.");
    OPERATOR_NEEDS_FEATURE(
        this->order_ == StorageOrder::NCHW,
        "Depthwise3x3ConvGradientOp only supports NCHW order");
    CUDNN_ENFORCE(cudnnCreateTensorDescriptor(&bias_desc_));
    CUDNN_ENFORCE(cudnnCreateTensorDescriptor(&top_desc_for_bias_));
  }

  ~Depthwise3x3ConvGradientOp() {
    CUDNN_ENFORCE(cudnnDestroyTensorDescriptor(bias_desc_));
    CUDNN_ENFORCE(cudnnDestroyTensorDescriptor(top_desc_for_bias_));
  }

  bool RunOnDeviceWithOrderNCHW() override {
    auto& X = Input(INPUT);
    auto& filter = Input(FILTER);
    auto& dY = Input(OUTPUT_GRAD);

    const int N = X.dim32(0), C = X.dim32(1);

    const vector<int> input_dims = this->GetDims(X);
    ConvPoolOpBase<CUDAContext>::ComputePads(input_dims);
    CAFFE_ENFORCE_EQ(X.dim(), filter.dim());
    const int M = filter.dim32(0);
    CAFFE_ENFORCE(filter.dim32(1) * group_ == C);
    CAFFE_ENFORCE(M % group_ == 0);
    auto* dfilter = Output(FILTER_GRAD, filter.sizes(), at::dtype<float>());
    DepthwiseArgs args;
    args.batch = X.dim32(0);
    args.in_rows = X.dim32(2);
    args.in_cols = X.dim32(3);
    args.in_depth = X.dim32(1);
    args.filter_cols = 3;
    args.filter_rows = 3;
    args.stride = this->stride_w();
    args.pad_rows = this->pad_t();
    args.pad_cols = this->pad_l();
    args.out_rows = dY.dim32(2);
    args.out_cols = dY.dim32(3);
    args.out_depth = dY.dim32(1);

    CAFFE_ENFORCE(OutputSize() == 3 || (no_bias_ && (OutputSize() == 2)));

    auto* dX = Output(
        no_bias_ ? BIAS_OR_INPUT_GRAD : INPUT_GRAD,
        X.sizes(),
        at::dtype<float>());
    math::Set<float, CUDAContext>(
        dfilter->size(), 0, dfilter->mutable_data<float>(), &context_);
    DepthwiseConv2dBackpropFilterGPUKernelNCHW<float, 3, 3>
        <<<CAFFE_GET_BLOCKS(dY.size()),
           CAFFE_CUDA_NUM_THREADS,
           0,
           context_.cuda_stream()>>>(
            args,
            dY.data<float>(),
            X.data<float>(),
            dfilter->mutable_data<float>(),
            dY.size());
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    DepthwiseConv2dBackpropInputGPUKernelNCHW<float, 3, 3>
        <<<CAFFE_GET_BLOCKS(dX->size()),
           CAFFE_CUDA_NUM_THREADS,
           0,
           context_.cuda_stream()>>>(
            args,
            dY.data<float>(),
            filter.data<float>(),
            dX->mutable_data<float>(),
            dX->size());
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    if (!no_bias_) {
      CUDNN_ENFORCE(cudnnSetTensor4dDescriptor(
          bias_desc_,
          GetCudnnTensorFormat(order_),
          cudnnTypeWrapper<float>::type,
          1,
          M,
          1,
          1));
      CUDNN_ENFORCE(cudnnSetTensor4dDescriptor(
          top_desc_for_bias_,
          GetCudnnTensorFormat(order_),
          cudnnTypeWrapper<float>::type,
          dY.dim32(0),
          M,
          dY.dim32(2),
          dY.dim32(3)));

      auto* dbias = Output(BIAS_OR_INPUT_GRAD, {M}, at::dtype<float>());
      CUDNN_ENFORCE(cudnnConvolutionBackwardBias(
          cudnn_wrapper_.inline_cudnn_handle(),
          cudnnTypeWrapper<float>::kOne(),
          top_desc_for_bias_,
          dY.data<float>(),
          cudnnTypeWrapper<float>::kZero(),
          bias_desc_,
          dbias->mutable_data<float>()));
    }
    return true;
  }

 private:
  CuDNNWrapper cudnn_wrapper_;
  cudnnTensorDescriptor_t bias_desc_;
  cudnnTensorDescriptor_t top_desc_for_bias_;

  bool no_bias_;

  INPUT_TAGS(INPUT, FILTER, OUTPUT_GRAD);
  OUTPUT_TAGS(FILTER_GRAD, BIAS_OR_INPUT_GRAD, INPUT_GRAD);
};

REGISTER_CUDA_OPERATOR_WITH_ENGINE(Conv, DEPTHWISE_3x3, Depthwise3x3ConvOp);
REGISTER_CUDA_OPERATOR_WITH_ENGINE(
    ConvGradient,
    DEPTHWISE_3x3,
    Depthwise3x3ConvGradientOp);

} // namespace caffe2
