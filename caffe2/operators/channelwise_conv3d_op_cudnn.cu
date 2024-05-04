#include "caffe2/core/common_cudnn.h"
#include "caffe2/core/context_gpu.h"
#include "caffe2/core/cudnn_wrappers.h"
#include "caffe2/operators/conv_op.h"
#include "caffe2/operators/conv_op_cache_cudnn.h"
#include "caffe2/operators/conv_pool_op_base.h"
#include "caffe2/utils/GpuAtomics.cuh"

// Adopted from caffe2 depthwise conv at
// pytorch/caffe2/caffe2/operators/depthwise_3x3_conv_op_cudnn.cu

namespace caffe2 {

struct DepthwiseArgs {
  // Input layer dimensions
  int batch{0};
  int in_rows{0};
  int in_cols{0};
  int in_length{0};
  int in_depth{0};

  // filter size
  int filter_rows{0};
  int filter_cols{0};
  int filter_length{0};

  // strides and pads
  int stride{0};
  int temporal_stride{0};
  int pad_rows{0};
  int pad_cols{0};
  int pad_length{0};

  // Output layer dimensions
  int out_rows{0};
  int out_cols{0};
  int out_length{0};
  int out_depth{0};
};

template <typename T>
__global__ void DepthwiseConv3dGPUKernelNCHW(
    const DepthwiseArgs args,
    const T* input,
    const T* filter,
    T* output,
    int num_outputs) {
  const int in_rows = args.in_rows;
  const int in_cols = args.in_cols;
  const int in_length = args.in_length;
  const int in_depth = args.in_depth;
  const int filter_rows = args.filter_rows;
  const int filter_cols = args.filter_cols;
  const int filter_length = args.filter_length;
  const int stride = args.stride;
  const int temporal_stride = args.temporal_stride;
  const int pad_rows = args.pad_rows;
  const int pad_cols = args.pad_cols;
  const int pad_length = args.pad_length;
  const int out_rows = args.out_rows;
  const int out_cols = args.out_cols;
  const int out_length = args.out_length;
  const int out_depth = args.out_depth;

  CUDA_1D_KERNEL_LOOP(thread_id, num_outputs) {
    const int OW = thread_id % out_cols;
    const int OH = (thread_id / out_cols) % out_rows;
    const int OL = (thread_id / out_cols / out_rows) % out_length;
    const int OC = (thread_id / out_cols / out_rows / out_length) % out_depth;
    const int OB = thread_id / out_cols / out_rows / out_length / out_depth;
    const int in_d = OC;

    const int input_offset_temp =
        (OB * in_depth + OC) * (in_length * in_rows * in_cols);
    const int input_row_start = OH * stride - pad_rows;
    const int input_col_start = OW * stride - pad_cols;
    const int input_length_start = OL * temporal_stride - pad_length;
    const int input_row_end = input_row_start + filter_rows;
    const int input_col_end = input_col_start + filter_cols;
    const int input_length_end = input_length_start + filter_length;
    const float* filter_start =
        filter + in_d * filter_rows * filter_cols * filter_length;

    T sum = 0;
    if (input_row_start >= 0 && input_col_start >= 0 &&
        input_length_start >= 0 && input_row_end < in_rows &&
        input_col_end < in_cols && input_length_end < in_length) {
// Loop that doesn't need to check for boundary conditions.
#pragma unroll
      for (int f_l = 0; f_l < filter_length; ++f_l) {
        const int in_l = input_length_start + f_l;
#pragma unroll
        for (int f_r = 0; f_r < filter_rows; ++f_r) {
          const int in_r = input_row_start + f_r;
          const float* filter_offset = filter_start +
              filter_cols * filter_rows * f_l + filter_cols * f_r;
#pragma unroll
          for (int f_c = 0; f_c < filter_cols; ++f_c) {
            const int in_c = input_col_start + f_c;

            const int input_offset = (input_offset_temp) +
                (in_l * in_cols * in_rows) + (in_r * in_cols) + in_c;
#if __CUDA_ARCH__ >= 350
            sum += __ldg(input + input_offset) * __ldg(filter_offset + f_c);
#else
            sum += input[input_offset] * filter_offset[f_c];
#endif
          }
        }
      }
    } else {
// Loop that needs to check for boundary conditions.
#pragma unroll
      for (int f_l = 0; f_l < filter_length; ++f_l) {
        const int in_l = input_length_start + f_l;
#pragma unroll
        for (int f_r = 0; f_r < filter_rows; ++f_r) {
          const int in_r = input_row_start + f_r;
          const float* filter_offset = filter_start +
              filter_cols * filter_rows * f_l + filter_cols * f_r;
#pragma unroll
          for (int f_c = 0; f_c < filter_cols; ++f_c) {
            const int in_c = input_col_start + f_c;
            if (in_r >= 0 && in_r < in_rows && in_c >= 0 && in_c < in_cols &&
                in_l >= 0 && in_l < in_length) {
              const int input_offset = (input_offset_temp) +
                  (in_l * in_cols * in_rows) + (in_r * in_cols) + in_c;
#if __CUDA_ARCH__ >= 350
              sum += __ldg(input + input_offset) * __ldg(filter_offset + f_c);
#else
              sum += input[input_offset] * filter_offset[f_c];
#endif
            }
          }
        }
      }
    }

    output[thread_id] = sum;
  }
}

// A Cuda kernel to compute the depthwise convolution backprop w.r.t. filter.
template <typename T>
__global__ void DepthwiseConv3dBackpropFilterGPUKernelNCHW(
    const DepthwiseArgs args,
    const T* out_backprop,
    const T* input,
    T* filter_backprop,
    int num_out_backprop) {
  const int in_rows = args.in_rows;
  const int in_cols = args.in_cols;
  const int in_length = args.in_length;
  const int in_depth = args.in_depth;
  const int filter_rows = args.filter_rows;
  const int filter_cols = args.filter_cols;
  const int filter_length = args.filter_length;
  const int stride = args.stride;
  const int temporal_stride = args.temporal_stride;
  const int pad_rows = args.pad_rows;
  const int pad_cols = args.pad_cols;
  const int pad_length = args.pad_length;
  const int out_rows = args.out_rows;
  const int out_cols = args.out_cols;
  const int out_length = args.out_length;
  const int out_depth = args.out_depth;

  CUDA_1D_KERNEL_LOOP(thread_id, num_out_backprop) {
    // Compute the indexes of this thread in the output.
    const int OW = thread_id % out_cols;
    const int OH = (thread_id / out_cols) % out_rows;
    const int OL = (thread_id / out_cols / out_rows) % out_length;
    const int OC = (thread_id / out_cols / out_rows / out_length) % out_depth;
    const int OB = thread_id / out_cols / out_rows / out_length / out_depth;

    // Compute the input depth and the index of depth multiplier.
    const int in_d = OC;

    // Decide if all input is valid, if yes, we can skip the boundary checks
    // for each input.
    const int in_r_start = OH * stride - pad_rows;
    const int in_c_start = OW * stride - pad_cols;
    const int in_l_start = OL * temporal_stride - pad_length;
    const int in_r_end = in_r_start + filter_rows;
    const int in_c_end = in_c_start + filter_cols;
    const int in_l_end = in_l_start + filter_length;

    const int out_backprop_offset =
        (OB * out_depth * out_length * out_rows * out_cols) +
        (OC * out_length * out_rows * out_cols) + (OL * out_rows * out_cols) +
        (OH * out_cols) + (OW);

#if __CUDA_ARCH__ >= 350
    const T out_bp = __ldg(out_backprop + out_backprop_offset);
#else
    const T out_bp = out_backprop[out_backprop_offset];
#endif
    if (in_r_start >= 0 && in_c_start >= 0 && in_r_end < in_rows &&
        in_c_end < in_cols && in_l_start >= 0 && in_l_end < in_length) {
#pragma unroll
      for (int f_l = 0; f_l < filter_length; ++f_l) {
        const int in_l = in_l_start + f_l;
#pragma unroll
        for (int f_r = 0; f_r < filter_rows; ++f_r) {
          const int in_r = in_r_start + f_r;
          // Avoid repeated computation.
          const int input_offset_temp =
              (OB * in_depth * in_length * in_rows * in_cols) +
              (OC * in_length * in_rows * in_cols) +
              (in_l * in_rows * in_cols) + (in_r * in_cols);

#pragma unroll
          for (int f_c = 0; f_c < filter_cols; ++f_c) {
            const int in_c = in_c_start + f_c;
            const int input_offset = input_offset_temp + in_c;
#if __CUDA_ARCH__ >= 350
            T partial_sum = __ldg(input + input_offset) * out_bp;
#else
            T partial_sum = input[input_offset] * out_bp;
#endif
            T* addr = filter_backprop +
                (in_d * filter_rows * filter_cols * filter_length) +
                (f_l * filter_rows * filter_cols) + (f_c + filter_cols * f_r);
            gpu_atomic_add(addr, partial_sum);
          }
        }
      }
    } else {
#pragma unroll
      for (int f_l = 0; f_l < filter_length; ++f_l) {
        const int in_l = in_l_start + f_l;
#pragma unroll
        for (int f_r = 0; f_r < filter_rows; ++f_r) {
          const int in_r = in_r_start + f_r;
          // Avoid repeated computation.
          const int input_offset_temp =
              (OB * in_depth * in_length * in_rows * in_cols) +
              (OC * in_length * in_rows * in_cols) +
              (in_l * in_rows * in_cols) + (in_r * in_cols);
#pragma unroll
          for (int f_c = 0; f_c < filter_cols; ++f_c) {
            const int in_c = in_c_start + f_c;

            if (in_r >= 0 && in_r < in_rows && in_c >= 0 && in_c < in_cols &&
                in_l >= 0 && in_l < in_length) {
              const int input_offset = input_offset_temp + in_c;
#if __CUDA_ARCH__ >= 350
              T partial_sum = __ldg(input + input_offset) * out_bp;
#else
              T partial_sum = input[input_offset] * out_bp;
#endif
              T* addr = filter_backprop +
                  (in_d * filter_rows * filter_cols * filter_length) +
                  (f_l * filter_rows * filter_cols) + (f_c + filter_cols * f_r);
              gpu_atomic_add(addr, partial_sum);
            }
          }
        }
      }
    }
  }
}

template <typename T>
__global__ void DepthwiseConv3dBackpropInputGPUKernelNCHW(
    const DepthwiseArgs args,
    const T* out_backprop,
    const T* filter,
    T* in_backprop,
    int num_in_backprop) {
  const int in_rows = args.in_rows;
  const int in_cols = args.in_cols;
  const int in_length = args.in_length;
  const int in_depth = args.in_depth;
  const int filter_rows = args.filter_rows;
  const int filter_cols = args.filter_cols;
  const int filter_length = args.filter_length;
  const int stride = args.stride;
  const int temporal_stride = args.temporal_stride;
  const int pad_rows = args.pad_rows;
  const int pad_cols = args.pad_cols;
  const int pad_length = args.pad_length;
  const int out_rows = args.out_rows;
  const int out_cols = args.out_cols;
  const int out_length = args.out_length;
  const int out_depth = args.out_depth;

  CUDA_1D_KERNEL_LOOP(thread_id, num_in_backprop) {
    const int IW = thread_id % in_cols;
    const int IH = (thread_id / in_cols) % in_rows;
    const int IL = (thread_id / in_cols / in_rows) % in_length;
    const int IC = (thread_id / in_cols / in_rows / in_length) % in_depth;
    const int IB = thread_id / in_cols / in_rows / in_length / in_depth;

    T sum = 0;

    const int out_r_start =
        max(0, (IH - filter_rows + pad_rows + stride) / stride);
    const int out_r_end = min(out_rows - 1, (IH + pad_rows) / stride);
    const int out_c_start =
        max(0, (IW - filter_cols + pad_cols + stride) / stride);
    const int out_c_end = min(out_cols - 1, (IW + pad_cols) / stride);
    const int out_l_start = max(
        0,
        (IL - filter_length + pad_length + temporal_stride) / temporal_stride);
    const int out_l_end =
        min(out_length - 1, (IL + pad_length) / temporal_stride);

#pragma unroll
    for (int out_l = out_l_start; out_l <= out_l_end; ++out_l) {
      const int f_l = IL + pad_length - out_l * temporal_stride;
      for (int out_r = out_r_start; out_r <= out_r_end; ++out_r) {
        const int f_r = IH + pad_rows - out_r * stride;
        for (int out_c = out_c_start; out_c <= out_c_end; ++out_c) {
          const int f_c = IW + pad_cols - out_c * stride;
          const int filter_offset =
              IC * filter_rows * filter_cols * filter_length +
              f_l * filter_cols * filter_rows + f_r * filter_cols + f_c;
          const int out_backprop_offset =
              (IB * out_depth * out_length * out_rows * out_cols) +
              (IC * out_length * out_rows * out_cols) +
              (out_l * out_rows * out_cols) + (out_r * out_cols) + (out_c);

#if __CUDA_ARCH__ >= 350
          sum += __ldg(out_backprop + out_backprop_offset) *
              __ldg(filter + filter_offset);
#else
          sum += out_backprop[out_backprop_offset] * filter[filter_offset];
#endif
        }
      }
    }
    const int in_backprop_offset =
        (IB * in_rows * in_cols * in_length * in_depth) +
        (IC * in_rows * in_cols * in_length) + (IL * in_rows * in_cols) +
        (IH * in_cols) + (IW);
    in_backprop[in_backprop_offset] = sum;
  }
}

class ChannelwiseConv3dOp final : public ConvPoolOpBase<CUDAContext> {
 public:
  USE_CONV_POOL_BASE_FUNCTIONS(CUDAContext);
  ChannelwiseConv3dOp(const OperatorDef& operator_def, Workspace* ws)
      : ConvPoolOpBase<CUDAContext>(operator_def, ws),
        cudnn_wrapper_(&context_) {
    OPERATOR_NEEDS_FEATURE(
        this->order_ == StorageOrder::NCHW,
        "ChannelwiseConv3dOp only supports NCHW order");
    CUDNN_ENFORCE(cudnnCreateTensorDescriptor(&bias_desc_));
    CUDNN_ENFORCE(cudnnCreateTensorDescriptor(&top_desc_for_bias_));
  }

  ~ChannelwiseConv3dOp() {
    CUDNN_ENFORCE(cudnnDestroyTensorDescriptor(bias_desc_));
    CUDNN_ENFORCE(cudnnDestroyTensorDescriptor(top_desc_for_bias_));
  }

  bool RunOnDeviceWithOrderNCHW() override {
    const Tensor& X = Input(0);
    auto& filter = Input(1);
    const int C = X.dim32(1);
    CAFFE_ENFORCE_EQ(X.ndim(), filter.ndim());
    const int M = filter.dim32(0); // number of output filters

    // enforce input/output filters are the same
    CAFFE_ENFORCE_EQ(M, X.dim32(1));
    CAFFE_ENFORCE_EQ(C, X.dim32(1));

    // check group parameters
    CAFFE_ENFORCE_EQ(C, this->group_);
    CAFFE_ENFORCE_GT(this->group_, 1);

    auto sizes = ConvPoolOpBase<CUDAContext>::GetOutputSize(X, filter.dim32(0));
    Tensor* Y = Output(0, sizes, at::dtype<float>());

    DepthwiseArgs args;
    args.batch = X.dim32(0);
    args.in_length = X.dim32(2);
    args.in_rows = X.dim32(3);
    args.in_cols = X.dim32(4);
    args.in_depth = X.dim32(1);

    CAFFE_ENFORCE_EQ(kernel_.size(), 3);
    args.filter_cols = kernel_[2];
    args.filter_rows = kernel_[1];
    args.filter_length = kernel_[0];

    CAFFE_ENFORCE_EQ(stride_.size(), 3);
    args.stride = stride_[1];
    CAFFE_ENFORCE_EQ(stride_[1], stride_[2]);
    args.temporal_stride = stride_[0];

    CAFFE_ENFORCE_EQ(pads_.size(), 6);
    args.pad_length = pads_[0];
    args.pad_rows = pads_[1];
    args.pad_cols = pads_[2];

    CAFFE_ENFORCE_EQ(Y->dim32(0), X.dim32(0));
    args.out_rows = Y->dim32(3);
    args.out_cols = Y->dim32(4);
    args.out_length = Y->dim32(2);
    args.out_depth = Y->dim32(1);

    DepthwiseConv3dGPUKernelNCHW<float>
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
      std::vector<int> bias_dims(X.ndim(), 1);
      bias_dims[1] = M;
      std::vector<int> strides = {M, 1, 1, 1, 1};
      CUDNN_ENFORCE(cudnnSetTensorNdDescriptor(
          bias_desc_,
          cudnnTypeWrapper<float>::type,
          X.ndim(),
          bias_dims.data(),
          strides.data()));

      vector<int> dims = {
          Y->dim32(0), M, Y->dim32(2), Y->dim32(3), Y->dim32(4)};
      strides = {M * Y->dim32(2) * Y->dim32(3) * Y->dim32(4),
                 Y->dim32(2) * Y->dim32(3) * Y->dim32(4),
                 Y->dim32(3) * Y->dim32(4),
                 Y->dim32(4),
                 1};
      CUDNN_ENFORCE(cudnnSetTensorNdDescriptor(
          top_desc_for_bias_,
          cudnnTypeWrapper<float>::type,
          X.ndim(),
          dims.data(),
          strides.data()));

      auto& bias = Input(2);
      CAFFE_ENFORCE_EQ(bias.ndim(), 1);
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

class ChannelwiseConv3dGradientOp final : public ConvPoolOpBase<CUDAContext> {
 public:
  USE_CONV_POOL_BASE_FUNCTIONS(CUDAContext);
  ChannelwiseConv3dGradientOp(const OperatorDef& operator_def, Workspace* ws)
      : ConvPoolOpBase<CUDAContext>(operator_def, ws),
        cudnn_wrapper_(&context_),
        no_bias_(OperatorBase::GetSingleArgument<int>("no_bias", 0)) {
    CAFFE_ENFORCE(
        !(no_bias_ && OutputSize() == 3),
        "If bias is not present, you should not have 3 grad output.");
    OPERATOR_NEEDS_FEATURE(
        this->order_ == StorageOrder::NCHW,
        "ChannelwiseConv3dGradientOp only supports NCHW order");
    CUDNN_ENFORCE(cudnnCreateTensorDescriptor(&bias_desc_));
    CUDNN_ENFORCE(cudnnCreateTensorDescriptor(&top_desc_for_bias_));
  }

  ~ChannelwiseConv3dGradientOp() {
    CUDNN_ENFORCE(cudnnDestroyTensorDescriptor(bias_desc_));
    CUDNN_ENFORCE(cudnnDestroyTensorDescriptor(top_desc_for_bias_));
  }

  bool RunOnDeviceWithOrderNCHW() override {
    auto& X = Input(INPUT);
    auto& filter = Input(FILTER);
    auto& dY = Input(OUTPUT_GRAD);
    auto* dfilter = Output(FILTER_GRAD);
    const int C = X.dim32(1);

    const vector<int> input_dims = this->GetDims(X);
    ConvPoolOpBase<CUDAContext>::ComputePads(input_dims);
    CAFFE_ENFORCE_EQ(X.ndim(), filter.ndim());
    const int M = filter.dim32(0);
    CAFFE_ENFORCE(filter.dim32(1) * group_ == C);
    CAFFE_ENFORCE(M % group_ == 0);
    dfilter->ResizeLike(filter);

    DepthwiseArgs args;
    args.batch = X.dim32(0);
    args.in_rows = X.dim32(3);
    args.in_cols = X.dim32(4);
    args.in_length = X.dim32(2);
    args.in_depth = X.dim32(1);

    args.filter_cols = kernel_[2];
    args.filter_rows = kernel_[1];
    args.filter_length = kernel_[0];

    args.stride = stride_[1];
    CAFFE_ENFORCE_EQ(stride_[1], stride_[2]);
    args.temporal_stride = stride_[0];

    args.pad_length = pads_[0];
    args.pad_rows = pads_[1];
    args.pad_cols = pads_[2];

    args.out_rows = dY.dim32(3);
    args.out_cols = dY.dim32(4);
    args.out_length = dY.dim32(2);
    args.out_depth = dY.dim32(1);

    CAFFE_ENFORCE(OutputSize() == 3 || (no_bias_ && (OutputSize() == 2)));
    auto* dX = Output(no_bias_ ? BIAS_OR_INPUT_GRAD : INPUT_GRAD);
    dX->ResizeLike(X);
    math::Set<float, CUDAContext>(
        dfilter->size(), 0, dfilter->mutable_data<float>(), &context_);

    DepthwiseConv3dBackpropFilterGPUKernelNCHW<float>
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

    DepthwiseConv3dBackpropInputGPUKernelNCHW<float>
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
      std::vector<int> bias_dims(X.ndim(), 1);
      bias_dims[1] = M;
      std::vector<int> strides = {M, 1, 1, 1, 1};
      CUDNN_ENFORCE(cudnnSetTensorNdDescriptor(
          bias_desc_,
          cudnnTypeWrapper<float>::type,
          X.ndim(),
          bias_dims.data(),
          strides.data()));

      std::vector<int> dims = {
          dY.dim32(0), M, dY.dim32(2), dY.dim32(3), dY.dim32(4)};
      strides = {M * dY.dim32(2) * dY.dim32(3) * dY.dim32(4),
                 dY.dim32(2) * dY.dim32(3) * dY.dim32(4),
                 dY.dim32(3) * dY.dim32(4),
                 dY.dim32(4),
                 1};
      CUDNN_ENFORCE(cudnnSetTensorNdDescriptor(
          top_desc_for_bias_,
          cudnnTypeWrapper<float>::type,
          X.ndim(),
          dims.data(),
          strides.data()));

      auto* dbias = Output(BIAS_OR_INPUT_GRAD);
      dbias->Resize(M);
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

REGISTER_CUDA_OPERATOR_WITH_ENGINE(Conv, CHANNELWISE_3D, ChannelwiseConv3dOp);
REGISTER_CUDA_OPERATOR_WITH_ENGINE(
    ConvGradient,
    CHANNELWISE_3D,
    ChannelwiseConv3dGradientOp);

} // namespace caffe2
