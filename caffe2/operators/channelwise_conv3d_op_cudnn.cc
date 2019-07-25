#include "caffe2/core/context.h"
#include "caffe2/core/logging.h"
#include "caffe2/core/operator.h"
#include "caffe2/operators/conv_pool_op_base.h"
#include "caffe2/utils/math.h"

// a CPU implementation of 3D depthwise conv

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
void DepthwiseConv3dCPUKernelNCHW(
    const DepthwiseArgs& args,
    const T* input,
    const T* filter,
    T* output) {
  const int batch = args.batch;
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

  int output_offsset = 0;
  for (int OB = 0; OB < batch; OB++) {
    for (int OC = 0; OC < out_depth; OC++) {
      for (int OL = 0; OL < out_length; OL++) {
        for (int OH = 0; OH < out_rows; OH++) {
          for (int OW = 0; OW < out_cols; OW++) {
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
              for (int f_l = 0; f_l < filter_length; ++f_l) {
                const int in_l = input_length_start + f_l;
                for (int f_r = 0; f_r < filter_rows; ++f_r) {
                  const int in_r = input_row_start + f_r;
                  const float* filter_offset = filter_start +
                      filter_cols * filter_rows * f_l + filter_cols * f_r;
                  for (int f_c = 0; f_c < filter_cols; ++f_c) {
                    const int in_c = input_col_start + f_c;
                    const int input_offset = (input_offset_temp) +
                        (in_l * in_cols * in_rows) + (in_r * in_cols) + in_c;
                    sum += input[input_offset] * filter_offset[f_c];
                  }
                }
              }
            } else {
              for (int f_l = 0; f_l < filter_length; ++f_l) {
                const int in_l = input_length_start + f_l;
                for (int f_r = 0; f_r < filter_rows; ++f_r) {
                  const int in_r = input_row_start + f_r;
                  const float* filter_offset = filter_start +
                      filter_cols * filter_rows * f_l + filter_cols * f_r;
                  for (int f_c = 0; f_c < filter_cols; ++f_c) {
                    const int in_c = input_col_start + f_c;
                    if (in_r >= 0 && in_r < in_rows && in_c >= 0 &&
                        in_c < in_cols && in_l >= 0 && in_l < in_length) {
                      const int input_offset = (input_offset_temp) +
                          (in_l * in_cols * in_rows) + (in_r * in_cols) + in_c;
                      sum += input[input_offset] * filter_offset[f_c];
                    }
                  }
                }
              }
            }
            output[output_offsset++] = sum;
          }
        }
      }
    }
  }
}

template <typename T>
void DepthwiseConv3dBackpropFilterCPUKernelNCHW(
    const DepthwiseArgs& args,
    const T* out_backprop,
    const T* input,
    T* filter_backprop) {
  const int batch = args.batch;
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

  for (int OB = 0; OB < batch; OB++) {
    for (int OC = 0; OC < out_depth; OC++) {
      for (int OL = 0; OL < out_length; OL++) {
        for (int OH = 0; OH < out_rows; OH++) {
          for (int OW = 0; OW < out_cols; OW++) {
            const int in_d = OC;
            const int in_r_start = OH * stride - pad_rows;
            const int in_c_start = OW * stride - pad_cols;
            const int in_l_start = OL * temporal_stride - pad_length;
            const int in_r_end = in_r_start + filter_rows;
            const int in_c_end = in_c_start + filter_cols;
            const int in_l_end = in_l_start + filter_length;

            // This can be further optimized
            // TODO(Du): abstract the multiplications
            const int out_backprop_offset =
                (OB * out_depth * out_length * out_rows * out_cols) +
                (OC * out_length * out_rows * out_cols) +
                (OL * out_rows * out_cols) + (OH * out_cols) + (OW);
            const T out_bp = out_backprop[out_backprop_offset];
            if (in_r_start >= 0 && in_c_start >= 0 && in_r_end < in_rows &&
                in_c_end < in_cols && in_l_start >= 0 && in_l_end < in_length) {
              for (int f_l = 0; f_l < filter_length; ++f_l) {
                const int in_l = in_l_start + f_l;
                for (int f_r = 0; f_r < filter_rows; ++f_r) {
                  const int in_r = in_r_start + f_r;
                  // TODO(Du): abstract the multiplications
                  const int input_offset_temp =
                      (OB * in_depth * in_length * in_rows * in_cols) +
                      (OC * in_length * in_rows * in_cols) +
                      (in_l * in_rows * in_cols) + (in_r * in_cols);
                  for (int f_c = 0; f_c < filter_cols; ++f_c) {
                    const int in_c = in_c_start + f_c;
                    const int input_offset = input_offset_temp + in_c;
                    T partial_sum = input[input_offset] * out_bp;
                    T* addr = filter_backprop +
                        (in_d * filter_rows * filter_cols * filter_length) +
                        (f_l * filter_rows * filter_cols) +
                        (f_c + filter_cols * f_r);
                    *addr = *addr + partial_sum;
                  }
                }
              }
            } else {
              for (int f_l = 0; f_l < filter_length; ++f_l) {
                const int in_l = in_l_start + f_l;
                for (int f_r = 0; f_r < filter_rows; ++f_r) {
                  const int in_r = in_r_start + f_r;

                  // TODO(Du): abstract the multiplications
                  const int input_offset_temp =
                      (OB * in_depth * in_length * in_rows * in_cols) +
                      (OC * in_length * in_rows * in_cols) +
                      (in_l * in_rows * in_cols) + (in_r * in_cols);
                  for (int f_c = 0; f_c < filter_cols; ++f_c) {
                    const int in_c = in_c_start + f_c;

                    if (in_r >= 0 && in_r < in_rows && in_c >= 0 &&
                        in_c < in_cols && in_l >= 0 && in_l < in_length) {
                      const int input_offset = input_offset_temp + in_c;
                      T partial_sum = input[input_offset] * out_bp;
                      T* addr = filter_backprop +
                          (in_d * filter_rows * filter_cols * filter_length) +
                          (f_l * filter_rows * filter_cols) +
                          (f_c + filter_cols * f_r);
                      *addr = *addr + partial_sum;
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
  }
}

template <typename T>
void DepthwiseConv3dBackpropInputCPUKernelNCHW(
    const DepthwiseArgs& args,
    const T* out_backprop,
    const T* filter,
    T* in_backprop) {
  const int batch = args.batch;
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

  for (int IB = 0; IB < batch; IB++) {
    for (int IC = 0; IC < in_depth; IC++) {
      for (int IL = 0; IL < in_length; IL++) {
        for (int IH = 0; IH < in_rows; IH++) {
          for (int IW = 0; IW < in_cols; IW++) {
            T sum = 0;

            const int out_r_start =
                std::max(0, (IH - filter_rows + pad_rows + stride) / stride);
            const int out_r_end =
                std::min(out_rows - 1, (IH + pad_rows) / stride);
            const int out_c_start =
                std::max(0, (IW - filter_cols + pad_cols + stride) / stride);
            const int out_c_end =
                std::min(out_cols - 1, (IW + pad_cols) / stride);
            const int out_l_start = std::max(
                0,
                (IL - filter_length + pad_length + temporal_stride) /
                    temporal_stride);
            const int out_l_end =
                std::min(out_length - 1, (IL + pad_length) / temporal_stride);
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
                      (out_l * out_rows * out_cols) + (out_r * out_cols) +
                      (out_c);

                  sum +=
                      out_backprop[out_backprop_offset] * filter[filter_offset];
                }
              }
            }
            const int in_backprop_offset =
                (IB * in_rows * in_cols * in_length * in_depth) +
                (IC * in_rows * in_cols * in_length) +
                (IL * in_rows * in_cols) + (IH * in_cols) + (IW);
            in_backprop[in_backprop_offset] = sum;
          }
        }
      }
    }
  }
}

class ChannelwiseConv3dOp final : public ConvPoolOpBase<CPUContext> {
 public:
  USE_CONV_POOL_BASE_FUNCTIONS(CPUContext);
  ChannelwiseConv3dOp(const OperatorDef& operator_def, Workspace* ws)
      : ConvPoolOpBase<CPUContext>(operator_def, ws) {
    OPERATOR_NEEDS_FEATURE(
        this->order_ == StorageOrder::NCHW,
        "ChannelwiseConv3dOp only supports NCHW order");
  }

  ~ChannelwiseConv3dOp() override {}

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

    auto sizes = ConvPoolOpBase<CPUContext>::GetOutputSize(X, filter.dim32(0));
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

    DepthwiseConv3dCPUKernelNCHW<float>(
        args, X.data<float>(), filter.data<float>(), Y->mutable_data<float>());

    // handle bias
    if (InputSize() == 3) {
      // TODO
    }
    return true;
  }

 private:
};

class ChannelwiseConv3dGradientOp final : public ConvPoolOpBase<CPUContext> {
 public:
  USE_CONV_POOL_BASE_FUNCTIONS(CPUContext);
  ChannelwiseConv3dGradientOp(const OperatorDef& operator_def, Workspace* ws)
      : ConvPoolOpBase<CPUContext>(operator_def, ws),
        no_bias_(OperatorBase::GetSingleArgument<int>("no_bias", 0)) {
    CAFFE_ENFORCE(
        !(no_bias_ && OutputSize() == 3),
        "If bias is not present, you should not have 3 grad output.");
    OPERATOR_NEEDS_FEATURE(
        this->order_ == StorageOrder::NCHW,
        "ChannelwiseConv3dGradientOp only supports NCHW order");
  }

  ~ChannelwiseConv3dGradientOp() override {}

  bool RunOnDeviceWithOrderNCHW() override {
    auto& X = Input(INPUT);
    auto& filter = Input(FILTER);
    auto& dY = Input(OUTPUT_GRAD);

    const int C = X.dim32(1);

    const vector<int> input_dims = this->GetDims(X);
    ConvPoolOpBase<CPUContext>::ComputePads(input_dims);
    CAFFE_ENFORCE_EQ(X.ndim(), filter.ndim());
    const int M = filter.dim32(0);
    CAFFE_ENFORCE(filter.dim32(1) * group_ == C);
    CAFFE_ENFORCE(M % group_ == 0);
    auto* dfilter = Output(FILTER_GRAD, filter.sizes(), at::dtype<float>());

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

    auto* dX = Output(
        no_bias_ ? BIAS_OR_INPUT_GRAD : INPUT_GRAD,
        X.sizes(),
        at::dtype<float>());

    math::Set<float, CPUContext>(
        dfilter->size(), 0, dfilter->mutable_data<float>(), &context_);

    DepthwiseConv3dBackpropFilterCPUKernelNCHW(
        args,
        dY.data<float>(),
        X.data<float>(),
        dfilter->mutable_data<float>());
    DepthwiseConv3dBackpropInputCPUKernelNCHW(
        args,
        dY.data<float>(),
        filter.data<float>(),
        dX->mutable_data<float>());

    if (!no_bias_) {
      // TODO
    }
    return true;
  }

 private:
  bool no_bias_;

  INPUT_TAGS(INPUT, FILTER, OUTPUT_GRAD);
  OUTPUT_TAGS(FILTER_GRAD, BIAS_OR_INPUT_GRAD, INPUT_GRAD);
};

REGISTER_CPU_OPERATOR_WITH_ENGINE(Conv, CHANNELWISE_3D, ChannelwiseConv3dOp);
REGISTER_CPU_OPERATOR_WITH_ENGINE(
    ConvGradient,
    CHANNELWISE_3D,
    ChannelwiseConv3dGradientOp);

} // namespace caffe2
