// TODO: reduce the apparent redundancy of all the code below.
#include "caffe2/operators/pool_op.h"
#include "caffe2/utils/cpu_neon.h"

namespace caffe2 {

using std::max;
using std::min;

namespace {
// These two classe are just used as template arguments passed to the PoolOp
// template to instantiate the different algorithms.
class AveragePool {};
class MaxPool {};

#ifdef __ARM_NEON__

bool isNeonEligible(int inputH, int inputW,
                    int outputH, int outputW,
                    int kH, int kW,
                    int strideH, int strideW,
                    int padT, int padL, int padB, int padR,
                    int dilationH, int dilationW,
                    const float* input,
                    float* output) {
  // Use this kernel only if:
  // Kernel width is 4x4
  // Kernel stride is 4x4
  // Padding is 0
  // Dilation is 1
  // Output width and height are even divisors of input width
  // Input width and height are divisible by 4 (should be implied by
  // all of the above, but just check again)
  // Input and output pointers are aligned by float32x4_t

  bool kernelOk = (kH == 4) && (kW == 4);
  bool strideOk = (strideH == 4) && (strideW == 4);
  bool padOk = (padT == 0) && (padL == 0) && (padB == 0) && (padR == 0);
  bool dilationOk = (dilationH == 1) && (dilationW == 1);

  bool outputOk = ((inputH % outputH) == 0) && ((inputW % outputW) == 0);
  bool inputOk = (inputW % 4 == 0) && (inputH % 4 == 0);
  bool alignOk = isPointerAligned(input, sizeof(float32x4_t)) &&
    isPointerAligned(output, sizeof(float32x4_t));

  return kernelOk && strideOk && padOk && dilationOk &&
    outputOk && inputOk && alignOk;
}

// Vectorizes 4x4p0s0 averge pooling for ARM NEON
void avgPoolNeon4x4p0s0Plane(int inputH, int inputW,
                             const float* input,
                             float* output) {
  constexpr int kKernelHeight = 4;
  constexpr int kKernelWidth = 4;
  constexpr float kDiv =
    (1.0f / ((float) kKernelHeight * (float) kKernelWidth));

  // Handle portion that can be unrolled by 4
  constexpr int kUnroll = 4;
  constexpr int kLoadSizeFloat = (sizeof(float32x4_t) / sizeof(float));
  constexpr int kLoadCols = kUnroll * kLoadSizeFloat;

  if (inputW % kLoadCols == 0) {
    //
    // Manually unroll by 4 (kUnroll)
    //

    for (int h = 0; h < inputH; h += kKernelHeight) {
      float* outputRow = output + (h / kKernelHeight) * (inputW / kKernelWidth);
      const float* curInput = input + h * inputW;

      for (int w = 0; w < inputW; w += kLoadCols) {
        float32x4_t out = {};

        {
          float32x4_t v0_0 = vld1q_f32_aligned(curInput + 0 * inputW);
          float32x4_t v0_1 = vld1q_f32_aligned(curInput + 1 * inputW);
          float32x4_t v0_2 = vld1q_f32_aligned(curInput + 2 * inputW);
          float32x4_t v0_3 = vld1q_f32_aligned(curInput + 3 * inputW);
          float v0 = horizontal_sum_f32(v0_0, v0_1, v0_2, v0_3);
          out = vsetq_lane_f32(v0, out, 0);
        }
        curInput += kLoadSizeFloat;

        {
          float32x4_t v0_0 = vld1q_f32_aligned(curInput + 0 * inputW);
          float32x4_t v0_1 = vld1q_f32_aligned(curInput + 1 * inputW);
          float32x4_t v0_2 = vld1q_f32_aligned(curInput + 2 * inputW);
          float32x4_t v0_3 = vld1q_f32_aligned(curInput + 3 * inputW);
          float v0 = horizontal_sum_f32(v0_0, v0_1, v0_2, v0_3);
          out = vsetq_lane_f32(v0, out, 1);
        }
        curInput += kLoadSizeFloat;

        {
          float32x4_t v0_0 = vld1q_f32_aligned(curInput + 0 * inputW);
          float32x4_t v0_1 = vld1q_f32_aligned(curInput + 1 * inputW);
          float32x4_t v0_2 = vld1q_f32_aligned(curInput + 2 * inputW);
          float32x4_t v0_3 = vld1q_f32_aligned(curInput + 3 * inputW);
          float v0 = horizontal_sum_f32(v0_0, v0_1, v0_2, v0_3);
          out = vsetq_lane_f32(v0, out, 2);
        }
        curInput += kLoadSizeFloat;

        {
          float32x4_t v0_0 = vld1q_f32_aligned(curInput + 0 * inputW);
          float32x4_t v0_1 = vld1q_f32_aligned(curInput + 1 * inputW);
          float32x4_t v0_2 = vld1q_f32_aligned(curInput + 2 * inputW);
          float32x4_t v0_3 = vld1q_f32_aligned(curInput + 3 * inputW);
          float v0 = horizontal_sum_f32(v0_0, v0_1, v0_2, v0_3);
          out = vsetq_lane_f32(v0, out, 3);
        }
        curInput += kLoadSizeFloat;

        out = vmulq_f32(out, vdupq_n_f32(kDiv));
        vst1q_f32_aligned(&outputRow[w / kKernelWidth], out);
      }
    }
  } else {
    //
    // Not unrolled
    //

    for (int h = 0; h < inputH; h += kKernelHeight) {
      const float* inputRow = input + h * inputW;
      float* outputRow = output + (h / kKernelHeight) * (inputW / kKernelWidth);

      for (int w = 0; w < inputW; w += kKernelWidth) {
        const float* curInput = inputRow + w;

        float32x4_t v0_0 = vld1q_f32_aligned(curInput + 0 * inputW);
        float32x4_t v0_1 = vld1q_f32_aligned(curInput + 1 * inputW);
        float32x4_t v0_2 = vld1q_f32_aligned(curInput + 2 * inputW);
        float32x4_t v0_3 = vld1q_f32_aligned(curInput + 3 * inputW);
        float v0 = horizontal_sum_f32(v0_0, v0_1, v0_2, v0_3) * kDiv;
        outputRow[w / kKernelWidth] = v0;
      }
    }
  }
}

void
runNeonAveragePool4x4p0s0NCHW(int N, int C, int inputH, int inputW,
                              const float* input,
                              float* output) {
  // We only have the 4x4p0s0 implementation at present, which is
  // checked at a higher level
  int outputH = inputH / 4;
  int outputW = inputW / 4;

  for (int n = 0; n < N; ++n) {
    for (int c = 0; c < C; ++c) {
      const float* curInput = input + (n * C + c) * inputH * inputW;
      float* curOutput = output + (n * C + c) * outputH * outputW;

      avgPoolNeon4x4p0s0Plane(inputH, inputW, curInput, curOutput);
    }
  }
}
#endif // __ARM_NEON__

}  // namespace

template <>
bool PoolOp<float, CPUContext, AveragePool>::RunOnDeviceWithOrderNCHW() {
  auto& X = Input(0);
  auto* Y = Output(0);
  ConvPoolOpBase::SetOutputSize(X, Y, X.dim32(1));

  const float* Xdata = X.data<float>();
  float* Ydata = Y->mutable_data<float>();
  math::Set<float, CPUContext>(
      Y->size(), 0, Ydata, &context_);
  // The main loop
  int channels = X.dim32(1);
  int height = X.dim32(2);
  int width = X.dim32(3);
  int pooled_height = Y->dim32(2);
  int pooled_width = Y->dim32(3);

#ifdef __ARM_NEON__
  // We specialize certain variants on ARM for vectorization
  if (isNeonEligible(X.dim32(2), X.dim32(3),
                     Y->dim32(2), Y->dim32(3),
                     kernel_h_, kernel_w_,
                     stride_h_, stride_w_,
                     pad_t_, pad_l_, pad_b_, pad_r_,
                     dilation_h_, dilation_w_,
                     Xdata, Ydata)) {
    runNeonAveragePool4x4p0s0NCHW(X.dim32(0), X.dim32(1),
                                  X.dim32(2), X.dim32(3),
                                  Xdata, Ydata);
    return true;
  }
#endif // __ARM_NEON__

  for (int n = 0; n < X.dim32(0); ++n) {
    for (int c = 0; c < channels; ++c) {
      for (int ph = 0; ph < pooled_height; ++ph) {
        for (int pw = 0; pw < pooled_width; ++pw) {
          int hstart = ph * stride_h_ - pad_t_;
          int wstart = pw * stride_w_ - pad_l_;
          int hend = min(hstart + kernel_h_, height);
          int wend = min(wstart + kernel_w_, width);
          hstart = max(hstart, 0);
          wstart = max(wstart, 0);
          const int pool_index = ph * pooled_width + pw;
          for (int h = hstart; h < hend; ++h) {
            for (int w = wstart; w < wend; ++w) {
              const int input_index = h * width + w;
              Ydata[pool_index] += Xdata[input_index];
            }
          }
          Ydata[pool_index] /= (hend - hstart) * (wend - wstart);
        }
      }
      // Do offset.
      Xdata += height * width;
      Ydata += pooled_height * pooled_width;
    }
  }
  return true;
}

template <>
bool PoolOp<float, CPUContext, AveragePool>::RunOnDeviceWithOrderNHWC() {
  auto& X = Input(0);
  auto* Y = Output(0);
  int height = X.dim32(1);
  int width = X.dim32(2);
  int channels = X.dim32(3);
  ConvPoolOpBase::SetOutputSize(X, Y, channels);
  const float* Xdata = X.data<float>();
  float* Ydata = Y->mutable_data<float>();
  math::Set<float, CPUContext>(Y->size(), 0, Ydata, &context_);
  // The main loop
  int pooled_height = Y->dim32(1);
  int pooled_width = Y->dim32(2);
  for (int n = 0; n < X.dim32(0); ++n) {
    for (int ph = 0; ph < pooled_height; ++ph) {
      for (int pw = 0; pw < pooled_width; ++pw) {
        int hstart = ph * stride_h_ - pad_t_;
        int wstart = pw * stride_w_ - pad_l_;
        int hend = min(hstart + kernel_h_, height);
        int wend = min(wstart + kernel_w_, width);
        hstart = max(hstart, 0);
        wstart = max(wstart, 0);
        const int pool_index = (ph * pooled_width + pw) * channels;
        for (int h = hstart; h < hend; ++h) {
          for (int w = wstart; w < wend; ++w) {
            const int input_index = (h * width + w) * channels;
            for (int c = 0; c < channels; ++c) {
              Ydata[pool_index + c] += Xdata[input_index + c];
            }
          }
        }
        float scale = 1. / (hend - hstart) / (wend - wstart);
        for (int c = 0; c < channels; ++c) {
          Ydata[pool_index + c] *= scale;
        }
      }
    }
    // Do offset.
    Xdata += X.size() / X.dim32(0);
    Ydata += Y->size() / Y->dim32(0);
  }
  return true;
}

template <>
bool PoolOp<float, CPUContext, MaxPool>::RunOnDeviceWithOrderNCHW() {
  auto& X = Input(0);
  auto* Y = Output(0);
  ConvPoolOpBase::SetOutputSize(X, Y, X.dim32(1));

  const float* Xdata = X.data<float>();
  float* Ydata = Y->mutable_data<float>();
  math::Set<float, CPUContext>(
      Y->size(), std::numeric_limits<float>::lowest(), Ydata, &context_);
  // The main loop
  int channels = X.dim32(1);
  int height = X.dim32(2);
  int width = X.dim32(3);
  int pooled_height = Y->dim32(2);
  int pooled_width = Y->dim32(3);
  for (int n = 0; n < X.dim32(0); ++n) {
    for (int c = 0; c < channels; ++c) {
      for (int ph = 0; ph < pooled_height; ++ph) {
        for (int pw = 0; pw < pooled_width; ++pw) {
          int hstart = ph * stride_h_ - pad_t_;
          int wstart = pw * stride_w_ - pad_l_;
          int hend = min(hstart + kernel_h_, height);
          int wend = min(wstart + kernel_w_, width);
          hstart = max(hstart, 0);
          wstart = max(wstart, 0);
          const int pool_index = ph * pooled_width + pw;
          for (int h = hstart; h < hend; ++h) {
            for (int w = wstart; w < wend; ++w) {
              const int input_index = h * width + w;
              if (Xdata[input_index] > Ydata[pool_index]) {
                Ydata[pool_index] = Xdata[input_index];
              }
            }
          }
        }
      }
      // Do offset.
      Xdata += height * width;
      Ydata += pooled_height * pooled_width;
    }
  }
  return true;
}

template <>
bool PoolOp<float, CPUContext, MaxPool>::RunOnDeviceWithOrderNHWC() {
  auto& X = Input(0);
  auto* Y = Output(0);
  int height = X.dim32(1);
  int width = X.dim32(2);
  int channels = X.dim32(3);
  ConvPoolOpBase::SetOutputSize(X, Y, channels);

  EigenMatrixMap<float> Ymat(
      Y->mutable_data<float>(), channels, Y->size() / channels);
  ConstEigenMatrixMap<float> Xmat(
      X.data<float>(), channels, X.size() / channels);
  int pooled_height = Y->dim32(1);
  int pooled_width = Y->dim32(2);

  // The main loop
  for (int n = 0; n < X.dim32(0); ++n) {
    for (int ph = 0; ph < pooled_height; ++ph) {
      int hstart = ph * stride_h_ - pad_t_;
      int hend = min(hstart + kernel_h_, height);
      hstart = max(hstart, 0);
      for (int pw = 0; pw < pooled_width; ++pw) {
        int wstart = pw * stride_w_ - pad_l_;
        int wend = min(wstart + kernel_w_, width);
        wstart = max(wstart, 0);
        // compute max in range X[n, hstart:hend, wstart:wend, :]
        auto Y_col = Ymat.col((n * pooled_height + ph) * pooled_width + pw);
        Y_col.setConstant(std::numeric_limits<float>::lowest());
        for (int h = hstart; h < hend; ++h) {
          for (int w = wstart; w < wend; ++w) {
            Y_col = Y_col.cwiseMax(Xmat.col((n * height + h) * width + w));
          }
        }
      }
    }
  }
  return true;
}

namespace {
REGISTER_CPU_OPERATOR(AveragePool, PoolOp<float, CPUContext, AveragePool>);

OPERATOR_SCHEMA(AveragePool)
  .NumInputs(1)
  .NumOutputs(1)
  .TensorInferenceFunction(ConvPoolOpBase<CPUContext>::TensorInferenceForPool)
  .SetDoc(R"DOC(
AveragePool consumes an input blob X and applies average pooling across the
the blob according to kernel sizes, stride sizes, and pad lengths defined by the
ConvPoolOpBase operator. Average pooling consisting of averaging all values of a
subset of the input tensor according to the kernel size and downsampling the
data into the output blob Y for further processing.
  )DOC")
  .Input(0, "X", "Input data tensor from the previous operator; dimensions "
  "depend on whether the NCHW or NHWC operators are being used. For example, "
  "in the former, the input has size (N x C x H x W), where N is the batch "
  "size, C is the number of channels, and H and W are the height and the width "
  "of the data. The corresponding permutation of dimensions is used in the "
  "latter case. ")
  .Output(0, "Y", "Output data tensor from average pooling across the input "
  "tensor. Dimensions will vary based on various kernel, stride, and pad "
  "sizes.");


REGISTER_CPU_OPERATOR(MaxPool, PoolOp<float, CPUContext, MaxPool>);

OPERATOR_SCHEMA(MaxPool)
  .NumInputs(1)
  .NumOutputs(1)
  .TensorInferenceFunction(ConvPoolOpBase<CPUContext>::TensorInferenceForPool)
  .SetDoc(R"DOC(
MaxPool consumes an input blob X and applies max pooling across the
the blob according to kernel sizes, stride sizes, and pad lengths defined by the
ConvPoolOpBase operator. Max pooling consisting of taking the maximumvalue of a
subset of the input tensor according to the kernel size and downsampling the
data into the output blob Y for further processing.
  )DOC")
  .Input(0, "X", "Input data tensor from the previous operator; dimensions "
  "depend on whether the NCHW or NHWC operators are being used. For example, "
  "in the former, the input has size (N x C x H x W), where N is the batch "
  "size, C is the number of channels, and H and W are the height and the width "
  "of the data. The corresponding permutation of dimensions is used in the "
  "latter case. ")
  .Output(0, "Y", "Output data tensor from max pooling across the input "
  "tensor. Dimensions will vary based on various kernel, stride, and pad "
  "sizes.");

}  // namespace
}  // namespace caffe2
