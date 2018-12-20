// TODO(ataei): reduce the apparent redundancy of all the code below.
#include "caffe2/operators/pool_op.h"
#include "caffe2/utils/cpu_neon.h"
#include "caffe2/utils/eigen_utils.h"

namespace caffe2 {

using std::max;
using std::min;

namespace {

#if defined(__ARM_NEON__) || defined(__ARM_NEON)

bool isNeon4x4p0s0Eligible(
    int inputH,
    int inputW,
    int outputH,
    int outputW,
    int kH,
    int kW,
    int strideH,
    int strideW,
    int padT,
    int padL,
    int padB,
    int padR,
    int dilationH,
    int dilationW,
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

  return kernelOk && strideOk && padOk && dilationOk && outputOk && inputOk &&
      alignOk;
}

// Vectorizes 4x4p0s0 averge pooling for ARM NEON
void avgPoolNeon4x4p0s0Plane(
    int inputH,
    int inputW,
    const float* input,
    float* output) {
  constexpr int kKernelHeight = 4;
  constexpr int kKernelWidth = 4;
  constexpr float kDiv = (1.0f / ((float)kKernelHeight * (float)kKernelWidth));

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

void runNeonAveragePool4x4p0s0NCHW(
    int N,
    int C,
    int inputH,
    int inputW,
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

bool isNeon2x2p0s0Eligible(
    int inputH,
    int inputW,
    int outputH,
    int outputW,
    int kH,
    int kW,
    int strideH,
    int strideW,
    int padT,
    int padL,
    int padB,
    int padR,
    int dilationH,
    int dilationW,
    const float* input,
    float* output) {
  // Use this kernel only if:
  // Kernel width is 2x2
  // Kernel stride is 2x2
  // Padding is 0
  // Dilation is 1
  // Output width and height are even divisors of input width
  // Input width and height are divisible by 4 (should be implied by
  // all of the above, but just check again)
  // Input and output pointers are aligned by float32x4_t

  bool kernelOk = (kH == 2) && (kW == 2);
  bool strideOk = (strideH == 2) && (strideW == 2);
  bool padOk = (padT == 0) && (padL == 0) && (padB == 0) && (padR == 0);
  bool dilationOk = (dilationH == 1) && (dilationW == 1);

  bool outputOk = ((inputH % outputH) == 0) && ((inputW % outputW) == 0);
  bool inputOk = (inputW % 4 == 0) && (inputH % 4 == 0);
  bool alignOk = isPointerAligned(input, sizeof(float32x4_t)) &&
      isPointerAligned(output, sizeof(float32x4_t));

  return kernelOk && strideOk && padOk && dilationOk && outputOk && inputOk &&
      alignOk;
}

// Vectorizes 2x2p0s0 averge pooling for ARM NEON
void maxPoolNeon2x2p0s0Plane(
    int inputH,
    int inputW,
    const float* input,
    float* output) {
  constexpr int kKernelHeight = 2;
  constexpr int kKernelWidth = 2;

  // Handle portion that can be unrolled by 4
  constexpr int kUnroll = 4;
  constexpr int kLoadSizeFloat = (sizeof(float32x4_t) / sizeof(float));
  constexpr int kLoadCols = kUnroll * kLoadSizeFloat;

  if (inputW % kLoadCols == 0) {
    for (int h = 0; h < inputH; h += kKernelHeight) {
      float* outputRow = output + (h / kKernelHeight) * (inputW / kKernelWidth);
      const float* curInput = input + h * inputW;

      for (int w = 0; w < inputW; w += kLoadCols) {
        float32x2_t hmax_0, hmax_1, hmax_2, hmax_3;
        {
          float32x4_t v0_0 = vld1q_f32_aligned(curInput + 0 * inputW);
          float32x4_t v0_1 = vld1q_f32_aligned(curInput + 1 * inputW);
          float32x4_t vmax = vmaxq_f32(v0_0, v0_1);
          hmax_0 = vpmax_f32(vget_low_f32(vmax), vget_high_f32(vmax));
        }
        curInput += kLoadSizeFloat;
        {
          float32x4_t v0_0 = vld1q_f32_aligned(curInput + 0 * inputW);
          float32x4_t v0_1 = vld1q_f32_aligned(curInput + 1 * inputW);
          float32x4_t vmax = vmaxq_f32(v0_0, v0_1);
          hmax_1 = vpmax_f32(vget_low_f32(vmax), vget_high_f32(vmax));
        }
        curInput += kLoadSizeFloat;
        {
          float32x4_t v0_0 = vld1q_f32_aligned(curInput + 0 * inputW);
          float32x4_t v0_1 = vld1q_f32_aligned(curInput + 1 * inputW);
          float32x4_t vmax = vmaxq_f32(v0_0, v0_1);
          hmax_2 = vpmax_f32(vget_low_f32(vmax), vget_high_f32(vmax));
        }
        curInput += kLoadSizeFloat;
        {
          float32x4_t v0_0 = vld1q_f32_aligned(curInput + 0 * inputW);
          float32x4_t v0_1 = vld1q_f32_aligned(curInput + 1 * inputW);
          float32x4_t vmax = vmaxq_f32(v0_0, v0_1);
          hmax_3 = vpmax_f32(vget_low_f32(vmax), vget_high_f32(vmax));
        }
        curInput += kLoadSizeFloat;

        float32x4_t out_0 = vcombine_f32(hmax_0, hmax_1);
        float32x4_t out_1 = vcombine_f32(hmax_2, hmax_3);
        vst1q_f32_aligned(&outputRow[w / kKernelWidth + 0], out_0);
        vst1q_f32_aligned(&outputRow[w / kKernelWidth + 4], out_1);
      }
    }
  } else {
    // Not unrolled
    for (int h = 0; h < inputH; h += kKernelHeight) {
      const float* inputRow = input + h * inputW;
      float* outputRow = output + (h / kKernelHeight) * (inputW / kKernelWidth);

      for (int w = 0; w < inputW; w += kKernelWidth * 2) {
        const float* curInput = inputRow + w;
        float32x4_t v0_0 = vld1q_f32_aligned(curInput + 0 * inputW);
        float32x4_t v0_1 = vld1q_f32_aligned(curInput + 1 * inputW);
        float32x4_t vmax = vmaxq_f32(v0_0, v0_1);
        float32x2_t hmax = vpmax_f32(vget_low_f32(vmax), vget_high_f32(vmax));
        vst1_f32(&outputRow[w / kKernelWidth], hmax);
      }
    }
  }
}

void runNeonMaxPool2x2p0s0NCHW(
    int N,
    int C,
    int inputH,
    int inputW,
    const float* input,
    float* output) {
  // We only have the 2x2p0s0 implementation at present, which is
  // checked at a higher level
  int outputH = inputH / 2;
  int outputW = inputW / 2;

  for (int n = 0; n < N; ++n) {
    for (int c = 0; c < C; ++c) {
      const float* curInput = input + (n * C + c) * inputH * inputW;
      float* curOutput = output + (n * C + c) * outputH * outputW;
      maxPoolNeon2x2p0s0Plane(inputH, inputW, curInput, curOutput);
    }
  }
}
#endif // defined(__ARM_NEON__) || defined(__ARM_NEON)

} // namespace

template <typename T>
class AveragePool {
 public:
  static float initialize() {
    return 0.0;
  }

  static void process(
      const int x_col,
      const int y_col,
      ConstEigenMatrixMap<float>& x_mat,
      EigenMatrixMap<float>& y_mat) {
    y_mat.col(y_col) += x_mat.col(x_col);
  }

  static void process(const T& x_data, T& y_data) {
    y_data += x_data;
  }

  static void finalize(const int size, T& y_data) {
    y_data /= size;
  }

  static void
  finalize(const int size, const int col, EigenMatrixMap<float>& y_mat) {
    y_mat.col(col) /= size;
  }

  static bool runSpecialized(
      int N,
      int C,
      int inputH,
      int inputW,
      int outputH,
      int outputW,
      int kH,
      int kW,
      int strideH,
      int strideW,
      int padT,
      int padL,
      int padB,
      int padR,
      int dilationH,
      int dilationW,
      const float* input,
      float* output) {
#if defined(__ARM_NEON__) || defined(__ARM_NEON)
    if (isNeon4x4p0s0Eligible(
            inputH,
            inputW,
            outputH,
            outputW,
            kH,
            kW,
            strideH,
            strideW,
            padT,
            padL,
            padB,
            padR,
            dilationH,
            dilationW,
            input,
            output)) {
      runNeonAveragePool4x4p0s0NCHW(N, C, inputH, inputW, input, output);
      return true;
    }
#else
    (void)N;
    (void)C;
    (void)inputH;
    (void)inputW;
    (void)outputH;
    (void)outputW;
    (void)kH;
    (void)kW;
    (void)strideH;
    (void)strideW;
    (void)padT;
    (void)padL;
    (void)padB;
    (void)padR;
    (void)dilationH;
    (void)dilationW;
    (void)input;
    (void)output;
#endif
    return false;
  }
};

template <typename T>
class MaxPool {
 public:
  static float initialize() {
    return std::numeric_limits<float>::lowest();
  }

  static void process(
      const int x_col,
      const int y_col,
      ConstEigenMatrixMap<float>& x_mat,
      EigenMatrixMap<float>& y_mat) {
    y_mat.col(y_col) = y_mat.col(y_col).cwiseMax(x_mat.col(x_col));
  }

  static void process(const T& x_data, T& y_data) {
    if (x_data > y_data) {
      y_data = x_data;
    }
  }

  static void finalize(const int /*size*/, T& /*y_data*/) {}

  static void finalize(
      const int /*size*/,
      const int /*col*/,
      EigenMatrixMap<float>& /*y_mat*/) {}

  static bool runSpecialized(
      int N,
      int C,
      int inputH,
      int inputW,
      int outputH,
      int outputW,
      int kH,
      int kW,
      int strideH,
      int strideW,
      int padT,
      int padL,
      int padB,
      int padR,
      int dilationH,
      int dilationW,
      const float* input,
      float* output) {
#if defined(__ARM_NEON__) || defined(__ARM_NEON)
    if (isNeon2x2p0s0Eligible(
            inputH,
            inputW,
            outputH,
            outputW,
            kH,
            kW,
            strideH,
            strideW,
            padT,
            padL,
            padB,
            padR,
            dilationH,
            dilationW,
            input,
            output)) {
      runNeonMaxPool2x2p0s0NCHW(N, C, inputH, inputW, input, output);
      return true;
    }
#else
    (void)N;
    (void)C;
    (void)inputH;
    (void)inputW;
    (void)outputH;
    (void)outputW;
    (void)kH;
    (void)kW;
    (void)strideH;
    (void)strideW;
    (void)padT;
    (void)padL;
    (void)padB;
    (void)padR;
    (void)dilationH;
    (void)dilationW;
    (void)input;
    (void)output;
#endif
    return false;
  }
};

template <typename T, class Context, typename PoolType>
bool PoolOp<T, Context, PoolType>::RunOnDeviceWithOrderNCHW() {
  auto& X = Input(0);
  auto* Y = Output(0);
  ConvPoolOpBase<Context>::SetOutputSize(X, Y, X.dim32(1));

  const float* Xdata = X.template data<float>();
  float* Ydata = Y->template mutable_data<float>();
  // The main loop
  int channels = X.dim32(1);
  int height = X.dim32(2);
  int width = kernel_.size() > 1 ? X.dim32(3) : 1;
  int depth = kernel_.size() > 2 ? X.dim32(4) : 1;
  int pooled_height = Y->dim32(2);
  int pooled_width = kernel_.size() > 1 ? Y->dim32(3) : 1;
  int pooled_depth = kernel_.size() > 2 ? Y->dim32(4) : 1;

  // We specialize certain variants on ARM for vectorization
  if (kernel_.size() == 2 &&
      PoolType::runSpecialized(
          X.dim32(0),
          X.dim32(1),
          X.dim32(2),
          X.dim32(3),
          Y->dim32(2),
          Y->dim32(3),
          kernel_h(),
          kernel_w(),
          stride_h(),
          stride_w(),
          pad_t(),
          pad_l(),
          pad_b(),
          pad_r(),
          dilation_h(),
          dilation_w(),
          Xdata,
          Ydata)) {
    return true;
  }

  switch (kernel_.size()) {
    case 1:
      for (int n = 0; n < X.dim32(0); ++n) {
        for (int c = 0; c < channels; ++c) {
          for (int ph = 0; ph < pooled_height; ++ph) {
            int hstart = ph * stride_h() - pad_t();
            int hend = min(hstart + kernel_h(), height);
            hstart = max(hstart, 0);
            T Yh = PoolType::initialize();
            for (int h = hstart; h < hend; ++h) {
              PoolType::process(Xdata[h], Yh);
            }
            PoolType::finalize(hend - hstart, Yh);
            Ydata[ph] = Yh;
          }
          // Do offset.
          Xdata += height;
          Ydata += pooled_height;
        }
      }
      break;
    case 2:
      for (int n = 0; n < X.dim32(0); ++n) {
        for (int c = 0; c < channels; ++c) {
          for (int ph = 0; ph < pooled_height; ++ph) {
            int hstart = ph * stride_h() - pad_t();
            int hend = min(hstart + kernel_h(), height);
            hstart = max(hstart, 0);
            for (int pw = 0; pw < pooled_width; ++pw) {
              int wstart = pw * stride_w() - pad_l();
              int wend = min(wstart + kernel_w(), width);
              wstart = max(wstart, 0);
              const int pool_index = ph * pooled_width + pw;
              T Yh = PoolType::initialize();
              for (int h = hstart; h < hend; ++h) {
                for (int w = wstart; w < wend; ++w) {
                  const int input_index = h * width + w;
                  PoolType::process(Xdata[input_index], Yh);
                }
              }
              PoolType::finalize((hend - hstart) * (wend - wstart), Yh);
              Ydata[pool_index] = Yh;
            }
          }
          // Do offset.
          Xdata += height * width;
          Ydata += pooled_height * pooled_width;
        }
      }
      break;
    case 3:
      for (int n = 0; n < X.dim32(0); ++n) {
        for (int c = 0; c < channels; ++c) {
          for (int ph = 0; ph < pooled_height; ++ph) {
            int hstart = ph * stride_h() - pad_t();
            int hend = min(hstart + kernel_h(), height);
            hstart = max(hstart, 0);
            for (int pw = 0; pw < pooled_width; ++pw) {
              int wstart = pw * stride_w() - pad_l();
              int wend = min(wstart + kernel_w(), width);
              wstart = max(wstart, 0);
              for (int pd = 0; pd < pooled_depth; ++pd) {
                int dstart = pd * stride_[2] - pads_[2];
                int dend = min(dstart + kernel_[2], depth);
                dstart = max(dstart, 0);
                const int pool_index =
                    ph * pooled_width * pooled_depth + pw * pooled_depth + pd;
                T Yh = PoolType::initialize();
                for (int h = hstart; h < hend; ++h) {
                  for (int w = wstart; w < wend; ++w) {
                    for (int d = dstart; d < dend; ++d) {
                      const int input_index = h * width * depth + w * depth + d;
                      PoolType::process(Xdata[input_index], Yh);
                    }
                  }
                }
                PoolType::finalize(
                    (hend - hstart) * (wend - wstart) * (dend - dstart), Yh);
                Ydata[pool_index] = Yh;
              }
            }
          }
          // Do offset.
          Xdata += height * width * depth;
          Ydata += pooled_height * pooled_width * pooled_depth;
        }
      }
      break;
    default:
      CAFFE_THROW("Unsupported pooling size : ", kernel_.size());
      return false;
  }
  return true;
}

template <typename T, class Context, typename PoolType>
bool PoolOp<T, Context, PoolType>::RunOnDeviceWithOrderNHWC() {
  auto& X = Input(0);
  auto* Y = Output(0);
  int height = X.dim32(1);
  int width = kernel_.size() > 1 ? X.dim32(2) : 1;
  int depth = kernel_.size() > 2 ? X.dim32(3) : 1;
  int channels = X.dim32(X.dim() - 1);
  ConvPoolOpBase<Context>::SetOutputSize(X, Y, channels);

  EigenMatrixMap<float> Ymat(
      Y->template mutable_data<float>(), channels, Y->numel() / channels);
  ConstEigenMatrixMap<float> Xmat(
      X.template data<float>(), channels, X.numel() / channels);
  int pooled_height = Y->dim32(1);
  int pooled_width = kernel_.size() > 1 ? Y->dim32(2) : 1;
  int pooled_depth = kernel_.size() > 2 ? Y->dim32(3) : 1;
  // The main loop
  switch (kernel_.size()) {
    case 1:
      for (int n = 0; n < X.dim32(0); ++n) {
        for (int ph = 0; ph < pooled_height; ++ph) {
          int hstart = ph * stride_h() - pad_t();
          int hend = min(hstart + kernel_h(), height);
          hstart = max(hstart, 0);
          const int y_col = n * pooled_height + ph;
          Ymat.col(y_col).setConstant(PoolType::initialize());
          for (int h = hstart; h < hend; ++h) {
            const int x_col = n * height + h;
            PoolType::process(x_col, y_col, Xmat, Ymat);
          }
          PoolType::finalize((hend - hstart), y_col, Ymat);
        }
      }
      break;
    case 2:
      for (int n = 0; n < X.dim32(0); ++n) {
        for (int ph = 0; ph < pooled_height; ++ph) {
          int hstart = ph * stride_h() - pad_t();
          int hend = min(hstart + kernel_h(), height);
          hstart = max(hstart, 0);
          for (int pw = 0; pw < pooled_width; ++pw) {
            int wstart = pw * stride_w() - pad_l();
            int wend = min(wstart + kernel_w(), width);
            wstart = max(wstart, 0);
            const int y_col = (n * pooled_height + ph) * pooled_width + pw;
            Ymat.col(y_col).setConstant(PoolType::initialize());
            for (int h = hstart; h < hend; ++h) {
              for (int w = wstart; w < wend; ++w) {
                const int x_col = (n * height + h) * width + w;
                PoolType::process(x_col, y_col, Xmat, Ymat);
              }
            }
            PoolType::finalize((hend - hstart) * (wend - wstart), y_col, Ymat);
          }
        }
      }
      break;
    case 3:
      for (int n = 0; n < X.dim32(0); ++n) {
        for (int ph = 0; ph < pooled_height; ++ph) {
          int hstart = ph * stride_h() - pad_t();
          int hend = min(hstart + kernel_h(), height);
          hstart = max(hstart, 0);
          for (int pw = 0; pw < pooled_width; ++pw) {
            int wstart = pw * stride_w() - pad_l();
            int wend = min(wstart + kernel_w(), width);
            wstart = max(wstart, 0);
            for (int pd = 0; pd < pooled_depth; ++pd) {
              int dstart = pd * stride_[2] - pads_[2];
              int dend = min(dstart + kernel_[2], depth);
              dstart = max(dstart, 0);
              const int y_col = ((n * pooled_height + ph) * pooled_width + pw) *
                      pooled_depth +
                  pd;
              Ymat.col(y_col).setConstant(PoolType::initialize());
              for (int h = hstart; h < hend; ++h) {
                for (int w = wstart; w < wend; ++w) {
                  for (int d = dstart; d < dend; ++d) {
                    const int x_col =
                        ((n * height + h) * width + w) * depth + d;
                    PoolType::process(x_col, y_col, Xmat, Ymat);
                  }
                }
              }
              PoolType::finalize(
                  (hend - hstart) * (wend - wstart) * (dend - dstart),
                  y_col,
                  Ymat);
            }
          }
        }
      }
      break;
    default:
      CAFFE_THROW("Unsupported pooling size : ", kernel_.size());
      return false;
  }
  return true;
}
const char kAveragePoolDoc[] = R"DOC(
consumes an input blob and applies average pooling across the the blob according
to kernel sizes, stride sizes, pad lengths and dilation. Average pooling consists
of taking the average value of a subset of the input tensor according to the kernel
size and downsampling the data into the output blob for further processing. The
`brew` module has a wrapper for this operator for use in a `ModelHelper` object.

Pooling layers reduce the spatial dimensionality of the input blob. Each of the
output blob's dimensions will reduce according to:

$$dim_{out}=\frac{dim_{in}-kernel+2*pad}{stride}+1$$

Github Links:

- https://github.com/pytorch/pytorch/blob/master/caffe2/operators/pool_op.h
- https://github.com/pytorch/pytorch/blob/master/caffe2/operators/pool_op.cc
- https://github.com/pytorch/pytorch/blob/master/caffe2/operators/conv_pool_op_base.h


<details>

<summary> <b>Example</b> </summary>

**Code**

```
workspace.ResetWorkspace()

op = core.CreateOperator(
    "AveragePool",
    ["X"],
    ["Y"],
    kernel=2,
    stride=2,
)

workspace.FeedBlob("X", np.random.randn(1, 1, 6, 6).astype(np.float32)) // NCHW
print("X:\n", workspace.FetchBlob("X"), "\n")
workspace.RunOperatorOnce(op)
print("Y:\n", workspace.FetchBlob("Y"))
```

**Result**

```
X:
 [[[[-0.2883434   0.43498734  0.05417408  1.912558    0.09390241
    -0.33173105]
   [ 1.633709    1.2047161   0.36964908  0.99961185  0.4184147
     0.9989975 ]
   [ 1.7644193   0.1789665   1.5812988  -0.6038542  -0.36090398
     0.33195344]
   [ 0.9457722  -0.95174325 -0.78124577  1.2062047   1.1903144
     0.2586746 ]
   [ 1.252104    0.32645547  1.8073524  -0.78397465  0.9978303
    -0.97614396]
   [ 0.5440196   1.5778259  -0.76750124  0.5051756   0.8838398
    -0.37085298]]]]

Y:
 [[[[0.7462672  0.83399826 0.2948959 ]
   [0.4843537  0.3506009  0.35500962]
   [0.9251013  0.19026303 0.13366827]]]]
```

</details>

)DOC";

const char kMaxPoolDoc[] = R"DOC(
consumes an input blob and applies max pooling across the the blob according to
kernel sizes, stride sizes, pad lengths and dilation. Max pooling consists of
taking the maximum value of a subset of the input tensor according to the kernel
size and downsampling the data into the output blob for further processing. The
`brew` module has a wrapper for this operator for use in a `ModelHelper` object.

Pooling layers reduce the spatial dimensionality of the input blob. Each of the
output blob's dimensions will reduce according to:

$$dim_{out}=\frac{dim_{in}-kernel+2*pad}{stride}+1$$

Github Links:

- https://github.com/pytorch/pytorch/blob/master/caffe2/operators/pool_op.h
- https://github.com/pytorch/pytorch/blob/master/caffe2/operators/pool_op.cc
- https://github.com/pytorch/pytorch/blob/master/caffe2/operators/conv_pool_op_base.h

<details>

<summary> <b>Example</b> </summary>

**Code**

```
workspace.ResetWorkspace()

op = core.CreateOperator(
    "MaxPool",
    ["X"],
    ["Y"],
    kernel=2,
    stride=2,
)

workspace.FeedBlob("X", np.random.randn(1, 1, 6, 6).astype(np.float32)) // NCHW
print("X:\n", workspace.FetchBlob("X"), "\n")
workspace.RunOperatorOnce(op)
print("Y:\n", workspace.FetchBlob("Y"))
```

**Result**

```
X:
 [[[[-2.8534958e-01 -1.7719941e+00 -8.2277227e-04  1.1088650e+00
    -2.1476576e+00 -3.5070452e-01]
   [-9.0058845e-01 -3.0070004e-01 -1.7907504e+00 -7.1746534e-01
     1.2798511e+00 -3.2214901e-01]
   [ 1.5806322e+00  1.6845188e+00 -2.6633200e-01 -3.8576153e-01
    -9.6424848e-02 -3.9696163e-01]
   [ 1.2572408e-01  6.3612902e-01 -3.9554062e-01 -6.9735396e-01
    -9.1898698e-01 -1.9609968e-01]
   [-1.1587460e+00  2.4605224e+00 -1.5497679e+00  1.3020347e-01
    -8.1293899e-01 -7.8803545e-01]
   [ 1.4323474e+00  1.3618395e+00  9.8975077e-02 -1.1307785e-01
     7.2035044e-01  2.7642491e-01]]]]

Y:
 [[[[-0.28534958  1.108865    1.2798511 ]
   [ 1.6845188  -0.266332   -0.09642485]
   [ 2.4605224   0.13020347  0.72035044]]]]

```

</details>

)DOC";

std::function<void(OpSchema&)> AveragePoolDocGenerator(const char* dim) {
  return [=](OpSchema& schema) {
    string doc = "AveragePool{dim} {pool_doc}";
    c10::ReplaceAll(doc, "{dim}", dim);
    c10::ReplaceAll(doc, "{pool_doc}", kAveragePoolDoc);
    schema.SetDoc(doc);
    schema.Input(
        0,
        "X",
        "*(type: Tensor`<float>`)* Input data tensor of shape NCHW or NHWC.");
    schema.Output(
        0,
        "Y",
        "*(type: Tensor`<float>`)* Output data tensor.");
    /*
    schema.Arg("kernel", "*(type: int)* Size of the window to take an average over.");
    schema.Arg("stride", "*(type: int)* Stride of the window.");
    schema.Arg("pad", "*(type: int)* Implicit zero padding to be added on both sides.");
    schema.Arg("dilation", "*(type: int)* Parameter that controls the stride of elements in the window.");
    schema.Arg("order", "*(type: string; default: 'NCHW')* Order of the blob dimensions.");
    */
  };
}

std::function<void(OpSchema&)> MaxPoolDocGenerator(const char* dim) {
  return [=](OpSchema& schema) {
    string doc = "MaxPool{dim} {pool_doc}";
    c10::ReplaceAll(doc, "{dim}", dim);
    c10::ReplaceAll(doc, "{pool_doc}", kMaxPoolDoc);
    schema.SetDoc(doc);
    schema.Input(
        0,
        "X",
        "*(type: Tensor`<float>`)* Input data tensor of shape NCHW or NHWC.");
    schema.Output(
        0,
        "Y",
        "*(type: Tensor`<float>`)* Output data tensor.");
    /*
    schema.Arg("kernel", "*(type: int)* Size of the window to take an average over.");
    schema.Arg("stride", "*(type: int)* Stride of the window.");
    schema.Arg("pad", "*(type: int)* Implicit zero padding to be added on both sides.");
    schema.Arg("dilation", "*(type: int)* Parameter that controls the stride of elements in the window.");
    schema.Arg("order", "*(type: string; default: 'NCHW')* Order of the blob dimensions.");
    */
  };
}
REGISTER_CPU_OPERATOR(
    AveragePool,
    PoolOp<float, CPUContext, AveragePool<float>>);

OPERATOR_SCHEMA(AveragePool)
    .NumInputs(1)
    .NumOutputs(1)
    .TensorInferenceFunction(ConvPoolOpBase<CPUContext>::TensorInferenceForPool)
    .FillUsing(AveragePoolDocGenerator(""))
    .InheritOnnxSchema();

REGISTER_CPU_OPERATOR(
    AveragePool1D,
    PoolOp<float, CPUContext, AveragePool<float>>);

OPERATOR_SCHEMA(AveragePool1D)
    .NumInputs(1)
    .NumOutputs(1)
    .TensorInferenceFunction(ConvPoolOpBase<CPUContext>::TensorInferenceForPool)
    .FillUsing(AveragePoolDocGenerator("1D"))
    .InheritOnnxSchema("AveragePool");

REGISTER_CPU_OPERATOR(
    AveragePool2D,
    PoolOp<float, CPUContext, AveragePool<float>>);

OPERATOR_SCHEMA(AveragePool2D)
    .NumInputs(1)
    .NumOutputs(1)
    .TensorInferenceFunction(ConvPoolOpBase<CPUContext>::TensorInferenceForPool)
    .FillUsing(AveragePoolDocGenerator("2D"))
    .InheritOnnxSchema("AveragePool");

REGISTER_CPU_OPERATOR(
    AveragePool3D,
    PoolOp<float, CPUContext, AveragePool<float>>);

OPERATOR_SCHEMA(AveragePool3D)
    .NumInputs(1)
    .NumOutputs(1)
    .TensorInferenceFunction(ConvPoolOpBase<CPUContext>::TensorInferenceForPool)
    .FillUsing(AveragePoolDocGenerator("3D"))
    .InheritOnnxSchema("AveragePool");

REGISTER_CPU_OPERATOR(MaxPool, PoolOp<float, CPUContext, MaxPool<float>>);

OPERATOR_SCHEMA(MaxPool)
    .NumInputs(1)
    .NumOutputs(1)
    .TensorInferenceFunction(ConvPoolOpBase<CPUContext>::TensorInferenceForPool)
    .FillUsing(MaxPoolDocGenerator(""))
    .InheritOnnxSchema();

REGISTER_CPU_OPERATOR(MaxPool1D, PoolOp<float, CPUContext, MaxPool<float>>);

OPERATOR_SCHEMA(MaxPool1D)
    .NumInputs(1)
    .NumOutputs(1)
    .TensorInferenceFunction(ConvPoolOpBase<CPUContext>::TensorInferenceForPool)
    .FillUsing(MaxPoolDocGenerator("1D"))
    .InheritOnnxSchema("MaxPool");

REGISTER_CPU_OPERATOR(MaxPool2D, PoolOp<float, CPUContext, MaxPool<float>>);

OPERATOR_SCHEMA(MaxPool2D)
    .NumInputs(1)
    .NumOutputs(1)
    .TensorInferenceFunction(ConvPoolOpBase<CPUContext>::TensorInferenceForPool)
    .FillUsing(MaxPoolDocGenerator("2D"))
    .InheritOnnxSchema("MaxPool");

REGISTER_CPU_OPERATOR(MaxPool3D, PoolOp<float, CPUContext, MaxPool<float>>);

OPERATOR_SCHEMA(MaxPool3D)
    .NumInputs(1)
    .NumOutputs(1)
    .TensorInferenceFunction(ConvPoolOpBase<CPUContext>::TensorInferenceForPool)
    .FillUsing(MaxPoolDocGenerator("3D"))
    .InheritOnnxSchema("MaxPool");
} // namespace caffe2
