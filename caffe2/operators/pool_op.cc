// TODO(ataei): reduce the apparent redundancy of all the code below.
#include "caffe2/operators/pool_op.h"
#include "caffe2/utils/cpu_neon.h"

namespace caffe2 {

using std::max;
using std::min;

namespace {

#ifdef __ARM_NEON__

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
#endif // __ARM_NEON__

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
#ifdef __ARM_NEON__
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
#ifdef __ARM_NEON__
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
  int channels = X.dim32(X.ndim() - 1);
  ConvPoolOpBase<Context>::SetOutputSize(X, Y, channels);

  EigenMatrixMap<float> Ymat(
      Y->template mutable_data<float>(), channels, Y->size() / channels);
  ConstEigenMatrixMap<float> Xmat(
      X.template data<float>(), channels, X.size() / channels);
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
const char* kAveragePoolDoc = R"DOC(
consumes an input blob X and applies average pooling across the
the blob according to kernel sizes, stride sizes, and pad lengths defined by the
ConvPoolOpBase operator. Average pooling consisting of averaging all values of a
subset of the input tensor according to the kernel size and downsampling the
data into the output blob Y for further processing.
)DOC";

const char* kMaxPoolDoc = R"DOC(
consumes an input blob X and applies max pooling across the
the blob according to kernel sizes, stride sizes, and pad lengths defined by the
ConvPoolOpBase operator. Max pooling consisting of taking the maximum value of a
subset of the input tensor according to the kernel size and downsampling the
data into the output blob Y for further processing.
)DOC";

std::function<void(OpSchema&)> AveragePoolDocGenerator(const char* dim) {
  return [=](OpSchema& schema) {
    string doc = "AveragePool{dim} {pool_doc}";
    ReplaceAll(doc, "{dim}", dim);
    ReplaceAll(doc, "{pool_doc}", kAveragePoolDoc);
    schema.SetDoc(doc);
    schema.Input(
        0,
        "X",
        "Input data tensor from the previous operator; dimensions depend on "
        "whether the NCHW or NHWC operators are being used. For example, in "
        "the former, the input has size (N x C x H x W), where N is the batch "
        "size, C is the number of channels, and H and W are the height and the "
        "width of the data. The corresponding permutation of dimensions is "
        "used in the latter case.");
    schema.Output(
        0,
        "Y",
        "Output data tensor from average pooling across the input "
        "tensor. Dimensions will vary based on various kernel, stride, and pad "
        "sizes.");
  };
}

std::function<void(OpSchema&)> MaxPoolDocGenerator(const char* dim) {
  return [=](OpSchema& schema) {
    string doc = "MaxPool{dim} {pool_doc}";
    ReplaceAll(doc, "{dim}", dim);
    ReplaceAll(doc, "{pool_doc}", kMaxPoolDoc);
    schema.SetDoc(doc);
    schema.Input(
        0,
        "X",
        "Input data tensor from the previous operator; dimensions depend on "
        "whether the NCHW or NHWC operators are being used. For example, in "
        "the former, the input has size (N x C x H x W), where N is the batch "
        "size, C is the number of channels, and H and W are the height and the "
        "width of the data. The corresponding permutation of dimensions is "
        "used in the latter case.");
    schema.Output(
        0,
        "Y",
        "Output data tensor from max pooling across the input "
        "tensor. Dimensions will vary based on various kernel, stride, and pad "
        "sizes.");
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
    .InheritOnnxSchema("AveragePool");

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
    .InheritOnnxSchema("MaxPool");

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
