#include "caffe2/core/operator.h"
#include "caffe2/utils/cpu_neon.h"
#include "caffe2/utils/math.h"

#ifdef CAFFE2_USE_MKLDNN
#include <caffe2/ideep/operators/operator_fallback_ideep.h>
#include <caffe2/ideep/utils/ideep_operator.h>
#endif

namespace caffe2 {

#if defined(__ARM_NEON__) || defined(__ARM_NEON)
namespace {

//
// ARM Neon code utilities
//

inline float32x4_t to_v4_f32(uint16x4_t v) {
  return vcvtq_f32_u32(vmovl_u16(v));
}

inline float32x4x4_t to_f32_v4_x4(uint8x16_t v) {
  float32x4x4_t out;

  uint16x8_t lo_u16 = vmovl_u8(vget_low_u8(v));

  out.val[0] = to_v4_f32(vget_low_u16(lo_u16));
  out.val[1] = to_v4_f32(vget_high_u16(lo_u16));

  uint16x8_t hi_u16 = vmovl_u8(vget_high_u8(v));

  out.val[2] = to_v4_f32(vget_low_u16(hi_u16));
  out.val[3] = to_v4_f32(vget_high_u16(hi_u16));

  return out;
}

inline void clamp(float32x4_t& v) {
  v = vmaxq_f32(v, vdupq_n_f32(0));
  v = vminq_f32(v, vdupq_n_f32((float)std::numeric_limits<uint8_t>::max()));
}

inline void addMeanAndClamp(float32x4_t& v, float mean) {
  v = vaddq_f32(v, vdupq_n_f32(mean));
  clamp(v);
}

inline uint8x8_t convertNarrowAndPack(float32x4_t v0, float32x4_t v1) {
  uint16x4_t u16_0 = vmovn_u32(vcvtq_u32_f32(v0));
  uint16x4_t u16_1 = vmovn_u32(vcvtq_u32_f32(v1));
  uint16x8_t u16_01 = vcombine_u16(u16_0, u16_1);
  return vmovn_u16(u16_01);
}

} // unnamed namespace
#endif // defined(__ARM_NEON__) || defined(__ARM_NEON)

class PackedInt8BGRANHWCToNCHWCStylizerPreprocessOp
    : public Operator<CPUContext> {
 public:
  // Expect this many channels as input
  static constexpr int kInputChannels = 4;

  // Expect this many channels as output
  static constexpr int kOutputChannels = 3;

  // We read this much noise per vectorized cycle
  static constexpr int kNeonNoiseReadSize = kOutputChannels * 16;

  USE_OPERATOR_FUNCTIONS(CPUContext);
  PackedInt8BGRANHWCToNCHWCStylizerPreprocessOp(
      const OperatorDef& operator_def,
      Workspace* ws)
      : Operator<CPUContext>(operator_def, ws), ws_(ws) {}

  bool RunOnDevice() override {
    const auto& X = Input(0);
    const auto& mean = Input(1);

    auto* noiseBlob = ws_->CreateBlob("__CAFFE2_STYLIZER_NOISE__");
    auto defaultNoiseSize = OperatorBase::GetSingleArgument<int>(
        "noise_size", 491 /* prime to avoid artifacts */);

    if (!BlobIsTensorType(*noiseBlob, CPU)) {
      // Initialize random noise on first use.
      // Cache it to maintain temporal consistency.
      auto* t = BlobGetMutableTensor(noiseBlob, CPU);

#if defined(__ARM_NEON__) || defined(__ARM_NEON)
      // Noise space is larger for vectorized code due to the
      // vectorized load
      initNoiseCPUNeon(t, defaultNoiseSize);
#else
      initNoiseCPU(t, defaultNoiseSize);
#endif
    }
    const auto& noise = noiseBlob->template Get<TensorCPU>();
    CAFFE_ENFORCE(noise.numel() >= defaultNoiseSize);

    CAFFE_ENFORCE(X.dim() == 4);
    const int N = X.dim32(0), H = X.dim32(1), W = X.dim32(2), C = X.dim32(3);
    // Assume BGR or BGRA
    CAFFE_ENFORCE(mean.numel() == kOutputChannels);

    CAFFE_ENFORCE(C == kInputChannels);
    auto* Y = Output(0, {N, kOutputChannels, H, W}, at::dtype<float>());

    runBatch(
        N,
        C,
        H,
        W,
        defaultNoiseSize,
        X.data<uint8_t>(),
        mean.data<float>(),
        noise.data<float>(),
        Y->template mutable_data<float>());

    return true;
  }

#if !defined(__ARM_NEON__) && !defined(__ARM_NEON)
  void initNoiseCPU(Tensor* noise, int size) {
    noise->Resize(size);

    math::RandGaussian<float, CPUContext>(
        size,
        0.0,
        OperatorBase::GetSingleArgument<float>("noise_std", 10.0),
        noise->template mutable_data<float>(),
        &context_);
  }
#endif // !defined(__ARM_NEON__) && !defined(__ARM_NEON)

#if defined(__ARM_NEON__) || defined(__ARM_NEON)
  void initNoiseCPUNeon(Tensor* noise, int size) {
    // For ARM NEON, we read in multiples of kNeonNoiseReadSize since
    // the inner loop is vectorized. Round up to the next highest
    // multiple of kNeonNoiseReadSize
    size = math::RoundUp(size, kNeonNoiseReadSize) + size;
    noise->Resize(size);

    math::RandGaussian<float, CPUContext>(
        size,
        0.0,
        OperatorBase::GetSingleArgument<float>("noise_std", 10.0),
        noise->template mutable_data<float>(),
        &context_);
  }
#endif // defined(__ARM_NEON__) || defined(__ARM_NEON)

  void runBatch(
      int N,
      int /*C*/,
      int H,
      int W,
      int noiseCycle,
      const uint8_t* input,
      const float* meanChannel,
      const float* noise,
      float* output) {
    int planeSize = H * W;

    for (int n = 0; n < N; ++n) {
      auto curInput = input + n * kInputChannels * planeSize;
      auto curOutput = output + n * kOutputChannels * planeSize;

#if defined(__ARM_NEON__) || defined(__ARM_NEON)
      runCPUNeon(H, W, noiseCycle, curInput, meanChannel, noise, curOutput);
#else
      runCPU(H, W, noiseCycle, curInput, meanChannel, noise, curOutput);
#endif // defined(__ARM_NEON__) || defined(__ARM_NEON)
    }
  }

#if !defined(__ARM_NEON__) && !defined(__ARM_NEON)
  void runCPU(
      int H,
      int W,
      int noiseCycle,
      const uint8_t* input,
      const float* meanChannel,
      const float* noise,
      float* output) {
    int planeSize = H * W;
    int noiseOffset = 0;

    for (int point = 0; point < planeSize; ++point) {
      for (int c = 0; c < kOutputChannels; ++c) {
        float v = (float)input[point * kInputChannels + c];
        output[c * planeSize + point] = v - meanChannel[c] + noise[noiseOffset];

        if (++noiseOffset >= noiseCycle) {
          noiseOffset = 0;
        }
      }
    }
  }
#endif // !defined(__ARM_NEON__) && !defined(__ARM_NEON)

#if defined(__ARM_NEON__) || defined(__ARM_NEON)
  void runCPUNeon(
      int H,
      int W,
      int noiseCycle,
      const uint8_t* input,
      const float* meanChannel,
      const float* noise,
      float* output) {
    // Vectorized load parameters:

    // Loop unroll factor
    // FIXME: this doesn't actually unroll; clang has per-loop unroll
    // pragmas but GCC does not
    constexpr int kUnroll = 1;

    // How much data we load for each inner loop
    constexpr int kInnerLoadSize = sizeof(uint8x16x4_t);

    // What we write out
    constexpr int kInnerStoreSize = sizeof(float32x4_t);

    // We load 16 pixels at a time, with 4 channels each
    constexpr int kLoadPixels = kInnerLoadSize / kInputChannels;
    static_assert(kLoadPixels == 16, "unexpected");

    // How many pixels we load per loop
    constexpr int kLoadPixelsPerLoop = kLoadPixels * kUnroll;

    // We need at least this much noise each loop through
    CAFFE_ENFORCE_GE(noiseCycle, kOutputChannels * kLoadPixelsPerLoop);

    int noiseUsed = 0;
    const float* curNoise = noise;

    float mean[kOutputChannels] = {
        meanChannel[0], meanChannel[1], meanChannel[2]};
    int planeSize = H * W;

    // Vectorized portion
    int point = 0;

    // If the slice is not aligned, then we have to use the
    // un-vectorized version
    bool isAligned = isPointerAligned(input, kInnerLoadSize) &&
        isPointerAligned(output, kInnerStoreSize) &&
        // Because we are writing to output at offsets of planeSize,
        // planeSize has to be an even multiple of kInnerStoreSize
        (planeSize % kInnerStoreSize == 0);

    // What portion the vectorized loop will handle
    int limit =
        isAligned ? (planeSize / kLoadPixelsPerLoop) * kLoadPixelsPerLoop : 0;

    for (; point < limit; point += kLoadPixelsPerLoop) {
      // Unroll load/update/store by kUnroll
      for (int j = 0; j < kUnroll; ++j) {
        // We load 16 pixels x 4 channels at a time
        const uint8_t* inputAligned = (const uint8_t*)__builtin_assume_aligned(
            input + (point + j * kLoadPixels) * kInputChannels,
            sizeof(uint8x16x4_t));
        uint8x16x4_t loadV = vld4q_u8(inputAligned);

        // The compiler doesn't want to unroll this when we put it in a
        // loop, and in GCC there's no per-loop unroll pragma, so we do
        // it manually.
        // This seems to involve no register spillage, crossing fingers
        // that it remains that way.
        {
          constexpr int kChannel = 0;
          float32x4_t noise0 = vld1q_f32(curNoise + j * 48 + 0);
          float32x4_t noise1 = vld1q_f32(curNoise + j * 48 + 4);
          float32x4_t noise2 = vld1q_f32(curNoise + j * 48 + 8);
          float32x4_t noise3 = vld1q_f32(curNoise + j * 48 + 12);

          float32x4x4_t outV = to_f32_v4_x4(loadV.val[kChannel]);
          float32x4_t meanV = vdupq_n_f32(mean[kChannel]);
          outV.val[0] = vsubq_f32(outV.val[0], meanV);
          outV.val[1] = vsubq_f32(outV.val[1], meanV);
          outV.val[2] = vsubq_f32(outV.val[2], meanV);
          outV.val[3] = vsubq_f32(outV.val[3], meanV);

          outV.val[0] = vaddq_f32(outV.val[0], noise0);
          outV.val[1] = vaddq_f32(outV.val[1], noise1);
          outV.val[2] = vaddq_f32(outV.val[2], noise2);
          outV.val[3] = vaddq_f32(outV.val[3], noise3);

          float* outputAligned = (float*)__builtin_assume_aligned(
              &output[kChannel * planeSize + (point + j * kLoadPixels)],
              sizeof(float32x4_t));

          vst1q_f32(outputAligned + 0, outV.val[0]);
          vst1q_f32(outputAligned + 4, outV.val[1]);
          vst1q_f32(outputAligned + 8, outV.val[2]);
          vst1q_f32(outputAligned + 12, outV.val[3]);
        }

        {
          constexpr int kChannel = 1;
          float32x4_t noise0 = vld1q_f32(curNoise + j * 48 + 16);
          float32x4_t noise1 = vld1q_f32(curNoise + j * 48 + 20);
          float32x4_t noise2 = vld1q_f32(curNoise + j * 48 + 24);
          float32x4_t noise3 = vld1q_f32(curNoise + j * 48 + 28);

          float32x4x4_t outV = to_f32_v4_x4(loadV.val[kChannel]);
          float32x4_t meanV = vdupq_n_f32(mean[kChannel]);
          outV.val[0] = vsubq_f32(outV.val[0], meanV);
          outV.val[1] = vsubq_f32(outV.val[1], meanV);
          outV.val[2] = vsubq_f32(outV.val[2], meanV);
          outV.val[3] = vsubq_f32(outV.val[3], meanV);

          outV.val[0] = vaddq_f32(outV.val[0], noise0);
          outV.val[1] = vaddq_f32(outV.val[1], noise1);
          outV.val[2] = vaddq_f32(outV.val[2], noise2);
          outV.val[3] = vaddq_f32(outV.val[3], noise3);

          float* outputAligned = (float*)__builtin_assume_aligned(
              &output[kChannel * planeSize + (point + j * kLoadPixels)],
              sizeof(float32x4_t));

          vst1q_f32(outputAligned + 0, outV.val[0]);
          vst1q_f32(outputAligned + 4, outV.val[1]);
          vst1q_f32(outputAligned + 8, outV.val[2]);
          vst1q_f32(outputAligned + 12, outV.val[3]);
        }

        {
          constexpr int kChannel = 2;
          float32x4_t noise0 = vld1q_f32(curNoise + j * 48 + 32);
          float32x4_t noise1 = vld1q_f32(curNoise + j * 48 + 36);
          float32x4_t noise2 = vld1q_f32(curNoise + j * 48 + 40);
          float32x4_t noise3 = vld1q_f32(curNoise + j * 48 + 44);

          float32x4x4_t outV = to_f32_v4_x4(loadV.val[kChannel]);
          float32x4_t meanV = vdupq_n_f32(mean[kChannel]);
          outV.val[0] = vsubq_f32(outV.val[0], meanV);
          outV.val[1] = vsubq_f32(outV.val[1], meanV);
          outV.val[2] = vsubq_f32(outV.val[2], meanV);
          outV.val[3] = vsubq_f32(outV.val[3], meanV);

          outV.val[0] = vaddq_f32(outV.val[0], noise0);
          outV.val[1] = vaddq_f32(outV.val[1], noise1);
          outV.val[2] = vaddq_f32(outV.val[2], noise2);
          outV.val[3] = vaddq_f32(outV.val[3], noise3);

          float* outputAligned = (float*)__builtin_assume_aligned(
              &output[kChannel * planeSize + (point + j * kLoadPixels)],
              sizeof(float32x4_t));

          vst1q_f32(outputAligned + 0, outV.val[0]);
          vst1q_f32(outputAligned + 4, outV.val[1]);
          vst1q_f32(outputAligned + 8, outV.val[2]);
          vst1q_f32(outputAligned + 12, outV.val[3]);
        }
      }

      curNoise += (kLoadPixels * kOutputChannels) * kUnroll;
      noiseUsed += (kLoadPixels * kOutputChannels) * kUnroll;

      if (noiseUsed >= noiseCycle) {
        noiseUsed = 0;
        curNoise = noise + ((curNoise - noise) % noiseCycle);
      }
    }

    // Epilogue: non-vectorized remainder
    for (; point < planeSize; ++point) {
      for (int c = 0; c < kOutputChannels; ++c) {
        float v = (float)input[point * kInputChannels + c];
        output[c * planeSize + point] = v - mean[c] + *curNoise++;
        ++noiseUsed;
      }

      if (noiseUsed >= noiseCycle) {
        noiseUsed = 0;
        curNoise = noise + ((curNoise - noise) % noiseCycle);
      }
    }
  }
#endif //  defined(__ARM_NEON__) || defined(__ARM_NEON)

 private:
  Workspace* ws_;
};

namespace {

template <typename T>
static inline T clamped_cast(float f) {
  if (f >= std::numeric_limits<T>::max()) {
    return std::numeric_limits<T>::max();
  }
  if (f <= std::numeric_limits<T>::min()) {
    return std::numeric_limits<T>::min();
  }
  return static_cast<T>(f);
}

} // unnamed namespace

class BRGNCHWCToPackedInt8BGRAStylizerDeprocessOp
    : public Operator<CPUContext> {
 public:
  using Operator<CPUContext>::Operator;

  // Expect this many channels as input
  static constexpr int kInputChannels = 3;

  // Expect this many channels as output
  static constexpr int kOutputChannels = 4;

  bool RunOnDevice() override {
    const auto& X = Input(0);
    const auto& mean = Input(1);

    CAFFE_ENFORCE(X.dim() == 4);
    const int N = X.dim32(0), C = X.dim32(1), H = X.dim32(2), W = X.dim32(3);
    // Assume BGR or BGRA
    CAFFE_ENFORCE(mean.numel() == kInputChannels);
    CAFFE_ENFORCE(C == kInputChannels);
    // RGB
    auto* Y = Output(0, {N, H, W, kOutputChannels}, at::dtype<uint8_t>());

    runBatch(
        N,
        C,
        H,
        W,
        X.data<float>(),
        mean.data<float>(),
        Y->template mutable_data<uint8_t>());

    return true;
  }

  void runBatch(
      int N,
      int /*C*/,
      int H,
      int W,
      const float* input,
      const float* meanChannel,
      uint8_t* output) {
    int planeSize = H * W;

    for (int n = 0; n < N; ++n) {
      auto curInput = input + n * kInputChannels * planeSize;
      auto curOutput = output + n * kOutputChannels * planeSize;

#if defined(__ARM_NEON__) || defined(__ARM_NEON)
      runCPUNeon(H, W, curInput, meanChannel, curOutput);
#else
      runCPU(H, W, curInput, meanChannel, curOutput);
#endif //  defined(__ARM_NEON__) || defined(__ARM_NEON)
    }
  }

#if !defined(__ARM_NEON__) && !defined(__ARM_NEON)
  void runCPU(
      int H,
      int W,
      const float* input,
      const float* meanChannel,
      uint8_t* output) {
    int planeSize = H * W;

    for (int point = 0; point < planeSize; ++point) {
      for (int c = 0; c < kInputChannels; ++c) {
        uint8_t v = clamped_cast<uint8_t>(
            input[c * planeSize + point] + meanChannel[c]);
        output[point * kOutputChannels + c] = v;
      }

      // alpha
      output[point * kOutputChannels + (kOutputChannels - 1)] =
          std::numeric_limits<uint8_t>::max();
    }
  }
#endif // !defined(__ARM_NEON__) && !defined(__ARM_NEON)

#if defined(__ARM_NEON__) || defined(__ARM_NEON)
  void runCPUNeon(
      int H,
      int W,
      const float* input,
      const float* meanChannel,
      uint8_t* output) {
    // Vectorized load parameters:

    // We load in chunks of this size
    constexpr int kLoadUnit = sizeof(float32x4_t);
    constexpr int kLoadFloats = (sizeof(float32x4_t) / sizeof(float));

    // We store in chunks of this size
    constexpr int kStoreUnit = sizeof(uint8x8x4_t);

    // The vector portion loads this many f32 pixels at a time (8)
    constexpr int kLoadPixels = 2 * kLoadFloats;

    float mean[kInputChannels] = {
        meanChannel[0], meanChannel[1], meanChannel[2]};
    int planeSize = H * W;

    // Vectorized portion
    int point = 0;

    // If the slice is not aligned, then we have to use the
    // un-vectorized version
    bool isAligned = isPointerAligned(input, kLoadUnit) &&
        isPointerAligned(output, kStoreUnit) &&
        // Because we are reading from input at offsets of planeSize,
        // planeSize has to be an even multiple of kLoadUnit
        (planeSize % kLoadUnit == 0);

    // What portion the vectorized loop will handle
    int limit = isAligned ? (planeSize / kLoadPixels) * kLoadPixels : 0;

    for (; point < limit; point += kLoadPixels) {
      // Load 8 f32 pixels from each channel; loading 16 involves
      // register spills it seems
      float32x4_t inputc0_0 =
          vld1q_f32_aligned(input + 0 * planeSize + point + 0 * kLoadFloats);
      float32x4_t inputc0_1 =
          vld1q_f32_aligned(input + 0 * planeSize + point + 1 * kLoadFloats);

      float32x4_t inputc1_0 =
          vld1q_f32_aligned(input + 1 * planeSize + point + 0 * kLoadFloats);
      float32x4_t inputc1_1 =
          vld1q_f32_aligned(input + 1 * planeSize + point + 1 * kLoadFloats);

      float32x4_t inputc2_0 =
          vld1q_f32_aligned(input + 2 * planeSize + point + 0 * kLoadFloats);
      float32x4_t inputc2_1 =
          vld1q_f32_aligned(input + 2 * planeSize + point + 1 * kLoadFloats);

      addMeanAndClamp(inputc0_0, mean[0]);
      addMeanAndClamp(inputc0_1, mean[0]);
      uint8x8_t u8_c0 = convertNarrowAndPack(inputc0_0, inputc0_1);

      addMeanAndClamp(inputc1_0, mean[1]);
      addMeanAndClamp(inputc1_1, mean[1]);
      uint8x8_t u8_c1 = convertNarrowAndPack(inputc1_0, inputc1_1);

      addMeanAndClamp(inputc2_0, mean[2]);
      addMeanAndClamp(inputc2_1, mean[2]);
      uint8x8_t u8_c2 = convertNarrowAndPack(inputc2_0, inputc2_1);

      // This is the alpha channel
      uint8x8_t u8_c3 = vdup_n_u8(std::numeric_limits<uint8_t>::max());

      // We now have 8 bytes of each channel in a separate vector
      // Write BGRA interleaved to output
      uint8x8x4_t u8_out = {{ u8_c0, u8_c1, u8_c2, u8_c3 }};
      vst4_u8_aligned(output + kOutputChannels * point, u8_out);
    }

    // Epilogue: non-vectorized remainder
    for (; point < planeSize; ++point) {
      for (int c = 0; c < kInputChannels; ++c) {
        uint8_t v =
            clamped_cast<uint8_t>(input[c * planeSize + point] + mean[c]);
        output[point * kOutputChannels + c] = v;
      }

      // alpha
      output[point * kOutputChannels + (kOutputChannels - 1)] =
          std::numeric_limits<uint8_t>::max();
    }
  }
#endif // defined(__ARM_NEON__) || defined(__ARM_NEON)
};

namespace {

REGISTER_CPU_OPERATOR(
    PackedInt8BGRANHWCToNCHWCStylizerPreprocess,
    PackedInt8BGRANHWCToNCHWCStylizerPreprocessOp);
OPERATOR_SCHEMA(PackedInt8BGRANHWCToNCHWCStylizerPreprocess)
    .NumInputs(2)
    .NumOutputs(1);
REGISTER_CPU_OPERATOR(
    BRGNCHWCToPackedInt8BGRAStylizerDeprocess,
    BRGNCHWCToPackedInt8BGRAStylizerDeprocessOp);
OPERATOR_SCHEMA(BRGNCHWCToPackedInt8BGRAStylizerDeprocess)
    .NumInputs(2)
    .NumOutputs(1);

#ifdef CAFFE2_USE_MKLDNN
REGISTER_IDEEP_OPERATOR(
    BRGNCHWCToPackedInt8BGRAStylizerDeprocess,
    IDEEPFallbackOp<BRGNCHWCToPackedInt8BGRAStylizerDeprocessOp, SkipIndices<0>>);
REGISTER_IDEEP_OPERATOR(
    PackedInt8BGRANHWCToNCHWCStylizerPreprocess,
    IDEEPFallbackOp<PackedInt8BGRANHWCToNCHWCStylizerPreprocessOp>);
#endif
} // namespace
} // namespace caffe2
