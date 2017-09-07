#include "caffe2/core/init.h"
#include "caffe2/core/operator.h"
#include "caffe2/core/tensor.h"
#include "caffe2/utils/proto_utils.h"
#include "gtest/gtest.h"

#include <cmath>
#include <random>

#include <nnpack/transform.h>
#ifdef __ARM_NEON__
#include <nnpack/arm_neon.h>
#endif

namespace caffe2 {

namespace {

void AddNoiseInput(const vector<TIndex>& shape, const string& name, Workspace* ws) {
  DeviceOption option;
  CPUContext context(option);
  Blob* blob = ws->CreateBlob(name);
  auto* tensor = blob->GetMutable<TensorCPU>();
  tensor->Resize(shape);

  math::RandGaussian<float, CPUContext>(
      tensor->size(), 0.0f, 3.0f, tensor->mutable_data<float>(), &context);
  for (auto i = 0; i < tensor->size(); ++i) {
    tensor->mutable_data<float>()[i] =
        std::min(-5.0f, std::max(5.0f, tensor->mutable_data<float>()[i]));
  }
}

inline float relativeError(float a, float b) {
  return std::abs(a - b) / (0.5f * (std::abs(a) + std::abs(b)));
}

void compare(int N,
             int inputC,
             int H,
             int W,
             int outputC,
             int kernelH,
             int kernelW,
             int strideH,
             int strideW,
             int padT,
             int padL,
             int padB,
             int padR,
             int group,
             const std::string& algo,
             const std::string& convolutionTransformStrategy,
             float maxRelErr,
             float absErrForRelErrFailure) {
  LOG(INFO) << "running N " << N << " inputC " << inputC << " H " << H << " W " << W << " outputC "
            << outputC << " kernelH " << kernelH << " kernelW " << kernelW << " strideH " << strideH
            << " strideW " << strideW << " padT " << padT << " padL " << padL << " padB " << padB
            << " padR " << padR << " group " << group;

  Workspace ws;

  OperatorDef def1;
  def1.set_name("test");
  def1.set_type("Conv");
  def1.set_engine("NNPACK");
  def1.add_input("X");
  def1.add_input("W");
  def1.add_input("B");
  def1.add_output("Y1");

  def1.add_arg()->CopyFrom(MakeArgument("kernel_h", kernelH));
  def1.add_arg()->CopyFrom(MakeArgument("kernel_w", kernelW));
  if (!algo.empty()) {
    def1.add_arg()->CopyFrom(MakeArgument("algo", algo));
  }
  if (!convolutionTransformStrategy.empty()) {
    def1.add_arg()->CopyFrom(
        MakeArgument("convolution_transform_strategy", convolutionTransformStrategy));
  }
  def1.add_arg()->CopyFrom(MakeArgument("stride_h", strideH));
  def1.add_arg()->CopyFrom(MakeArgument("stride_w", strideW));
  def1.add_arg()->CopyFrom(MakeArgument("pad_t", padT));
  def1.add_arg()->CopyFrom(MakeArgument("pad_l", padL));
  def1.add_arg()->CopyFrom(MakeArgument("pad_b", padB));
  def1.add_arg()->CopyFrom(MakeArgument("pad_r", padR));
  def1.add_arg()->CopyFrom(MakeArgument("group", group));

  AddNoiseInput(vector<TIndex>{N, inputC, H, W}, "X", &ws);
  AddNoiseInput(vector<TIndex>{outputC, inputC / group , kernelH, kernelW}, "W", &ws);
  AddNoiseInput(vector<TIndex>{outputC}, "B", &ws);

  unique_ptr<OperatorBase> op1(CreateOperator(def1, &ws));
  EXPECT_NE(nullptr, op1.get());

  OperatorDef def2;
  def2.set_name("test");
  def2.set_type("Conv");
  def2.add_input("X");
  def2.add_input("W");
  def2.add_input("B");
  def2.add_output("Y2");

  def2.add_arg()->CopyFrom(MakeArgument("kernel_h", kernelH));
  def2.add_arg()->CopyFrom(MakeArgument("kernel_w", kernelW));
  def2.add_arg()->CopyFrom(MakeArgument("stride_h", strideH));
  def2.add_arg()->CopyFrom(MakeArgument("stride_w", strideW));
  def2.add_arg()->CopyFrom(MakeArgument("pad_t", padT));
  def2.add_arg()->CopyFrom(MakeArgument("pad_l", padL));
  def2.add_arg()->CopyFrom(MakeArgument("pad_b", padB));
  def2.add_arg()->CopyFrom(MakeArgument("pad_r", padR));
  def2.add_arg()->CopyFrom(MakeArgument("group", group));

  unique_ptr<OperatorBase> op2(CreateOperator(def2, &ws));
  EXPECT_NE(nullptr, op2.get());

  for (auto i = 0; i < 10; ++i) {
    EXPECT_TRUE(op1->Run());
  }
  Blob* Y1blob = ws.GetBlob("Y1");
  EXPECT_NE(nullptr, Y1blob);
  auto& Y1 = Y1blob->Get<TensorCPU>();

  for (auto i = 0; i < 10; ++i) {
    EXPECT_TRUE(op2->Run());
  }

  Blob* Y2blob = ws.GetBlob("Y2");
  EXPECT_NE(nullptr, Y2blob);
  auto& Y2 = Y2blob->Get<TensorCPU>();

  // Compare all output points
  for (int n = 0; n < Y1.dim32(0); ++n) {
    for (int c = 0; c < Y1.dim32(1); ++c) {
      for (int h = 0; h < Y1.dim32(2); ++h) {
        for (int w = 0; w < Y1.dim32(3); ++w) {
          int offset = n * Y1.dim32(1) * Y1.dim32(2) * Y1.dim32(3) + c * Y1.dim32(2) * Y1.dim32(3) +
                       h * Y1.dim32(3) + w;

          auto v1 = Y1.data<float>()[offset];
          auto v2 = Y2.data<float>()[offset];

          float relErr = relativeError(v1, v2);
          float absErr = std::abs(v1 - v2);

          // For small values / small difference, the relative error
          // can be huge but the absolute error will be small
          EXPECT_TRUE(relErr <= maxRelErr ||
                      (relErr > maxRelErr && absErr <= absErrForRelErrFailure))
              << v1 << " " << v2 << " (rel err " << relErr << ") "
              << "(" << n << " " << c << " " << h << " " << w << ") "
              << "running N " << N << " inputC " << inputC << " H " << H << " W " << W
              << " outputC " << outputC << " kernelH " << kernelH << " kernelW " << kernelW
              << " strideH " << strideH << " strideW " << strideW << " padT " << padT << " padL "
              << padL << " padB " << padB << " padR " << padR << " group " << group << " algo "
              << algo << " convolutionTransformStrategy " << convolutionTransformStrategy;
        }
      }
    }
  }
}

int randInt(int a, int b) {
  static std::random_device rd;
  static std::mt19937 gen(rd());

  return std::uniform_int_distribution<int>(a, b)(gen);
}

void runConv(int kernelH,
             int kernelW,
             int strideH,
             int strideW,
             int group = 1,
             std::string algo = "",
             int planesIn = randInt(1, 6),
             int planesOut = randInt(1, 6),
             int n = randInt(1, 2),
             std::string convolutionTransformStrategy = "COMPUTE") {
  int h = randInt(20, 100);
  int w = randInt(20, 100);
  // This pad restriction is imposed by NNPACK
  int padT = std::min(randInt(0, 3), kernelH - 1);
  int padB = std::min(randInt(0, 3), kernelH - 1);
  int padL = std::min(randInt(0, 3), kernelW - 1);
  int padR = std::min(randInt(0, 3), kernelW - 1);

  caffe2::compare(n,
                  planesIn,
                  h,
                  w,
                  planesOut,
                  kernelH,
                  kernelW,
                  strideH,
                  strideW,
                  padT,
                  padL,
                  padB,
                  padR,
                  group,
                  algo,
                  convolutionTransformStrategy,
                  0.05f,
                  0.1f);
}

} // unnamed namespace

constexpr size_t kIters = 20;

// TODO(#14383029) cblas_sgemm not yet implemented on limited mobile cases.
#if !defined(CAFFE2_FB_LIMITED_MOBILE_CAPABILITY)


TEST(MobileNNPACK, Conv_3x3s1) {
  for (int i = 0; i < kIters; ++i) {
    runConv(3, 3, 1, 1);
  }
}

TEST(MobileNNPACK, Conv_3x3s1_precompute) {
  for (int i = 0; i < kIters; ++i) {
    int group = randInt(1, 2);
    runConv(3,
            3,
            1,
            1,
            group,
            "WINOGRAD",
            group * randInt(1, 8),
            group * randInt(1, 8),
            1,
            "PRECOMPUTE");
  }
}

TEST(MobileNNPACK, Conv_3x3s1_FP16) {
  for (int i = 0; i < kIters; ++i) {
    runConv(3, 3, 1, 1, 1, "WINOGRAD_FP16");
  }
}

TEST(MobileNNPACK, Conv_3x3s1_FP16_precompute) {
  for (int i = 0; i < kIters; ++i) {
    int group = randInt(1, 2);
    runConv(3,
            3,
            1,
            1,
            group,
            "WINOGRAD_FP16",
            group * randInt(1, 8),
            group * randInt(1, 8),
            1,
            "PRECOMPUTE");
  }
}

TEST(MobileNNPACK, Conv_NxNs1) {
  for (int i = 0; i < kIters; ++i) {
    int kernel = randInt(2, 10);
    runConv(kernel, kernel, 1, 1);
  }
}

TEST(MobileNNPACK, Conv_1x1s1) {
  for (int i = 0; i < kIters; ++i) {
    auto group = randInt(1, 3);
    auto inChannels = randInt(1, 8) * group;
    auto outChannels = randInt(1, 8) * group;
    auto n = 1;
    runConv(1, 1, 1, 1, group, "DIRECT", inChannels, outChannels, n);
  }
}

TEST(MobileNNPACK, Conv_1x1s1_precompute) {
  for (int i = 0; i < kIters; ++i) {
    auto group = randInt(1, 3);
    auto inChannels = randInt(1, 8) * group;
    auto outChannels = randInt(1, 8) * group;
    auto n = 1;
    runConv(1, 1, 1, 1, group, "DIRECT", inChannels, outChannels, n, "PRECOMPUTE");
  }
}

TEST(MobileNNPACK, Conv_NxNs_grouped) {
  for (int i = 0; i < kIters; ++i) {
    int group = randInt(2, 3);
    int iC = randInt(1, 6) * group;
    int oC = randInt(1, 6) * group;
    int kernel = randInt(2, 10);
    int n = randInt(1, 2);
    runConv(kernel, kernel, 1, 1, group, "", iC, oC, n);
  }
}

TEST(MobileNNPACK, Conv_NxNs_grouped_precompute) {
  for (int i = 0; i < kIters; ++i) {
    int group = randInt(2, 3);
    int iC = randInt(1, 6) * group;
    int oC = randInt(1, 6) * group;
    int kernel = randInt(2, 10);
    int n = randInt(1, 2);
    runConv(kernel, kernel, 1, 1, group, "", iC, oC, n, "PRECOMPUTE");
  }
}

// FIXME: base Caffe2 conv appears broken?
// TEST(MobileNNPACK, Conv_NxNsW) {
//   for (int i = 0; i < 3; ++i) {
//     int kernel = randInt(3, 5);
//     int stride = randInt(1, kernel - 1);
//     runConv(kernel, kernel, stride, stride);
//   }
// }

// FIXME: base Caffe2 conv appears broken?
// TEST(MobileNNPACK, Conv_HxWsHxW) {
//   for (int i = 0; i < 3; ++i) {
//     int kernelH = randInt(2, 5);
//     int kernelW = randInt(2, 5);
//     int strideH = randInt(1, kernelH - 1);
//     int strideW = randInt(1, kernelW - 1);
//     runConv(kernelH, kernelW, strideH, strideW);
//   }
// }

#endif

#ifdef __ARM_NEON__

TEST(MobileNNPACK, PSIMDvsNEONReferenceTesting) {
  const auto kIters = 200;
  auto randomTensor = [](int seed) {
    auto t = caffe2::make_unique<caffe2::TensorCPU>();
    t->Resize(20000);
    DeviceOption dopt;
    dopt.set_random_seed(seed);
    CPUContext ctx(dopt);
    math::RandGaussian<float, CPUContext>(20000, 0.0, 1.0, t->mutable_data<float>(), &ctx);
    for (auto i = 0; i < t->size(); ++i) {
      t->mutable_data<float>()[i] = std::min(-5.0f, std::max(5.0f, t->mutable_data<float>()[i]));
    }
    return t;
  };

#define EXPECT_TENSOR_APPROX_EQUAL(lhs, rhs, tol)                                          \
  do {                                                                                     \
    EXPECT_TRUE((lhs).dims() == (rhs).dims());                                             \
    for (auto i = 0; i < (lhs).size(); ++i) {                                              \
      const auto lhsd = (lhs).data<float>()[i];                                            \
      const auto rhsd = (rhs).data<float>()[i];                                            \
      const auto absd = std::abs<float>(lhsd - rhsd);                                      \
      EXPECT_TRUE(absd < tol + tol * std::abs(lhsd)) << i << ", " << lhsd << ", " << rhsd; \
    }                                                                                      \
  } while (0);

  {
    auto X = randomTensor(5);
    auto WD = randomTensor(6);
    auto WD2 = randomTensor(6);
    auto WD16 = randomTensor(6);
    auto WD16_32 = randomTensor(6);

    for (auto i = 0; i < WD->size(); i += 4) {
      const float* srcOffset = WD->data<float>() + i;
      float32x4_t src = vld1q_f32(srcOffset);
      float16_t* dst = ((float16_t*)(WD16->mutable_data<float>())) + i;
      vst1q_f32_f16(dst, src);
    }

    for (auto i = 0; i < kIters; ++i) {
      auto dataStride = randInt(1, 32);
      auto transformStride = randInt(4, 16);
      transformStride *= 4;
      auto rowCount = randInt(1, 8);
      auto columnCount = randInt(1, 8);
      auto rowOffset = randInt(0, 4);
      auto columnOffset = randInt(0, 4);
      if (columnCount + columnOffset > 8 || rowCount + rowOffset > 8) {
        continue;
      }
      nnp_iwt8x8_3x3__psimd(X->data<float>(),
                            WD->mutable_data<float>(),
                            dataStride,
                            transformStride,
                            rowCount,
                            columnCount,
                            rowOffset,
                            columnOffset);
      nnp_iwt8x8_3x3__neon(X->data<float>(),
                           WD2->mutable_data<float>(),
                           dataStride,
                           transformStride,
                           rowCount,
                           columnCount,
                           rowOffset,
                           columnOffset);
      EXPECT_TENSOR_APPROX_EQUAL(*WD, *WD2, 1e-5);

      nnp_iwt8x8_3x3__neonfp16(X->data<float>(),
                               WD16->mutable_data<float>(),
                               dataStride,
                               transformStride / 2,
                               rowCount,
                               columnCount,
                               rowOffset,
                               columnOffset);
      EXPECT_EQ(WD16->size() % 4, 0);
      for (auto i = 0; i < WD16->size(); i += 4) {
        const float16_t* srcOffset = ((float16_t*)(WD16->mutable_data<float>())) + i;
        const float32x4_t src = vld1q_f32_f16(srcOffset);
        float* dst = WD16_32->mutable_data<float>() + i;
        vst1q_f32(dst, src);
      }
      EXPECT_TENSOR_APPROX_EQUAL(*WD, *WD16_32, 1e-2);
    }
  }

  {
    auto X = randomTensor(5);
    auto WD = randomTensor(6);
    auto WD2 = randomTensor(6);
    auto WD16 = randomTensor(6);
    auto WD16_32 = randomTensor(6);

    for (auto i = 0; i < WD->size(); i += 4) {
      const float* srcOffset = WD->data<float>() + i;
      float32x4_t src = vld1q_f32(srcOffset);
      float16_t* dst = ((float16_t*)(WD16->mutable_data<float>())) + i;
      vst1q_f32_f16(dst, src);
    }

    for (auto i = 0; i < kIters; ++i) {
      auto dataStride = randInt(1, 32);
      auto transformStride = randInt(4, 16);
      transformStride *= 4;
      auto rowCount = randInt(1, 8);
      auto columnCount = randInt(1, 8);
      auto rowOffset = randInt(0, 4);
      auto columnOffset = randInt(0, 4);
      nnp_kwt8x8_3x3__psimd(X->data<float>(),
                            WD->mutable_data<float>(),
                            dataStride,
                            transformStride,
                            rowCount,
                            columnCount,
                            rowOffset,
                            columnOffset);
      nnp_kwt8x8_3x3__neon(X->data<float>(),
                           WD2->mutable_data<float>(),
                           dataStride,
                           transformStride,
                           rowCount,
                           columnCount,
                           rowOffset,
                           columnOffset);
      EXPECT_TENSOR_APPROX_EQUAL(*WD, *WD2, 1e-5);

      nnp_kwt8x8_3x3__neonfp16(X->data<float>(),
                               WD16->mutable_data<float>(),
                               dataStride,
                               transformStride / 2,
                               rowCount,
                               columnCount,
                               rowOffset,
                               columnOffset);
      EXPECT_EQ(WD16->size() % 4, 0);
      for (auto i = 0; i < WD16->size(); i += 4) {
        const float16_t* srcOffset = ((float16_t*)(WD16->mutable_data<float>())) + i;
        const float32x4_t src = vld1q_f32_f16(srcOffset);
        float* dst = WD16_32->mutable_data<float>() + i;
        vst1q_f32(dst, src);
      }
      EXPECT_TENSOR_APPROX_EQUAL(*WD, *WD16_32, 1e-2);
    }
  }

  {
    auto X = randomTensor(5);
    auto XF16 = randomTensor(5);

    auto B = randomTensor(9);
    auto WD = randomTensor(6);
    auto WD2 = randomTensor(6);
    EXPECT_EQ(WD2->size() % 4, 0);

    for (auto i = 0; i < X->size(); i += 4) {
      const float* srcOffset = X->data<float>() + i;
      float32x4_t src = vld1q_f32(srcOffset);
      float16_t* dst = ((float16_t*)(XF16->mutable_data<float>())) + i;
      vst1q_f32_f16(dst, src);
    }

    for (auto i = 0; i < X->size(); i += 4) {
      const float* srcOffset = X->data<float>() + i;
      const float32x4_t src = vld1q_f32(srcOffset);
      const float16_t* dstOffset = ((const float16_t*)(XF16->mutable_data<float>())) + i;
      const float32x4_t dst = vld1q_f32_f16(dstOffset);
      for (auto i = 0; i < 4; ++i) {
        EXPECT_NEAR(src[i], dst[i], 1e-3);
      }
    }

    for (auto i = 0; i < kIters; ++i) {
      auto dataStride = randInt(1, 32);
      auto transformStride = randInt(4, 16);
      transformStride *= 4;
      auto rowCount = randInt(1, 6);
      auto columnCount = randInt(1, 6);
      nnp_owt8x8_3x3_with_bias__psimd(X->data<float>(),
                                      WD->mutable_data<float>(),
                                      B->data<float>(),
                                      transformStride,
                                      dataStride,
                                      rowCount,
                                      columnCount);
      nnp_owt8x8_3x3_with_bias__neon(X->data<float>(),
                                     WD2->mutable_data<float>(),
                                     B->data<float>(),
                                     transformStride,
                                     dataStride,
                                     rowCount,
                                     columnCount);
      EXPECT_TENSOR_APPROX_EQUAL(*WD, *WD2, 1e-5);

      nnp_owt8x8_3x3_with_bias__neonfp16(XF16->data<float>(),
                                         WD2->mutable_data<float>(),
                                         B->data<float>(),
                                         transformStride / 2,
                                         dataStride,
                                         rowCount,
                                         columnCount);
      EXPECT_TENSOR_APPROX_EQUAL(*WD, *WD2, 1e-2);

      nnp_owt8x8_3x3__psimd(X->data<float>(),
                            WD->mutable_data<float>(),
                            transformStride,
                            dataStride,
                            rowCount,
                            columnCount,
                            0,
                            0);
      nnp_owt8x8_3x3__neon(X->data<float>(),
                           WD2->mutable_data<float>(),
                           transformStride,
                           dataStride,
                           rowCount,
                           columnCount,
                           0,
                           0);
      EXPECT_TENSOR_APPROX_EQUAL(*WD, *WD2, 1e-5);

      nnp_owt8x8_3x3__neonfp16(XF16->data<float>(),
                               WD2->mutable_data<float>(),
                               transformStride / 2,
                               dataStride,
                               rowCount,
                               columnCount,
                               0,
                               0);
      EXPECT_TENSOR_APPROX_EQUAL(*WD, *WD2, 1e-3);
    }
  }
}

#endif

} // namespace caffe2
