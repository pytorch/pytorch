#include "caffe2/core/context.h"
#include "caffe2/core/timer.h"
#include "caffe2/operators/conv_op.h"
#include "caffe2/operators/conv_pool_op_base.h"

#include "c10/macros/Macros.h"

#ifdef __ARM_NEON__
#include <arm_neon.h>
#endif

C10_DEFINE_bool(caffe2_profile_depthwise, false, "");

namespace caffe2 {

namespace {
struct DepthwiseArgs {
  // Input layer dimensions
  int batch{0};
  int in_rows{0};
  int in_cols{0};
  int stride{0};
  int pad_rows{0};
  int pad_cols{0};

  // Output layer dimensions
  int out_rows{0};
  int out_cols{0};
};

#ifdef __ARM_NEON__

static inline void winograd_f2k3_input_transform_inplace__neon(
    float32x4_t* d0,
    float32x4_t* d1,
    float32x4_t* d2,
    float32x4_t* d3) {
  //*d7 = wd7;
  float32x4_t wd0 = *d0 - *d2;
  float32x4_t wd1 = *d1 + *d2;
  float32x4_t wd2 = -*d1 + *d2;
  float32x4_t wd3 = *d1 - *d3;
  *d0 = wd0;
  *d1 = wd1;
  *d2 = wd2;
  *d3 = wd3;
}

static inline void winograd_f2k3_output_transform_inplace__neon(
    float32x4_t* m0,
    float32x4_t* m1,
    float32x4_t* m2,
    float32x4_t* m3) {
  *m0 = *m0 + *m1 + *m2;
  *m1 = *m1 - *m2 - *m3;
}

static inline float32x4_t
vmuladdq_f32(float32x4_t c, float32x4_t a, float32x4_t b) {
#if defined(__aarch64__)
  return vfmaq_f32(c, a, b);
#else
  return vmlaq_f32(c, a, b);
#endif
}

static inline float32x4_t
vmulsubq_f32(float32x4_t c, float32x4_t a, float32x4_t b) {
#if defined(__aarch64__)
  return vfmsq_f32(c, a, b);
#else
  return vmlsq_f32(c, a, b);
#endif
}

static inline void winograd_f2k3_kernel_transform__neon(
    const float32x4_t g0,
    const float32x4_t g1,
    const float32x4_t g2,
    float32x4_t* transform0,
    float32x4_t* transform1,
    float32x4_t* transform2,
    float32x4_t* transform3) {
  const float32x4_t const_half = vdupq_n_f32(0.5f);
  float32x4_t half_g0_plus_g2 = const_half * (g0 + g2);
  *transform0 = g0;
  *transform1 = vmuladdq_f32(half_g0_plus_g2, const_half, g1);
  *transform2 = vmulsubq_f32(half_g0_plus_g2, const_half, g1);
  *transform3 = g2;
}

static inline float32x4x4_t v4f_transpose4x4__neon(float32x4x4_t m) {
  float32x4x4_t ret;
  vst4q_f32((float*)(&ret), m);
  return ret;
}

void runDepthwise3x3Conv(
    const DepthwiseArgs& args,
    const float* input,
    const float* kernel,
    const float* bias,
    float* output) {
  const float32x4_t vbias = vsetq_lane_f32(*bias, vdupq_n_f32(0.0), 1);
  float32x4x4_t kernel_tile;
  {
    const float32x4_t g0 = vld1q_f32(kernel);
    const float32x4_t g1 = vld1q_f32(kernel + 3);
    // g2[3] is junk
    const float32x4_t g2 =
        vextq_f32(vld1q_f32(kernel + 5), vld1q_f32(kernel + 5), 1);
    float32x4x4_t w;
    winograd_f2k3_kernel_transform__neon(
        g0, g1, g2, &w.val[0], &w.val[1], &w.val[2], &w.val[3]);
    w = v4f_transpose4x4__neon(w);

    winograd_f2k3_kernel_transform__neon(
        w.val[0],
        w.val[1],
        w.val[2],
        &kernel_tile.val[0],
        &kernel_tile.val[1],
        &kernel_tile.val[2],
        &kernel_tile.val[3]);
  }

#define TILE                                                  \
  winograd_f2k3_input_transform_inplace__neon(                \
      &input_tile.val[0],                                     \
      &input_tile.val[1],                                     \
      &input_tile.val[2],                                     \
      &input_tile.val[3]);                                    \
  input_tile = v4f_transpose4x4__neon(input_tile);            \
  winograd_f2k3_input_transform_inplace__neon(                \
      &input_tile.val[0],                                     \
      &input_tile.val[1],                                     \
      &input_tile.val[2],                                     \
      &input_tile.val[3]);                                    \
                                                              \
  for (int row = 0; row < 4; ++row) {                         \
    input_tile.val[row] =                                     \
        vmulq_f32(input_tile.val[row], kernel_tile.val[row]); \
  }                                                           \
                                                              \
  input_tile.val[1] = input_tile.val[1] + vbias;              \
  winograd_f2k3_output_transform_inplace__neon(               \
      &input_tile.val[0],                                     \
      &input_tile.val[1],                                     \
      &input_tile.val[2],                                     \
      &input_tile.val[3]);                                    \
  input_tile = v4f_transpose4x4__neon(input_tile);            \
  winograd_f2k3_output_transform_inplace__neon(               \
      &input_tile.val[0],                                     \
      &input_tile.val[1],                                     \
      &input_tile.val[2],                                     \
      &input_tile.val[3])

  // Non-padded regime.

  // Iterate over non-padded output tiles.
  // TODO: avoid spilling W by breaking out the non-padded vs padded case.
  for (int oth = 0; oth < (args.out_rows + 1) / 2; ++oth) {
    for (int otw = 0; otw < (args.out_cols + 1) / 2; ++otw) {
      // load input tile for [oth, otw];
      int ih = oth * 2 - args.pad_rows;
      int iw = otw * 2 - args.pad_cols;
      // fast-path, all accesses in-bounds
      if (C10_LIKELY(
              ih >= 0 && iw >= 0 && ih + 3 < args.in_rows &&
                  iw + 3 < args.in_cols && 2 * oth + 1 < args.out_rows &&
                  2 * otw + 1 < args.out_cols
              )) {
        float32x4x4_t input_tile;
        for (int row = 0; row < 4; ++row) {
          input_tile.val[row] =
              vld1q_f32(input + (ih + row) * args.in_cols + iw);
        }

        TILE;

        for (size_t row = 0; row < 2; ++row) {
          vst1_f32(
              output + (oth * 2 + row) * args.out_cols + otw * 2,
              vget_low_f32(input_tile.val[row]));
        }
      } else {
        float block[4][4];
        for (int row = 0; row < 4; ++row) {
          for (int col = 0; col < 4; ++col) {
            if (ih + row >= 0 && iw + col >= 0 && ih + row < args.in_rows &&
                iw + col < args.in_cols) {
              block[row][col] = input[(ih + row) * args.in_cols + iw + col];
            } else {
              block[row][col] = 0.0;
            }
          }
        }

        float32x4x4_t input_tile;
        for (int row = 0; row < 4; ++row) {
          input_tile.val[row] = vld1q_f32(&block[row][0]);
        }

        TILE;

        float oblock[2][2];
        for (int row = 0; row < 2; ++row) {
          vst1_f32(&oblock[row][0], vget_low_f32(input_tile.val[row]));
        }
        for (int row = 0; row < 2; ++row) {
          for (int col = 0; col < 2; ++col) {
            if (2 * oth + row < args.out_rows &&
                2 * otw + col < args.out_cols) {
              output[(2 * oth + row) * args.out_cols + 2 * otw + col] =
                  oblock[row][col];
            }
          }
        }
      }
    }
  }
}

#else

#define PSIMD_INTRINSIC inline static __attribute__((__always_inline__))
typedef float psimd_f32 __attribute__((vector_size(16), aligned(1)));
typedef int psimd_s32 __attribute__((__vector_size__(16)));

PSIMD_INTRINSIC void psimd_store_f32(void* address, psimd_f32 value) {
  *((psimd_f32*)address) = value;
}

PSIMD_INTRINSIC psimd_f32 psimd_load_f32(const void* address) {
  return *((const psimd_f32*)address);
}

PSIMD_INTRINSIC psimd_f32 psimd_splat_f32(float c) {
  return (psimd_f32){c, c, c, c};
}

#if defined(__clang__)

PSIMD_INTRINSIC psimd_f32 psimd_interleave_lo_f32(psimd_f32 a, psimd_f32 b) {
  return __builtin_shufflevector(a, b, 0, 4 + 0, 1, 4 + 1);
}

PSIMD_INTRINSIC psimd_f32 psimd_interleave_hi_f32(psimd_f32 a, psimd_f32 b) {
  return __builtin_shufflevector(a, b, 2, 4 + 2, 3, 4 + 3);
}

PSIMD_INTRINSIC psimd_f32 psimd_concat_lo_f32(psimd_f32 a, psimd_f32 b) {
  return __builtin_shufflevector(a, b, 0, 1, 4 + 0, 4 + 1);
}

PSIMD_INTRINSIC psimd_f32 psimd_concat_hi_f32(psimd_f32 a, psimd_f32 b) {
  return __builtin_shufflevector(a, b, 2, 3, 4 + 2, 4 + 3);
}

#else

PSIMD_INTRINSIC psimd_f32 psimd_interleave_lo_f32(psimd_f32 a, psimd_f32 b) {
  return __builtin_shuffle(a, b, (psimd_s32){0, 4 + 0, 1, 4 + 1});
}

PSIMD_INTRINSIC psimd_f32 psimd_interleave_hi_f32(psimd_f32 a, psimd_f32 b) {
  return __builtin_shuffle(a, b, (psimd_s32){2, 4 + 2, 3, 4 + 3});
}
PSIMD_INTRINSIC psimd_f32 psimd_concat_lo_f32(psimd_f32 a, psimd_f32 b) {
  return __builtin_shuffle(a, b, (psimd_s32){0, 1, 4 + 0, 4 + 1});
}

PSIMD_INTRINSIC psimd_f32 psimd_concat_hi_f32(psimd_f32 a, psimd_f32 b) {
  return __builtin_shuffle(a, b, (psimd_s32){2, 3, 4 + 2, 4 + 3});
}

#endif

static inline void psimd_transpose4x4_f32(
    const psimd_f32 row0,
    const psimd_f32 row1,
    const psimd_f32 row2,
    const psimd_f32 row3,
    psimd_f32* col0,
    psimd_f32* col1,
    psimd_f32* col2,
    psimd_f32* col3) {
  const psimd_f32 row01lo = psimd_interleave_lo_f32(row0, row1);
  const psimd_f32 row01hi = psimd_interleave_hi_f32(row0, row1);
  const psimd_f32 row23lo = psimd_interleave_lo_f32(row2, row3);
  const psimd_f32 row23hi = psimd_interleave_hi_f32(row2, row3);
  *col0 = psimd_concat_lo_f32(row01lo, row23lo);
  *col1 = psimd_concat_hi_f32(row01lo, row23lo);
  *col2 = psimd_concat_lo_f32(row01hi, row23hi);
  *col3 = psimd_concat_hi_f32(row01hi, row23hi);
}

static inline void winograd_f2k3_input_transform(
    const psimd_f32 d0,
    const psimd_f32 d1,
    const psimd_f32 d2,
    const psimd_f32 d3,
    psimd_f32* transform0,
    psimd_f32* transform1,
    psimd_f32* transform2,
    psimd_f32* transform3) {
  *transform0 = d0 - d2;
  *transform1 = d1 + d2;
  *transform2 = -d1 + d2;
  *transform3 = d1 - d3;
}

static inline void winograd_f2k3_kernel_transform(
    const psimd_f32 g0,
    const psimd_f32 g1,
    const psimd_f32 g2,
    psimd_f32* transform0,
    psimd_f32* transform1,
    psimd_f32* transform2,
    psimd_f32* transform3) {
  const psimd_f32 const_half = psimd_splat_f32(0.5);
  const psimd_f32 half_g0_plus_g2 = const_half * (g0 + g2);
  *transform0 = g0;
  *transform1 = half_g0_plus_g2 + const_half * g1;
  *transform2 = half_g0_plus_g2 - const_half * g1;
  *transform3 = g2;
}

static inline void winograd_f2k3_output_transform(
    const psimd_f32 m0,
    const psimd_f32 m1,
    const psimd_f32 m2,
    const psimd_f32 m3,
    psimd_f32* output0,
    psimd_f32* output1) {
  *output0 = m0 + m1 + m2;
  *output1 = m1 - m2 - m3;
}

void runDepthwise3x3Conv(
    const DepthwiseArgs& args,
    const float* input,
    const float* kernel,
    const float* bias,
    float* output) {
  const psimd_f32 vbias = {0, *bias, 0, 0};
  const psimd_f32 g0 = psimd_load_f32(kernel);
  const psimd_f32 g1 = psimd_load_f32(kernel + 3);
  const psimd_f32 g5678 = psimd_load_f32(kernel + 5);
#ifdef __clang__
  const psimd_f32 g2 = __builtin_shufflevector(g5678, g5678, 1, 2, 3, -1);
#else
  const psimd_f32 g2 =
      __builtin_shuffle(g5678, g5678, (psimd_s32){1, 2, 3, -1});
#endif
  psimd_f32 w[4];
  winograd_f2k3_kernel_transform(g0, g1, g2, &w[0], &w[1], &w[2], &w[3]);
  psimd_transpose4x4_f32(w[0], w[1], w[2], w[3], &w[0], &w[1], &w[2], &w[3]);
  psimd_f32 wg[4];
  winograd_f2k3_kernel_transform(
      w[0], w[1], w[2], &wg[0], &wg[1], &wg[2], &wg[3]);

  // Iterate over non-padded output tiles.
  for (int oth = 0; oth < (args.out_rows + 1) / 2; ++oth) {
    for (int otw = 0; otw < (args.out_cols + 1) / 2; ++otw) {
      // load input tile for [oth, otw], i.e. [2 * oth - 1:2 * oth - 1 + 2, 2 *
      // otw - 1:2 * otw - 1 + 2]]
      int ih = oth * 2 - args.pad_rows;
      int iw = otw * 2 - args.pad_cols;
      // fast-path, all accesses in-bounds
      float block[4][4];
      for (int row = 0; row < 4; ++row) {
        for (int col = 0; col < 4; ++col) {
          if (ih + row >= 0 && iw + col >= 0 && ih + row < args.in_rows &&
              iw + col < args.in_cols) {
            block[row][col] = input[(ih + row) * args.in_cols + iw + col];
          } else {
            block[row][col] = 0.0;
          }
        }
      }
      psimd_f32 wd[4];
      winograd_f2k3_input_transform(
          psimd_load_f32(&block[0]),
          psimd_load_f32(&block[1]),
          psimd_load_f32(&block[2]),
          psimd_load_f32(&block[3]),
          &wd[0],
          &wd[1],
          &wd[2],
          &wd[3]);
      psimd_transpose4x4_f32(
          wd[0], wd[1], wd[2], wd[3], &wd[0], &wd[1], &wd[2], &wd[3]);
      winograd_f2k3_input_transform(
          wd[0], wd[1], wd[2], wd[3], &wd[0], &wd[1], &wd[2], &wd[3]);

      for (int row = 0; row < 4; ++row) {
        wd[row] = wg[row] * wd[row];
      }
      wd[1] += vbias;
      psimd_f32 s[4] = {{0}};
      winograd_f2k3_output_transform(wd[0], wd[1], wd[2], wd[3], &s[0], &s[1]);
      psimd_transpose4x4_f32(
          s[0], s[1], s[2], s[3], &s[0], &s[1], &s[2], &s[3]);

      psimd_f32 t0, t1;
      winograd_f2k3_output_transform(s[0], s[1], s[2], s[3], &t0, &t1);

      float oblock[2][4];
      psimd_store_f32(&oblock[0], t0);
      psimd_store_f32(&oblock[1], t1);
      for (int row = 0; row < 2; ++row) {
        for (int col = 0; col < 2; ++col) {
          if (2 * oth + row >= 0 && 2 * otw + col >= 0 &&
              2 * oth + row < args.out_rows && 2 * otw + col < args.out_cols) {
            output[(2 * oth + row) * args.out_cols + 2 * otw + col] =
                oblock[row][col];
          }
        }
      }
    }
  }
}

#endif

class Depthwise3x3ConvOp final : public ConvPoolOpBase<CPUContext> {
 public:
  USE_CONV_POOL_BASE_FUNCTIONS(CPUContext);
  Depthwise3x3ConvOp(const OperatorDef& operator_def, Workspace* ws)
      : ConvPoolOpBase<CPUContext>(operator_def, ws) {
    OPERATOR_NEEDS_FEATURE(
        this->order_ == StorageOrder::NCHW,
        "Depthwise3x3ConvOp only supports NCHW order");
    OPERATOR_NEEDS_FEATURE(this->group_ > 1);
    OPERATOR_NEEDS_FEATURE(this->kernel_w() == 3);
    OPERATOR_NEEDS_FEATURE(this->kernel_h() == 3);
    OPERATOR_NEEDS_FEATURE(this->stride_h() == 1);
    OPERATOR_NEEDS_FEATURE(this->stride_w() == 1);
  }

  bool RunOnDeviceWithOrderNCHW() override {
    const Tensor& X = Input(0);
    auto& filter = Input(1);
    const int N = X.dim32(0), C = X.dim32(1);
    CAFFE_ENFORCE_EQ(X.ndim(), filter.ndim());
    const int M = filter.dim32(0);

    CAFFE_ENFORCE_EQ(M, X.dim32(1));
    CAFFE_ENFORCE_EQ(C, X.dim32(1));
    CAFFE_ENFORCE_EQ(C, this->group_);
    CAFFE_ENFORCE_EQ(M, this->group_);

    auto sizes = ConvPoolOpBase<CPUContext>::GetOutputSize(X, filter.dim32(0));
    Tensor* Y = Output(0, sizes, at::dtype<float>());

    DepthwiseArgs args;
    args.batch = X.dim32(0);
    args.in_rows = X.dim32(2);
    args.in_cols = X.dim32(3);
    args.stride = this->stride_w();
    args.pad_rows = this->pad_t();
    args.pad_cols = this->pad_l();
    args.out_rows = Y->dim32(2);
    args.out_cols = Y->dim32(3);

    const auto G = this->group_;
    const auto IS = X.dim32(2) * X.dim32(3);
    const auto OS = Y->dim32(2) * Y->dim32(3);

    if (InputSize() != 3 && bias_.size() != M) {
      // no bias.
      bias_.Resize(M);
      math::Set<float, CPUContext>(
          M, 0.0, bias_.mutable_data<float>(), &context_);
    }
    const auto* bias =
        InputSize() == 3 ? Input(2).data<float>() : bias_.data<float>();

    auto f = [&](int n, int g) {
      runDepthwise3x3Conv(
          args,
          X.data<float>() + g * IS + n * G * IS,
          filter.data<float>() + g * 3 * 3,
          bias + g,
          Y->mutable_data<float>() + g * OS + n * G * OS);
    };

    Timer t;

#ifdef C10_MOBILE
    ws_->GetThreadPool()->run(
        [&](int, int n_g) {
          const int g = n_g / N;
          const int n = n_g % N;
          f(n, g);
        },
        N * G);
#else
    for (auto n = 0; n < N; ++n) {
      for (auto g = 0; g < G; ++g) {
        f(n, g);
      }
    }
#endif
    if (FLAGS_caffe2_profile_depthwise) {
      char buffer[1024];
      const double gmacs = double(
                               Y->dim32(2) * Y->dim32(3) * Y->dim32(1) *
                               kernel_w() * kernel_h()) /
          1.0E9;
      const double gflops = 2 * gmacs / t.Seconds();
      auto ret = snprintf(
          buffer,
          sizeof(buffer),
          "H: %3zu, W: %3zu, iC: %3zu, oC: %3zu, K: %1zu, S: %1zu, P: %1zu, GMACs: "
          "%4.2f, totalT: %6.3f, inputT: %6.3f, "
          "kernelT: %6.3f, blockT: %6.3f, outputT: %6.3f, GFLOPS: %6.3f",
          size_t(X.dim(2)),
          size_t(X.dim(3)),
          size_t(X.dim(1)),
          size_t(Y->dim(1)),
          size_t(kernel_w()),
          size_t(stride_w()),
          size_t(pad_t()),
          gmacs,
          t.Seconds() * 1E3,
          0 * 1E3,
          0 * 1E3,
          0 * 1E3,
          0 * 1E3,
          gflops);
      CAFFE_ENFORCE(ret > 0);
      LOG(INFO) << buffer;
    }
    return true;
  }

 private:
  Tensor bias_{CPU};
};

REGISTER_CPU_OPERATOR_WITH_ENGINE(Conv, DEPTHWISE_3x3, Depthwise3x3ConvOp);
} // namespace
} // namespace caffe2
