#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/native/cpu/DepthwiseConvKernel.h>
#include <ATen/core/Tensor.h>
#include <ATen/Parallel.h>
#include <c10/util/irange.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#else
#include <ATen/ops/empty.h>
#include <ATen/ops/zeros.h>
#endif

#ifdef __ARM_NEON__
#include <arm_neon.h>
#endif

namespace at::native {
namespace {

struct Arguments final {
  // Input layer dimensions
  int64_t batch;
  int64_t in_rows;
  int64_t in_cols;
  int64_t stride;
  int64_t pad_rows;
  int64_t pad_cols;

  // Output layer dimensions
  int64_t out_rows;
  int64_t out_cols;
  int64_t out_channels;
};

inline std::vector<int64_t> calculate_conv_output_size(
    const IntArrayRef input_size,
    const IntArrayRef weight_size,
    const IntArrayRef stride,
    const IntArrayRef padding) {
  const auto calc_output_dimension = [](
    const int64_t input, const int64_t kernel, const int64_t stride, const int64_t padding) {
    return 1 + (input - kernel + 2 * padding) / stride;
  };

  return std::vector<int64_t> {
    input_size[0],
    weight_size[0],
    calc_output_dimension(input_size[2], weight_size[2], stride[0], padding[0]),
    calc_output_dimension(input_size[3], weight_size[3], stride[1], padding[1]),
  };
}

#ifdef __ARM_NEON__

inline void winograd_f2k3_input_transform_inplace__neon(
    float32x4_t* const d0,
    float32x4_t* const d1,
    float32x4_t* const d2,
    float32x4_t* const d3) {
  const float32x4_t wd0 = *d0 - *d2;
  const float32x4_t wd1 = *d1 + *d2;
  const float32x4_t wd2 = -*d1 + *d2;
  const float32x4_t wd3 = *d1 - *d3;
  *d0 = wd0;
  *d1 = wd1;
  *d2 = wd2;
  *d3 = wd3;
}

inline void winograd_f2k3_output_transform_inplace__neon(
    float32x4_t* const m0,
    float32x4_t* const m1,
    const float32x4_t* const m2,
    const float32x4_t* const m3) {
  *m0 = *m0 + *m1 + *m2;
  *m1 = *m1 - *m2 - *m3;
}

inline float32x4_t
vmuladdq_f32(const float32x4_t c, const float32x4_t a, const float32x4_t b) {
#if defined(__aarch64__)
  return vfmaq_f32(c, a, b);
#else
  return vmlaq_f32(c, a, b);
#endif
}

inline float32x4_t
vmulsubq_f32(const float32x4_t c, const float32x4_t a, const float32x4_t b) {
#if defined(__aarch64__)
  return vfmsq_f32(c, a, b);
#else
  return vmlsq_f32(c, a, b);
#endif
}

inline void winograd_f2k3_kernel_transform__neon(
    const float32x4_t g0,
    const float32x4_t g1,
    const float32x4_t g2,
    float32x4_t* const transform0,
    float32x4_t* const transform1,
    float32x4_t* const transform2,
    float32x4_t* const transform3) {
  const float32x4_t const_half = vdupq_n_f32(0.5f);
  float32x4_t half_g0_plus_g2 = const_half * (g0 + g2);
  *transform0 = g0;
  *transform1 = vmuladdq_f32(half_g0_plus_g2, const_half, g1);
  *transform2 = vmulsubq_f32(half_g0_plus_g2, const_half, g1);
  *transform3 = g2;
}

inline float32x4x4_t v4f_transpose4x4__neon(const float32x4x4_t m) {
  float32x4x4_t ret;
  vst4q_f32((float*)(&ret), m);
  return ret;
}

void convolution_depthwise3x3_winograd_impl(
    const Arguments& args,
    const float* const input,
    const float* const kernel,
    const float* const bias,
    float* const output) {
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
  for (const auto row : c10::irange(4)) {                         \
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
  for (int64_t oth = 0; oth < (args.out_rows + 1) / 2; ++oth) {
    for (int64_t otw = 0; otw < (args.out_cols + 1) / 2; ++otw) {
      // load input tile for [oth, otw];
      int64_t ih = oth * 2 - args.pad_rows;
      int64_t iw = otw * 2 - args.pad_cols;
      // fast-path, all accesses in-bounds
      if (C10_LIKELY(
              ih >= 0 && iw >= 0 && ih + 3 < args.in_rows &&
                  iw + 3 < args.in_cols && 2 * oth + 1 < args.out_rows &&
                  2 * otw + 1 < args.out_cols
              )) {
        float32x4x4_t input_tile;
        for (const auto row : c10::irange(4)) {
          input_tile.val[row] =
              vld1q_f32(input + (ih + row) * args.in_cols + iw);
        }

        TILE;

        for (const auto row : c10::irange(2)) {
          vst1_f32(
              output + (oth * 2 + row) * args.out_cols + otw * 2,
              vget_low_f32(input_tile.val[row]));
        }
      } else {
        float block[4][4];
        for (const auto row : c10::irange(4)) {
          for (const auto col : c10::irange(4)) {
            if (ih + row >= 0 && iw + col >= 0 && ih + row < args.in_rows &&
                iw + col < args.in_cols) {
              block[row][col] = input[(ih + row) * args.in_cols + iw + col];
            } else {
              block[row][col] = 0.0;
            }
          }
        }

        float32x4x4_t input_tile;
        for (const auto row : c10::irange(4)) {
          input_tile.val[row] = vld1q_f32(&block[row][0]);
        }

        TILE;

        float oblock[2][2];
        for (const auto row : c10::irange(2)) {
          vst1_f32(&oblock[row][0], vget_low_f32(input_tile.val[row]));
        }
        for (const auto row : c10::irange(2)) {
          for (const auto col : c10::irange(2)) {
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

void convolution_depthwise3x3_winograd_impl(
    const Arguments&,
    const float* const,
    const float* const,
    const float* const,
    float* const) {
}

#endif /* __ARM_NEON__ */

Tensor _convolution_depthwise3x3_winograd(
    const Tensor & input,
    const Tensor & kernel,
    const Tensor & bias_potentially_undefined,
    const IntArrayRef stride,
    const IntArrayRef padding,
    const int64_t groups)
{
  const IntArrayRef input_sizes = input.sizes();
  const IntArrayRef kernel_sizes = kernel.sizes();

  Tensor output = at::empty(
    calculate_conv_output_size(input_sizes, kernel_sizes, stride, padding),
    input.options());

  const IntArrayRef output_sizes = output.sizes();

  const Arguments args {
      input_sizes[0],     // Input N
      input_sizes[2],     // Input H
      input_sizes[3],     // Input W
      stride[0],          // Stride
      padding[0],         // Padding Rows
      padding[1],         // Padding Columns
      output_sizes[2],    // Output H
      output_sizes[3],    // Output W
      output_sizes[1],    // Output C
  };

  const int64_t input_hxw = args.in_rows * args.in_cols;
  const int64_t output_hxw = args.out_rows * args.out_cols;

  const Tensor bias = bias_potentially_undefined.defined() ?
                      bias_potentially_undefined :
                      at::zeros({kernel_sizes[0]}, input.options());

  auto input_data = input.const_data_ptr<float>();
  auto kernel_data = kernel.const_data_ptr<float>();
  auto bias_data = bias.const_data_ptr<float>();
  auto output_data = output.data_ptr<float>();

  at::parallel_for(0, args.batch * args.out_channels, 0, [&](int64_t start, int64_t end) {
    for (const auto k : c10::irange(start, end)) {
      const int64_t g = k % args.out_channels;
      const int64_t i = k / (args.out_channels / groups);
      convolution_depthwise3x3_winograd_impl(
          args,
          input_data + i * input_hxw,
          kernel_data + g * 3 * 3,
          bias_data + g,
          output_data + k * output_hxw);
    }
  });

  return output;
}

}  // namespace

ALSO_REGISTER_AVX512_DISPATCH(convolution_depthwise3x3_winograd_stub, &_convolution_depthwise3x3_winograd);

}  // namespace at::native
