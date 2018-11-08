
#include "DataTransfer.h"
#include "GLLogging.h"

#include "caffe2/core/common.h"

inline uint16x4x4_t vld4_u16_aligned16(const uint16_t* address) {
  return vld4_u16(static_cast<const uint16_t*>(__builtin_assume_aligned(address, 16)));
}

inline uint16x4_t vld1_u16_aligned8(const uint16_t* address) {
  return vld1_u16(static_cast<const uint16_t*>(__builtin_assume_aligned(address, 8)));
}

inline void vst4_u16_aligned16(uint16_t* address, uint16x4x4_t data) {
  vst4_u16(static_cast<uint16_t*>(__builtin_assume_aligned(address, 16)), data);
}

inline void vst1_u16_aligned8(uint16_t* address, uint16x4_t data) {
  vst1_u16(static_cast<uint16_t*>(__builtin_assume_aligned(address, 8)), data);
}

template <int input_channels>
static void interleaveSlice(
    void* output, const float* input, size_t width, size_t height, size_t row_stride) {
  const float* input_r = input;
  const float* input_g = input_r + height * width;
  const float* input_b = input_g + height * width;
  const float* input_a = input_b + height * width;
  uint16_t* output_f16 = static_cast<uint16_t*>(output);
  if (width >= 4) {
    for (size_t y = 0; y < height; y++) {
      size_t nx = width;
      while (nx >= 4) {
        const uint16x4_t r = uint16x4_t(vcvt_f16_f32(vld1q_f32(input_r)));
        input_r += 4;
        uint16x4_t g, b, a;
        g = b = a = vdup_n_u16(0);
        if (input_channels >= 2) {
          g = uint16x4_t(vcvt_f16_f32(vld1q_f32(input_g)));
          input_g += 4;
          if (input_channels >= 3) {
            b = uint16x4_t(vcvt_f16_f32(vld1q_f32(input_b)));
            input_b += 4;
            if (input_channels >= 4) {
              a = uint16x4_t(vcvt_f16_f32(vld1q_f32(input_a)));
              input_a += 4;
            }
          }
        }

        const uint16x4x4_t rgba = (uint16x4x4_t){{r, g, b, a}};
        vst4_u16_aligned16(output_f16, rgba);
        output_f16 += 4 * 4;

        nx -= 4;
      }
      if (nx != 0) {
        output_f16 -= (4 - nx) * 4;
        input_r -= 4 - nx;
        if (input_channels >= 2) {
          input_g -= 4 - nx;
          if (input_channels >= 3) {
            input_b -= 4 - nx;
            if (input_channels >= 4) {
              input_a -= 4 - nx;
            }
          }
        }

        const uint16x4_t r = uint16x4_t(vcvt_f16_f32(vld1q_f32(input_r)));
        input_r += 4;
        uint16x4_t g, b, a;
        g = b = a = vdup_n_u16(0);
        if (input_channels >= 2) {
          g = uint16x4_t(vcvt_f16_f32(vld1q_f32(input_g)));
          input_g += 4;
          if (input_channels >= 3) {
            b = uint16x4_t(vcvt_f16_f32(vld1q_f32(input_b)));
            input_b += 4;
            if (input_channels >= 4) {
              a = uint16x4_t(vcvt_f16_f32(vld1q_f32(input_a)));
              input_a += 4;
            }
          }
        }

        const uint16x4x4_t rgba = (uint16x4x4_t){{r, g, b, a}};
        vst4_u16_aligned16(output_f16, rgba);
        output_f16 += 4 * 4;
      }
      output_f16 += (row_stride - width) * 4;
    }
  } else {
    for (size_t y = 0; y < height; y++) {
      for (size_t x = 0; x < width; x++) {
        float32x4_t rgba = vld1q_dup_f32(input_r++);
        if (input_channels >= 2) {
          rgba = vld1q_lane_f32(input_g++, rgba, 1);
          if (input_channels >= 3) {
            rgba = vld1q_lane_f32(input_b++, rgba, 2);
            if (input_channels >= 4) {
              rgba = vld1q_lane_f32(input_a++, rgba, 3);
            }
          }
        }
        vst1_u16_aligned8(output_f16, uint16x4_t(vcvt_f16_f32(rgba)));
        output_f16 += 4;
      }
      output_f16 += (row_stride - width) * 4;
    }
  }
}

void interleaveSlice(void* output,
                     const float* input,
                     size_t width,
                     size_t height,
                     size_t row_stride,
                     uint16_t input_channels) {
  switch (input_channels) {
  case 1:
    interleaveSlice<1>(output, input, width, height, row_stride);
    break;
  case 2:
    interleaveSlice<2>(output, input, width, height, row_stride);
    break;
  case 3:
    interleaveSlice<3>(output, input, width, height, row_stride);
    break;
  case 4:
    interleaveSlice<4>(output, input, width, height, row_stride);
    break;
  }
}

template <int output_channels>
static void deInterleaveSlice(
    float* output, const void* input, size_t width, size_t height, size_t row_stride) {
  float* output_r = output;
  float* output_g = output_r + height * width;
  float* output_b = output_g + height * width;
  float* output_a = output_b + height * width;
  const uint16_t* input_f16 = static_cast<const uint16_t*>(input);
  if (width >= 4) {
    for (size_t y = 0; y < height; y++) {
      size_t nx = width;
      while (nx >= 4) {
        const uint16x4x4_t rgba = vld4_u16_aligned16(input_f16);
        input_f16 += 4 * 4;
        const float32x4_t r = vcvt_f32_f16(float16x4_t(rgba.val[0]));
        vst1q_f32(output_r, r);
        output_r += 4;
        if (output_channels >= 2) {
          const float32x4_t g = vcvt_f32_f16(float16x4_t(rgba.val[1]));
          vst1q_f32(output_g, g);
          output_g += 4;
          if (output_channels >= 3) {
            const float32x4_t b = vcvt_f32_f16(float16x4_t(rgba.val[2]));
            vst1q_f32(output_b, b);
            output_b += 4;
            if (output_channels >= 4) {
              const float32x4_t a = vcvt_f32_f16(float16x4_t(rgba.val[3]));
              vst1q_f32(output_a, a);
              output_a += 4;
            }
          }
        }

        nx -= 4;
      }
      if (nx != 0) {
        input_f16 -= (4 - nx) * 4;
        output_r -= 4 - nx;
        if (output_channels >= 2) {
          output_g -= 4 - nx;
          if (output_channels >= 3) {
            output_b -= 4 - nx;
            if (output_channels >= 4) {
              output_a -= 4 - nx;
            }
          }
        }

        const uint16x4x4_t rgba = vld4_u16_aligned16(input_f16);
        input_f16 += 4 * 4;
        const float32x4_t r = vcvt_f32_f16(float16x4_t(rgba.val[0]));
        vst1q_f32(output_r, r);
        output_r += 4;
        if (output_channels >= 2) {
          const float32x4_t g = vcvt_f32_f16(float16x4_t(rgba.val[1]));
          vst1q_f32(output_g, g);
          output_g += 4;
          if (output_channels >= 3) {
            const float32x4_t b = vcvt_f32_f16(float16x4_t(rgba.val[2]));
            vst1q_f32(output_b, b);
            output_b += 4;
            if (output_channels >= 4) {
              const float32x4_t a = vcvt_f32_f16(float16x4_t(rgba.val[3]));
              vst1q_f32(output_a, a);
              output_a += 4;
            }
          }
        }
      }
      input_f16 += (row_stride - width) * 4;
    }
  } else {
    for (size_t y = 0; y < height; y++) {
      for (size_t x = 0; x < width; x++) {
        const float32x4_t rgba = vcvt_f32_f16(float16x4_t(vld1_u16_aligned8(input_f16)));
        input_f16 += 4;
        vst1q_lane_f32(output_r++, rgba, 0);
        if (output_channels >= 2) {
          vst1q_lane_f32(output_g++, rgba, 1);
          if (output_channels >= 3) {
            vst1q_lane_f32(output_b++, rgba, 2);
            if (output_channels >= 4) {
              vst1q_lane_f32(output_a++, rgba, 3);
            }
          }
        }
      }
      input_f16 += (row_stride - width) * 4;
    }
  }
}

void deInterleaveSlice(float* output,
                       const void* input,
                       size_t width,
                       size_t height,
                       size_t row_stride,
                       uint32_t output_channels) {
  switch (output_channels) {
  case 1:
    deInterleaveSlice<1>(output, input, width, height, row_stride);
    break;
  case 2:
    deInterleaveSlice<2>(output, input, width, height, row_stride);
    break;
  case 3:
    deInterleaveSlice<3>(output, input, width, height, row_stride);
    break;
  case 4:
    deInterleaveSlice<4>(output, input, width, height, row_stride);
    break;
  }
}
