#import <ATen/native/metal/MetalUtils.h>
#import <ATen/native/metal/mpscnn/MPSCNNContext.h>
#import <Accelerate/Accelerate.h>

namespace at {
namespace native {
namespace metal {

std::vector<fp16_t> Fp32ToFp16(const std::vector<float>& src) {
    unsigned long count = src.size();
    std::vector<fp16_t> output(count, 0);
    vImage_Buffer float32{(void*)src.data(), 1, count, count * sizeof(float)};
    vImage_Buffer float16{(void*)output.data(), 1, count, count * sizeof(fp16_t)};
    if (vImageConvert_PlanarFtoPlanar16F(&float32, &float16, 0) !=
        kvImageNoError) {
      TORCH_CHECK(false);
    }
  return output;
}

std::vector<float> Fp16ToFp32(const std::vector<fp16_t>& src) {
  unsigned long count = src.size();
  std::vector<float> output(count, 0);
  vImage_Buffer float16{(void*)src.data(), 1, count, count * sizeof(fp16_t)};
  vImage_Buffer float32{(void*)output.data(), 1, count, count * sizeof(float)};
  if (vImageConvert_Planar16FtoPlanarF(&float16, &float32, 0) !=
      kvImageNoError) {
    TORCH_CHECK(false);
  }
  return output;
}

std::vector<float> NCHWToNC4(
    const float* src,
    const std::vector<int64_t>& sizes) {
  int64_t N = sizes[0];
  int64_t C = sizes[1];
  int64_t H = sizes[2];
  int64_t W = sizes[3];
  int64_t src_image_count = C * H * W;
  int64_t src_count = N * src_image_count;
  int64_t slices = (C + 3) / 4;
  int64_t numComponents = C < 3 ? C : 4;
  int64_t dst_image_count = slices * numComponents * W * H;
  int64_t dst_count = N * dst_image_count;
  std::vector<float> output(dst_count, 0.0f);
  for (int n = 0; n < N; ++n) {
    int64_t src_image = n * src_image_count;
    int64_t dst_image = n * dst_image_count;
    for (int i = 0; i < slices; ++i) {
      int64_t slice = i * W * H * numComponents;
      for (int j = 0; j < W * H; ++j) {
        for (int k = 0; k < numComponents; ++k) {
          int ii = src_image + slice + k * W * H + j;
          int oi = dst_image + slice + j * numComponents + k;
          if (k < C && ii < src_count) {
            output[oi] = src[ii];
          }
        }
      }
    }
  }
  return output;
}

std::vector<float> NC4ToNCHW(
    const float* src,
    const std::vector<int64_t>& sizes) {
  int64_t N = sizes[0];
  int64_t C = sizes[1];
  int64_t H = sizes[2];
  int64_t W = sizes[3];
  int64_t slices = (C + 3) / 4;
  int64_t numComponents = C < 3 ? C : 4;
  int64_t src_image_count = slices * numComponents * W * H;
  int64_t dst_image_count = C * H * W;
  int64_t dst_count = N * dst_image_count;
  std::vector<float> output(dst_count, 0.0f);
  for (int n = 0; n < N; ++n) {
    int64_t src_image = n * src_image_count;
    int64_t dst_image = n * dst_image_count;
    for (int i = 0; i < slices; ++i) {
      int64_t slice = i * W * H * numComponents;
      for (int j = 0; j < numComponents; ++j) {
        for (int k = 0; k < W * H; ++k) {
          int ii = src_image + slice + k * numComponents + j;
          int oi = dst_image + slice + j * W * H + k;
          if (j < C && oi < dst_count) {
            output[oi] = src[ii];
          }
        }
      }
    }
  }
  return output;
}

}
}
}
