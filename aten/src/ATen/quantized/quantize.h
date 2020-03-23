#include <ATen/ATen.h>

// Quantize a float value into a *int* value given scale and zero_point
namespace at {

template <typename T>
CAFFE2_API T quantize_val(double scale, int64_t zero_point, float value);
template <typename T, int precision=8>
void quantize_vec(double scale, int64_t zero_point, const float *src, T *dst, size_t count=8);
template <typename T>
CAFFE2_API float dequantize_val(double scale, int64_t zero_point, T value);
template <typename T>
CAFFE2_API float dequantize_vec(double scale, int64_t zero_point, const T* src, float* dst, size_t count=8);
template <typename SRC_T, typename DST_T>
CAFFE2_API DST_T requantize_val(double, int64_t, double, int64_t, SRC_T src);

// Given a multiplier and a zero_point, requantize int32_t computed values back
// to quantized values. See comment above
// make_per_tensor_affine_quantizer function for the usage of int64_t
template <typename DST_T>
CAFFE2_API DST_T
requantize_from_int(double multiplier, int64_t zero_point, int64_t src);

}
