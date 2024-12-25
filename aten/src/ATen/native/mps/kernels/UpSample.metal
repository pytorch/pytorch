#include <metal_stdlib>
using namespace metal;

// Atomic operations helper
template <typename T>
struct AtomicType {};
template <typename T>
using AtomicType_t = typename AtomicType<T>::type;

template <>
struct AtomicType<float> {
  using type = atomic<float>;
  static inline void atomic_add(device type* data, long offset, float value) {
    atomic_fetch_add_explicit(data + offset, value, memory_order_relaxed);
  }
};

// As of Metal3.2 atomic operations are not supported on half-precision floats,
// so they must be simulated Using atomic compare and exchange over 32-bit
// atomic type
template <typename T>
static inline void atomic_add_helper(
    device atomic<int>* data,
    long offset,
    float value) {
  auto ptr = data + (offset >> 1);
  auto old = atomic_load_explicit(ptr, memory_order_relaxed);
  union {
    int i;
    T t[2];
  } val;
  do {
    val.i = old;
    val.t[offset & 1] += static_cast<T>(value);
  } while (!atomic_compare_exchange_weak_explicit(
      ptr, &old, val.i, memory_order_relaxed, memory_order_relaxed));
}

template <>
struct AtomicType<half> {
  using type = atomic<int>;
  static inline void atomic_add(device type* data, long offset, float value) {
    atomic_add_helper<half>(data, offset, value);
  }
};

#if __METAL_VERSION__ >= 310
template <>
struct AtomicType<bfloat> {
  using type = atomic<int>;
  static inline void atomic_add(device type* data, long offset, float value) {
    atomic_add_helper<bfloat>(data, offset, value);
  }
};
#endif

// Based on
// https://en.wikipedia.org/wiki/Bicubic_interpolation#Bicubic_convolution_algorithm
template <typename accscalar_t>
accscalar_t cubic_convolution1(accscalar_t x, accscalar_t A) {
  return ((A + 2) * x - (A + 3)) * x * x + 1;
}

template <typename accscalar_t>
accscalar_t cubic_convolution2(accscalar_t x, accscalar_t A) {
  return ((A * x - 5 * A) * x + 8 * A) * x - 4 * A;
}

template <typename accscalar_t>
void get_cubic_upsampling_coefficients(accscalar_t coeffs[4], accscalar_t t) {
  accscalar_t A = -0.75;

  accscalar_t x1 = t;
  coeffs[0] = cubic_convolution2<accscalar_t>(x1 + 1.0, A);
  coeffs[1] = cubic_convolution1<accscalar_t>(x1, A);

  // opposite coefficients
  accscalar_t x2 = 1.0 - t;
  coeffs[2] = cubic_convolution1<accscalar_t>(x2, A);
  coeffs[3] = cubic_convolution2<accscalar_t>(x2 + 1.0, A);
}

template <typename scalar_t, typename accscalar_t>
accscalar_t cubic_interp1d(
    scalar_t x0,
    scalar_t x1,
    scalar_t x2,
    scalar_t x3,
    accscalar_t t) {
  accscalar_t coeffs[4];
  get_cubic_upsampling_coefficients<accscalar_t>(coeffs, t);

  return x0 * coeffs[0] + x1 * coeffs[1] + x2 * coeffs[2] + x3 * coeffs[3];
}

template <typename accscalar_t>
accscalar_t area_pixel_compute_source_index(
    accscalar_t scale,
    int dst_index,
    bool align_corners,
    bool cubic) {
  if (align_corners) {
    return scale * dst_index;
  } else {
    accscalar_t src_idx = scale * (dst_index + static_cast<accscalar_t>(0.5)) -
        static_cast<accscalar_t>(0.5);
    // See Note[Follow Opencv resize logic]
    return (!cubic && src_idx < static_cast<accscalar_t>(0))
        ? static_cast<accscalar_t>(0)
        : src_idx;
  }
}

template <typename scalar_t>
scalar_t upsample_get_value_bounded(
    constant scalar_t* data,
    long2 dim,
    ulong4 strides,
    long n,
    long c,
    long y,
    long x) {
  int access_y = max(min(y, dim.y - 1), 0L);
  int access_x = max(min(x, dim.x - 1), 0L);
  return data
      [n * strides.w + c * strides.z + access_y * strides.y +
       access_x * strides.x];
}

template <typename scalar_t>
void upsample_increment_value_bounded(
    device AtomicType_t<scalar_t>* data,
    long2 dim,
    ulong4 strides,
    long n,
    long c,
    long y,
    long x,
    float value) {
  int access_y = max(min(y, dim.y - 1), 0L);
  int access_x = max(min(x, dim.x - 1), 0L);
  AtomicType<scalar_t>::atomic_add(
      data,
      n * strides.w + c * strides.z + access_y * strides.y +
          access_x * strides.x,
      value);
}

template <typename T>
kernel void upsample_bicubic2d(
    constant T* inputData [[buffer(0)]],
    device T* outputData [[buffer(1)]],
    constant ulong4& input_strides [[buffer(2)]],
    constant ulong4& output_strides [[buffer(3)]],
    constant long4& input_sizes [[buffer(4)]],
    constant long4& output_sizes [[buffer(5)]],
    constant float2& scales [[buffer(6)]],
    constant bool& align_corners [[buffer(7)]],
    uint thread_index [[thread_position_in_grid]]) {
  auto output_x = thread_index % output_sizes.x;
  auto output_y = thread_index / output_sizes.x;
  auto real_x = area_pixel_compute_source_index(
      scales.x, output_x, align_corners, /*cubic=*/true);
  int in_x = floor(real_x);
  auto t_x = real_x - in_x;

  auto real_y = area_pixel_compute_source_index(
      scales.y, output_y, align_corners, /*cubic=*/true);
  int in_y = floor(real_y);
  auto t_y = real_y - in_y;
  for (int n = 0; n < output_sizes.w; n++) {
    for (int c = 0; c < output_sizes.z; c++) {
      float coefficients[4];
      for (int k = 0; k < 4; k++) {
        coefficients[k] = cubic_interp1d(
            upsample_get_value_bounded<T>(
                inputData,
                input_sizes.xy,
                input_strides,
                n,
                c,
                in_y - 1 + k,
                in_x - 1),
            upsample_get_value_bounded<T>(
                inputData,
                input_sizes.xy,
                input_strides,
                n,
                c,
                in_y - 1 + k,
                in_x + 0),
            upsample_get_value_bounded<T>(
                inputData,
                input_sizes.xy,
                input_strides,
                n,
                c,
                in_y - 1 + k,
                in_x + 1),
            upsample_get_value_bounded<T>(
                inputData,
                input_sizes.xy,
                input_strides,
                n,
                c,
                in_y - 1 + k,
                in_x + 2),
            t_x);
      }
      auto inp = static_cast<T>(cubic_interp1d(
          coefficients[0],
          coefficients[1],
          coefficients[2],
          coefficients[3],
          t_y));
      outputData
          [n * output_strides.w + c * output_strides.z +
           output_x * output_strides.x + output_y * output_strides.y] = inp;
    }
  }
}

template <typename T>
kernel void upsample_bicubic2d_backward(
    device AtomicType_t<T>* gradInputData [[buffer(0)]],
    constant T* gradOutputData [[buffer(1)]],
    constant ulong4& input_strides [[buffer(2)]],
    constant ulong4& output_strides [[buffer(3)]],
    constant long4& input_sizes [[buffer(4)]],
    constant long4& output_sizes [[buffer(5)]],
    constant float2& scales [[buffer(6)]],
    constant bool& align_corners [[buffer(7)]],
    uint thread_index [[thread_position_in_grid]]) {
  auto output_x = thread_index % output_sizes.x;
  auto output_y = thread_index / output_sizes.x;
  auto real_x = area_pixel_compute_source_index<float>(
      scales.x, output_x, align_corners, /*cubic=*/true);
  int input_x = floor(real_x);
  float t_x = real_x - input_x;

  auto real_y = area_pixel_compute_source_index<float>(
      scales.y, output_y, align_corners, /*cubic=*/true);
  int input_y = floor(real_y);
  float t_y = real_y - input_y;

  float x_coeffs[4];
  float y_coeffs[4];

  get_cubic_upsampling_coefficients(x_coeffs, t_x);
  get_cubic_upsampling_coefficients(y_coeffs, t_y);

  for (int n = 0; n < output_sizes.w; n++) {
    for (int c = 0; c < output_sizes.z; ++c) {
      auto out_value = gradOutputData
          [n * output_strides.w + c * output_strides.z +
           output_x * output_strides.x + output_y * output_strides.y];
      for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
          upsample_increment_value_bounded<T>(
              gradInputData,
              input_sizes.xy,
              input_strides,
              n,
              c,
              input_y - 1 + i,
              input_x - 1 + j,
              out_value * y_coeffs[i] * x_coeffs[j]);
        }
      }
    }
  }
}

#define INSTANTIATE_UPSAMPLE_BICUBIC(DTYPE)                        \
  template [[host_name("upsample_bicubic2d_" #DTYPE)]] kernel void \
  upsample_bicubic2d<DTYPE>(                                       \
      constant DTYPE * inputData [[buffer(0)]],                    \
      device DTYPE * outputData [[buffer(1)]],                     \
      constant ulong4 & input_strides [[buffer(2)]],               \
      constant ulong4 & output_strides [[buffer(3)]],              \
      constant long4 & input_sizes [[buffer(4)]],                  \
      constant long4 & output_sizes [[buffer(5)]],                 \
      constant float2 & scales [[buffer(6)]],                      \
      constant bool& align_corners [[buffer(7)]],                  \
      uint thread_index [[thread_position_in_grid]])

#define INSTANTIATE_UPSAMPLE_BICUBIC_BACKWARD(DTYPE)                        \
  template [[host_name("upsample_bicubic2d_backward_" #DTYPE)]] kernel void \
  upsample_bicubic2d_backward<DTYPE>(                                       \
      device AtomicType_t<DTYPE> * gradInputData [[buffer(0)]],             \
      constant DTYPE * gradOutputData [[buffer(1)]],                        \
      constant ulong4 & input_strides [[buffer(2)]],                        \
      constant ulong4 & output_strides [[buffer(3)]],                       \
      constant long4 & input_sizes [[buffer(4)]],                           \
      constant long4 & output_sizes [[buffer(5)]],                          \
      constant float2 & scales [[buffer(6)]],                               \
      constant bool& align_corners [[buffer(7)]],                           \
      uint thread_index [[thread_position_in_grid]])

INSTANTIATE_UPSAMPLE_BICUBIC(float);
INSTANTIATE_UPSAMPLE_BICUBIC_BACKWARD(float);
INSTANTIATE_UPSAMPLE_BICUBIC(half);
INSTANTIATE_UPSAMPLE_BICUBIC_BACKWARD(half);
#if __METAL_VERSION__ >= 310
INSTANTIATE_UPSAMPLE_BICUBIC(bfloat);
INSTANTIATE_UPSAMPLE_BICUBIC_BACKWARD(bfloat);
#endif
