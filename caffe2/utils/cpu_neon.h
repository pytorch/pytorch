#ifndef CAFFE2_UTILS_CPU_NEON_H_
#define CAFFE2_UTILS_CPU_NEON_H_

// Provides a variety of ARM NEON-specific utility functions
#ifdef __ARM_NEON__
#include <arm_neon.h>

namespace caffe2 {

template <typename T>
inline bool isPointerAligned(T* p, size_t align) {
  return (reinterpret_cast<uintptr_t>(p) % align == 0);
}

inline float32x4_t vert_sum_f32(float32x4_t v0,
                                float32x4_t v1,
                                float32x4_t v2,
                                float32x4_t v3) {
  v0 = vaddq_f32(v0, v1);
  v2 = vaddq_f32(v2, v3);
  return vaddq_f32(v0, v2);
}

inline float horizontal_sum_f32(float32x4_t v0,
                                float32x4_t v1,
                                float32x4_t v2,
                                float32x4_t v3) {
  v0 = vert_sum_f32(v0, v1, v2, v3);
  float32x2_t v = vadd_f32(vget_high_f32(v0), vget_low_f32(v0));
  return vget_lane_f32(vpadd_f32(v, v), 0);
}

// Load/store functions that assume alignment

inline float32x4_t vld1q_f32_aligned(const float* p) {
  return vld1q_f32((const float*)
                   __builtin_assume_aligned(p, sizeof(float32x4_t)));
}

inline void vst1q_f32_aligned(float* p, float32x4_t v) {
  vst1q_f32((float*) __builtin_assume_aligned(p, sizeof(float32x4_t)), v);
}

inline void vst4_u8_aligned(uint8_t* p, uint8x8x4_t v) {
  vst4_u8((uint8_t*)
          __builtin_assume_aligned(p, sizeof(uint8x8x4_t)), v);
}

}  // namespace caffe2

#endif // __ARM_NEON__

#endif  // CAFFE2_UTILS_CPU_NEON_H_
