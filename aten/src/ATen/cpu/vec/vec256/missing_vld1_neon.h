/* Workaround for missing vld1_*_x2 and vst1_*_x2 intrinsics in gcc-7.  */

__extension__ extern __inline uint8x8x2_t
__attribute__ ((__always_inline__, __gnu_inline__, __artificial__))
vld1_u8_x2 (const uint8_t *__a)
{
  uint8x8x2_t ret;
  asm volatile("ld1 {%S0.8b - %T0.8b}, %1" : "=w" (ret) : "Q"(*__a));
  return ret;
}

__extension__ extern __inline int8x8x2_t
__attribute__ ((__always_inline__, __gnu_inline__, __artificial__))
vld1_s8_x2 (const int8_t *__a)
{
  int8x8x2_t ret;
  asm volatile("ld1 {%S0.8b - %T0.8b}, %1" : "=w" (ret) : "Q"(*__a));
  return ret;
}

__extension__ extern __inline uint16x4x2_t
__attribute__ ((__always_inline__, __gnu_inline__, __artificial__))
vld1_u16_x2 (const uint16_t *__a)
{
  uint16x4x2_t ret;
  asm volatile("ld1 {%S0.4h - %T0.4h}, %1" : "=w" (ret) : "Q"(*__a));
  return ret;
}

__extension__ extern __inline int16x4x2_t
__attribute__ ((__always_inline__, __gnu_inline__, __artificial__))
vld1_s16_x2 (const int16_t *__a)
{
  int16x4x2_t ret;
  asm volatile("ld1 {%S0.4h - %T0.4h}, %1" : "=w" (ret) : "Q"(*__a));
  return ret;
}

__extension__ extern __inline uint32x2x2_t
__attribute__ ((__always_inline__, __gnu_inline__, __artificial__))
vld1_u32_x2 (const uint32_t *__a)
{
  uint32x2x2_t ret;
  asm volatile("ld1 {%S0.2s - %T0.2s}, %1" : "=w" (ret) : "Q"(*__a));
  return ret;
}

__extension__ extern __inline int32x2x2_t
__attribute__ ((__always_inline__, __gnu_inline__, __artificial__))
vld1_s32_x2 (const int32_t *__a)
{
  int32x2x2_t ret;
  asm volatile("ld1 {%S0.2s - %T0.2s}, %1" : "=w" (ret) : "Q"(*__a));
  return ret;
}

__extension__ extern __inline uint64x1x2_t
__attribute__ ((__always_inline__, __gnu_inline__, __artificial__))
vld1_u64_x2 (const uint64_t *__a)
{
  uint64x1x2_t ret;
  asm volatile("ld1 {%S0.1d - %T0.1d}, %1" : "=w" (ret) : "Q"(*__a));
  return ret;
}

__extension__ extern __inline int64x1x2_t
__attribute__ ((__always_inline__, __gnu_inline__, __artificial__))
vld1_s64_x2 (const int64_t *__a)
{
  int64x1x2_t ret;
  __builtin_aarch64_simd_oi __o;
  asm volatile("ld1 {%S0.1d - %T0.1d}, %1" : "=w" (ret) : "Q"(*__a));
  return ret;
}

__extension__ extern __inline float16x4x2_t
__attribute__ ((__always_inline__, __gnu_inline__, __artificial__))
vld1_f16_x2 (const float16_t *__a)
{
  float16x4x2_t ret;
  asm volatile("ld1 {%S0.4h - %T0.4h}, %1" : "=w" (ret) : "Q"(*__a));
  return ret;
}

__extension__ extern __inline float32x2x2_t
__attribute__ ((__always_inline__, __gnu_inline__, __artificial__))
vld1_f32_x2 (const float32_t *__a)
{
  float32x2x2_t ret;
  asm volatile("ld1 {%S0.2s - %T0.2s}, %1" : "=w" (ret) : "Q"(*__a));
  return ret;
}

__extension__ extern __inline float64x1x2_t
__attribute__ ((__always_inline__, __gnu_inline__, __artificial__))
vld1_f64_x2 (const float64_t *__a)
{
  float64x1x2_t ret;
  asm volatile("ld1 {%S0.1d - %T0.1d}, %1" : "=w" (ret) : "Q"(*__a));
  return ret;
}

__extension__ extern __inline poly8x8x2_t
__attribute__ ((__always_inline__, __gnu_inline__, __artificial__))
vld1_p8_x2 (const poly8_t *__a)
{
  poly8x8x2_t ret;
  asm volatile("ld1 {%S0.8b - %T0.8b}, %1" : "=w" (ret) : "Q"(*__a));
  return ret;
}

__extension__ extern __inline poly16x4x2_t
__attribute__ ((__always_inline__, __gnu_inline__, __artificial__))
vld1_p16_x2 (const poly16_t *__a)
{
  poly16x4x2_t ret;
  asm volatile("ld1 {%S0.4h - %T0.4h}, %1" : "=w" (ret) : "Q"(*__a));
  return ret;
}

__extension__ extern __inline poly64x1x2_t
__attribute__ ((__always_inline__, __gnu_inline__, __artificial__))
vld1_p64_x2 (const poly64_t *__a)
{
  poly64x1x2_t ret;
  asm volatile("ld1 {%S0.1d - %T0.1d}, %1" : "=w" (ret) : "Q"(*__a));
  return ret;
}

__extension__ extern __inline uint8x16x2_t
__attribute__ ((__always_inline__, __gnu_inline__, __artificial__))
vld1q_u8_x2 (const uint8_t *__a)
{
  uint8x16x2_t ret;
  asm volatile("ld1 {%S0.16b - %T0.16b}, %1" : "=w" (ret) : "Q"(*__a));
  return ret;
}

__extension__ extern __inline int8x16x2_t
__attribute__ ((__always_inline__, __gnu_inline__, __artificial__))
vld1q_s8_x2 (const int8_t *__a)
{
  int8x16x2_t ret;
  asm volatile("ld1 {%S0.16b - %T0.16b}, %1" : "=w" (ret) : "Q"(*__a));
  return ret;
}

__extension__ extern __inline uint16x8x2_t
__attribute__ ((__always_inline__, __gnu_inline__, __artificial__))
vld1q_u16_x2 (const uint16_t *__a)
{
  uint16x8x2_t ret;
  asm volatile("ld1 {%S0.8h - %T0.8h}, %1" : "=w" (ret) : "Q"(*__a));
  return ret;
}

__extension__ extern __inline int16x8x2_t
__attribute__ ((__always_inline__, __gnu_inline__, __artificial__))
vld1q_s16_x2 (const int16_t *__a)
{
  int16x8x2_t ret;
  asm volatile("ld1 {%S0.8h - %T0.8h}, %1" : "=w" (ret) : "Q"(*__a));
  return ret;
}

__extension__ extern __inline uint32x4x2_t
__attribute__ ((__always_inline__, __gnu_inline__, __artificial__))
vld1q_u32_x2 (const uint32_t *__a)
{
  uint32x4x2_t ret;
  asm volatile("ld1 {%S0.4s - %T0.4s}, %1" : "=w" (ret) : "Q"(*__a));
  return ret;
}

__extension__ extern __inline int32x4x2_t
__attribute__ ((__always_inline__, __gnu_inline__, __artificial__))
vld1q_s32_x2 (const int32_t *__a)
{
  int32x4x2_t ret;
  asm volatile("ld1 {%S0.4s - %T0.4s}, %1" : "=w" (ret) : "Q"(*__a));
  return ret;
}

__extension__ extern __inline uint64x2x2_t
__attribute__ ((__always_inline__, __gnu_inline__, __artificial__))
vld1q_u64_x2 (const uint64_t *__a)
{
  uint64x2x2_t ret;
  asm volatile("ld1 {%S0.2d - %T0.2d}, %1" : "=w" (ret) : "Q"(*__a));
  return ret;
}

__extension__ extern __inline int64x2x2_t
__attribute__ ((__always_inline__, __gnu_inline__, __artificial__))
vld1q_s64_x2 (const int64_t *__a)
{
  int64x2x2_t ret;
  asm volatile("ld1 {%S0.2d - %T0.2d}, %1" : "=w" (ret) : "Q"(*__a));
  return ret;
}

__extension__ extern __inline float16x8x2_t
__attribute__ ((__always_inline__, __gnu_inline__, __artificial__))
vld1q_f16_x2 (const float16_t *__a)
{
  float16x8x2_t ret;
  asm volatile("ld1 {%S0.8h - %T0.8h}, %1" : "=w" (ret) : "Q"(*__a));
  return ret;
}

__extension__ extern __inline float32x4x2_t
__attribute__ ((__always_inline__, __gnu_inline__, __artificial__))
vld1q_f32_x2 (const float32_t *__a)
{
  float32x4x2_t ret;
  asm volatile("ld1 {%S0.4s - %T0.4s}, %1" : "=w" (ret) : "Q"(*__a));
  return ret;
}

__extension__ extern __inline float64x2x2_t
__attribute__ ((__always_inline__, __gnu_inline__, __artificial__))
vld1q_f64_x2 (const float64_t *__a)
{
  float64x2x2_t ret;
  asm volatile("ld1 {%S0.2d - %T0.2d}, %1" : "=w" (ret) : "Q"(*__a));
  return ret;
}

__extension__ extern __inline poly8x16x2_t
__attribute__ ((__always_inline__, __gnu_inline__, __artificial__))
vld1q_p8_x2 (const poly8_t *__a)
{
  poly8x16x2_t ret;
  asm volatile("ld1 {%S0.16b - %T0.16b}, %1" : "=w" (ret) : "Q"(*__a));
  return ret;
}

__extension__ extern __inline poly16x8x2_t
__attribute__ ((__always_inline__, __gnu_inline__, __artificial__))
vld1q_p16_x2 (const poly16_t *__a)
{
  poly16x8x2_t ret;
  asm volatile("ld1 {%S0.8h - %T0.8h}, %1" : "=w" (ret) : "Q"(*__a));
  return ret;
}

__extension__ extern __inline poly64x2x2_t
__attribute__ ((__always_inline__, __gnu_inline__, __artificial__))
vld1q_p64_x2 (const poly64_t *__a)
{
  poly64x2x2_t ret;
  asm volatile("ld1 {%S0.2d - %T0.2d}, %1" : "=w" (ret) : "Q"(*__a));
  return ret;
}

/* vst1x2 */

__extension__ extern __inline void
__attribute__ ((__always_inline__, __gnu_inline__, __artificial__))
vst1_s64_x2 (int64_t * __a, int64x1x2_t val)
{
  asm volatile("st1 {%S1.1d - %T1.1d}, %0" : "=Q" (*__a) : "w" (val));
}

__extension__ extern __inline void
__attribute__ ((__always_inline__, __gnu_inline__, __artificial__))
vst1_u64_x2 (uint64_t * __a, uint64x1x2_t val)
{
  asm volatile("st1 {%S1.1d - %T1.1d}, %0" : "=Q" (*__a) : "w" (val));
}

__extension__ extern __inline void
__attribute__ ((__always_inline__, __gnu_inline__, __artificial__))
vst1_f64_x2 (float64_t * __a, float64x1x2_t val)
{
  asm volatile("st1 {%S1.1d - %T1.1d}, %0" : "=Q" (*__a) : "w" (val));
}

__extension__ extern __inline void
__attribute__ ((__always_inline__, __gnu_inline__, __artificial__))
vst1_s8_x2 (int8_t * __a, int8x8x2_t val)
{
  asm volatile("st1 {%S1.8b - %T1.8b}, %0" : "=Q" (*__a) : "w" (val));
}

__extension__ extern __inline void
__attribute__ ((__always_inline__, __gnu_inline__, __artificial__))
vst1_p8_x2 (poly8_t * __a, poly8x8x2_t val)
{
  asm volatile("st1 {%S1.8b - %T1.8b}, %0" : "=Q" (*__a) : "w" (val));
}

__extension__ extern __inline void
__attribute__ ((__always_inline__, __gnu_inline__, __artificial__))
vst1_s16_x2 (int16_t * __a, int16x4x2_t val)
{
  asm volatile("st1 {%S1.4h - %T1.4h}, %0" : "=Q" (*__a) : "w" (val));
}

__extension__ extern __inline void
__attribute__ ((__always_inline__, __gnu_inline__, __artificial__))
vst1_p16_x2 (poly16_t * __a, poly16x4x2_t val)
{
  asm volatile("st1 {%S1.4h - %T1.4h}, %0" : "=Q" (*__a) : "w" (val));
}

__extension__ extern __inline void
__attribute__ ((__always_inline__, __gnu_inline__, __artificial__))
vst1_s32_x2 (int32_t * __a, int32x2x2_t val)
{
  asm volatile("st1 {%S1.2s - %T1.2s}, %0" : "=Q" (*__a) : "w" (val));
}

__extension__ extern __inline void
__attribute__ ((__always_inline__, __gnu_inline__, __artificial__))
vst1_u8_x2 (uint8_t * __a, uint8x8x2_t val)
{
  asm volatile("st1 {%S1.8b - %T1.8b}, %0" : "=Q" (*__a) : "w" (val));
}

__extension__ extern __inline void
__attribute__ ((__always_inline__, __gnu_inline__, __artificial__))
vst1_u16_x2 (uint16_t * __a, uint16x4x2_t val)
{
  asm volatile("st1 {%S1.4h - %T1.4h}, %0" : "=Q" (*__a) : "w" (val));
}

__extension__ extern __inline void
__attribute__ ((__always_inline__, __gnu_inline__, __artificial__))
vst1_u32_x2 (uint32_t * __a, uint32x2x2_t val)
{
  asm volatile("st1 {%S1.2s - %T1.2s}, %0" : "=Q" (*__a) : "w" (val));
}

__extension__ extern __inline void
__attribute__ ((__always_inline__, __gnu_inline__, __artificial__))
vst1_f16_x2 (float16_t * __a, float16x4x2_t val)
{
  asm volatile("st1 {%S1.4h - %T1.4h}, %0" : "=Q" (*__a) : "w" (val));
}

__extension__ extern __inline void
__attribute__ ((__always_inline__, __gnu_inline__, __artificial__))
vst1_f32_x2 (float32_t * __a, float32x2x2_t val)
{
  asm volatile("st1 {%S1.2s - %T1.2s}, %0" : "=Q" (*__a) : "w" (val));
}

__extension__ extern __inline void
__attribute__ ((__always_inline__, __gnu_inline__, __artificial__))
vst1_p64_x2 (poly64_t * __a, poly64x1x2_t val)
{
  asm volatile("st1 {%S1.1d - %T1.1d}, %0" : "=Q" (*__a) : "w" (val));
}

__extension__ extern __inline void
__attribute__ ((__always_inline__, __gnu_inline__, __artificial__))
vst1q_s8_x2 (int8_t * __a, int8x16x2_t val)
{
  asm volatile("st1 {%S1.16b - %T1.16b}, %0" : "=Q" (*__a) : "w" (val));
}

__extension__ extern __inline void
__attribute__ ((__always_inline__, __gnu_inline__, __artificial__))
vst1q_p8_x2 (poly8_t * __a, poly8x16x2_t val)
{
  asm volatile("st1 {%S1.16b - %T1.16b}, %0" : "=Q" (*__a) : "w" (val));
}

__extension__ extern __inline void
__attribute__ ((__always_inline__, __gnu_inline__, __artificial__))
vst1q_s16_x2 (int16_t * __a, int16x8x2_t val)
{
  asm volatile("st1 {%S1.8h - %T1.8h}, %0" : "=Q" (*__a) : "w" (val));
}

__extension__ extern __inline void
__attribute__ ((__always_inline__, __gnu_inline__, __artificial__))
vst1q_p16_x2 (poly16_t * __a, poly16x8x2_t val)
{
  asm volatile("st1 {%S1.8h - %T1.8h}, %0" : "=Q" (*__a) : "w" (val));
}

__extension__ extern __inline void
__attribute__ ((__always_inline__, __gnu_inline__, __artificial__))
vst1q_s32_x2 (int32_t * __a, int32x4x2_t val)
{
  asm volatile("st1 {%S1.4s - %T1.4s}, %0" : "=Q" (*__a) : "w" (val));
}

__extension__ extern __inline void
__attribute__ ((__always_inline__, __gnu_inline__, __artificial__))
vst1q_s64_x2 (int64_t * __a, int64x2x2_t val)
{
  asm volatile("st1 {%S1.2d - %T1.2d}, %0" : "=Q" (*__a) : "w" (val));
}

__extension__ extern __inline void
__attribute__ ((__always_inline__, __gnu_inline__, __artificial__))
vst1q_u8_x2 (uint8_t * __a, uint8x16x2_t val)
{
  asm volatile("st1 {%S1.16b - %T1.16b}, %0" : "=Q" (*__a) : "w" (val));
}

__extension__ extern __inline void
__attribute__ ((__always_inline__, __gnu_inline__, __artificial__))
vst1q_u16_x2 (uint16_t * __a, uint16x8x2_t val)
{
  asm volatile("st1 {%S1.8h - %T1.8h}, %0" : "=Q" (*__a) : "w" (val));
}

__extension__ extern __inline void
__attribute__ ((__always_inline__, __gnu_inline__, __artificial__))
vst1q_u32_x2 (uint32_t * __a, uint32x4x2_t val)
{
  asm volatile("st1 {%S1.4s - %T1.4s}, %0" : "=Q" (*__a) : "w" (val));
}

__extension__ extern __inline void
__attribute__ ((__always_inline__, __gnu_inline__, __artificial__))
vst1q_u64_x2 (uint64_t * __a, uint64x2x2_t val)
{
  asm volatile("st1 {%S1.2d - %T1.2d}, %0" : "=Q" (*__a) : "w" (val));
}

__extension__ extern __inline void
__attribute__ ((__always_inline__, __gnu_inline__, __artificial__))
vst1q_f16_x2 (float16_t * __a, float16x8x2_t val)
{
  asm volatile("st1 {%S1.8h - %T1.8h}, %0" : "=Q" (*__a) : "w" (val));
}

__extension__ extern __inline void
__attribute__ ((__always_inline__, __gnu_inline__, __artificial__))
vst1q_f32_x2 (float32_t * __a, float32x4x2_t val)
{
  asm volatile("st1 {%S1.4s - %T1.4s}, %0" : "=Q" (*__a) : "w" (val));
}

__extension__ extern __inline void
__attribute__ ((__always_inline__, __gnu_inline__, __artificial__))
vst1q_f64_x2 (float64_t * __a, float64x2x2_t val)
{
  asm volatile("st1 {%S1.2d - %T1.2d}, %0" : "=Q" (*__a) : "w" (val));
}

__extension__ extern __inline void
__attribute__ ((__always_inline__, __gnu_inline__, __artificial__))
vst1q_p64_x2 (poly64_t * __a, poly64x2x2_t val)
{
  asm volatile("st1 {%S1.2d - %T1.2d}, %0" : "=Q" (*__a) : "w" (val));
}
