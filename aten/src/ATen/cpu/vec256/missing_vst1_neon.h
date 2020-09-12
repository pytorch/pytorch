/* Workaround for missing vst1q_f32_x2 in gcc-8.  */

__extension__ extern __inline void
__attribute__ ((__always_inline__, __gnu_inline__, __artificial__))
vst1q_f32_x2 (float32_t * __a, float32x4x2_t val)
{
  asm ("st1 {%S0.4s - %T0.4s}, [%1]" :: "w" (val), "r"(__a) :);
}

