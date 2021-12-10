
#define __NVFUSER_BFLOAT_TO_US(var) *(reinterpret_cast<unsigned short*>(&(var)))
#define __NVFUSER_BFLOAT_TO_CUS(var) \
  *(reinterpret_cast<const unsigned short*>(&(var)))

struct __bfloat;
__device__ __bfloat __float2bfloat(const float);

struct __align__(2) __bfloat {
  __bfloat() = default;

  __device__ __bfloat(const float f) {
    __x = __float2bfloat(f).__x;
  }

 protected:
  unsigned short __x;
};

__device__ __bfloat __float2bfloat(const float f) {
  __bfloat val;
  asm("{  cvt.rn.bf16.f32 %0, %1;}\n"
      : "=h"(__NVFUSER_BFLOAT_TO_US(val))
      : "f"(f));
  return val;
}

__device__ float __bfloat2float(const __bfloat h) {
  float val;
  asm("{  mov.b32 %0, {0,%1};}\n"
      : "=f"(val)
      : "h"(__NVFUSER_BFLOAT_TO_CUS(h)));
  return val;
}
