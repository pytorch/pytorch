
struct __align__(2) __bfloat {
  __bfloat() = default;

  inline __device__ __bfloat(const float f) {
    if (f != f) {
      __x = uint16_t(0x7FC0);
    } else {
      union {
        uint32_t U32;
        float F32;
      };

      F32 = f;
      uint32_t rounding_bias = ((U32 >> 16) & 1) + uint32_t(0x7FFF);
      __x = static_cast<uint16_t>((U32 + rounding_bias) >> 16);
    }
  }

  inline __device__ operator float() const {
    float res = 0;
    uint32_t tmp = __x;
    tmp <<= 16;
    float* tempRes = reinterpret_cast<float*>(&tmp);
    res = *tempRes;
    return res;
  }

 protected:
  unsigned short __x;
};

__device__ __bfloat __float2bfloat(const float f) {
  return __bfloat(f);
}

__device__ float __bfloat2float(const __bfloat h) {
  return float(h);
}
