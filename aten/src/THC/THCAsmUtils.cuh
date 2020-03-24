#ifndef THC_ASM_UTILS_INC
#define THC_ASM_UTILS_INC

// Collection of direct PTX functions

template <typename T>
struct Bitfield {};

template <>
struct Bitfield<unsigned int> {
  static __device__ __forceinline__
  unsigned int getBitfield(unsigned int val, int pos, int len) {
#if defined(__HIP_PLATFORM_HCC__)
    pos &= 0xff;
    len &= 0xff;

    unsigned int m = (1u << len) - 1u;
    return (val >> pos) & m;
#else
    unsigned int ret;
    asm("bfe.u32 %0, %1, %2, %3;" : "=r"(ret) : "r"(val), "r"(pos), "r"(len));
    return ret;
#endif
  }

  static __device__ __forceinline__
  unsigned int setBitfield(unsigned int val, unsigned int toInsert, int pos, int len) {
#if defined(__HIP_PLATFORM_HCC__)
    pos &= 0xff;
    len &= 0xff;

    unsigned int m = (1u << len) - 1u;
    toInsert &= m;
    toInsert <<= pos;
    m <<= pos;

    return (val & ~m) | toInsert;
#else
    unsigned int ret;
    asm("bfi.b32 %0, %1, %2, %3, %4;" :
        "=r"(ret) : "r"(toInsert), "r"(val), "r"(pos), "r"(len));
    return ret;
#endif
  }
};

template <>
struct Bitfield<uint64_t> {
  static __device__ __forceinline__
  uint64_t getBitfield(uint64_t val, int pos, int len) {
#if defined(__HIP_PLATFORM_HCC__)
    pos &= 0xff;
    len &= 0xff;

    uint64_t m = (1u << len) - 1u;
    return (val >> pos) & m;
#else
    uint64_t ret;
    asm("bfe.u64 %0, %1, %2, %3;" : "=l"(ret) : "l"(val), "r"(pos), "r"(len));
    return ret;
#endif
  }

  static __device__ __forceinline__
  uint64_t setBitfield(uint64_t val, uint64_t toInsert, int pos, int len) {
#if defined(__HIP_PLATFORM_HCC__)
    pos &= 0xff;
    len &= 0xff;

    uint64_t m = (1u << len) - 1u;
    toInsert &= m;
    toInsert <<= pos;
    m <<= pos;

    return (val & ~m) | toInsert;
#else
    uint64_t ret;
    asm("bfi.b64 %0, %1, %2, %3, %4;" :
        "=l"(ret) : "l"(toInsert), "l"(val), "r"(pos), "r"(len));
    return ret;
#endif
  }
};

__device__ __forceinline__ int getLaneId() {
#if defined(__HIP_PLATFORM_HCC__)
  return __lane_id();
#else
  int laneId;
  asm("mov.s32 %0, %%laneid;" : "=r"(laneId) );
  return laneId;
#endif
}

#if defined(__HIP_PLATFORM_HCC__)
__device__ __forceinline__ unsigned long long int getLaneMaskLt() {
  std::uint64_t m = (1ull << getLaneId()) - 1ull;
  return m;
#else
__device__ __forceinline__ unsigned getLaneMaskLt() {
  unsigned mask;
  asm("mov.u32 %0, %%lanemask_lt;" : "=r"(mask));
  return mask;
#endif
}

#if defined (__HIP_PLATFORM_HCC__)
__device__ __forceinline__ unsigned long long int getLaneMaskLe() {
  std::uint64_t m = UINT64_MAX >> (sizeof(std::uint64_t) * CHAR_BIT - (getLaneId() + 1));
  return m;
}
#else
__device__ __forceinline__ unsigned getLaneMaskLe() {
  unsigned mask;
  asm("mov.u32 %0, %%lanemask_le;" : "=r"(mask));
  return mask;
}
#endif

#if defined(__HIP_PLATFORM_HCC__)
__device__ __forceinline__ unsigned long long int getLaneMaskGt() {
  std::uint64_t m = getLaneMaskLe();
  return m ? ~m : m;
#else
__device__ __forceinline__ unsigned getLaneMaskGt() {
  unsigned mask;
  asm("mov.u32 %0, %%lanemask_gt;" : "=r"(mask));
  return mask;
#endif
}

#if defined(__HIP_PLATFORM_HCC__)
__device__ __forceinline__ unsigned long long int getLaneMaskGe() {
  std::uint64_t m = getLaneMaskLt();
  return ~m;
#else
__device__ __forceinline__ unsigned getLaneMaskGe() {
  unsigned mask;
  asm("mov.u32 %0, %%lanemask_ge;" : "=r"(mask));
  return mask;
#endif
}


#endif // THC_ASM_UTILS_INC
