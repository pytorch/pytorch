/*This file contains various adjustments required to interoperate with the Bitfield and Lanes for the AMD Architecture.*/

#ifndef THC_ASM_UTILS_INC
#define THC_ASM_UTILS_INC

// Collection of direct PTX functions

template <typename T>
struct Bitfield {};

template <>
struct Bitfield<unsigned int> {
  static __device__
  inline
  unsigned int getBitfield(unsigned int val, int pos, int len)
  {
    pos &= 0x1f;
    len &= 0x1f;

    unsigned int m = (1u << len) - 1u;
    m <<= pos;
    return val & m;
  }

 static  __device__
  inline
  unsigned int setBitfield(
    unsigned int val, unsigned int toInsert, int pos, int len)
  {
    pos &= 0x1f;
    len &= 0x1f;

    unsigned int m = (1u << len) - 1u;
    toInsert &= m;
    toInsert <<= pos;
    m <<= pos;

    return (val & ~m) | toInsert;
  }
};

template<>
struct Bitfield<uint64_t>{
  static __device__
  inline
  uint64_t getBitfield(uint64_t val, int pos, int len)
  {
    pos &= 0x1f;
    len &= 0x1f;

    uint64_t m = (1u << len) - 1u;
    m <<= pos;
    return val & m;
  }

  static __device__
  inline
  uint64_t setBitfield(
    uint64_t val, uint64_t toInsert, int pos, int len)
  {
    pos &= 0x1f;
    len &= 0x1f;

    uint64_t m = (1u << len) - 1u;
    toInsert &= m;
    toInsert <<= pos;
    m <<= pos;

    return (val & ~m) | toInsert;
  }
};

__device__ __forceinline__ inline int getLaneId() {
    return hc::__lane_id();
}

__device__ inline std::uint64_t getLaneMaskLt()
{
  std::uint64_t m = (1ull << getLaneId()) - 1ull;
  return m;
}

__device__ inline std::uint64_t getLaneMaskLe()
{
  std::uint64_t m = (1ull << (getLaneId() + 1ull)) - 1ull;
  return m;
}

__device__ inline std::uint64_t getLaneMaskGt()
{
  std::uint64_t m = getLaneMaskLe();
  return m ? ~m : m;
}

__device__ inline std::uint64_t getLaneMaskGe()
{
  std::uint64_t m = getLaneMaskLt();
  return ~m;
}

#endif // THC_ASM_UTILS_INC
