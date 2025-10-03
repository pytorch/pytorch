#pragma once

#include <ATen/cpu/vec/intrinsics.h>
#include <ATen/cpu/vec/sve/sve_helper.h>
#include <ATen/cpu/vec/vec_base.h>
#include <c10/macros/Macros.h>
#include <c10/util/irange.h>

namespace at::vec {
// Note [CPU_CAPABILITY namespace]
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// This header, and all of its subheaders, will be compiled with
// different architecture flags for each supported set of vector
// intrinsics. So we need to make sure they aren't inadvertently
// linked together. We do this by declaring objects in an `inline
// namespace` which changes the name mangling, but can still be
// accessed as `at::vec`.
inline namespace CPU_CAPABILITY {

#if defined(CPU_CAPABILITY_SVE)

#define VEC_UINT_SVE_TEMPLATE(vl, bit)                                        \
  template <>                                                                 \
  struct is_vec_specialized_for<uint##bit##_t> : std::bool_constant<true> {}; \
                                                                              \
  template <>                                                                 \
  class Vectorized<uint##bit##_t> {                                           \
   using neon_type = uint##bit##x##vl##_t;                                    \
   private:                                                                   \
    neon_type values;                                                         \
                                                                              \
   public:                                                                    \
    using value_type = uint##bit##_t;                                         \
    using size_type = int;                                                    \
    static constexpr size_type size() {                                       \
      return vl;                                                              \
    }                                                                         \
    Vectorized() {                                                            \
      values = vdupq_n_u##bit(0);                                             \
    }                                                                         \
    Vectorized(neon_type v) : values(v) {}                                    \
    Vectorized(svuint##bit##_t v) : values(svget_neonq(v)) {}                 \
    Vectorized(value_type val) {                                              \
      values = svget_neonq(svdup_n_u##bit(val));                              \
    }                                                                         \
    template <                                                                \
        typename... Args,                                                     \
        typename = std::enable_if_t<(sizeof...(Args) == size())>>             \
    Vectorized(Args... vals) {                                                \
      __at_align__ value_type buffer[size()] = {vals...};                     \
      values = vld1q_u##bit(buffer);                                          \
    }                                                                         \
    operator neon_type() const {                                              \
      return values;                                                          \
    }                                                                         \
    operator svuint##bit##_t() const {                                        \
      return svset_neonq(svundef_u##bit(), values);                           \
    }                                                                         \
    template <int64_t mask>                                                   \
    static Vectorized<uint##bit##_t> blend(                                   \
      const Vectorized<uint##bit##_t>& a,                                     \
      const Vectorized<uint##bit##_t>& b);                                    \
    static Vectorized<uint##bit##_t> blendv(                                  \
        const Vectorized<uint##bit##_t>& a,                                   \
        const Vectorized<uint##bit##_t>& b,                                   \
        const Vectorized<uint##bit##_t>& mask_) {                             \
      return vbslq_u##bit(mask_.values, b, a);                                \
    }                                                                         \
    template <typename step_t>                                                \
    static Vectorized<uint##bit##_t> arange(                                  \
        value_type base = 0,                                                  \
        step_t step = static_cast<step_t>(1));                                \
    static Vectorized<uint##bit##_t> set(                                     \
        const Vectorized<uint##bit##_t>& a,                                   \
        const Vectorized<uint##bit##_t>& b,                                   \
        uint##bit##_t count = size()) {                                       \
      if (count == 0) {                                                       \
        return a;                                                             \
      } else if (count < size()) {                                            \
        return svsel_u##bit(svwhilelt_b##bit(0ull, count), b, a);             \
      }                                                                       \
      return b;                                                               \
    }                                                                         \
    static Vectorized<uint##bit##_t> loadu(                                   \
        const void* ptr,                                                      \
        int64_t count = size()) {                                             \
      if (count == size())                                                    \
        return vld1q_u##bit(reinterpret_cast<const uint##bit##_t*>(ptr));     \
      svbool_t pg = svwhilelt_b##bit(0ull, count);                            \
      return svld1_u##bit(pg, reinterpret_cast<const uint##bit##_t*>(ptr));   \
    }                                                                         \
    void store(void* ptr, int64_t count = size()) const {                     \
      if (count == size()) {                                                  \
        vst1q_u##bit(reinterpret_cast<uint##bit##_t*>(ptr), values);          \
      } else {                                                                \
        svbool_t pg = svwhilelt_b##bit(0ull, count);                          \
        auto dstPtr = reinterpret_cast<uint##bit##_t*>(ptr);                  \
        svst1_u##bit(pg, dstPtr, svset_neonq(svundef_u##bit(), values));      \
      }                                                                       \
    }                                                                         \
    const uint##bit##_t& operator[](int idx) const = delete;                  \
    uint##bit##_t& operator[](int idx) = delete;                              \
    Vectorized<uint##bit##_t> abs() const {                                   \
      return values;                                                          \
    }                                                                         \
    Vectorized<uint##bit##_t> real() const {                                  \
      return values;                                                          \
    }                                                                         \
    Vectorized<uint##bit##_t> imag() const {                                  \
      return vdupq_n_u##bit(0);                                               \
    }                                                                         \
    Vectorized<uint##bit##_t> conj() const {                                  \
      return values;                                                          \
    }                                                                         \
    Vectorized<uint##bit##_t> neg() const {                                   \
      return vreinterpretq_u##bit##_s##bit(                                   \
        vnegq_s##bit(vreinterpretq_s##bit##_u##bit(values)));                 \
    }                                                                         \
    uint##bit##_t reduce_add() const {                                        \
    return vaddvq_u##bit(values);                                             \
    }                                                                         \
    uint##bit##_t reduce_max() const;                                         \
    Vectorized<uint##bit##_t> operator==(                                     \
        const Vectorized<uint##bit##_t>& other) const {                       \
      return Vectorized<value_type>(vceqq_u##bit(values, other.values));      \
    }                                                                         \
    Vectorized<uint##bit##_t> operator!=(                                     \
        const Vectorized<uint##bit##_t>& other) const;                        \
    Vectorized<uint##bit##_t> operator<(                                      \
        const Vectorized<uint##bit##_t>& other) const {                       \
      return Vectorized<value_type>(vcltq_u##bit(values, other.values));      \
    }                                                                         \
    Vectorized<uint##bit##_t> operator<=(                                     \
        const Vectorized<uint##bit##_t>& other) const {                       \
      return Vectorized<value_type>(vcleq_u##bit(values, other.values));      \
    }                                                                         \
    Vectorized<uint##bit##_t> operator>(                                      \
        const Vectorized<uint##bit##_t>& other) const {                       \
      return Vectorized<value_type>(vcgtq_u##bit(values, other.values));      \
    }                                                                         \
    Vectorized<uint##bit##_t> operator>=(                                     \
        const Vectorized<uint##bit##_t>& other) const {                       \
      return Vectorized<value_type>(vcgeq_u##bit(values, other.values));      \
    }                                                                         \
    Vectorized<uint##bit##_t> eq(const Vectorized<uint##bit##_t>& other) const; \
    Vectorized<uint##bit##_t> ne(const Vectorized<uint##bit##_t>& other) const; \
    Vectorized<uint##bit##_t> gt(const Vectorized<uint##bit##_t>& other) const; \
    Vectorized<uint##bit##_t> ge(const Vectorized<uint##bit##_t>& other) const; \
    Vectorized<uint##bit##_t> lt(const Vectorized<uint##bit##_t>& other) const; \
    Vectorized<uint##bit##_t> le(const Vectorized<uint##bit##_t>& other) const; \
  };                                                                            \
  template <>                                                                   \
  Vectorized<uint##bit##_t> inline operator+(                                   \
      const Vectorized<uint##bit##_t>& a, const Vectorized<uint##bit##_t>& b) { \
    return vaddq_u##bit(a, b);                                                  \
  }                                                                             \
  template <>                                                                   \
  Vectorized<uint##bit##_t> inline operator-(                                   \
      const Vectorized<uint##bit##_t>& a, const Vectorized<uint##bit##_t>& b) { \
    return vsubq_u##bit(a, b);                                                  \
  }                                                                             \
  template <>                                                                   \
  Vectorized<uint##bit##_t> inline operator&(                                   \
      const Vectorized<uint##bit##_t>& a, const Vectorized<uint##bit##_t>& b) { \
    return vandq_u##bit(a, b);                                                  \
  }                                                                             \
  template <>                                                                   \
  Vectorized<uint##bit##_t> inline operator|(                                   \
      const Vectorized<uint##bit##_t>& a, const Vectorized<uint##bit##_t>& b) { \
    return vorrq_u##bit(a, b);                                                  \
  }                                                                             \
  template <>                                                                   \
  Vectorized<uint##bit##_t> inline operator^(                                   \
      const Vectorized<uint##bit##_t>& a, const Vectorized<uint##bit##_t>& b) { \
    return veorq_u##bit(a, b);                                                  \
  }                                                                             \
  Vectorized<uint##bit##_t> inline Vectorized<uint##bit##_t>::eq(               \
      const Vectorized<uint##bit##_t>& other) const {                           \
    return svand_n_u##bit##_x(ptrue, *this == other, 1);                        \
  }                                                                             \
  Vectorized<uint##bit##_t> inline Vectorized<uint##bit##_t>::ne(               \
      const Vectorized<uint##bit##_t>& other) const {                           \
    return svand_n_u##bit##_x(ptrue, *this != other, 1);                        \
  }                                                                             \
  Vectorized<uint##bit##_t> inline Vectorized<uint##bit##_t>::gt(               \
      const Vectorized<uint##bit##_t>& other) const {                           \
    return svand_n_u##bit##_x(ptrue, *this > other, 1);                         \
  }                                                                             \
  Vectorized<uint##bit##_t> inline Vectorized<uint##bit##_t>::ge(               \
      const Vectorized<uint##bit##_t>& other) const {                           \
    return svand_n_u##bit##_x(ptrue, *this >= other, 1);                        \
  }                                                                             \
  Vectorized<uint##bit##_t> inline Vectorized<uint##bit##_t>::lt(               \
      const Vectorized<uint##bit##_t>& other) const {                           \
    return svand_n_u##bit##_x(ptrue, *this < other, 1);                         \
  }                                                                             \
  Vectorized<uint##bit##_t> inline Vectorized<uint##bit##_t>::le(               \
      const Vectorized<uint##bit##_t>& other) const {                           \
    return svand_n_u##bit##_x(ptrue, *this <= other, 1);                        \
  }

VEC_UINT_SVE_TEMPLATE(2, 64)
VEC_UINT_SVE_TEMPLATE(4, 32)
VEC_UINT_SVE_TEMPLATE(8, 16)
VEC_UINT_SVE_TEMPLATE(16, 8)

inline uint64_t Vectorized<uint64_t>::reduce_max() const {                                         
  return svmaxv_u64(ptrue, svset_neonq(svundef_u64(), values));                                 
}

inline uint32_t Vectorized<uint32_t>::reduce_max() const {                                         
  return vmaxvq_u32(values);                                 
}

inline uint16_t Vectorized<uint16_t>::reduce_max() const {                                         
  return vmaxvq_u16(values);                                 
}

inline uint8_t Vectorized<uint8_t>::reduce_max() const {                                         
  return vmaxvq_u8(values);                                 
}                                                                         

template <>                                                                 
Vectorized<uint64_t> inline operator*(                                  
    const Vectorized<uint64_t>& a, const Vectorized<uint64_t>& b) { 
  return svmul_u64_x(ptrue, a, b);                                                
} 

template <>                                                                 
Vectorized<uint32_t> inline operator*(                                  
    const Vectorized<uint32_t>& a, const Vectorized<uint32_t>& b) { 
  return vmulq_u32(a, b);                                                
} 

template <>                                                                 
Vectorized<uint16_t> inline operator*(                                  
    const Vectorized<uint16_t>& a, const Vectorized<uint16_t>& b) { 
  return vmulq_u16(a, b);                                                
} 

template <>                                                                 
Vectorized<uint8_t> inline operator*(                                  
    const Vectorized<uint8_t>& a, const Vectorized<uint8_t>& b) { 
  return vmulq_u8(a, b);                                                
}  

template <>
Vectorized<uint64_t> inline operator/(
    const Vectorized<uint64_t>& a,
    const Vectorized<uint64_t>& b) {
  return svdiv_u64_x(ptrue, a, b);
}

template <>
Vectorized<uint32_t> inline operator/(
    const Vectorized<uint32_t>& a,
    const Vectorized<uint32_t>& b) {
  return svdiv_u32_x(ptrue, a, b);
}

template <>
Vectorized<uint16_t> inline operator/(
    const Vectorized<uint16_t>& a,
    const Vectorized<uint16_t>& b) {
  Vectorized<uint32_t> highBitsA = vmovl_high_u16(a);
  Vectorized<uint32_t> highBitsB = vmovl_high_u16(b);
  Vectorized<uint32_t> lowBitsA = vmovl_u16(vget_low_u16(a));
  Vectorized<uint32_t> lowBitsB = vmovl_u16(vget_low_u16(b));
  int32x4_t highBitsResult = highBitsA / highBitsB;
  int32x4_t lowBitsResult = lowBitsA / lowBitsB;
  return vuzp1q_u16(vreinterpretq_u16_u32(lowBitsResult), vreinterpretq_u16_u32(highBitsResult));
}

template <>
Vectorized<uint8_t> inline operator/(
    const Vectorized<uint8_t>& a,
    const Vectorized<uint8_t>& b) {
  Vectorized<uint16_t> highBitsA = vmovl_high_u8(a);
  Vectorized<uint16_t> highBitsB = vmovl_high_u8(b);
  Vectorized<uint16_t> lowBitsA = vmovl_u8(vget_low_u8(a));
  Vectorized<uint16_t> lowBitsB = vmovl_u8(vget_low_u8(b));
  int16x8_t highBitsResult = highBitsA / highBitsB;
  int16x8_t lowBitsResult = lowBitsA / lowBitsB;
  return vuzp1q_u8(vreinterpretq_u8_u16(lowBitsResult), vreinterpretq_u8_u16(highBitsResult));
}                                                                           

template <>
inline Vectorized<uint64_t> operator~(                                  
    const Vectorized<uint64_t>& a) {
  uint64x2_t val = a;                                    
  return ~val;                                                   
}
 
template <>
inline Vectorized<uint32_t> operator~(                                  
    const Vectorized<uint32_t>& a) {                                    
  return vmvnq_u32(a);                                                   
}

template <>                                                                
inline Vectorized<uint16_t> operator~(                                  
    const Vectorized<uint16_t>& a) {                                    
  return vmvnq_u16(a);                                                   
} 

template <>
inline Vectorized<uint8_t> operator~(                                  
    const Vectorized<uint8_t>& a) {                                    
  return vmvnq_u8(a);                                                   
}

inline Vectorized<uint64_t> Vectorized<uint64_t>::operator!=(                                  
    const Vectorized<uint64_t>& other) const {                                    
  return  ~(*this == other);                                                   
} 
                                                                 
inline Vectorized<uint32_t> Vectorized<uint32_t>::operator!=(                                  
    const Vectorized<uint32_t>& other) const {                                    
  return  ~(*this == other);                                                   
} 
                                                                
inline Vectorized<uint16_t> Vectorized<uint16_t>::operator!=(                                  
    const Vectorized<uint16_t>& other) const {                                    
  return  ~(*this == other);                                                   
} 
                                                              
inline Vectorized<uint8_t> Vectorized<uint8_t>::operator!=(                                  
    const Vectorized<uint8_t>& other) const {                                    
  return  ~(*this == other);                                                   
}

template <>                                                                 
Vectorized<uint64_t> inline minimum(                                    
    const Vectorized<uint64_t>& a, const Vectorized<uint64_t>& b) { 
  return svmin_u64_x(ptrue, a, b);                                                
}

template <>                                                                 
Vectorized<uint32_t> inline minimum(                                    
    const Vectorized<uint32_t>& a, const Vectorized<uint32_t>& b) { 
  return vminq_u32(a, b);                                                
}   

template <>                                                                 
Vectorized<uint16_t> inline minimum(                                    
    const Vectorized<uint16_t>& a, const Vectorized<uint16_t>& b) { 
  return vminq_u16(a, b);                                                
}   

template <>                                                                 
Vectorized<uint8_t> inline minimum(                                    
    const Vectorized<uint8_t>& a, const Vectorized<uint8_t>& b) { 
  return vminq_u8(a, b);                                                
}

template <>                                                                 
Vectorized<uint64_t> inline maximum(                                    
    const Vectorized<uint64_t>& a, const Vectorized<uint64_t>& b) { 
  return svmax_u64_x(ptrue, a, b);                                                
}

template <>                                                                 
Vectorized<uint32_t> inline maximum(                                    
    const Vectorized<uint32_t>& a, const Vectorized<uint32_t>& b) { 
  return vmaxq_u32(a, b);                                                
}   

template <>                                                                 
Vectorized<uint16_t> inline maximum(                                    
    const Vectorized<uint16_t>& a, const Vectorized<uint16_t>& b) { 
  return vmaxq_u16(a, b);                                                
}   

template <>                                                                 
Vectorized<uint8_t> inline maximum(                                    
    const Vectorized<uint8_t>& a, const Vectorized<uint8_t>& b) { 
  return vmaxq_u8(a, b);                                                
}

template <>                                                                 
Vectorized<uint64_t> inline clamp(                                      
    const Vectorized<uint64_t>& a,                                      
    const Vectorized<uint64_t>& min,                                    
    const Vectorized<uint64_t>& max) {                                  
  return minimum(max, maximum(min, a));                                     
}

template <>                                                                 
Vectorized<uint32_t> inline clamp(                                      
    const Vectorized<uint32_t>& a,                                      
    const Vectorized<uint32_t>& min,                                    
    const Vectorized<uint32_t>& max) {                                  
  return minimum(max, maximum(min, a));                                     
}

template <>                                                                 
Vectorized<uint16_t> inline clamp(                                      
    const Vectorized<uint16_t>& a,                                      
    const Vectorized<uint16_t>& min,                                    
    const Vectorized<uint16_t>& max) {                                  
  return minimum(max, maximum(min, a));                                     
}

template <>                                                                 
Vectorized<uint8_t> inline clamp(                                      
    const Vectorized<uint8_t>& a,                                      
    const Vectorized<uint8_t>& min,                                    
    const Vectorized<uint8_t>& max) {                                  
  return minimum(max, maximum(min, a));                                     
}

template <>                                                                 
Vectorized<uint64_t> inline clamp_max(                                  
    const Vectorized<uint64_t>& a,                                      
    const Vectorized<uint64_t>& max) {                                  
  return minimum(max, a);                                                   
}

template <>                                                                 
Vectorized<uint32_t> inline clamp_max(                                  
    const Vectorized<uint32_t>& a,                                      
    const Vectorized<uint32_t>& max) {                                  
  return minimum(max, a);                                                   
}

template <>                                                                 
Vectorized<uint16_t> inline clamp_max(                                  
    const Vectorized<uint16_t>& a,                                      
    const Vectorized<uint16_t>& max) {                                  
  return minimum(max, a);                                                   
}

template <>                                                                 
Vectorized<uint8_t> inline clamp_max(                                  
    const Vectorized<uint8_t>& a,                                      
    const Vectorized<uint8_t>& max) {                                  
  return minimum(max, a);                                                   
}

template <>                                                                 
Vectorized<uint64_t> inline clamp_min(                                  
    const Vectorized<uint64_t>& a,                                      
    const Vectorized<uint64_t>& min) {                                  
  return maximum(min, a);                                                   
} 

template <>                                                                 
Vectorized<uint32_t> inline clamp_min(                                  
    const Vectorized<uint32_t>& a,                                      
    const Vectorized<uint32_t>& min) {                                  
  return maximum(min, a);                                                   
} 

template <>                                                                 
Vectorized<uint16_t> inline clamp_min(                                  
    const Vectorized<uint16_t>& a,                                      
    const Vectorized<uint16_t>& min) {                                  
  return maximum(min, a);                                                   
} 

template <>                                                                 
Vectorized<uint8_t> inline clamp_min(                                  
    const Vectorized<uint8_t>& a,                                      
    const Vectorized<uint8_t>& min) {                                  
  return maximum(min, a);                                                   
}

template <int64_t mask>
  Vectorized<uint64_t> Vectorized<uint64_t>::blend(
      const Vectorized<uint64_t>& a,
      const Vectorized<uint64_t>& b) {
    // Build an array of flags: each bit of element is 1 if the corresponding bit in
    // 'mask' is set, 0 otherwise.
    uint64x2_t maskArray = {
      (mask & 1ULL) ? 0xFFFFFFFFFFFFFFFF : 0, 
      (mask & 2ULL) ? 0xFFFFFFFFFFFFFFFF : 0};
    // Use BSL to select elements from b where the mask is 1, else from a
    return vbslq_u64(maskArray, b.values, a.values);
}

template <int64_t mask>
  Vectorized<uint32_t> Vectorized<uint32_t>::blend(
      const Vectorized<uint32_t>& a,
      const Vectorized<uint32_t>& b) {
    // Build an array of flags: each bit of element is 1 if the corresponding bit in
    // 'mask' is set, 0 otherwise.
    uint32x4_t maskArray = {
      (mask &     1ULL) ? 0xFFFFFFFF : 0, 
      (mask &     2ULL) ? 0xFFFFFFFF : 0, 
      (mask &     4ULL) ? 0xFFFFFFFF : 0, 
      (mask &     8ULL) ? 0xFFFFFFFF : 0};
    // Use BSL to select elements from b where the mask is 1, else from a
    return vbslq_u32(maskArray, b.values, a.values);
}

template <int64_t mask>
  Vectorized<uint16_t> Vectorized<uint16_t>::blend(
      const Vectorized<uint16_t>& a,
      const Vectorized<uint16_t>& b) {
    // Build an array of flags: each bit of element is 1 if the corresponding bit in
    // 'mask' is set, 0 otherwise.
    uint16x8_t maskArray = {
      (mask &     1ULL) ? 0xFFFF : 0, 
      (mask &     2ULL) ? 0xFFFF : 0, 
      (mask &     4ULL) ? 0xFFFF : 0, 
      (mask &     8ULL) ? 0xFFFF : 0,
      (mask &    16ULL) ? 0xFFFF : 0, 
      (mask &    32ULL) ? 0xFFFF : 0, 
      (mask &    64ULL) ? 0xFFFF : 0, 
      (mask &   128ULL) ? 0xFFFF : 0};
    // Use BSL to select elements from b where the mask is 1, else from a
    return vbslq_u16(maskArray, b.values, a.values);
}

template <int64_t mask>
  Vectorized<uint8_t> Vectorized<uint8_t>::blend(
      const Vectorized<uint8_t>& a,
      const Vectorized<uint8_t>& b) {
    // Build an array of flags: each bit of element is 1 if the corresponding bit in
    // 'mask' is set, 0 otherwise.
    uint8x16_t maskArray = {
      (mask &     1ULL) ? 0xFF : 0, 
      (mask &     2ULL) ? 0xFF : 0, 
      (mask &     4ULL) ? 0xFF : 0, 
      (mask &     8ULL) ? 0xFF : 0,
      (mask &    16ULL) ? 0xFF : 0, 
      (mask &    32ULL) ? 0xFF : 0, 
      (mask &    64ULL) ? 0xFF : 0, 
      (mask &   128ULL) ? 0xFF : 0,
      (mask &   256ULL) ? 0xFF : 0, 
      (mask &   512ULL) ? 0xFF : 0, 
      (mask &  1024ULL) ? 0xFF : 0, 
      (mask &  2048ULL) ? 0xFF : 0,
      (mask &  4096ULL) ? 0xFF : 0, 
      (mask &  8192ULL) ? 0xFF : 0, 
      (mask & 16384ULL) ? 0xFF : 0, 
      (mask & 32768ULL) ? 0xFF : 0};
    // Use BSL to select elements from b where the mask is 1, else from a
    return vbslq_u8(maskArray, b.values, a.values);
}

template <typename step_t>                                                
inline Vectorized<uint64_t> Vectorized<uint64_t>::arange(                                   
    uint64_t base,                                                
    step_t step) {                               
      const Vectorized<uint64_t> base_vec(base);                        
      const Vectorized<uint64_t> step_vec(step);                        
      const Vectorized<uint64_t> step_sizes = svindex_u64(0, 1);     
      return svmla_u64_x(ptrue, base_vec, step_sizes, step_vec);                
}

template <typename step_t>                                                
inline Vectorized<uint32_t> Vectorized<uint32_t>::arange(                                   
    uint32_t base,                                                
    step_t step) {                               
      const Vectorized<uint32_t> base_vec(base);                        
      const Vectorized<uint32_t> step_vec(step);                        
      const Vectorized<uint32_t> step_sizes = svindex_u32(0, 1);     
      return vmlaq_u32(base_vec, step_sizes, step_vec);                
}

template <typename step_t>                                                
inline Vectorized<uint16_t> Vectorized<uint16_t>::arange(                                   
    uint16_t base,                                                
    step_t step) {                               
      const Vectorized<uint16_t> base_vec(base);                        
      const Vectorized<uint16_t> step_vec(step);                        
      const Vectorized<uint16_t> step_sizes = svindex_u16(0, 1);     
      return vmlaq_u16(base_vec, step_sizes, step_vec);                
} 

template <typename step_t>                                                
inline Vectorized<uint8_t> Vectorized<uint8_t>::arange(                                   
    uint8_t base,                                                
    step_t step) {                               
      const Vectorized<uint8_t> base_vec(base);                        
      const Vectorized<uint8_t> step_vec(step);                        
      const Vectorized<uint8_t> step_sizes = svindex_u8(0, 1);     
      return vmlaq_u8(base_vec, step_sizes, step_vec);                
} 

template <>
Vectorized<uint64_t> inline operator<<(
    const Vectorized<uint64_t>& a,
    const Vectorized<uint64_t>& b) {
  return vshlq_u64(a, b);
}

template <>
Vectorized<uint32_t> inline operator<<(
    const Vectorized<uint32_t>& a,
    const Vectorized<uint32_t>& b) {
  return vshlq_u32(a, b);
}

template <>
Vectorized<uint16_t> inline operator<<(
    const Vectorized<uint16_t>& a,
    const Vectorized<uint16_t>& b) {
  return vshlq_u16(a, b);
}

template <>
Vectorized<uint8_t> inline operator<<(
    const Vectorized<uint8_t>& a,
    const Vectorized<uint8_t>& b) {
  return vshlq_u8(a, b);
}

template <>
Vectorized<uint64_t> inline operator>>(
    const Vectorized<uint64_t>& a,
    const Vectorized<uint64_t>& b) {
  return svlsr_u64_x(ptrue, a, b);
}

template <>
Vectorized<uint32_t> inline operator>>(
    const Vectorized<uint32_t>& a,
    const Vectorized<uint32_t>& b) {
  return svlsr_u32_x(ptrue, a, b);
}

template <>
Vectorized<uint16_t> inline operator>>(
    const Vectorized<uint16_t>& a,
    const Vectorized<uint16_t>& b) {
  return svlsr_u16_x(ptrue, a, b);
}

template <>
Vectorized<uint8_t> inline operator>>(
    const Vectorized<uint8_t>& a,
    const Vectorized<uint8_t>& b) {
  return svlsr_u8_x(ptrue, a, b);
}

#endif // defined(CPU_CAPABILITY_SVE)

} // namespace CPU_CAPABILITY
} // namespace at::vec
