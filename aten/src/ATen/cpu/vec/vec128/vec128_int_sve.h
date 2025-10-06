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

#define VEC_INT_SVE_TEMPLATE(vl, bit)                                         \
  template <>                                                                 \
  struct is_vec_specialized_for<int##bit##_t> : std::bool_constant<true> {};  \
                                                                              \
  template <>                                                                 \
  class Vectorized<int##bit##_t> {                                            \
   using neon_type = int##bit##x##vl##_t;                                     \
   private:                                                                   \
    neon_type values;                                                         \
                                                                              \
   public:                                                                    \
    using value_type = int##bit##_t;                                          \
    using size_type = int;                                                    \
    static constexpr size_type size() {                                       \
      return vl;                                                              \
    }                                                                         \
    Vectorized() {                                                            \
      values = vdupq_n_s##bit(0);                                             \
    }                                                                         \
    Vectorized(neon_type v) : values(v) {}                                    \
    Vectorized(svint##bit##_t v) : values(svget_neonq(v)) {}                  \
    Vectorized(int##bit##_t val) {                                            \
      values = svget_neonq(svdup_n_s##bit(val));                              \
    }                                                                         \
    template <                                                                \
        typename... Args,                                                     \
        typename = std::enable_if_t<(sizeof...(Args) == size())>>             \
    Vectorized(Args... vals) {                                                \
      __at_align__ int##bit##_t buffer[size()] = {vals...};                   \
      values = vld1q_s##bit(buffer);                                          \
    }                                                                         \
    operator neon_type() const {                                              \
      return values;                                                          \
    }                                                                         \
    operator svint##bit##_t() const {                                         \
      return svset_neonq(svundef_s##bit(), values);                           \
    }                                                                         \
    template <int64_t mask>                                                   \
    static Vectorized<int##bit##_t> blend(                                    \
      const Vectorized<int##bit##_t>& a,                                      \
      const Vectorized<int##bit##_t>& b);                                     \
    static Vectorized<int##bit##_t> blendv(                                   \
        const Vectorized<int##bit##_t>& a,                                    \
        const Vectorized<int##bit##_t>& b,                                    \
        const Vectorized<int##bit##_t>& mask_) {                              \
      return vbslq_s##bit(vreinterpretq_u##bit##_s##bit(mask_.values), b, a); \
    }                                                                         \
    template <typename step_t>                                                \
    static Vectorized<int##bit##_t> arange(                                   \
        value_type base = 0,                                                  \
        step_t step = static_cast<step_t>(1));                                \
    static Vectorized<int##bit##_t> set(                                      \
        const Vectorized<int##bit##_t>& a,                                    \
        const Vectorized<int##bit##_t>& b,                                    \
        int##bit##_t count = size()) {                                        \
      if (count == 0) {                                                       \
        return a;                                                             \
      } else if (count < size()) {                                            \
        return svsel_s##bit(svwhilelt_b##bit(0ull, count), b, a);             \
      }                                                                       \
      return b;                                                               \
    }                                                                         \
    static Vectorized<int##bit##_t> loadu(                                    \
        const void* ptr,                                                      \
        int64_t count = size()) {                                             \
      if (count == size())                                                    \
        return vld1q_s##bit(reinterpret_cast<const int##bit##_t*>(ptr));      \
      svbool_t pg = svwhilelt_b##bit(0ull, count);                            \
      return svld1_s##bit(pg, reinterpret_cast<const int##bit##_t*>(ptr));    \
    }                                                                         \
    void store(void* ptr, int64_t count = size()) const {                     \
      if (count == size()) {                                                  \
        vst1q_s##bit(reinterpret_cast<int##bit##_t*>(ptr), values);           \
      } else {                                                                \
        svbool_t pg = svwhilelt_b##bit(0ull, count);                          \
        auto dstPtr = reinterpret_cast<int##bit##_t*>(ptr);                   \
        svst1_s##bit(pg, dstPtr, svset_neonq(svundef_s##bit(), values));      \
      }                                                                       \
    }                                                                         \
    const int##bit##_t& operator[](int idx) const = delete;                   \
    int##bit##_t& operator[](int idx) = delete;                               \
    Vectorized<int##bit##_t> abs() const {                                    \
      return vabsq_s##bit(values);                                            \
    }                                                                         \
    Vectorized<int##bit##_t> real() const {                                   \
      return values;                                                          \
    }                                                                         \
    Vectorized<int##bit##_t> imag() const {                                   \
      return svdup_n_s##bit(0);                                               \
    }                                                                         \
    Vectorized<int##bit##_t> conj() const {                                   \
      return values;                                                          \
    }                                                                         \
    Vectorized<int##bit##_t> neg() const {                                    \
      return vnegq_s##bit(values);                                            \
    }                                                                         \
    int##bit##_t reduce_add() const {                                         \
    return vaddvq_s##bit(values);                                             \
    }                                                                         \
    int##bit##_t reduce_max() const;                                          \
    Vectorized<int##bit##_t> operator==(                                      \
        const Vectorized<int##bit##_t>& other) const {                        \
      return Vectorized<value_type>(                                          \
        vreinterpretq_s##bit##_u##bit(vceqq_s##bit(values, other.values)));   \
    }                                                                         \
    Vectorized<int##bit##_t> operator!=(                                      \
        const Vectorized<int##bit##_t>& other) const;                         \
    Vectorized<int##bit##_t> operator<(                                       \
        const Vectorized<int##bit##_t>& other) const {                        \
      return Vectorized<value_type>(                                          \
        vreinterpretq_s##bit##_u##bit(vcltq_s##bit(values, other.values)));   \
    }                                                                         \
    Vectorized<int##bit##_t> operator<=(                                      \
        const Vectorized<int##bit##_t>& other) const {                        \
      return Vectorized<value_type>(                                          \
        vreinterpretq_s##bit##_u##bit(vcleq_s##bit(values, other.values)));   \
    }                                                                         \
    Vectorized<int##bit##_t> operator>(                                       \
        const Vectorized<int##bit##_t>& other) const {                        \
      return Vectorized<value_type>(                                          \
        vreinterpretq_s##bit##_u##bit(vcgtq_s##bit(values, other.values)));   \
    }                                                                         \
    Vectorized<int##bit##_t> operator>=(                                      \
        const Vectorized<int##bit##_t>& other) const {                        \
      return Vectorized<value_type>(                                          \
        vreinterpretq_s##bit##_u##bit(vcgeq_s##bit(values, other.values)));   \
    }                                                                         \
    Vectorized<int##bit##_t> eq(const Vectorized<int##bit##_t>& other) const; \
    Vectorized<int##bit##_t> ne(const Vectorized<int##bit##_t>& other) const; \
    Vectorized<int##bit##_t> gt(const Vectorized<int##bit##_t>& other) const; \
    Vectorized<int##bit##_t> ge(const Vectorized<int##bit##_t>& other) const; \
    Vectorized<int##bit##_t> lt(const Vectorized<int##bit##_t>& other) const; \
    Vectorized<int##bit##_t> le(const Vectorized<int##bit##_t>& other) const; \
  };                                                                          \
  template <>                                                                 \
  Vectorized<int##bit##_t> inline operator+(                                  \
      const Vectorized<int##bit##_t>& a, const Vectorized<int##bit##_t>& b) { \
    return vaddq_s##bit(a, b);                                                \
  }                                                                           \
  template <>                                                                 \
  Vectorized<int##bit##_t> inline operator-(                                  \
      const Vectorized<int##bit##_t>& a, const Vectorized<int##bit##_t>& b) { \
    return vsubq_s##bit(a, b);                                                \
  }                                                                           \
  template <>                                                                 \
  Vectorized<int##bit##_t> inline operator&(                                  \
      const Vectorized<int##bit##_t>& a, const Vectorized<int##bit##_t>& b) { \
    return vandq_s##bit(a, b);                                                \
  }                                                                           \
  template <>                                                                 \
  Vectorized<int##bit##_t> inline operator|(                                  \
      const Vectorized<int##bit##_t>& a, const Vectorized<int##bit##_t>& b) { \
    return vorrq_s##bit(a, b);                                                \
  }                                                                           \
  template <>                                                                 \
  Vectorized<int##bit##_t> inline operator^(                                  \
      const Vectorized<int##bit##_t>& a, const Vectorized<int##bit##_t>& b) { \
    return veorq_s##bit(a, b);                                                \
  }                                                                           \
  Vectorized<int##bit##_t> inline Vectorized<int##bit##_t>::eq(               \
      const Vectorized<int##bit##_t>& other) const {                          \
    return svand_n_s##bit##_x(ptrue, *this == other, 1);                      \
  }                                                                           \
  Vectorized<int##bit##_t> inline Vectorized<int##bit##_t>::ne(               \
      const Vectorized<int##bit##_t>& other) const {                          \
    return svand_n_s##bit##_x(ptrue, *this != other, 1);                      \
  }                                                                           \
  Vectorized<int##bit##_t> inline Vectorized<int##bit##_t>::gt(               \
      const Vectorized<int##bit##_t>& other) const {                          \
    return svand_n_s##bit##_x(ptrue, *this > other, 1);                       \
  }                                                                           \
  Vectorized<int##bit##_t> inline Vectorized<int##bit##_t>::ge(               \
      const Vectorized<int##bit##_t>& other) const {                          \
    return svand_n_s##bit##_x(ptrue, *this >= other, 1);                      \
  }                                                                           \
  Vectorized<int##bit##_t> inline Vectorized<int##bit##_t>::lt(               \
      const Vectorized<int##bit##_t>& other) const {                          \
    return svand_n_s##bit##_x(ptrue, *this < other, 1);                       \
  }                                                                           \
  Vectorized<int##bit##_t> inline Vectorized<int##bit##_t>::le(               \
      const Vectorized<int##bit##_t>& other) const {                          \
    return svand_n_s##bit##_x(ptrue, *this <= other, 1);                      \
  }

VEC_INT_SVE_TEMPLATE(2, 64)
VEC_INT_SVE_TEMPLATE(4, 32)
VEC_INT_SVE_TEMPLATE(8, 16)
VEC_INT_SVE_TEMPLATE(16, 8)

inline int64_t Vectorized<int64_t>::reduce_max() const {                                         
  return svmaxv_s64(ptrue, svset_neonq(svundef_s64(), values));                                 
}

inline int32_t Vectorized<int32_t>::reduce_max() const {                                         
  return vmaxvq_s32(values);                                 
}

inline int16_t Vectorized<int16_t>::reduce_max() const {                                         
  return vmaxvq_s16(values);                                 
}

inline int8_t Vectorized<int8_t>::reduce_max() const {                                         
  return vmaxvq_s8(values);                                 
}                                                                         

template <>                                                                 
Vectorized<int64_t> inline operator*(                                  
    const Vectorized<int64_t>& a, const Vectorized<int64_t>& b) { 
  return svmul_s64_x(ptrue, a, b);                                                
} 

template <>                                                                 
Vectorized<int32_t> inline operator*(                                  
    const Vectorized<int32_t>& a, const Vectorized<int32_t>& b) { 
  return vmulq_s32(a, b);                                                
} 

template <>                                                                 
Vectorized<int16_t> inline operator*(                                  
    const Vectorized<int16_t>& a, const Vectorized<int16_t>& b) { 
  return vmulq_s16(a, b);                                                
} 

template <>                                                                 
Vectorized<int8_t> inline operator*(                                  
    const Vectorized<int8_t>& a, const Vectorized<int8_t>& b) { 
  return vmulq_s8(a, b);                                                
}  

template <>
Vectorized<int64_t> inline operator/(
    const Vectorized<int64_t>& a,
    const Vectorized<int64_t>& b) {
  return svdiv_s64_x(ptrue, a, b);
}

template <>
Vectorized<int32_t> inline operator/(
    const Vectorized<int32_t>& a,
    const Vectorized<int32_t>& b) {
  return svdiv_s32_x(ptrue, a, b);
}

template <>
Vectorized<int16_t> inline operator/(
    const Vectorized<int16_t>& a,
    const Vectorized<int16_t>& b) {
  Vectorized<int32_t> highBitsA = vmovl_high_s16(a);
  Vectorized<int32_t> highBitsB = vmovl_high_s16(b);
  Vectorized<int32_t> lowBitsA = vmovl_s16(vget_low_s16(a));
  Vectorized<int32_t> lowBitsB = vmovl_s16(vget_low_s16(b));
  int32x4_t highBitsResult = highBitsA / highBitsB;
  int32x4_t lowBitsResult = lowBitsA / lowBitsB;
  return vuzp1q_s16(vreinterpretq_s16_s32(lowBitsResult), vreinterpretq_s16_s32(highBitsResult));
}

template <>
Vectorized<int8_t> inline operator/(
    const Vectorized<int8_t>& a,
    const Vectorized<int8_t>& b) {
  Vectorized<int16_t> highBitsA = vmovl_high_s8(a);
  Vectorized<int16_t> highBitsB = vmovl_high_s8(b);
  Vectorized<int16_t> lowBitsA = vmovl_s8(vget_low_s8(a));
  Vectorized<int16_t> lowBitsB = vmovl_s8(vget_low_s8(b));
  int16x8_t highBitsResult = highBitsA / highBitsB;
  int16x8_t lowBitsResult = lowBitsA / lowBitsB;
  return vuzp1q_s8(vreinterpretq_s8_s16(lowBitsResult), vreinterpretq_s8_s16(highBitsResult));
}                                                                           

template <>
inline Vectorized<int64_t> operator~(                                  
    const Vectorized<int64_t>& a) {
  int64x2_t val = a;                                    
  return ~val;                                                   
} 
 
template <>
inline Vectorized<int32_t> operator~(                                  
    const Vectorized<int32_t>& a) {                                    
  return vmvnq_s32(a);                                                   
}

template <>                                                                
inline Vectorized<int16_t> operator~(                                  
    const Vectorized<int16_t>& a) {                                    
  return vmvnq_s16(a);                                                   
} 

template <>
inline Vectorized<int8_t> operator~(                                  
    const Vectorized<int8_t>& a) {                                    
  return vmvnq_s8(a);                                                   
}

inline Vectorized<int64_t> Vectorized<int64_t>::operator!=(                                  
    const Vectorized<int64_t>& other) const {                                    
  return  ~(*this == other);                                                   
} 
                                                                 
inline Vectorized<int32_t> Vectorized<int32_t>::operator!=(                                  
    const Vectorized<int32_t>& other) const {                                    
  return  ~(*this == other);                                                   
} 
                                                                
inline Vectorized<int16_t> Vectorized<int16_t>::operator!=(                                  
    const Vectorized<int16_t>& other) const {                                    
  return  ~(*this == other);                                                   
} 
                                                              
inline Vectorized<int8_t> Vectorized<int8_t>::operator!=(                                  
    const Vectorized<int8_t>& other) const {                                    
  return  ~(*this == other);                                                   
}

template <>                                                                 
Vectorized<int64_t> inline minimum(                                    
    const Vectorized<int64_t>& a, const Vectorized<int64_t>& b) { 
  return svmin_s64_x(ptrue, a, b);                                                
}

template <>                                                                 
Vectorized<int32_t> inline minimum(                                    
    const Vectorized<int32_t>& a, const Vectorized<int32_t>& b) { 
  return vminq_s32(a, b);                                                
}   

template <>                                                                 
Vectorized<int16_t> inline minimum(                                    
    const Vectorized<int16_t>& a, const Vectorized<int16_t>& b) { 
  return vminq_s16(a, b);                                                
}   

template <>                                                                 
Vectorized<int8_t> inline minimum(                                    
    const Vectorized<int8_t>& a, const Vectorized<int8_t>& b) { 
  return vminq_s8(a, b);                                                
}

template <>                                                                 
Vectorized<int64_t> inline maximum(                                    
    const Vectorized<int64_t>& a, const Vectorized<int64_t>& b) { 
  return svmax_s64_x(ptrue, a, b);                                                
}

template <>                                                                 
Vectorized<int32_t> inline maximum(                                    
    const Vectorized<int32_t>& a, const Vectorized<int32_t>& b) { 
  return vmaxq_s32(a, b);                                                
}   

template <>                                                                 
Vectorized<int16_t> inline maximum(                                    
    const Vectorized<int16_t>& a, const Vectorized<int16_t>& b) { 
  return vmaxq_s16(a, b);                                                
}   

template <>                                                                 
Vectorized<int8_t> inline maximum(                                    
    const Vectorized<int8_t>& a, const Vectorized<int8_t>& b) { 
  return vmaxq_s8(a, b);                                                
}

template <>                                                                 
Vectorized<int64_t> inline clamp(                                      
    const Vectorized<int64_t>& a,                                      
    const Vectorized<int64_t>& min,                                    
    const Vectorized<int64_t>& max) {                                  
  return minimum(max, maximum(min, a));                                     
}

template <>                                                                 
Vectorized<int32_t> inline clamp(                                      
    const Vectorized<int32_t>& a,                                      
    const Vectorized<int32_t>& min,                                    
    const Vectorized<int32_t>& max) {                                  
  return minimum(max, maximum(min, a));                                     
}

template <>                                                                 
Vectorized<int16_t> inline clamp(                                      
    const Vectorized<int16_t>& a,                                      
    const Vectorized<int16_t>& min,                                    
    const Vectorized<int16_t>& max) {                                  
  return minimum(max, maximum(min, a));                                     
}

template <>                                                                 
Vectorized<int8_t> inline clamp(                                      
    const Vectorized<int8_t>& a,                                      
    const Vectorized<int8_t>& min,                                    
    const Vectorized<int8_t>& max) {                                  
  return minimum(max, maximum(min, a));                                     
}

template <>                                                                 
Vectorized<int64_t> inline clamp_max(                                  
    const Vectorized<int64_t>& a,                                      
    const Vectorized<int64_t>& max) {                                  
  return minimum(max, a);                                                   
}

template <>                                                                 
Vectorized<int32_t> inline clamp_max(                                  
    const Vectorized<int32_t>& a,                                      
    const Vectorized<int32_t>& max) {                                  
  return minimum(max, a);                                                   
}

template <>                                                                 
Vectorized<int16_t> inline clamp_max(                                  
    const Vectorized<int16_t>& a,                                      
    const Vectorized<int16_t>& max) {                                  
  return minimum(max, a);                                                   
}

template <>                                                                 
Vectorized<int8_t> inline clamp_max(                                  
    const Vectorized<int8_t>& a,                                      
    const Vectorized<int8_t>& max) {                                  
  return minimum(max, a);                                                   
}

template <>                                                                 
Vectorized<int64_t> inline clamp_min(                                  
    const Vectorized<int64_t>& a,                                      
    const Vectorized<int64_t>& min) {                                  
  return maximum(min, a);                                                   
} 

template <>                                                                 
Vectorized<int32_t> inline clamp_min(                                  
    const Vectorized<int32_t>& a,                                      
    const Vectorized<int32_t>& min) {                                  
  return maximum(min, a);                                                   
} 

template <>                                                                 
Vectorized<int16_t> inline clamp_min(                                  
    const Vectorized<int16_t>& a,                                      
    const Vectorized<int16_t>& min) {                                  
  return maximum(min, a);                                                   
} 

template <>                                                                 
Vectorized<int8_t> inline clamp_min(                                  
    const Vectorized<int8_t>& a,                                      
    const Vectorized<int8_t>& min) {                                  
  return maximum(min, a);                                                   
}

template <int64_t mask>
  Vectorized<int64_t> Vectorized<int64_t>::blend(
      const Vectorized<int64_t>& a,
      const Vectorized<int64_t>& b) {
    // Build an array of flags: each bit of element is 1 if the corresponding bit in
    // 'mask' is set, 0 otherwise.
    uint64x2_t maskArray = {
      (mask & 1ULL) ? 0xFFFFFFFFFFFFFFFF : 0, 
      (mask & 2ULL) ? 0xFFFFFFFFFFFFFFFF : 0};
    // Use BSL to select elements from b where the mask is 1, else from a
    return vbslq_s64(maskArray, b.values, a.values);
}

template <int64_t mask>
  Vectorized<int32_t> Vectorized<int32_t>::blend(
      const Vectorized<int32_t>& a,
      const Vectorized<int32_t>& b) {
    // Build an array of flags: each bit of element is 1 if the corresponding bit in
    // 'mask' is set, 0 otherwise.
    uint32x4_t maskArray = {
      (mask &     1ULL) ? 0xFFFFFFFF : 0, 
      (mask &     2ULL) ? 0xFFFFFFFF : 0, 
      (mask &     4ULL) ? 0xFFFFFFFF : 0, 
      (mask &     8ULL) ? 0xFFFFFFFF : 0};
    // Use BSL to select elements from b where the mask is 1, else from a
    return vbslq_s32(maskArray, b.values, a.values);
}

template <int64_t mask>
  Vectorized<int16_t> Vectorized<int16_t>::blend(
      const Vectorized<int16_t>& a,
      const Vectorized<int16_t>& b) {
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
    return vbslq_s16(maskArray, b.values, a.values);
}

template <int64_t mask>
  Vectorized<int8_t> Vectorized<int8_t>::blend(
      const Vectorized<int8_t>& a,
      const Vectorized<int8_t>& b) {
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
    return vbslq_s8(maskArray, b.values, a.values);
}

template <typename step_t>                                                
inline Vectorized<int64_t> Vectorized<int64_t>::arange(                                   
    int64_t base,                                                
    step_t step) {                               
      const Vectorized<int64_t> base_vec(base);                        
      const Vectorized<int64_t> step_vec(step);                        
      const Vectorized<int64_t> step_sizes = svindex_s64(0, 1);     
      return svmla_s64_x(ptrue, base_vec, step_sizes, step_vec);                
}

template <typename step_t>                                                
inline Vectorized<int32_t> Vectorized<int32_t>::arange(                                   
    int32_t base,                                                
    step_t step) {                               
      const Vectorized<int32_t> base_vec(base);                        
      const Vectorized<int32_t> step_vec(step);                        
      const Vectorized<int32_t> step_sizes = svindex_s32(0, 1);     
      return vmlaq_s32(base_vec, step_sizes, step_vec);                
}

template <typename step_t>                                                
inline Vectorized<int16_t> Vectorized<int16_t>::arange(                                   
    int16_t base,                                                
    step_t step) {                               
      const Vectorized<int16_t> base_vec(base);                        
      const Vectorized<int16_t> step_vec(step);                        
      const Vectorized<int16_t> step_sizes = svindex_s16(0, 1);     
      return vmlaq_s16(base_vec, step_sizes, step_vec);                
} 

template <typename step_t>                                                
inline Vectorized<int8_t> Vectorized<int8_t>::arange(                                   
    int8_t base,                                                
    step_t step) {                               
      const Vectorized<int8_t> base_vec(base);                        
      const Vectorized<int8_t> step_vec(step);                        
      const Vectorized<int8_t> step_sizes = svindex_s8(0, 1);     
      return vmlaq_s8(base_vec, step_sizes, step_vec);                
} 

template <>
inline void convert(const int32_t* src, int64_t* dst, int64_t n) {
  constexpr uint64_t oneRegElemCount = Vectorized<int64_t>::size();
  constexpr uint64_t twoRegsElemCount = oneRegElemCount * 2;
  const uint64_t count = static_cast<uint64_t>(n);
  const uint64_t remainder = count % twoRegsElemCount;
  for (uint64_t iters = count / twoRegsElemCount; iters > 0; --iters) {
    auto vec1 = svget_neonq(svld1sw_s64(svptrue_b8(), src));
    auto vec2 = svget_neonq(svld1sw_s64(svptrue_b8(), src + oneRegElemCount));
    vst1q_s64(dst, vec1);
    vst1q_s64(dst + oneRegElemCount, vec2);
    src += twoRegsElemCount;
    dst += twoRegsElemCount;
  }
  if (remainder > 0) {
    svbool_t pa = svwhilelt_b64_u64(0, remainder);
    svbool_t pb = svwhilelt_b64_u64(oneRegElemCount, remainder);
    auto vec1 = svld1sw_s64(pa, src);
    auto vec2 = svld1sw_s64(pb, src + oneRegElemCount);
    svst1_s64(pa, dst, vec1);
    svst1_s64(pb, dst + oneRegElemCount, vec2);
  }
}

template <>
inline void convert(const int64_t* src, float* dst, int64_t n) {
  constexpr uint64_t oneRegElemCount = Vectorized<int64_t>::size();
  constexpr uint64_t twoRegsElemCount = oneRegElemCount * 2;
  constexpr uint64_t fourRegsElemCount = twoRegsElemCount * 2;
  const uint64_t count = static_cast<uint64_t>(n);
  int32_t* dstPtr = reinterpret_cast<int32_t*>(dst);
  for (uint64_t iters = count / fourRegsElemCount; iters > 0; --iters) {
    auto vec1 = vld1q_s64(src);
    auto vec2 = vld1q_s64(src + oneRegElemCount);
    auto vec3 = vld1q_s64(src + twoRegsElemCount);
    auto vec4 = vld1q_s64(src + (twoRegsElemCount + oneRegElemCount));
    auto convertedVec1 = svcvt_f32_s64_x(ptrue, svset_neonq(svundef_s64(), vec1));
    auto convertedVec2 = svcvt_f32_s64_x(ptrue, svset_neonq(svundef_s64(), vec2));
    auto convertedVec3 = svcvt_f32_s64_x(ptrue, svset_neonq(svundef_s64(), vec3));
    auto convertedVec4 = svcvt_f32_s64_x(ptrue, svset_neonq(svundef_s64(), vec4));
    svst1w_s64(ptrue, dstPtr, svreinterpret_s64_f32(convertedVec1));
    svst1w_s64(ptrue, dstPtr + oneRegElemCount, svreinterpret_s64_f32(convertedVec2));
    svst1w_s64(ptrue, dstPtr + twoRegsElemCount, svreinterpret_s64_f32(convertedVec3));
    svst1w_s64(ptrue, dstPtr + (twoRegsElemCount + oneRegElemCount), svreinterpret_s64_f32(convertedVec4));
    src += fourRegsElemCount;
    dstPtr += fourRegsElemCount;
  }
  dst = reinterpret_cast<float*>(dstPtr);
#pragma clang loop vectorize(disable)
#pragma clang loop unroll(disable)
  for (uint64_t remainder = count % fourRegsElemCount; remainder > 0; --remainder) {
    *dst = *src;
    src += 1;
    dst += 1;
  }
}

template <>
inline void convert(const int32_t* src, float* dst, int64_t n) {
  constexpr uint64_t oneRegElemCount = Vectorized<int32_t>::size();
  constexpr uint64_t twoRegsElemCount = oneRegElemCount * 2;
  constexpr uint64_t fourRegsElemCount = twoRegsElemCount * 2;
  const uint64_t count = static_cast<uint64_t>(n);
  for (uint64_t iters = count / fourRegsElemCount; iters > 0; --iters) {
    auto vec1 = vld1q_s32(src);
    auto vec2 = vld1q_s32(src + oneRegElemCount);
    auto vec3 = vld1q_s32(src + twoRegsElemCount);
    auto vec4 = vld1q_s32(src + (twoRegsElemCount + oneRegElemCount));
    auto convertedVec1 = vcvtq_f32_s32(vec1);
    auto convertedVec2 = vcvtq_f32_s32(vec2);
    auto convertedVec3 = vcvtq_f32_s32(vec3);
    auto convertedVec4 = vcvtq_f32_s32(vec4);
    vst1q_f32(dst, convertedVec1);
    vst1q_f32(dst + oneRegElemCount, convertedVec2);
    vst1q_f32(dst + twoRegsElemCount, convertedVec3);
    vst1q_f32(dst + (twoRegsElemCount + oneRegElemCount), convertedVec4);
    src += fourRegsElemCount;
    dst += fourRegsElemCount;
  }
#pragma clang loop vectorize(disable)
#pragma clang loop unroll(disable)
  for (uint64_t remainder = count % fourRegsElemCount; remainder > 0; --remainder) {
    *dst = *src;
    src += 1;
    dst += 1;
  }
}

template <>
inline void convert(const int32_t* src, double* dst, int64_t n) {
  constexpr uint64_t oneRegElemCount = Vectorized<int64_t>::size();
  constexpr uint64_t twoRegsElemCount = oneRegElemCount * 2;
  constexpr uint64_t fourRegsElemCount = twoRegsElemCount * 2;
  const uint64_t count = static_cast<uint64_t>(n);
  for (uint64_t iters = count / fourRegsElemCount; iters > 0; --iters) {
    auto vec1 = svld1sw_s64(ptrue, src);
    auto vec2 = svld1sw_s64(ptrue, src + oneRegElemCount);
    auto vec3 = svld1sw_s64(ptrue, src + twoRegsElemCount);
    auto vec4 = svld1sw_s64(ptrue, src + (twoRegsElemCount + oneRegElemCount));
    auto convertedVec1 = vcvtq_f64_s64(svget_neonq(vec1));
    auto convertedVec2 = vcvtq_f64_s64(svget_neonq(vec2));
    auto convertedVec3 = vcvtq_f64_s64(svget_neonq(vec3));
    auto convertedVec4 = vcvtq_f64_s64(svget_neonq(vec4));
    vst1q_f64(dst, convertedVec1);
    vst1q_f64(dst + oneRegElemCount, convertedVec2);
    vst1q_f64(dst + twoRegsElemCount, convertedVec3);
    vst1q_f64(dst + (twoRegsElemCount + oneRegElemCount), convertedVec4);
    src += fourRegsElemCount;
    dst += fourRegsElemCount;
  }
#pragma clang loop vectorize(disable)
#pragma clang loop unroll(disable)
  for (uint64_t remainder = count % fourRegsElemCount; remainder > 0; --remainder) {
    *dst = *src;
    src += 1;
    dst += 1;
  }
}

template <>
inline void convert(const bool* src, int64_t* dst, int64_t n) {
  constexpr uint64_t oneRegElemCount = Vectorized<int64_t>::size();
  constexpr uint64_t twoRegsElemCount = oneRegElemCount * 2;
  const uint64_t count = static_cast<uint64_t>(n);
  const uint64_t remainder = count % twoRegsElemCount;
  const uint8_t* srcPtr = reinterpret_cast<const uint8_t*>(src);
  for (uint64_t iters = count / twoRegsElemCount; iters > 0; --iters) {
    auto vec1 = svget_neonq(svld1ub_s64(svptrue_b8(), srcPtr));
    auto vec2 = svget_neonq(svld1ub_s64(svptrue_b8(), srcPtr + oneRegElemCount));
    auto vec1Mask = vtstq_s64(vec1, vec1);
    auto vec2Mask = vtstq_s64(vec2, vec2);
    vec1Mask = svget_neonq(svand_n_s64_x(svptrue_b8(), svset_neonq(svundef_s64(), vec1Mask), 1));
    vec2Mask = svget_neonq(svand_n_s64_x(svptrue_b8(), svset_neonq(svundef_s64(), vec2Mask), 1));
    vst1q_s64(dst, vec1Mask);
    vst1q_s64(dst + oneRegElemCount, vec2Mask);
    srcPtr += twoRegsElemCount;
    dst += twoRegsElemCount;
  }
  if (remainder > 0) {
    svbool_t pa = svwhilelt_b64_u64(0, remainder);
    svbool_t pb = svwhilelt_b64_u64(oneRegElemCount, remainder);
    auto vec1 = svget_neonq(svld1ub_s64(pa, srcPtr));
    auto vec2 = svget_neonq(svld1ub_s64(pb, srcPtr + oneRegElemCount));
    auto vec1Mask = vtstq_s64(vec1, vec1);
    auto vec2Mask = vtstq_s64(vec2, vec2);
    auto vec1MaskSv = svand_n_s64_x(svptrue_b8(), svset_neonq(svundef_s64(), vec1Mask), 1);
    auto vec2MaskSv = svand_n_s64_x(svptrue_b8(), svset_neonq(svundef_s64(), vec2Mask), 1);
    svst1_s64(pa, dst, vec1MaskSv);
    svst1_s64(pb, dst + oneRegElemCount, vec2MaskSv);
  }
}

template <>
inline void convert(const bool* src, int32_t* dst, int64_t n) {
  constexpr uint64_t oneRegElemCount = Vectorized<int32_t>::size();
  constexpr uint64_t twoRegsElemCount = oneRegElemCount * 2;
  const uint64_t count = static_cast<uint64_t>(n);
  const uint64_t remainder = count % twoRegsElemCount;
  const uint8_t* srcPtr = reinterpret_cast<const uint8_t*>(src);
  for (uint64_t iters = count / twoRegsElemCount; iters > 0; --iters) {
    auto vec1 = svget_neonq(svld1ub_s32(svptrue_b8(), srcPtr));
    auto vec2 = svget_neonq(svld1ub_s32(svptrue_b8(), srcPtr + oneRegElemCount));
    auto vec1Mask = vtstq_s32(vec1, vec1);
    auto vec2Mask = vtstq_s32(vec2, vec2);
    vec1Mask = svget_neonq(svand_n_s32_x(svptrue_b8(), svset_neonq(svundef_s32(), vec1Mask), 1));
    vec2Mask = svget_neonq(svand_n_s32_x(svptrue_b8(), svset_neonq(svundef_s32(), vec2Mask), 1));
    vst1q_s32(dst, vec1Mask);
    vst1q_s32(dst + oneRegElemCount, vec2Mask);
    srcPtr += twoRegsElemCount;
    dst += twoRegsElemCount;
  }
  if (remainder > 0) {
    svbool_t pa = svwhilelt_b32_u64(0, remainder);
    svbool_t pb = svwhilelt_b32_u64(oneRegElemCount, remainder);
    auto vec1 = svget_neonq(svld1ub_s32(pa, srcPtr));
    auto vec2 = svget_neonq(svld1ub_s32(pb, srcPtr + oneRegElemCount));
    auto vec1Mask = vtstq_s32(vec1, vec1);
    auto vec2Mask = vtstq_s32(vec2, vec2);
    auto vec1MaskSv = svand_n_s32_x(pa, svset_neonq(svundef_s32(), vec1Mask), 1);
    auto vec2MaskSv = svand_n_s32_x(pb, svset_neonq(svundef_s32(), vec2Mask), 1);
    svst1_s32(pa, dst, vec1MaskSv);
    svst1_s32(pb, dst + oneRegElemCount, vec2MaskSv);
  }
}

template <>
inline void convert(const uint8_t* src, bool* dst, int64_t n) {
  constexpr uint64_t oneRegElemCount = Vectorized<uint8_t>::size();
  constexpr uint64_t twoRegsElemCount = oneRegElemCount * 2;
  constexpr uint64_t fourRegsElemCount = twoRegsElemCount * 2;
  const uint64_t count = static_cast<uint64_t>(n);
  uint8_t* dstPtr = reinterpret_cast<uint8_t*>(dst);
  const uint64_t remainder = count % fourRegsElemCount;
  for (uint64_t iters = count / fourRegsElemCount; iters > 0; --iters) {
    auto vec1 = vld1q_u8(src);
    auto vec2 = vld1q_u8(src + oneRegElemCount);
    auto vec3 = vld1q_u8(src + twoRegsElemCount);
    auto vec4 = vld1q_u8(src + (twoRegsElemCount + oneRegElemCount));
    auto vec1Mask = vtstq_u8(vec1, vec1);
    auto vec2Mask = vtstq_u8(vec2, vec2);
    auto vec3Mask = vtstq_u8(vec3, vec3);
    auto vec4Mask = vtstq_u8(vec4, vec4);
    vst1q_u8(dstPtr, vec1Mask);
    vst1q_u8(dstPtr + oneRegElemCount, vec2Mask);
    vst1q_u8(dstPtr + twoRegsElemCount, vec3Mask);
    vst1q_u8(dstPtr + (twoRegsElemCount + oneRegElemCount), vec4Mask);
    src += fourRegsElemCount;
    dstPtr += fourRegsElemCount;
  }
  if (remainder > 0) {
    svbool_t pa = svwhilelt_b8_u64(0, remainder);
    svbool_t pb = svwhilelt_b8_u64(oneRegElemCount, remainder);
    svbool_t pc = svwhilelt_b8_u64(twoRegsElemCount, remainder);
    svbool_t pd = svwhilelt_b8_u64((twoRegsElemCount + oneRegElemCount), remainder);
    auto vec1 = svget_neonq(svld1_u8(pa, src));
    auto vec2 = svget_neonq(svld1_u8(pb, src + oneRegElemCount));
    auto vec3 = svget_neonq(svld1_u8(pc, src + twoRegsElemCount));
    auto vec4 = svget_neonq(svld1_u8(pd, src + (twoRegsElemCount + oneRegElemCount)));
    auto vec1Mask = vtstq_u8(vec1, vec1);
    auto vec2Mask = vtstq_u8(vec2, vec2);
    auto vec3Mask = vtstq_u8(vec3, vec3);
    auto vec4Mask = vtstq_u8(vec4, vec4);
    svst1_u8(pa, dstPtr, svset_neonq(svundef_u8(), vec1Mask));
    svst1_u8(pb, dstPtr + oneRegElemCount, svset_neonq(svundef_u8(), vec2Mask));
    svst1_u8(pc, dstPtr + twoRegsElemCount, svset_neonq(svundef_u8(), vec3Mask));
    svst1_u8(pd, dstPtr + (twoRegsElemCount + oneRegElemCount), svset_neonq(svundef_u8(), vec4Mask));
  }
}

template <>
Vectorized<int64_t> inline operator<<(
    const Vectorized<int64_t>& a,
    const Vectorized<int64_t>& b) {
  return vshlq_s64(a, b);
}

template <>
Vectorized<int32_t> inline operator<<(
    const Vectorized<int32_t>& a,
    const Vectorized<int32_t>& b) {
  return vshlq_s32(a, b);
}

template <>
Vectorized<int16_t> inline operator<<(
    const Vectorized<int16_t>& a,
    const Vectorized<int16_t>& b) {
  return vshlq_s16(a, b);
}

template <>
Vectorized<int8_t> inline operator<<(
    const Vectorized<int8_t>& a,
    const Vectorized<int8_t>& b) {
  return vshlq_s8(a, b);
}

template <>
Vectorized<int64_t> inline operator>>(
    const Vectorized<int64_t>& a,
    const Vectorized<int64_t>& b) {
  return svasr_s64_x(ptrue, a, svreinterpret_u64_s64(b));
}

template <>
Vectorized<int32_t> inline operator>>(
    const Vectorized<int32_t>& a,
    const Vectorized<int32_t>& b) {
  return svasr_s32_x(ptrue, a, svreinterpret_u32_s32(b));
}

template <>
Vectorized<int16_t> inline operator>>(
    const Vectorized<int16_t>& a,
    const Vectorized<int16_t>& b) {
  return svasr_s16_x(ptrue, a, svreinterpret_u16_s16(b));
}

template <>
Vectorized<int8_t> inline operator>>(
    const Vectorized<int8_t>& a,
    const Vectorized<int8_t>& b) {
  return svasr_s8_x(ptrue, a, svreinterpret_u8_s8(b));
}

#endif // defined(CPU_CAPABILITY_SVE)

} // namespace CPU_CAPABILITY
} // namespace at::vec
