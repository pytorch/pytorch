#pragma once

#include<stdint.h>
#include <stdexcept>
#include <string>
#include "TensorLib/HalfConvert.h"

#ifdef TENSORLIB_CUDA_ENABLED
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#endif

#if defined(__GNUC__)
#define TLIB_ALIGN(n) __attribute__((aligned(n)))
#elif defined(_WIN32)
#define TLIB_ALIGN(n) __declspec(align(n))
#else
#define TLIB_ALIGN(n)
#endif



namespace tlib {


template<typename To, typename From> To convert(From f) {
  return static_cast<To>(f);
}

typedef struct  TLIB_ALIGN(2) {
  unsigned short x;
#ifdef TENSORLIB_CUDA_ENABLED
  operator half() { return half { x }; }
#endif
  operator double();
} Half;

template<> Half convert(double f);
template<> double convert(Half f);
template<> Half convert(int64_t f);
template<> int64_t convert(Half f);

inline Half::operator double() {
  return convert<double,Half>(*this);
}
#ifdef TENSORLIB_CUDA_ENABLED
template<> half convert(double d);
#endif

#define TLIB_SCALAR_TYPES(_) \
_(uint8_t,Byte,i) \
_(int8_t,Char,i) \
_(double,Double,d) \
_(float,Float,d) \
_(int,Int,i) \
_(int64_t,Long,i) \
_(int16_t,Short,i) \
_(Half,Half,d)

class Scalar {
public:
#define DEFINE_IMPLICIT_CTOR(type,name,member) \
  Scalar(type v) \
  : tag(Tag::HAS_##member) { \
    member = convert<decltype(member),type>(v); \
  }

  TLIB_SCALAR_TYPES(DEFINE_IMPLICIT_CTOR)

#ifdef TENSORLIB_CUDA_ENABLED
  Scalar(half v)
  : tag(Tag::HAS_d) {
    d = convert<double,Half>(Half{v.x});
  }
#endif

#undef DEFINE_IMPLICIT_CTOR

#define DEFINE_ACCESSOR(type,name,member) \
  type to##name () { \
    if (Tag::HAS_d == tag) { \
      auto casted = convert<type,double>(d); \
      if(convert<double,type>(casted) != d) { \
        throw std::domain_error(std::string("value cannot be losslessly represented in type " #name ": ") + std::to_string(d) ); \
      } \
      return casted; \
    } else { \
      auto casted = convert<type,int64_t>(i); \
      if(convert<int64_t,type>(casted) != i) { \
        throw std::domain_error(std::string("value cannot be losslessly represented in type " #name ": ") + std::to_string(i)); \
      } \
      return casted; \
    } \
  }

  TLIB_SCALAR_TYPES(DEFINE_ACCESSOR)

#undef DEFINE_ACCESSOR
  bool isFloatingPoint() {
    return Tag::HAS_d == tag;
  }
  bool isIntegral() {
    return Tag::HAS_i == tag;
  }
private:
  enum class Tag { HAS_d, HAS_i };
  Tag tag;
  union {
    double d;
    int64_t i;
  };
};

}
