#pragma once

#include<stdint.h>
#include <stdexcept>
#include <string>
#include "ATen/HalfConvert.h"

#ifdef AT_CUDA_ENABLED
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#endif

#if defined(__GNUC__)
#define AT_ALIGN(n) __attribute__((aligned(n)))
#elif defined(_WIN32)
#define AT_ALIGN(n) __declspec(align(n))
#else
#define AT_ALIGN(n)
#endif



namespace at {


template<typename To, typename From> To convert(From f) {
  return static_cast<To>(f);
}

typedef struct  AT_ALIGN(2) {
  unsigned short x;
#ifdef AT_CUDA_ENABLED
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
#ifdef AT_CUDA_ENABLED
template<> half convert(double d);
#endif

#define AT_FORALL_SCALAR_TYPES(_) \
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
  Scalar(type vv) \
  : tag(Tag::HAS_##member) { \
    v . member = convert<decltype(v.member),type>(vv); \
  }

  AT_FORALL_SCALAR_TYPES(DEFINE_IMPLICIT_CTOR)

#ifdef AT_CUDA_ENABLED
  Scalar(half vv)
  : tag(Tag::HAS_d) {
    v.d = convert<double,Half>(Half{vv.x});
  }
#endif

#undef DEFINE_IMPLICIT_CTOR

#define DEFINE_ACCESSOR(type,name,member) \
  type to##name () { \
    if (Tag::HAS_d == tag) { \
      auto casted = convert<type,double>(v.d); \
      if(convert<double,type>(casted) != v.d) { \
        throw std::domain_error(std::string("value cannot be losslessly represented in type " #name ": ") + std::to_string(v.d) ); \
      } \
      return casted; \
    } else { \
      auto casted = convert<type,int64_t>(v.i); \
      if(convert<int64_t,type>(casted) != v.i) { \
        throw std::domain_error(std::string("value cannot be losslessly represented in type " #name ": ") + std::to_string(v.i)); \
      } \
      return casted; \
    } \
  }

  AT_FORALL_SCALAR_TYPES(DEFINE_ACCESSOR)

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
  } v;
};

}
