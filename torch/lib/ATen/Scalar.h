#pragma once

#include<stdint.h>
#include <stdexcept>
#include <string>
#include "ATen/Half.h"
#include "ATen/Type.h"
#include "ATen/Utils.h"
#include "ATen/Tensor.h"

namespace at {

class Scalar {
public:
  explicit Scalar(const Tensor & t)
  : tag(Tag::HAS_t), t(t) {
    AT_ASSERT(t.dim() == 0,"Attempting to create a Scalar from a %d dim tensor",t.dim());
  }
#define DEFINE_IMPLICIT_CTOR(type,name,member) \
  Scalar(type vv) \
  : tag(Tag::HAS_##member) { \
    v . member = convert<decltype(v.member),type>(vv); \
  }

  AT_FORALL_SCALAR_TYPES(DEFINE_IMPLICIT_CTOR)

#ifdef AT_CUDA_ENABLED
  Scalar(half vv)
  : tag(Tag::HAS_d) {
#if CUDA_VERSION < 9000
    v.d  = convert<double, Half>(Half{vv.x});
#else
    __half_raw vv_raw(vv);
    v.d = convert<double,Half>(Half{vv_raw.x});
#endif
  }
#endif

#undef DEFINE_IMPLICIT_CTOR

  // return a new scalar that is guaranteed to be not backed by a tensor.
  Scalar local() {
    if (Tag::HAS_t != tag) {
      return *this;
    }
    return t.pImpl->localScalar();
  }

#define DEFINE_ACCESSOR(type,name,member) \
  type to##name () { \
    if (Tag::HAS_t == tag) { \
      return local().to##name(); \
    } else if (Tag::HAS_d == tag) { \
      auto casted = convert<type,double>(v.d); \
      if(convert<double,type>(casted) != v.d) { \
        throw std::domain_error(std::string("value cannot be losslessly represented in type " #name ": ") + std::to_string(v.d) ); \
      } \
      return casted; \
    } else { \
      assert(Tag::HAS_i == tag); \
      auto casted = convert<type,int64_t>(v.i); \
      if(convert<int64_t,type>(casted) != v.i) { \
        throw std::domain_error(std::string("value cannot be losslessly represented in type " #name ": ") + std::to_string(v.i)); \
      } \
      return casted; \
    } \
  }

  AT_FORALL_SCALAR_TYPES(DEFINE_ACCESSOR)

  //also support scalar.to<int64_t>();
  template<typename T>
  T to();

#undef DEFINE_ACCESSOR
  bool isFloatingPoint() {
    return Tag::HAS_d == tag;
  }
  bool isIntegral() {
    return Tag::HAS_i == tag;
  }
  bool isBackedByTensor() {
    return Tag::HAS_t == tag;
  }

private:
  enum class Tag { HAS_d, HAS_i, HAS_t };
  Tag tag;
  union {
    double d;
    int64_t i;
  } v;
  Tensor t; //Note: cannot be in union be cause of copy/destruct behavior
            //ideally we try to pack this structure tighter if it becomes
            //a performance problem.
  friend struct Type;
};

// define the scalar.to<int64_t>() specializations
template<typename T>
inline T Scalar::to() {
  throw std::runtime_error("to() cast to unexpected type.");
}

#define DEFINE_TO(T,name,_) \
template<> \
inline T Scalar::to<T>() { \
  return to##name(); \
}
AT_FORALL_SCALAR_TYPES(DEFINE_TO)
#undef DEFINE_TO

}
