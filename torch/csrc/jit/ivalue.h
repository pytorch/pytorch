#pragma once

#include "torch/csrc/jit/assertions.h"

#include <ATen/ATen.h>

#include <type_traits>

namespace torch { namespace jit {

// smart pointer to hold onto at::Retainable objects in a generic way
// this is close to the implementation of boost's intrusive_ptr
template<typename PointerType>
struct Shared {
  Shared(): Shared(nullptr, false) {}
  Shared(PointerType * self, bool retain)
  : pImpl(self) {
    if(retain && pImpl)
      pImpl->retain();
  }
  Shared(const Shared & rhs)
  : pImpl(rhs.pImpl) {
    if (pImpl)
      pImpl->retain();
  }
  Shared(Shared && rhs) noexcept
  : pImpl(rhs.pImpl) {
    rhs.pImpl = nullptr;
  }
  ~Shared() {
    if (pImpl)
      pImpl->release();
  }
  Shared & operator=(Shared && rhs) & {
    rhs.swap(*this);
    return *this;
  }
  Shared & operator=(Shared const & rhs) & {
      //Shared ctor retains original rhs.pImpl
      //then rhs.pImpl is swapped with this->pImpl
      //finally Shared dtor releases rhs.pImpl, which was originally this->pImpl
      Shared(rhs).swap(*this);
      return *this;
  }
  void reset() {
    Shared().swap(*this);
  }
  void reset(PointerType * rhs) {
    Shared(rhs, true).swap(*this);
  }
  void reset(PointerType * rhs, bool retain) {
    Shared(rhs, retain).swap(*this);
  }
  void swap(Shared & rhs) {
    PointerType * tmp = pImpl;
    pImpl = rhs.pImpl;
    rhs.pImpl = tmp;
  }
  PointerType* get() const {
    return pImpl;
  }
  PointerType* detach() {
    PointerType * ret = pImpl;
    pImpl = nullptr;
    return ret;
  }
  PointerType& operator*() const {
    return  *get();
  }
  PointerType* operator->() const {
    return get();
  }
  operator bool() const {
    return pImpl != nullptr;
  }
private:
  PointerType * pImpl;
};


template<typename T>
struct ConstantList;
struct IValue;
using Tuple = ConstantList<IValue>;
using IntList = ConstantList<int64_t>;
using DoubleList = ConstantList<double>;

// IValue is the generic tagged union used by the interpreter to hold
// all value types.
// It is a 16-byte object with an 8-byte payload and an 8-byte tag.
// The tag is currently 4 bytes to determine the type, and 1 byte
// to mark whether that type is a subtype of at::Retainable and needs
// retain/release calls.

#define TORCH_FORALL_TAGS(_) \
  _(None) _(Tensor) _(Double) _(Int) _(Tuple) _(IntList) _(DoubleList)

struct IValue {
  IValue()
  : payload(0)
  , tag(Tag::None)
  , retainable(false) {}
  IValue(const IValue& rhs)
      : payload(rhs.payload),
        tag(rhs.tag),
        retainable(rhs.retainable) {
    if (retainable)
      as_retainable->retain();
  }
  IValue(IValue&& rhs) noexcept : IValue() {
    swap(rhs);
  }
  ~IValue() {
    if (retainable) {
      as_retainable->release();
    }
  }
  IValue & operator=(IValue && rhs) & {
    rhs.swap(*this);
    return *this;
  }
  IValue & operator=(IValue const & rhs) & {
      IValue(rhs).swap(*this);
      return *this;
  }
  void swap(IValue & rhs) {
    std::swap(payload, rhs.payload);
    std::swap(retainable, rhs.retainable);
    std::swap(tag, rhs.tag);
  }
  // Accessors for subtypes are arragned together below
  // While some of these accessors could be generated through templates,
  // we prefer to write them manually for clarity

  // Tensor
  IValue(at::Tensor t)
  : tag(Tag::Tensor), retainable(t.defined())  {
    // note: the undefined tensor is not refcounted, so while it
    // is tagged as a tensor, retainable is set to false.
    as_tensor_impl = t.at::detail::TensorBase::detach();
  }
  bool isTensor() const { return Tag::Tensor == tag; }
  at::Tensor toTensor() && {
    JIT_ASSERT(isTensor());
    at::Tensor t(as_tensor_impl, /*retain=*/false);
    clearToNone();
    return t;
  }
  at::Tensor toTensor() const & {
    JIT_ASSERT(isTensor());
    return at::Tensor(as_tensor_impl, /*retain=*/true);
  }

  // Tuple
  IValue(Shared<Tuple> v);
  bool isTuple() const { return Tag::Tuple == tag; }
  Shared<Tuple> toTuple() && {
    JIT_ASSERT(isTuple());
    return moveToRetainable<Tuple>();
  }
  Shared<Tuple> toTuple() const & {
    JIT_ASSERT(isTuple());
    return toRetainable<Tuple>();
  }

  // Double
  IValue(double d)
  : tag(Tag::Double), retainable(false) {
    as_double = d;
  }
  bool isDouble() const { return Tag::Double == tag; }
  double toDouble() const {
    JIT_ASSERT(isDouble());
    return as_double;
  }

  // Int
  IValue(int64_t i)
  : tag(Tag::Int), retainable(false) {
    as_int = i;
  }

  // allow you to pass literals (3, 4) without ambiguity
  IValue(int32_t i)
  : IValue(static_cast<int64_t>(i)) {}
  IValue(bool b)
  : IValue(static_cast<int64_t>(b)) {}

  bool isInt() const { return Tag::Int == tag; }

  int64_t toInt() const {
    JIT_ASSERT(isInt());
    return as_int;
  }

  // IntList
  IValue(Shared<IntList> v);
  IValue(std::vector<int64_t> v);
  IValue(at::ArrayRef<int64_t> v)
  : IValue(std::vector<int64_t>(v.begin(), v.end())) {}
  bool isIntList() const { return Tag::IntList == tag; }
  Shared<IntList> toIntList() && {
    JIT_ASSERT(isIntList());
    return moveToRetainable<IntList>();
  }
  Shared<IntList> toIntList() const & {
    JIT_ASSERT(isIntList());
    return toRetainable<IntList>();
  }

  std::vector<int64_t> copyToIntList() const;

  // DoubleList
  IValue(Shared<DoubleList> v);
  IValue(std::vector<double> v);
  bool isDoubleList() const { return Tag::DoubleList == tag; }
  Shared<DoubleList> toDoubleList() && {
    JIT_ASSERT(isDoubleList());
    return moveToRetainable<DoubleList>();
  }
  Shared<DoubleList> toDoubleList() const & {
    JIT_ASSERT(isDoubleList());
    return toRetainable<DoubleList>();
  }

  bool isNone() {
    return Tag::None == tag;
  }

  // Scalar, which gets encoded as either an Int or a Double
  IValue(at::Scalar s)
  : IValue() {
    if(s.isFloatingPoint()) {
      *this = s.toDouble();
    } else {
      *this = s.toLong();
    }
  }
  bool isScalar() {
    return isDouble() || isInt();
  }
  at::Scalar toScalar() const {
    if(isDouble())
      return toDouble();
    else if(isInt())
      return toInt();
    else
      throw std::runtime_error("IValue is not a Scalar");
  }

  // for debugging
  std::string tagKind() {
    switch(tag) {
      #define DEFINE_CASE(x) case Tag::x: return #x;
      TORCH_FORALL_TAGS(DEFINE_CASE)
      #undef DEFINE_CASE
    }
    return "Invalid Tag";
  }

  // generic v.to<at::Tensor>() implementations
  // that can be used in special functions like pop/push
  // that use template meta-programming.
  // prefer the directly named methods when you can,
  // since they are simpler to understand

  // Note: if you get linker errors saying one of these is missing,
  // change it to ... && = delete; and you will see better error messages for why
  // However, we cannot commit this because some compiler versions barf on it.
  template<typename T>
  T to() &&;
  template<typename T>
  T to() const &;

private:
  template<typename T>
  Shared<T> moveToRetainable() {
    Shared<T> t(static_cast<T*>(as_retainable), false);
    clearToNone();
    return t;
  }
  template<typename T>
  Shared<T> toRetainable() const {
    return Shared<T>(static_cast<T*>(as_retainable), true);
  }
  void clearToNone() {
    payload = 0;
    tag = Tag::None;
    retainable = false;
  }
  enum class Tag : uint32_t {
    #define DEFINE_TAG(x) x,
    TORCH_FORALL_TAGS(DEFINE_TAG)
    #undef DEFINE_TAG
  };
  union {
    at::TensorImpl* as_tensor_impl;
    at::Retainable* as_retainable;
    double as_double;
    int64_t as_int;
    // this type should be as big as all the other types because it will
    // be used to copy the union's value in certain cases
    int64_t payload;
  };
  Tag tag;
  bool retainable;
};

#undef TORCH_FORALL_TAGS


#define DEFINE_TO(type, method_name) \
template<> \
inline type IValue::to<type>() && { \
  return std::move(*this).method_name(); \
} \
template<> \
inline type IValue::to<type>() const & { \
  return this->method_name(); \
}
DEFINE_TO(at::Tensor, toTensor)
DEFINE_TO(Shared<Tuple>, toTuple)
DEFINE_TO(double, toDouble)
DEFINE_TO(int64_t, toInt)
DEFINE_TO(Shared<DoubleList>, toDoubleList)
DEFINE_TO(Shared<IntList>, toIntList)
DEFINE_TO(at::Scalar, toScalar)
DEFINE_TO(bool, toInt)
DEFINE_TO(std::vector<int64_t>, copyToIntList)

#undef DEFINE_TO

// non-mutable list
template<typename Elem>
struct ConstantList : at::Retainable {
 private:
  ConstantList(std::vector<Elem> elements_)
  : elements_(std::move(elements_)) {}
  std::vector<Elem> elements_;
 public:
  static Shared<ConstantList<Elem>> create(std::vector<Elem> elements_) {
    return Shared<ConstantList<Elem>>(
        new ConstantList<Elem>(std::move(elements_)), false);
  }
  at::ArrayRef<Elem> elements() const {
    return elements_;
  }
  operator at::ArrayRef<Elem>() const {
    return elements();
  }
};

inline IValue::IValue(Shared<Tuple> v)
: tag(Tag::Tuple), retainable(true) {
  as_retainable = v.detach();
}

inline IValue::IValue(Shared<IntList> v)
: tag(Tag::IntList), retainable(true) {
  as_retainable = v.detach();
}
inline IValue::IValue(std::vector<int64_t> v)
: IValue(IntList::create(std::move(v))) {}

inline IValue::IValue(Shared<DoubleList> v)
: tag(Tag::DoubleList), retainable(true) {
  as_retainable = v.detach();
}
inline IValue::IValue(std::vector<double> v)
: IValue(DoubleList::create(std::move(v))) {}

inline std::vector<int64_t> IValue::copyToIntList() const {
  return std::vector<int64_t>(toIntList()->elements());
}

}}
