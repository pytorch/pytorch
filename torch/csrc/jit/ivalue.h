#pragma once

#include "torch/csrc/jit/assertions.h"
#include "torch/csrc/WindowsTorchApiMacro.h"

#include <ATen/ATen.h>

#include <type_traits>

namespace torch { namespace jit {

template <typename T>
using Shared = c10::intrusive_ptr<T>;

// string
struct TORCH_API ConstantString : c10::intrusive_ptr_target {
 private:
  const std::string str_;
 public:
  ConstantString(std::string str)
  : str_(std::move(str)) {}
  static c10::intrusive_ptr<ConstantString> create(const std::string str_) {
    return c10::make_intrusive<ConstantString>(str_);
  }
  const std::string & string() const {
    return str_;
  }
  operator const std::string & () const {
    return string();
  }
  TORCH_API friend std::ostream& operator<<(std::ostream& out, const ConstantString & v);
};


// non-mutable list
template<typename Elem>
struct TORCH_API ConstantList : c10::intrusive_ptr_target {
 private:
  std::vector<Elem> elements_;
 public:
  ConstantList(std::vector<Elem> elements_)
  : elements_(std::move(elements_)) {}
  static c10::intrusive_ptr<ConstantList<Elem>> create(std::vector<Elem> elements_) {
    return c10::make_intrusive<ConstantList<Elem>>(std::move(elements_));
  }
  const std::vector<Elem>& elements() const {
    return elements_;
  }
  operator const std::vector<Elem>&() const {
    return elements();
  }
};

struct IValue;
using Tuple = ConstantList<IValue>;
using IntList = ConstantList<int64_t>;
using TensorList = ConstantList<at::Tensor>;
using DoubleList = ConstantList<double>;

// IValue is the generic tagged union used by the interpreter to hold
// all value types.
// It is a 16-byte object with an 8-byte payload and an 8-byte tag.
// The tag is currently 4 bytes to determine the type, and 1 byte
// to mark whether that type is a subtype of c10::intrusive_ptr_target and needs
// retain/release calls.

#define TORCH_FORALL_TAGS(_) \
  _(None) _(Tensor) _(Double) _(Int) _(Tuple) _(IntList) _(DoubleList) _(String) _(TensorList)

struct TORCH_API IValue {
  IValue()
  : payload(0)
  , tag(Tag::None)
  , is_intrusive_ptr(false) {}
  IValue(const IValue& rhs)
      : payload(rhs.payload),
        tag(rhs.tag),
        is_intrusive_ptr(rhs.is_intrusive_ptr) {
    if (is_intrusive_ptr) {
      c10::raw::intrusive_ptr::incref(as_intrusive_ptr);
    }
  }
  IValue(IValue&& rhs) noexcept : IValue() {
    swap(rhs);
  }
  ~IValue() {
    if (is_intrusive_ptr) {
      c10::raw::intrusive_ptr::decref(as_intrusive_ptr);
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
    std::swap(is_intrusive_ptr, rhs.is_intrusive_ptr);
    std::swap(tag, rhs.tag);
  }
  // Accessors for subtypes are arranged together below
  // While some of these accessors could be generated through templates,
  // we prefer to write them manually for clarity

  // Tensor
  IValue(at::Tensor t)
  : tag(Tag::Tensor), is_intrusive_ptr(t.defined())  {
    // Note: the undefined tensor is not refcounted, so while it
    // is tagged as a tensor, is_intrusive_ptr is set to false.
    // This is not an optional optimization: our incref call
    // *will not* do the right thing when called on an
    // undefined tensor.
    as_tensor_impl = t.unsafeReleaseTensorImpl();
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
  IValue(c10::intrusive_ptr<Tuple> v);
  bool isTuple() const { return Tag::Tuple == tag; }
  c10::intrusive_ptr<Tuple> toTuple() && {
    JIT_ASSERT(isTuple());
    return moveToIntrusivePtr<Tuple>();
  }
  c10::intrusive_ptr<Tuple> toTuple() const & {
    JIT_ASSERT(isTuple());
    return toIntrusivePtr<Tuple>();
  }

  // Double
  IValue(double d)
  : tag(Tag::Double), is_intrusive_ptr(false) {
    as_double = d;
  }
  bool isDouble() const { return Tag::Double == tag; }
  double toDouble() const {
    JIT_ASSERT(isDouble());
    return as_double;
  }

  // Int
  IValue(int64_t i)
  : tag(Tag::Int), is_intrusive_ptr(false) {
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
  IValue(c10::intrusive_ptr<IntList> v);
  IValue(std::vector<int64_t> v);
  IValue(at::ArrayRef<int64_t> v)
  : IValue(std::vector<int64_t>(v.begin(), v.end())) {}
  bool isIntList() const { return Tag::IntList == tag; }
  c10::intrusive_ptr<IntList> toIntList() && {
    JIT_ASSERT(isIntList());
    return moveToIntrusivePtr<IntList>();
  }
  c10::intrusive_ptr<IntList> toIntList() const & {
    JIT_ASSERT(isIntList());
    return toIntrusivePtr<IntList>();
  }

  const std::vector<int64_t>& toIntListRef() const;
  const std::vector<double>& toDoubleListRef() const;
  const std::vector<at::Tensor>& toTensorListRef() const;

  // ConstantString
  IValue(c10::intrusive_ptr<ConstantString> v);
  IValue(const std::string& v);
  bool isString() const { return Tag::String == tag; }
  c10::intrusive_ptr<ConstantString> toString() && {
    JIT_ASSERT(isString());
    return moveToIntrusivePtr<ConstantString>();
  }
  c10::intrusive_ptr<ConstantString> toString() const & {
    JIT_ASSERT(isString());
    return toIntrusivePtr<ConstantString>();
  }

  // DoubleList
  IValue(c10::intrusive_ptr<DoubleList> v);
  IValue(std::vector<double> v);
  bool isDoubleList() const { return Tag::DoubleList == tag; }
  c10::intrusive_ptr<DoubleList> toDoubleList() && {
    JIT_ASSERT(isDoubleList());
    return moveToIntrusivePtr<DoubleList>();
  }
  c10::intrusive_ptr<DoubleList> toDoubleList() const & {
    JIT_ASSERT(isDoubleList());
    return toIntrusivePtr<DoubleList>();
  }

  //TensorList
  IValue(c10::intrusive_ptr<TensorList> v);
  IValue(std::vector<at::Tensor> v);
  bool isTensorList() const { return Tag::TensorList == tag; }
  c10::intrusive_ptr<TensorList> toTensorList() && {
    JIT_ASSERT(isTensorList());
    return moveToIntrusivePtr<TensorList>();
  }
  c10::intrusive_ptr<TensorList> toTensorList() const & {
    JIT_ASSERT(isTensorList());
    return toIntrusivePtr<TensorList>();
  }

  // None
  bool isNone() {
    return Tag::None == tag;
  }
  std::string toNone() const {
    return "None";
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
  std::string tagKind() const {
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

  TORCH_API friend std::ostream& operator<<(std::ostream & out, const IValue & v);

private:
  // NOTE: IValue tags are intentionally private. In the future we may encode
  // this value different (e.g. using NaN boxing), and this would make it more
  // costly to determine the tag for all types vs just determining if something
  // is a particular type. Instead we want clients to use the `isX` methods when
  // possible. If for perf. reasons you really, absolutely, must have a jump
  // table, then we can revisit this.
  enum class Tag : uint32_t {
#define DEFINE_TAG(x) x,
    TORCH_FORALL_TAGS(DEFINE_TAG)
#undef DEFINE_TAG
  };

  template<typename T>
  c10::intrusive_ptr<T> moveToIntrusivePtr() {
    auto t = c10::intrusive_ptr<T>::reclaim(static_cast<T*>(as_intrusive_ptr));
    clearToNone();
    return t;
  }
  template<typename T>
  c10::intrusive_ptr<T> toIntrusivePtr() const {
    auto r = c10::intrusive_ptr<T>::reclaim(static_cast<T*>(as_intrusive_ptr));
    auto p = r;
    r.release();
    return p;
  }
  void clearToNone() {
    payload = 0;
    tag = Tag::None;
    is_intrusive_ptr = false;
  }
  union {
    at::TensorImpl* as_tensor_impl;
    c10::intrusive_ptr_target* as_intrusive_ptr;
    double as_double;
    int64_t as_int;
    // this type should be as big as all the other types because it will
    // be used to copy the union's value in certain cases
    int64_t payload;
  };
  Tag tag;
  bool is_intrusive_ptr;
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
DEFINE_TO(c10::intrusive_ptr<Tuple>, toTuple)
DEFINE_TO(double, toDouble)
DEFINE_TO(int64_t, toInt)
DEFINE_TO(c10::intrusive_ptr<DoubleList>, toDoubleList)
DEFINE_TO(c10::intrusive_ptr<IntList>, toIntList)
DEFINE_TO(c10::intrusive_ptr<TensorList>, toTensorList)
DEFINE_TO(c10::intrusive_ptr<ConstantString>, toString)
DEFINE_TO(at::Scalar, toScalar)
DEFINE_TO(bool, toInt)
DEFINE_TO(std::vector<int64_t>, toIntListRef)
DEFINE_TO(std::vector<double>, toDoubleListRef)
DEFINE_TO(std::vector<at::Tensor>, toTensorListRef)

#undef DEFINE_TO

inline IValue::IValue(c10::intrusive_ptr<Tuple> v)
: tag(Tag::Tuple), is_intrusive_ptr(true) {
  as_intrusive_ptr = v.release();
}

inline IValue::IValue(c10::intrusive_ptr<IntList> v)
: tag(Tag::IntList), is_intrusive_ptr(true) {
  as_intrusive_ptr = v.release();
}
inline IValue::IValue(std::vector<int64_t> v)
: IValue(IntList::create(std::move(v))) {}

inline IValue::IValue(c10::intrusive_ptr<ConstantString> v)
: tag(Tag::String), is_intrusive_ptr(true) {
  as_intrusive_ptr = v.release();
}
inline IValue::IValue(const std::string& v)
: IValue(ConstantString::create(v)) {}

inline IValue::IValue(c10::intrusive_ptr<DoubleList> v)
: tag(Tag::DoubleList), is_intrusive_ptr(true) {
  as_intrusive_ptr = v.release();
}
inline IValue::IValue(std::vector<double> v)
: IValue(DoubleList::create(std::move(v))) {}

inline IValue::IValue(c10::intrusive_ptr<TensorList> v)
: tag(Tag::TensorList), is_intrusive_ptr(true) {
  as_intrusive_ptr = v.release();
}
inline IValue::IValue(std::vector<at::Tensor> v)
: IValue(TensorList::create(std::move(v))) {}

inline const std::vector<int64_t>& IValue::toIntListRef() const {
  return toIntList()->elements();
}

inline const std::vector<double>& IValue::toDoubleListRef() const {
  return toDoubleList()->elements();
}

inline const std::vector<at::Tensor>& IValue::toTensorListRef() const {
  return toTensorList()->elements();
}


}}
