#pragma once

#include <ATen/core/Scalar.h>
#include <ATen/core/Tensor.h>
#include <ATen/core/TensorImpl.h>
#include <ATen/core/UndefinedTensorImpl.h>
#include <ATen/core/blob.h>
#include <ATen/core/intrusive_ptr.h>

#include <type_traits>

namespace torch { namespace jit {

template <typename T>
using Shared = c10::intrusive_ptr<T>;

// string
struct CAFFE2_API ConstantString final : c10::intrusive_ptr_target {
 private:
  const std::string str_;
 public:
  ConstantString(std::string str)
  : str_(std::move(str)) {}
  static c10::intrusive_ptr<ConstantString> create(std::string str_);
  const std::string & string() const {
    return str_;
  }
  operator const std::string & () const {
    return string();
  }
  CAFFE2_API friend std::ostream& operator<<(
      std::ostream& out,
      const ConstantString& v);
};

template <typename Elem>
struct C10_EXPORT List : c10::intrusive_ptr_target {
 private:
  std::vector<Elem> elements_;

 public:
  typedef Elem ElemType;

  List(std::vector<Elem> elements_) : elements_(std::move(elements_)) {}
  static c10::intrusive_ptr<List<Elem>> create(std::vector<Elem> elements_) {
    return c10::make_intrusive<List<Elem>>(std::move(elements_));
  }
  const std::vector<Elem>& elements() const {
    return elements_;
  }
  operator const std::vector<Elem>&() const {
    return elements();
  }

  std::vector<Elem>& elements() {
    return elements_;
  }
  operator std::vector<Elem>&() {
    return elements();
  }
};

struct World {
  int64_t world_id;
};

struct IValue;
struct C10_EXPORT Tuple : public List<IValue> {
  using List<IValue>::List;
  static c10::intrusive_ptr<Tuple> create(std::vector<IValue> elements_) {
    return c10::make_intrusive<Tuple>(std::move(elements_));
  }
};
using IntList = List<int64_t>;
using TensorList = List<at::Tensor>;
using DoubleList = List<double>;
using BoolList = List<bool>;
using GenericList = List<IValue>;

// IValue is the generic tagged union used by the interpreter to hold
// all value types.
// It is a 16-byte object with an 8-byte payload and an 8-byte tag.
// The tag is currently 4 bytes to determine the type, and 1 byte
// to mark whether that type is a subtype of c10::intrusive_ptr_target and needs
// retain/release calls.

#define TORCH_FORALL_TAGS(_) \
  _(None) \
  _(Tensor) \
  _(Double) \
  _(Int) \
  _(Bool) \
  _(Tuple) \
  _(IntList) \
  _(DoubleList) \
  _(BoolList) \
  _(String) \
  _(TensorList) \
  _(Blob) \
  _(GenericList) \
  _(World) \

struct CAFFE2_API IValue final {
  IValue()
  : payload{0}
  , tag(Tag::None)
  , is_intrusive_ptr(false) {}
  IValue(const IValue& rhs)
      : payload(rhs.payload),
        tag(rhs.tag),
        is_intrusive_ptr(rhs.is_intrusive_ptr) {
    if (is_intrusive_ptr) {
      c10::raw::intrusive_ptr::incref(payload.as_intrusive_ptr);
    }
  }
  IValue(IValue&& rhs) noexcept : IValue() {
    swap(rhs);
  }
  ~IValue() {
    if (is_intrusive_ptr) {
      c10::raw::intrusive_ptr::decref(payload.as_intrusive_ptr);
    }
  }
  IValue & operator=(IValue && rhs) & noexcept {
    IValue(std::move(rhs)).swap(*this); // this also sets rhs to None
    return *this;
  }
  IValue & operator=(IValue const & rhs) & {
    IValue(rhs).swap(*this);
    return *this;
  }
  void swap(IValue & rhs) noexcept {
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
    payload.as_intrusive_ptr = t.unsafeReleaseTensorImpl();
  }
  bool isTensor() const { return Tag::Tensor == tag; }
  at::Tensor toTensor() && {
    AT_ASSERT(isTensor());
    return at::Tensor(moveToIntrusivePtr<at::TensorImpl, at::UndefinedTensorImpl>());
  }
  at::Tensor toTensor() const & {
    AT_ASSERT(isTensor());
    return at::Tensor(toIntrusivePtr<at::TensorImpl, at::UndefinedTensorImpl>());
  }

  const IValue& toIValue() const {
    return *this;
  }
  IValue& toIValue() {
    return *this;
  }

  IValue(caffe2::Blob blob) : tag(Tag::Blob), is_intrusive_ptr(true) {
    // TODO (after Tensor merge) If we pass in a Blob holding a Tensor, extract
    // and
    //      store it as a Tensor instead.
    payload.as_intrusive_ptr =
        c10::make_intrusive<caffe2::Blob>(std::move(blob)).release();
  }
  bool isBlob() const {
    return Tag::Blob == tag;
  }
  caffe2::Blob& toBlob() & {
    AT_ASSERT(isBlob());
    return *static_cast<caffe2::Blob*>(payload.as_intrusive_ptr);
  }
  const caffe2::Blob& toBlob() const& {
    AT_ASSERT(isBlob());
    return *static_cast<caffe2::Blob*>(payload.as_intrusive_ptr);
  }

  // Tuple
  IValue(c10::intrusive_ptr<Tuple> v);
  bool isTuple() const { return Tag::Tuple == tag; }
  c10::intrusive_ptr<Tuple> toTuple() && {
    AT_ASSERT(isTuple());
    return moveToIntrusivePtr<Tuple>();
  }
  c10::intrusive_ptr<Tuple> toTuple() const & {
    AT_ASSERT(isTuple());
    return toIntrusivePtr<Tuple>();
  }

  // Double
  IValue(double d)
  : tag(Tag::Double), is_intrusive_ptr(false) {
    payload.as_double = d;
  }
  bool isDouble() const { return Tag::Double == tag; }
  double toDouble() const {
    AT_ASSERT(isDouble());
    return payload.as_double;
  }

  // World
  IValue(World w)
  : tag(Tag::World), is_intrusive_ptr(false) {
    payload.as_world = w;
  }
  bool isWorld() const { return Tag::World == tag; }
  World toWorld() const {
    AT_ASSERT(isWorld());
    return payload.as_world;
  }

  // Int
  IValue(int64_t i)
  : tag(Tag::Int), is_intrusive_ptr(false) {
    payload.as_int = i;
  }

  // allow you to pass literals (3, 4) without ambiguity
  IValue(int32_t i)
  : IValue(static_cast<int64_t>(i)) {}

  bool isInt() const { return Tag::Int == tag; }

  int64_t toInt() const {
    AT_ASSERT(isInt());
    return payload.as_int;
  }

  // Bool
  IValue(bool b)
  : tag(Tag::Bool), is_intrusive_ptr(false) {
    payload.as_bool = b;
  }
   bool isBool() const { return Tag::Bool == tag; }
   bool toBool() const {
    AT_ASSERT(isBool());
    return payload.as_bool;
  }

  // IntList
  IValue(c10::intrusive_ptr<IntList> v);
  IValue(std::vector<int64_t> v);
  IValue(at::ArrayRef<int64_t> v)
  : IValue(v.vec()) {}
  bool isIntList() const { return Tag::IntList == tag; }
  c10::intrusive_ptr<IntList> toIntList() && {
    AT_ASSERT(isIntList());
    return moveToIntrusivePtr<IntList>();
  }
  c10::intrusive_ptr<IntList> toIntList() const & {
    AT_ASSERT(isIntList());
    return toIntrusivePtr<IntList>();
  }

  const std::vector<int64_t>& toIntListRef() const;
  const std::vector<double>& toDoubleListRef() const;
  const std::vector<bool>& toBoolListRef() const;
  const std::vector<at::Tensor>& toTensorListRef() const;
  const std::vector<IValue>& toGenericListRef() const;

  // ConstantString
  IValue(c10::intrusive_ptr<ConstantString> v);
  IValue(std::string v);
  bool isString() const { return Tag::String == tag; }
  c10::intrusive_ptr<ConstantString> toString() && {
    AT_ASSERT(isString());
    return moveToIntrusivePtr<ConstantString>();
  }
  c10::intrusive_ptr<ConstantString> toString() const & {
    AT_ASSERT(isString());
    return toIntrusivePtr<ConstantString>();
  }

  // DoubleList
  IValue(c10::intrusive_ptr<DoubleList> v);
  IValue(std::vector<double> v);
  bool isDoubleList() const { return Tag::DoubleList == tag; }
  c10::intrusive_ptr<DoubleList> toDoubleList() && {
    AT_ASSERT(isDoubleList());
    return moveToIntrusivePtr<DoubleList>();
  }
  c10::intrusive_ptr<DoubleList> toDoubleList() const & {
    AT_ASSERT(isDoubleList());
    return toIntrusivePtr<DoubleList>();
  }

  // BoolList
  IValue(c10::intrusive_ptr<BoolList> v);
  IValue(std::vector<bool> v);
  bool isBoolList() const { return Tag::BoolList == tag; }
  c10::intrusive_ptr<BoolList> toBoolList() && {
    AT_ASSERT(isBoolList());
    return moveToIntrusivePtr<BoolList>();
  }
  c10::intrusive_ptr<BoolList> toBoolList() const & {
    AT_ASSERT(isBoolList());
    return toIntrusivePtr<BoolList>();
  }

  //TensorList
  IValue(c10::intrusive_ptr<TensorList> v);
  IValue(std::vector<at::Tensor> v);
  bool isTensorList() const { return Tag::TensorList == tag; }
  c10::intrusive_ptr<TensorList> toTensorList() && {
    AT_ASSERT(isTensorList());
    return moveToIntrusivePtr<TensorList>();
  }
  c10::intrusive_ptr<TensorList> toTensorList() const & {
    AT_ASSERT(isTensorList());
    return toIntrusivePtr<TensorList>();
  }

  //GenericList
  IValue(c10::intrusive_ptr<GenericList> v);
  IValue(std::vector<IValue> v);
  bool isGenericList() const { return Tag::GenericList == tag; }
  c10::intrusive_ptr<GenericList> toGenericList() && {
    AT_ASSERT(isGenericList());
    return moveToIntrusivePtr<GenericList>();
  }
  c10::intrusive_ptr<GenericList> toGenericList() const & {
    AT_ASSERT(isGenericList());
    return toIntrusivePtr<GenericList>();
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
    return isDouble() || isInt() || isBool();
  }
  at::Scalar toScalar() const {
    if(isDouble())
      return toDouble();
    else if(isInt())
      return toInt();
    else if (isBool())
      return int(toBool());
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

  CAFFE2_API friend std::ostream& operator<<(
      std::ostream& out,
      const IValue& v);

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

  template<class T, class NullType = c10::detail::intrusive_target_default_null_type<T>>
  c10::intrusive_ptr<T, NullType> moveToIntrusivePtr() {
    auto t = c10::intrusive_ptr<T, NullType>::reclaim(static_cast<T*>(payload.as_intrusive_ptr));
    clearToNone();
    return t;
  }
  template<typename T, class NullType = c10::detail::intrusive_target_default_null_type<T>>
  c10::intrusive_ptr<T, NullType> toIntrusivePtr() const {
    auto r = c10::intrusive_ptr<T, NullType>::reclaim(static_cast<T*>(payload.as_intrusive_ptr));
    auto p = r;
    r.release();
    return p;
  }
  void clearToNone() {
    payload.as_int = 0;
    tag = Tag::None;
    is_intrusive_ptr = false;
  }
  union {
    int64_t as_int;
    double as_double;
    bool as_bool;
    c10::intrusive_ptr_target* as_intrusive_ptr;
    World as_world;
  } payload;
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
DEFINE_TO(bool, toBool)
DEFINE_TO(c10::intrusive_ptr<DoubleList>, toDoubleList)
DEFINE_TO(c10::intrusive_ptr<IntList>, toIntList)
DEFINE_TO(c10::intrusive_ptr<TensorList>, toTensorList)
DEFINE_TO(c10::intrusive_ptr<GenericList>, toGenericList)
DEFINE_TO(c10::intrusive_ptr<ConstantString>, toString)
DEFINE_TO(at::Scalar, toScalar)
DEFINE_TO(std::vector<int64_t>, toIntListRef)
DEFINE_TO(std::vector<double>, toDoubleListRef)
DEFINE_TO(std::vector<bool>, toBoolListRef)
DEFINE_TO(std::vector<at::Tensor>, toTensorListRef)
DEFINE_TO(std::vector<IValue>, toGenericListRef)
DEFINE_TO(World, toWorld)
DEFINE_TO(IValue, toIValue)

#undef DEFINE_TO

#define DEFINE_TO_WITH_BODY(type, body) \
template<> \
inline type IValue::to<type>() && { \
  body(std::move(*this)); \
} \
template<> \
inline type IValue::to<type>() const & { \
  body((*this)); \
}

#define SCALAR_TYPE_BODY(this) return static_cast<at::ScalarType>(this.toInt());
#define LAYOUT_BODY(this) return static_cast<at::Layout>(this.toInt());
#define DEVICE_BODY(this)                                           \
  /* NB: const_list might be a move of the vector, so we need to */ \
  /*     assign it to prevent its deallocation.                  */ \
  auto&& const_list = this.toIntList();                             \
  const auto& elems = const_list->elements();                       \
  AT_ASSERT(elems.size() == 2);                                     \
  return at::Device(static_cast<at::Device::Type>(elems[0]), elems[1]);

DEFINE_TO_WITH_BODY(at::ScalarType, SCALAR_TYPE_BODY)
DEFINE_TO_WITH_BODY(at::Layout, LAYOUT_BODY)
DEFINE_TO_WITH_BODY(at::Device, DEVICE_BODY)

#undef DEFINE_TO_WITH_BODY
#undef SCALAR_TYPE_BODY
#undef LAYOUT_BODY
#undef DEVICE_BODY

inline IValue::IValue(c10::intrusive_ptr<Tuple> v)
: tag(Tag::Tuple), is_intrusive_ptr(true) {
  payload.as_intrusive_ptr = v.release();
}

inline IValue::IValue(c10::intrusive_ptr<IntList> v)
: tag(Tag::IntList), is_intrusive_ptr(true) {
  payload.as_intrusive_ptr = v.release();
}
inline IValue::IValue(std::vector<int64_t> v)
: IValue(IntList::create(std::move(v))) {}

inline IValue::IValue(c10::intrusive_ptr<ConstantString> v)
: tag(Tag::String), is_intrusive_ptr(true) {
  payload.as_intrusive_ptr = v.release();
}
inline IValue::IValue(std::string v)
: IValue(ConstantString::create(std::move(v))) {}

inline IValue::IValue(c10::intrusive_ptr<DoubleList> v)
: tag(Tag::DoubleList), is_intrusive_ptr(true) {
  payload.as_intrusive_ptr = v.release();
}
inline IValue::IValue(std::vector<double> v)
: IValue(DoubleList::create(std::move(v))) {}

inline IValue::IValue(c10::intrusive_ptr<BoolList> v)
: tag(Tag::BoolList), is_intrusive_ptr(true) {
  payload.as_intrusive_ptr = v.release();
}
inline IValue::IValue(std::vector<bool> v)
: IValue(BoolList::create(std::move(v))) {}

inline IValue::IValue(c10::intrusive_ptr<TensorList> v)
: tag(Tag::TensorList), is_intrusive_ptr(true) {
  payload.as_intrusive_ptr = v.release();
}
inline IValue::IValue(std::vector<at::Tensor> v)
: IValue(TensorList::create(std::move(v))) {}

inline IValue::IValue(c10::intrusive_ptr<GenericList> v)
: tag(Tag::GenericList), is_intrusive_ptr(true) {
  payload.as_intrusive_ptr = v.release();
}
inline IValue::IValue(std::vector<IValue> v)
: IValue(GenericList::create(std::move(v))) {}


inline const std::vector<int64_t>& IValue::toIntListRef() const {
  return toIntList()->elements();
}

inline const std::vector<double>& IValue::toDoubleListRef() const {
  return toDoubleList()->elements();
}

inline const std::vector<at::Tensor>& IValue::toTensorListRef() const {
  return toTensorList()->elements();
}

inline const std::vector<bool>& IValue::toBoolListRef() const {
  return toBoolList()->elements();
}

inline const std::vector<IValue>& IValue::toGenericListRef() const {
  return toGenericList()->elements();
}


}}
