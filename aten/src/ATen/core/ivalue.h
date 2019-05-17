#pragma once

#include <ATen/core/blob.h>
#include <c10/util/intrusive_ptr.h>
#include <ATen/core/Tensor.h>

namespace torch {
namespace jit {
namespace script {
struct Function;
}
} // namespace jit
} // namespace torch
namespace c10 {
template<class Key, class Value> class Dict;
struct IValue;
namespace ivalue {
struct Tuple;
template<class Elem> struct List;
using IntList = List<int64_t>;
using TensorList = List<at::Tensor>;
using DoubleList = List<double>;
using BoolList = List<bool>;
using GenericList = List<IValue>;
struct Future;
struct ConstantString;
struct GenericDict;
struct Object;
}

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
  _(GenericDict) \
  _(Future) \
  _(Device) \
  _(Object)

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

  void dump() const;

  bool isAliasOf(const IValue& rhs) const {
    if (this->tag != rhs.tag) {
      // Trivially don't alias if the type is different
      return false;
    }

    if (!this->is_intrusive_ptr) {
      // Primitive types don't alias anything
      return false;
    }

    AT_ASSERT(rhs.is_intrusive_ptr);

    // Tensors should be compared based on internal storage
    if (this->isTensor()) {
      const auto thisTensor = this->toTensor();
      const auto rhsTensor = rhs.toTensor();
      return thisTensor.is_alias_of(rhsTensor);
    }

    // Other types can be compared by their ptr value
    return this->payload.as_intrusive_ptr == rhs.payload.as_intrusive_ptr;
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
  at::Tensor toTensor() &&;
  at::Tensor toTensor() const &;

  const IValue& toIValue() const {
    return *this;
  }
  IValue& toIValue() {
    return *this;
  }

  IValue(intrusive_ptr<caffe2::Blob> blob)
  : tag(Tag::Blob), is_intrusive_ptr(true) {
    // TODO (after Tensor merge) If we pass in a Blob holding a Tensor, extract
    // and store it as a Tensor instead.
    payload.as_intrusive_ptr = blob.release();
  }
  bool isBlob() const {
    return Tag::Blob == tag;
  }
  c10::intrusive_ptr<caffe2::Blob> toBlob() &&;
  c10::intrusive_ptr<caffe2::Blob> toBlob() const &;

  // Tuple
  IValue(c10::intrusive_ptr<ivalue::Tuple> v);
  bool isTuple() const { return Tag::Tuple == tag; }
  c10::intrusive_ptr<ivalue::Tuple> toTuple() &&;
  c10::intrusive_ptr<ivalue::Tuple> toTuple() const &;

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

  // Future
  IValue(c10::intrusive_ptr<ivalue::Future> v);
  bool isFuture() const { return Tag::Future == tag; }
  c10::intrusive_ptr<ivalue::Future> toFuture() &&;
  c10::intrusive_ptr<ivalue::Future> toFuture() const &;

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
  IValue(c10::intrusive_ptr<ivalue::IntList> v);
  IValue(std::vector<int64_t> v);
  IValue(at::ArrayRef<int64_t> v)
  : IValue(v.vec()) {}
  bool isIntList() const { return Tag::IntList == tag; }
  c10::intrusive_ptr<ivalue::IntList> toIntList() &&;
  c10::intrusive_ptr<ivalue::IntList> toIntList() const &;

  const std::vector<int64_t>& toIntListRef() const;
  const std::vector<double>& toDoubleListRef() const;
  const std::vector<bool>& toBoolListRef() const;
  const std::vector<at::Tensor>& toTensorListRef() const;
  const std::vector<IValue>& toGenericListRef() const;
  const c10::Dict<IValue, IValue>& toGenericDictRef() const;
  const std::string& toStringRef() const;

  // ConstantString
  IValue(c10::intrusive_ptr<ivalue::ConstantString> v);
  IValue(std::string v);
  IValue(const char* v): IValue(std::string(v)) {}
  bool isString() const { return Tag::String == tag; }
  c10::intrusive_ptr<ivalue::ConstantString> toString() &&;
  c10::intrusive_ptr<ivalue::ConstantString> toString() const &;

  // DoubleList
  IValue(c10::intrusive_ptr<ivalue::DoubleList> v);
  IValue(std::vector<double> v);
  bool isDoubleList() const { return Tag::DoubleList == tag; }
  c10::intrusive_ptr<ivalue::DoubleList> toDoubleList() &&;
  c10::intrusive_ptr<ivalue::DoubleList> toDoubleList() const &;

  // BoolList
  IValue(c10::intrusive_ptr<ivalue::BoolList> v);
  IValue(std::vector<bool> v);
  bool isBoolList() const { return Tag::BoolList == tag; }
  c10::intrusive_ptr<ivalue::BoolList> toBoolList() &&;
  c10::intrusive_ptr<ivalue::BoolList> toBoolList() const &;

  //TensorList
  IValue(c10::intrusive_ptr<ivalue::TensorList> v);
  IValue(std::vector<at::Tensor> v);
  bool isTensorList() const { return Tag::TensorList == tag; }
  c10::intrusive_ptr<ivalue::TensorList> toTensorList() &&;
  c10::intrusive_ptr<ivalue::TensorList> toTensorList() const &;

  //GenericList
  IValue(c10::intrusive_ptr<ivalue::GenericList> v);
  IValue(std::vector<IValue> v);
  bool isGenericList() const { return Tag::GenericList == tag; }
  c10::intrusive_ptr<ivalue::GenericList> toGenericList() &&;
  c10::intrusive_ptr<ivalue::GenericList> toGenericList() const &;

  // GenericDict
  IValue(c10::intrusive_ptr<ivalue::GenericDict> v);
  IValue(c10::Dict<IValue, IValue> v);
  bool isGenericDict() const { return Tag::GenericDict == tag; }
  c10::intrusive_ptr<ivalue::GenericDict> toGenericDict() &&;
  c10::intrusive_ptr<ivalue::GenericDict> toGenericDict() const &;

  // ClassType
  IValue(c10::intrusive_ptr<ivalue::Object> v);
  bool isObject() const { return tag == Tag::Object; }
  c10::intrusive_ptr<ivalue::Object> toObject() &&;
  c10::intrusive_ptr<ivalue::Object> toObject() const & ;

  // None
  bool isNone() const {
    return Tag::None == tag;
  }
  std::string toNone() const {
    AT_ASSERT(isNone());
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
  bool isScalar() const {
    return isDouble() || isInt();
  }
  at::Scalar toScalar() const {
    if(isDouble())
      return toDouble();
    else if(isInt())
      return toInt();
    throw std::runtime_error("IValue is not a Scalar");
  }

  // Device
  IValue(c10::Device d)
  : tag(Tag::Device), is_intrusive_ptr(false) {
    payload.as_device.type = d.type();
    payload.as_device.index = d.index();
  }
  bool isDevice() const { return Tag::Device == tag; }
  c10::Device toDevice() const {
    AT_ASSERT(isDevice());
    return c10::Device(payload.as_device.type, payload.as_device.index);
  }

  // ScalarType
  at::ScalarType toScalarType() const {
    return static_cast<at::ScalarType>(toInt());
  }

  // Layout
  at::Layout toLayout() const {
    return static_cast<at::Layout>(toInt());
  }

  // MemoryFormat
  at::MemoryFormat toMemoryFormat() const {
    return static_cast<at::MemoryFormat>(toInt());
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

  // ToOptional: convert a IValue to the Optional obj that accepts both T and None
  template<typename T>
  optional<T> toOptional();

  // this is a shallow comparison of two IValues to test the object identity
  bool isSameIdentity(const IValue& rhs) const;

  CAFFE2_API friend std::ostream& operator<<(
      std::ostream& out,
      const IValue& v);

  bool isPtrType() const {
    return is_intrusive_ptr;
  }

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
  c10::intrusive_ptr<T, NullType> moveToIntrusivePtr();
  template<typename T, class NullType = c10::detail::intrusive_target_default_null_type<T>>
  c10::intrusive_ptr<T, NullType> toIntrusivePtr() const;

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
    struct {
      DeviceType type;
      DeviceIndex index;
    } as_device;
  } payload;
  Tag tag;
  bool is_intrusive_ptr;
};

}

#include <ATen/core/ivalue_inl.h>
