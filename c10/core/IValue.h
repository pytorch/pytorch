#pragma once

#include <c10/core/Tensor.h>
#include <c10/core/blob.h>
#include <c10/util/intrusive_ptr.h>

namespace c10 {
struct IValue;

namespace core {

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
  _(Future) \
  _(Device)

/**
 * This is a somewhat narrowed down version of IValue. Ideally, this should be
 * merged with IValue (after IValue is moved to c10).
 *
 * It was split from IValue to break the IValue->at::Tensor dependency.
 * We needed to move IValue to c10 so the dispatcher can use it, but IValue
 * depended on at::Tensor. While we wanted to move at::Tensor to c10 eventually,
 * we couldn't do that yet because we first needed to get rid of its dependency
 * on at::Type, which in turn needs the dispatcher. This is a cyclic dependency
 * in implementation order.
 *
 * To break this cycle, we split IValue into IValue and C10IValue.
 * C10IValue doesn't have the at::Tensor dependency and IValue uses C10IValue
 * internally. The plan is to move C10IValue to c10,
 * then we can set up the dispatcher, move at::Tensor to c10,
 * move IValue to c10, and re-merge C10IValue and IValue.
 */
class C10_API C10IValue final {
private:
  friend struct c10::IValue;

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

public:

  C10IValue(const C10IValue& rhs)
  : payload(rhs.payload),
    tag(rhs.tag),
    is_intrusive_ptr(rhs.is_intrusive_ptr) {
    if (is_intrusive_ptr) {
      c10::raw::intrusive_ptr::incref(payload.as_intrusive_ptr);
    }
  }

  C10IValue(C10IValue&& rhs) noexcept : C10IValue() {
    swap(rhs);
  }

  ~C10IValue() {
    if (is_intrusive_ptr) {
      c10::raw::intrusive_ptr::decref(payload.as_intrusive_ptr);
    }
  }

  C10IValue & operator=(C10IValue && rhs) & noexcept {
    C10IValue(std::move(rhs)).swap(*this); // this also sets rhs to None
    return *this;
  }

  C10IValue & operator=(C10IValue const & rhs) & {
    C10IValue(rhs).swap(*this);
    return *this;
  }

  void swap(C10IValue& rhs) noexcept {
    std::swap(payload, rhs.payload);
    std::swap(is_intrusive_ptr, rhs.is_intrusive_ptr);
    std::swap(tag, rhs.tag);
  }

private:
  // intrusive_ptr based types.
  // This is a private extension point for IValue to add accessors
  // for types like TensorList.
  template<class T, class NullType>
  explicit C10IValue(Tag value_tag, intrusive_ptr<T, NullType> value)
  : tag(value_tag)
  , is_intrusive_ptr(true) {
    payload.as_intrusive_ptr = value.release();
  }
  bool isIntrusivePtr(Tag expected_tag) const { return expected_tag == tag; }
  template<class T, class NullType = detail::intrusive_target_default_null_type<T>>
  intrusive_ptr<T, NullType> toIntrusivePtr(Tag expected_tag) && {
    AT_ASSERT(tag == expected_tag);
    // move out the intrusive_ptr
    auto t = intrusive_ptr<T, NullType>::reclaim(static_cast<T*>(payload.as_intrusive_ptr));
    clearToNone();
    return t;
  }
  template<typename T, class NullType = detail::intrusive_target_default_null_type<T>>
  intrusive_ptr<T, NullType> toIntrusivePtr(Tag expected_tag) const & {
    AT_ASSERT(tag == expected_tag);
    // copy out the intrusive_ptr
    auto r = intrusive_ptr<T, NullType>::reclaim(static_cast<T*>(payload.as_intrusive_ptr));
    auto p = r;
    r.release();
    return p;
  }
  bool isPtrType() const noexcept {
    return is_intrusive_ptr;
  }
  bool isSameIntrusivePtr(const C10IValue& rhs) const noexcept {
    return is_intrusive_ptr && rhs.is_intrusive_ptr
        && payload.as_intrusive_ptr == rhs.payload.as_intrusive_ptr;
  }

public:
  // None
  C10IValue()
  : payload{0}
  , tag(Tag::None)
  , is_intrusive_ptr(false) {}
  bool isNone() const {
    return Tag::None == tag;
  }
  std::string toNone() const {
    AT_ASSERT(isNone());
    return "None";
  }

  // Tensor
  explicit C10IValue(C10Tensor t)
  : tag(Tag::Tensor), is_intrusive_ptr(t.defined())  {
    // Note: the undefined tensor is not refcounted, so while it
    // is tagged as a tensor, is_intrusive_ptr is set to false.
    // This is not an optional optimization: our incref call
    // *will not* do the right thing when called on an
    // undefined tensor.
    payload.as_intrusive_ptr = std::move(t).impl().release();
  }
  bool isTensor() const { return Tag::Tensor == tag; }
  C10Tensor toTensor() && {
    return C10Tensor(std::move(*this).toIntrusivePtr<TensorImpl, UndefinedTensorImpl>(Tag::Tensor));
  }
  C10Tensor toTensor() const & {
    return C10Tensor(toIntrusivePtr<TensorImpl, UndefinedTensorImpl>(Tag::Tensor));
  }

  // Blob
  explicit C10IValue(Blob blob)
  : tag(Tag::Blob), is_intrusive_ptr(true) {
    // TODO (after Tensor merge) If we pass in a Blob holding a Tensor, extract
    // and store it as a Tensor instead. Or maybe rather assert against it?
    payload.as_intrusive_ptr =
        c10::make_intrusive<Blob>(std::move(blob)).release();
  }
  bool isBlob() const {
    return Tag::Blob == tag;
  }
  Blob& toBlob() & {
    AT_ASSERT(isBlob());
    return *static_cast<Blob*>(payload.as_intrusive_ptr);
  }
  const Blob& toBlob() const& {
    AT_ASSERT(isBlob());
    return *static_cast<Blob*>(payload.as_intrusive_ptr);
  }

  // Double
  explicit C10IValue(double d)
  : tag(Tag::Double), is_intrusive_ptr(false) {
    payload.as_double = d;
  }
  bool isDouble() const { return Tag::Double == tag; }
  double toDouble() const {
    AT_ASSERT(isDouble());
    return payload.as_double;
  }

  // Int
  explicit C10IValue(int64_t i)
  : tag(Tag::Int), is_intrusive_ptr(false) {
    payload.as_int = i;
  }
  // allow you to pass literals (3, 4) without ambiguity
  explicit C10IValue(int32_t i) : C10IValue(static_cast<int64_t>(i)) {}
  bool isInt() const { return Tag::Int == tag; }
  int64_t toInt() const {
    AT_ASSERT(isInt());
    return payload.as_int;
  }

  // Bool
  explicit C10IValue(bool b)
  : tag(Tag::Bool), is_intrusive_ptr(false) {
    payload.as_bool = b;
  }
  bool isBool() const { return Tag::Bool == tag; }
  bool toBool() const {
    AT_ASSERT(isBool());
    return payload.as_bool;
  }

  // Device
  explicit C10IValue(c10::Device d)
  : tag(Tag::Device), is_intrusive_ptr(false) {
    payload.as_device.type = d.type();
    payload.as_device.index = d.index();
  }
  bool isDevice() const { return Tag::Device == tag; }
  c10::Device toDevice() const {
    AT_ASSERT(isDevice());
    return c10::Device(payload.as_device.type, payload.as_device.index);
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

private:

  union {
    int64_t as_int;
    double as_double;
    bool as_bool;
    intrusive_ptr_target* as_intrusive_ptr;
    struct {
      DeviceType type;
      DeviceIndex index;
    } as_device;
  } payload;
  Tag tag;
  bool is_intrusive_ptr;

  void clearToNone() {
    payload.as_int = 0;
    tag = Tag::None;
    is_intrusive_ptr = false;
  }
};

#undef TORCH_FORALL_TAGS

}
}
