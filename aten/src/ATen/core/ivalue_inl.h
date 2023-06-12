#pragma once

#include <condition_variable>
#include <memory>
#include <type_traits>
#include <utility>

#include <ATen/core/Dict.h>
#include <ATen/core/List.h>
#include <ATen/core/IListRef.h>
#include <ATen/core/functional.h>
#include <ATen/core/jit_type.h>
#include <ATen/core/qualified_name.h>
#include <ATen/core/rref_interface.h>
#include <ATen/core/symbol.h>
#include <c10/core/DeviceGuard.h>
#include <c10/core/Event.h>
#include <c10/core/Scalar.h>
#include <c10/core/Stream.h>
#include <c10/core/StreamGuard.h>
#include <c10/core/TensorImpl.h>
#include <c10/core/UndefinedTensorImpl.h>
#include <c10/core/impl/DeviceGuardImplInterface.h>
#include <c10/util/FunctionRef.h>
#include <c10/util/Logging.h>
#include <c10/util/hash.h>
#include <c10/util/intrusive_ptr.h>
#include <c10/util/irange.h>

namespace torch {
namespace jit {
struct Function;
struct CompilationUnit;
} // namespace jit
TORCH_API bool isCustomClass(const c10::IValue& v);
} // namespace torch
namespace c10 {
struct IValue;
struct ClassType;
struct TupleType;
struct EnumType;
struct InferredType;

// For custom class __init__ registration, we need to pass in a function
// that looks like this: [](IValue x, args...)

// However, make_boxed_from_unboxed_functor.h automatically sets the input types
// of the function by introspecting the types of the functor (which is IValue in
// this case). However, we need the type it binds to be Foo.

// Instead, we pass in a lambda [](ivalue_holder<CurClass> x, args...) from
// which getTypePtr can recover the original class pointer.

template <typename TaggedCapsuleType>
struct tagged_capsule {
  IValue ivalue;
};

template <class T, class NullType>
c10::intrusive_ptr<T, NullType> IValue::moveToIntrusivePtr() {
  auto t = c10::intrusive_ptr<T, NullType>::reclaim(
      payload.u.as_intrusive_ptr == c10::UndefinedTensorImpl::singleton()
      ? NullType::singleton()
      : static_cast<T*>(payload.u.as_intrusive_ptr));
  clearToNone();
  return t;
}
template <typename T, class NullType>
c10::intrusive_ptr<T, NullType> IValue::toIntrusivePtr() const {
  if (payload.u.as_intrusive_ptr == c10::UndefinedTensorImpl::singleton()) {
    return c10::intrusive_ptr<T, NullType>();
  }
  c10::raw::intrusive_ptr::incref(payload.u.as_intrusive_ptr);
  return c10::intrusive_ptr<T, NullType>::reclaim(
      static_cast<T*>(payload.u.as_intrusive_ptr));
}

template <class T, class U>
intrusive_ptr<T> static_intrusive_pointer_cast(intrusive_ptr<U> r) {
  return intrusive_ptr<T>::reclaim(static_cast<T*>(r.release()));
}

template <class T, class U>
intrusive_ptr<T> dynamic_intrusive_pointer_cast(intrusive_ptr<U> r) {
  return intrusive_ptr<T>::reclaim(dynamic_cast<T*>(r.release()));
}

inline c10::intrusive_ptr<ivalue::Future> IValue::toFuture() && {
  AT_ASSERT(isFuture(), "Expected Future but got ", tagKind());
  return moveToIntrusivePtr<ivalue::Future>();
}
inline c10::intrusive_ptr<ivalue::Future> IValue::toFuture() const& {
  AT_ASSERT(isFuture(), "Expected Future but got ", tagKind());
  return toIntrusivePtr<ivalue::Future>();
}
inline c10::intrusive_ptr<ivalue::Await> IValue::toAwait() && {
  AT_ASSERT(isAwait(), "Expected Await but got ", tagKind());
  return moveToIntrusivePtr<ivalue::Await>();
}
inline c10::intrusive_ptr<ivalue::Await> IValue::toAwait() const& {
  AT_ASSERT(isAwait(), "Expected Await but got ", tagKind());
  return toIntrusivePtr<ivalue::Await>();
}
inline c10::intrusive_ptr<c10::RRefInterface> IValue::toRRef() && {
  AT_ASSERT(isRRef(), "Expected RRef but got ", tagKind());
  return moveToIntrusivePtr<c10::RRefInterface>();
}
inline c10::intrusive_ptr<c10::RRefInterface> IValue::toRRef() const& {
  AT_ASSERT(isRRef(), "Expected RRef but got ", tagKind());
  return toIntrusivePtr<c10::RRefInterface>();
}
inline c10::intrusive_ptr<at::Quantizer> IValue::toQuantizer() && {
  AT_ASSERT(isQuantizer(), "Expected Quantizer but got ", tagKind());
  return moveToIntrusivePtr<at::Quantizer>();
}
inline c10::intrusive_ptr<at::Quantizer> IValue::toQuantizer() const& {
  AT_ASSERT(isQuantizer(), "Expected Quantizer but got ", tagKind());
  return toIntrusivePtr<at::Quantizer>();
}
inline c10::intrusive_ptr<ivalue::ConstantString> IValue::toString() && {
  AT_ASSERT(isString(), "Expected String but got ", tagKind());
  return moveToIntrusivePtr<ivalue::ConstantString>();
}
inline c10::intrusive_ptr<ivalue::ConstantString> IValue::toString() const& {
  AT_ASSERT(isString(), "Expected String but got ", tagKind());
  return toIntrusivePtr<ivalue::ConstantString>();
}
inline c10::intrusive_ptr<ivalue::Object> IValue::toObject() && {
  AT_ASSERT(isObject(), "Expected Object but got ", tagKind());
  return moveToIntrusivePtr<ivalue::Object>();
}
inline c10::intrusive_ptr<ivalue::Object> IValue::toObject() const& {
  AT_ASSERT(isObject(), "Expected Object but got ", tagKind());
  return toIntrusivePtr<ivalue::Object>();
}
inline c10::intrusive_ptr<ivalue::PyObjectHolder> IValue::
    toPyObjectHolder() && {
  TORCH_INTERNAL_ASSERT(isPyObject(), "Expected PyObject but got ", tagKind());
  return moveToIntrusivePtr<ivalue::PyObjectHolder>();
}
inline c10::intrusive_ptr<ivalue::PyObjectHolder> IValue::toPyObjectHolder()
    const& {
  TORCH_INTERNAL_ASSERT(isPyObject(), "Expected PyObject but got ", tagKind());
  return toIntrusivePtr<ivalue::PyObjectHolder>();
}
inline c10::intrusive_ptr<ivalue::EnumHolder> IValue::toEnumHolder() && {
  TORCH_INTERNAL_ASSERT(isEnum(), "Expected Enum but got ", tagKind());
  return moveToIntrusivePtr<ivalue::EnumHolder>();
}
inline c10::intrusive_ptr<ivalue::EnumHolder> IValue::toEnumHolder() const& {
  TORCH_INTERNAL_ASSERT(isEnum(), "Expected Enum but got ", tagKind());
  return toIntrusivePtr<ivalue::EnumHolder>();
}
inline c10::complex<double> IValue::toComplexDouble() const {
  TORCH_INTERNAL_ASSERT(isComplexDouble(), "Expected ComplexDouble but got ", tagKind());
  auto ptr = toIntrusivePtr<ivalue::ComplexHolder>();
  return (*ptr).val;
}
inline at::Tensor IValue::toTensor() && {
  if (C10_UNLIKELY(!isTensor())) {
    reportToTensorTypeError();
  }
  auto result = std::move(payload.as_tensor);
  // As far as I can tell, omitting the usual explicit destructor call
  // is not UB in and of itself, and it's a slight perf win. The
  // destructor is a no-op, because the moved-from Tensor is
  // effectively an intrusive_ptr in the null state, so we don't need
  // the behavior for correctness reasons either. Leaving this
  // explanatory comment, including commented-out destructor call, to
  // make this abundantly clear.
  //
  // payload.as_tensor.~Tensor();
  clearToNone();
  return result;
}
inline at::Tensor& IValue::toTensor() & {
  if (C10_UNLIKELY(!isTensor())) {
    reportToTensorTypeError();
  }
  return payload.as_tensor;
}
inline const at::Tensor& IValue::toTensor() const& {
  if (C10_UNLIKELY(!isTensor())) {
    reportToTensorTypeError();
  }
  return payload.as_tensor;
}
inline c10::Storage IValue::toStorage() && {
  AT_ASSERT(isStorage(), "Expected Storage but got ", tagKind());
  return c10::Storage(
      moveToIntrusivePtr<at::StorageImpl>());
}
inline c10::Storage IValue::toStorage() const& {
  AT_ASSERT(isStorage(), "Expected Storage but got ", tagKind());
  return c10::Storage(toIntrusivePtr<at::StorageImpl>());
}
inline c10::Stream IValue::toStream() && {
  AT_ASSERT(isStream(), "Expected Stream but got ", tagKind());
  auto ptr = toIntrusivePtr<ivalue::StreamData3Holder>();
  return c10::Stream::unpack3((*ptr).val.stream_id,
                              (*ptr).val.device_index,
                              (*ptr).val.device_type);
}
inline c10::Stream IValue::toStream() const& {
  AT_ASSERT(isStream(), "Expected Stream but got ", tagKind());
  auto ptr = toIntrusivePtr<ivalue::StreamData3Holder>();
  return c10::Stream::unpack3((*ptr).val.stream_id,
                              (*ptr).val.device_index,
                              (*ptr).val.device_type);
}
inline c10::intrusive_ptr<caffe2::Blob> IValue::toBlob() && {
  AT_ASSERT(isBlob(), "Expected Blob but got ", tagKind());
  return moveToIntrusivePtr<caffe2::Blob>();
}
inline c10::intrusive_ptr<caffe2::Blob> IValue::toBlob() const& {
  AT_ASSERT(isBlob(), "Expected Blob but got ", tagKind());
  return toIntrusivePtr<caffe2::Blob>();
  ;
}
inline c10::intrusive_ptr<torch::CustomClassHolder> IValue::toCapsule() && {
  TORCH_INTERNAL_ASSERT(isCapsule());
  return moveToIntrusivePtr<torch::CustomClassHolder>();
}
inline c10::intrusive_ptr<torch::CustomClassHolder> IValue::toCapsule() const& {
  TORCH_INTERNAL_ASSERT(isCapsule());
  return toIntrusivePtr<torch::CustomClassHolder>();
}
inline at::Generator IValue::toGenerator() && {
  AT_ASSERT(isGenerator(), "Expected Generator but got ", tagKind());
  return at::Generator(moveToIntrusivePtr<at::GeneratorImpl>());
}
inline at::Generator IValue::toGenerator() const& {
  AT_ASSERT(isGenerator(), "Expected Generator but got ", tagKind());
  return at::Generator(toIntrusivePtr<at::GeneratorImpl>());
}
inline c10::SymInt IValue::toSymInt() && {
  AT_ASSERT(isSymInt() || isInt(), "Expected SymInt or int but got ", tagKind());
  if (isSymInt()) {
    return c10::SymInt(moveToIntrusivePtr<c10::SymNodeImpl>());
  } else {
    return c10::SymInt(payload.u.as_int);
  }
}
inline c10::SymInt IValue::toSymInt() const& {
  AT_ASSERT(isSymInt() || isInt(), "Expected SymInt or int but got ", tagKind());
  if (isSymInt()) {
    return c10::SymInt(toIntrusivePtr<c10::SymNodeImpl>());
  } else {
    return c10::SymInt(payload.u.as_int);
  }
}
inline c10::SymFloat IValue::toSymFloat() && {
  AT_ASSERT(isSymFloat() || isDouble(), "Expected SymFloat or double but got ", tagKind());
  if (isSymFloat()) {
    return c10::SymFloat(moveToIntrusivePtr<c10::SymNodeImpl>());
  } else {
    return c10::SymFloat(payload.u.as_double);
  }
}
inline c10::SymFloat IValue::toSymFloat() const& {
  AT_ASSERT(isSymFloat() || isDouble(), "Expected SymFloat or double but got ", tagKind());
  if (isSymFloat()) {
    return c10::SymFloat(toIntrusivePtr<c10::SymNodeImpl>());
  } else {
    return c10::SymFloat(payload.u.as_double);
  }
}
inline c10::SymBool IValue::toSymBool() && {
  AT_ASSERT(isSymBool() || isBool(), "Expected SymBool or boolean but got ", tagKind());
  if (isSymBool()) {
    return c10::SymBool(moveToIntrusivePtr<c10::SymNodeImpl>());
  } else {
    return c10::SymBool(payload.u.as_bool);
  }
}

inline c10::SymBool IValue::toSymBool() const& {
  AT_ASSERT(isSymBool() || isBool(), "Expected SymBool or boolean but got ", tagKind());
  if (isSymBool()) {
    return c10::SymBool(toIntrusivePtr<c10::SymNodeImpl>());
  } else {
    return c10::SymBool(payload.u.as_bool);
  }
}

namespace ivalue {

void TORCH_API
checkCustomClassType(const ClassType* expected_type, const Type* actual_type);

template <typename T>
using Shared = c10::intrusive_ptr<T>;

// string
struct TORCH_API ConstantString final : c10::intrusive_ptr_target {
 private:
  const std::string str_;

 public:
  ConstantString(std::string str) : str_(std::move(str)) {}
  ConstantString(c10::string_view str) : str_(std::string(str)) {}
  static c10::intrusive_ptr<ConstantString> create(std::string str_);
  static c10::intrusive_ptr<ConstantString> create(c10::string_view str_);
  static c10::intrusive_ptr<ConstantString> create(const char* str_);

  const std::string& string() const {
    return str_;
  }
  c10::string_view string_view() const {
    return str_;
  }

  operator const std::string&() const {
    return string();
  }
  TORCH_API friend std::ostream& operator<<(
      std::ostream& out,
      const ConstantString& v);
};

struct Future;

struct TORCH_API TupleElements {
 private:
  size_t inlineSize_;
  // We represent TupleElements this way to save doing a heap
  // allocation in the common (at least for unpickling) case where we
  // have only 3 elements. We have our own union instead of
  // c10::SmallVector<IValue> because c10::SmallVector<IValue> always
  // stores the begin/end/capacity pointers, which would be a waste of
  // space in our use case.
  union {
    std::vector<IValue> elementsVector_;
    // Don't want to declare a std::array because the convenient
    // iteration and size members are a footgun in this case -- the
    // actual size of the array may be smaller than 3!
    // NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays)
    IValue elementsInline_[3];
  };

  void destroyInline() {
   for (const auto ii : c10::irange(inlineSize_)) {
     elementsInline_[ii].~IValue();
   }
  }
 public:

  using iterator = IValue*;
  using const_iterator = const IValue*;

  TupleElements() : inlineSize_(0) {
    new (&elementsVector_) std::vector<IValue>();
  }

  explicit TupleElements(std::vector<IValue> elements)
  : inlineSize_(0), elementsVector_(std::move(elements)) {}

  explicit TupleElements(c10::ArrayRef<IValue> elements)
  : inlineSize_(elements.size() <= 3 ? elements.size() : 0) {
    switch (inlineSize_) {
      case 3:
        new (&elementsInline_[2]) IValue(elements[2]);
        C10_FALLTHROUGH;
      case 2:
        new (&elementsInline_[1]) IValue(elements[1]);
        C10_FALLTHROUGH;
      case 1:
        new (&elementsInline_[0]) IValue(elements[0]);
        break;
      case 0:
        new (&elementsVector_) std::vector<IValue>(elements.begin(), elements.end());
        break;
    }
  }

  explicit TupleElements(IValue&& e1)
  : inlineSize_(1) {
    new (&elementsInline_[0]) IValue(std::move(e1));
  }

  explicit TupleElements(IValue&& e1, IValue&& e2)
  : inlineSize_(2) {
    new (&elementsInline_[0]) IValue(std::move(e1));
    new (&elementsInline_[1]) IValue(std::move(e2));
  }

  explicit TupleElements(IValue&& e1, IValue&& e2, IValue&& e3)
  : inlineSize_(3) {
    new (&elementsInline_[0]) IValue(std::move(e1));
    new (&elementsInline_[1]) IValue(std::move(e2));
    new (&elementsInline_[2]) IValue(std::move(e3));
  }

  ~TupleElements() {
    if (inlineSize_) {
      destroyInline();
    } else {
      elementsVector_.~vector();
    }
  }

  // It would be nice to make this noncopyable to prevent people from
  // writing code like `auto output =
  // forward(...).toTupleRef().elements()` (which does refcount bumps on
  // each element, unlike the more efficient but verbose
  // ```
  // auto outputIntrusivePtr = forward(...).toTuple();
  // const auto& output = outputIntrusivePtr->elements();
  // ```
  // ), but there is simply an overwhelming amount of code that does
  // it the inefficient way.
  // See also operator std::vector below.
  TupleElements(const TupleElements& rhs)
  : inlineSize_(rhs.inlineSize_) {
    if (rhs.inlineSize_) {
      for (const auto  ii : c10::irange(inlineSize_)) {
        new (&elementsInline_[ii]) IValue(rhs.elementsInline_[ii]);
      }
    } else {
      new (&elementsVector_) std::vector<IValue>(rhs.elementsVector_);
    }
  }

  TupleElements& operator=(const TupleElements& rhs) {
    if (inlineSize_) {
      if (rhs.inlineSize_) {
        for (const auto ii : c10::irange(std::min(inlineSize_, rhs.inlineSize_))) {
          elementsInline_[ii] = rhs.elementsInline_[ii];
        }
        if (rhs.inlineSize_ > inlineSize_) {
          for (const auto ii : c10::irange(inlineSize_, rhs.inlineSize_)) {
            new (&elementsInline_[ii]) IValue(rhs.elementsInline_[ii]);
          }
        } else {
          for (const auto ii : c10::irange(rhs.inlineSize_, inlineSize_)) {
            elementsInline_[ii].~IValue();
          }
        }
      } else {
        destroyInline();
        new (&elementsVector_) std::vector<IValue>(rhs.elementsVector_);
      }
    } else {
      if (rhs.inlineSize_) {
        elementsVector_.~vector();
        for (const auto ii : c10::irange(rhs.inlineSize_)) {
          new (&elementsInline_[ii]) IValue(rhs.elementsInline_[ii]);
        }
      } else {
        elementsVector_ = rhs.elementsVector_;
      }
    }
    inlineSize_ = rhs.inlineSize_;
    return *this;
  }

  TupleElements(TupleElements&& rhs) noexcept
  : inlineSize_(rhs.inlineSize_) {
    if (inlineSize_) {
      for (const auto ii : c10::irange(inlineSize_)) {
        new (&elementsInline_[ii]) IValue(std::move(rhs.elementsInline_[ii]));
      }
    } else {
      new (&elementsVector_) std::vector<IValue>(std::move(rhs.elementsVector_));
    }
  }

  TupleElements& operator=(TupleElements&& rhs) noexcept {
    if (inlineSize_) {
      if (rhs.inlineSize_) {
        for (const auto ii : c10::irange(std::min(inlineSize_, rhs.inlineSize_))) {
          elementsInline_[ii] = std::move(rhs.elementsInline_[ii]);
        }
        if (rhs.inlineSize_ > inlineSize_) {
          for (const auto ii : c10::irange(inlineSize_, rhs.inlineSize_)) {
            new (&elementsInline_[ii]) IValue(std::move(rhs.elementsInline_[ii]));
          }
        } else {
          for (const auto ii : c10::irange(rhs.inlineSize_, inlineSize_)) {
            elementsInline_[ii].~IValue();
          }
        }
      } else {
        destroyInline();
        new (&elementsVector_) std::vector<IValue>(std::move(rhs.elementsVector_));
      }
    } else {
      if (rhs.inlineSize_) {
        elementsVector_.~vector();
        for (const auto ii : c10::irange(rhs.inlineSize_)) {
          new (&elementsInline_[ii]) IValue(std::move(rhs.elementsInline_[ii]));
        }
      } else {
        elementsVector_ = std::move(rhs.elementsVector_);
      }
    }
    inlineSize_ = rhs.inlineSize_;
    return *this;
  }

  C10_NODISCARD c10::ArrayRef<IValue> asArrayRef() const {
    if (inlineSize_) {
      return c10::ArrayRef<IValue>(elementsInline_, inlineSize_);
    } else {
      return elementsVector_;
    }
  }

  // Mimic implicit conversion from std::vector to ArrayRef.
  operator c10::ArrayRef<IValue>() const {
    return asArrayRef();
  }

  static size_t hash(const TupleElements& v) {
    return c10::hash<c10::ArrayRef<IValue>>()(v.asArrayRef());
  }

  void setContents(std::vector<IValue>&& contents) {
    if (inlineSize_) {
      destroyInline();
      new (&elementsVector_) std::vector<IValue>(std::move(contents));
      inlineSize_ = 0;
    } else {
      elementsVector_ = std::move(contents);
    }
  }

  C10_NODISCARD bool empty() const {
    return inlineSize_ ? false : elementsVector_.empty();
  }

  C10_NODISCARD size_t size() const {
    return inlineSize_ ? inlineSize_ : elementsVector_.size();
  }

  C10_NODISCARD IValue& operator[](size_t idx) {
    if (inlineSize_) {
      return elementsInline_[idx];
    } else {
      return elementsVector_[idx];
    }
  }

  C10_NODISCARD const IValue& operator[](size_t idx) const {
    if (inlineSize_) {
      return elementsInline_[idx];
    } else {
      return elementsVector_[idx];
    }
  }

  C10_NODISCARD IValue& at(size_t idx) {
    if (inlineSize_) {
      TORCH_INTERNAL_ASSERT_DEBUG_ONLY(inlineSize_ <= 3);
      TORCH_CHECK(idx < inlineSize_, "TupleElements: invalid index Index = ", idx, "; Length = ", inlineSize_);
      return elementsInline_[idx];
    } else {
      return elementsVector_.at(idx);
    }
  }

  C10_NODISCARD const IValue& at(size_t idx) const {
    if (inlineSize_) {
      TORCH_INTERNAL_ASSERT_DEBUG_ONLY(inlineSize_ <= 3);
      TORCH_CHECK(idx < inlineSize_, "TupleElements: invalid index Index = ", idx, "; Length = ", inlineSize_);
      return elementsInline_[idx];
    } else {
      TORCH_CHECK(idx < elementsVector_.size(), "TupleElements: invalid index Index = ", idx, "; Length = ", elementsVector_.size());
      return elementsVector_.at(idx);
    }
  }

  C10_NODISCARD iterator begin() {
    if (inlineSize_) {
      return elementsInline_;
    } else {
      return elementsVector_.data();
    }
  }

  C10_NODISCARD iterator end() {
    if (inlineSize_) {
      return elementsInline_ + inlineSize_;
    } else {
      return elementsVector_.data() + elementsVector_.size();
    }
  }

  C10_NODISCARD const_iterator begin() const {
    if (inlineSize_) {
      return elementsInline_;
    } else {
      return elementsVector_.data();
    }
  }

  C10_NODISCARD const_iterator end() const {
    if (inlineSize_) {
      return elementsInline_ + inlineSize_;
    } else {
      return elementsVector_.data() + elementsVector_.size();
    }
  }

  C10_NODISCARD const_iterator cbegin() const {
    return begin();
  }

  C10_NODISCARD const_iterator cend() const {
    return end();
  }

  C10_NODISCARD std::vector<IValue> vec() const & {
    return asArrayRef().vec();
  }

  C10_NODISCARD IValue& back() {
    return *(end() - 1);
  }

  C10_NODISCARD const IValue& back() const {
    return *(end() - 1);
  }

  C10_NODISCARD std::vector<IValue> vec() && {
    std::vector<IValue> result;
    result.reserve(size());
    for (auto&& iv : *this) {
      result.push_back(std::move(iv));
    }
    return result;
  }

  // More compatibility shims for the overwhelming amount of code that
  // likes to copy tuple elements into a vector; see comment above the
  // copy constructor.
  operator std::vector<IValue>() const & {
    return vec();
  }

  operator std::vector<IValue>() && {
    return vec();
  }
};

template <typename T>
struct TupleTypeFactory {};

template <>
struct TORCH_API TupleTypeFactory<TupleType> {
  static TupleTypePtr create(std::vector<TypePtr> types) {
    return TupleType::create(std::move(types));
  }
  static TupleTypePtr fallback(const Type& type);
};

template <>
struct TORCH_API TupleTypeFactory<c10::DynamicType> {
  static DynamicTypePtr create(std::vector<TypePtr> elemTypes);
  static DynamicTypePtr fallback(const Type&);
};

struct TORCH_API Tuple : c10::intrusive_ptr_target {
 private:
  TupleElements elements_;
  mutable c10::TypePtr type_; // lazily computed for unnamed tuples

 public:
  // named tuples have additional type information, so we
  // directly create them tagged
  static c10::intrusive_ptr<Tuple> createNamed(
      std::vector<IValue> elements_,
      c10::TypePtr type_) {
    return c10::make_intrusive<Tuple>(std::move(elements_), std::move(type_));
  }

  static c10::intrusive_ptr<Tuple> createNamed(
      TupleElements elements_,
      std::shared_ptr<TupleType> type_) {
    return c10::make_intrusive<Tuple>(std::move(elements_), std::move(type_));
  }

  static c10::intrusive_ptr<Tuple> createNamed(
      std::initializer_list<IValue> elements_,
      std::shared_ptr<TupleType> type_) {
    return createNamed(TupleElements(c10::ArrayRef<IValue>(elements_)), std::move(type_));
  }

  // MSVC apparently can't disambiguate the other two overloads of
  // create when passed an initializer_list without this.
  static c10::intrusive_ptr<Tuple> create(std::initializer_list<IValue> elements_) {
    return create(c10::ArrayRef<IValue>(elements_));
  }

  static c10::intrusive_ptr<Tuple> create(std::vector<IValue> elements_) {
    return c10::make_intrusive<Tuple>(std::move(elements_));
  }

  static c10::intrusive_ptr<Tuple> create(TupleElements elements_) {
    return c10::make_intrusive<Tuple>(std::move(elements_));
  }

  static c10::intrusive_ptr<Tuple> create(c10::ArrayRef<IValue> elements_) {
    return create(TupleElements(elements_));
  }

  static c10::intrusive_ptr<Tuple> create(IValue e1) {
    return c10::make_intrusive<Tuple>(std::move(e1));
  }

  static c10::intrusive_ptr<Tuple> create(IValue e1, IValue e2) {
    return c10::make_intrusive<Tuple>(std::move(e1), std::move(e2));
  }

  static c10::intrusive_ptr<Tuple> create(IValue e1, IValue e2, IValue e3) {
    return c10::make_intrusive<Tuple>(std::move(e1), std::move(e2), std::move(e3));
  }

 private:
  // Workaround inability to use `>` operator in template argument list.
  template <typename... Args>
  static constexpr bool hasMoreThanThreeArgs() {
    return sizeof...(Args) > 3;
  }

 public:
  template <typename... Args>
  static c10::intrusive_ptr<Tuple> create(Args&&... elements_) {
    switch (sizeof...(Args)) {
      case 1:
      case 2:
      case 3:
        return create(IValue(std::forward<Args>(elements_))...);
      default:
        return create(
            std::vector<IValue>{IValue(std::forward<Args>(elements_))...});
    }
  }

  // Again, it would be nice to make this noncopyable, but there's a
  // lot of extant code that copies Tuples.
  // Tuple(const Tuple& rhs) = delete;

  const TupleElements& elements() const& {
    return elements_;
  }

  TupleElements elements() && {
    return std::move(elements_);
  }

  void setElements(std::vector<IValue>&& elements) {
    elements_.setContents(std::move(elements));
  }

  void setElements(TupleElements&& elements) {
    elements_ = std::move(elements);
  }

  void unsafeSetElement(size_t idx, const IValue& element) {
    elements_[idx] = element;
  }

  void unsafeSetElement(size_t idx, IValue&& element) {
    elements_[idx] = std::move(element);
  }

  size_t size() const {
    return elements_.size();
  }

  template <typename T = c10::TupleType>
  std::shared_ptr<T> type() const {
    if (!type_) {
      type_ = TupleTypeFactory<T>::create(fmap(elements(), [&](const IValue& v) {
        return v.type<typename T::ElementType>();
      }));
    }
    if (auto t = type_->cast<T>()) {
      return t;
    }
    return TupleTypeFactory<T>::fallback(*type_);
  }

  static size_t hash(const Tuple& t) {
    return c10::get_hash(t.elements());
  }

  TORCH_API friend bool operator==(
      const ivalue::Tuple& lhs,
      const ivalue::Tuple& rhs);

 private:
  // NOTE: If we try to avoid the overloads without
  // `std::shared_ptr<TupleType> type` by defaulting it to nullptr, we
  // end up having to call (part of) the shared_ptr destructor for
  // `type` even though we should know statically it won't do
  // anything.
  explicit Tuple(std::vector<IValue> elements)
    : elements_(std::move(elements)){}

  explicit Tuple(std::vector<IValue> elements, c10::TypePtr type)
    : elements_(std::move(elements)), type_(std::move(type)) {}

  explicit Tuple(TupleElements&& elements)
    : elements_(std::move(elements)) {}

  explicit Tuple(TupleElements&& elements, std::shared_ptr<TupleType> type)
    : elements_(std::move(elements)), type_(std::move(type)) {}

  explicit Tuple(IValue&& e1)
    : elements_(std::move(e1)) {}

  explicit Tuple(IValue&& e1, std::shared_ptr<TupleType> type)
    : elements_(std::move(e1)), type_(std::move(type)) {}

  explicit Tuple(IValue&& e1, IValue&& e2)
    : elements_(std::move(e1), std::move(e2)) {}

  explicit Tuple(IValue&& e1, IValue&& e2, std::shared_ptr<TupleType> type)
    : elements_(std::move(e1), std::move(e2)), type_(std::move(type)) {}

  explicit Tuple(IValue&& e1, IValue&& e2, IValue&& e3)
    : elements_(std::move(e1), std::move(e2), std::move(e3)) {}

  explicit Tuple(IValue&& e1, IValue&& e2, IValue&& e3, std::shared_ptr<TupleType> type)
    : elements_(std::move(e1), std::move(e2), std::move(e3)), type_(std::move(type)) {}

  friend class c10::intrusive_ptr<Tuple>;
};

struct Object;
struct PyObjectHolder;
struct EnumHolder;
} // namespace ivalue

// Future
struct C10_EXPORT ivalue::Future final : c10::intrusive_ptr_target {
 private:
  // Keep this private in order to force users to go through make_intrusive and
  // thus prevent creating a Future that's not held by an intrusive_ptr.
  explicit Future(TypePtr type, std::vector<c10::Device> devices={})
      : type_(std::move(type)),
        impl_(getTypeOfDevices(devices)),
        devices_(sortAndDeduplicateDevices(impl_, std::move(devices))) {}

  friend c10::intrusive_ptr<Future>;

 public:
  Future(const Future&) = delete;
  Future(Future&&) = delete;
  Future& operator=(const Future&) = delete;
  Future& operator=(Future&&) = delete;

  struct TORCH_API FutureError final : public std::exception {
    explicit FutureError(std::string&& error_msg_)
        : error_msg(std::move(error_msg_)) {}

    FutureError() = default;

    const char* what() const noexcept override {
      return error_msg.c_str();
    }

    std::string error_msg;
  };

  /**
   * Wait on the future until it completes.
   */
  void wait() {
    std::unique_lock<std::mutex> lock(mutex_);
    finished_cv_.wait(lock, [&]() -> bool { return completed_; });
    synchronizeWithCurrentStreams();
  }

  /**
   * Wait on the future until it completes and throw an
   * exception if an error exists.
   */
  void waitAndThrow() {
    wait();

    if (eptr_) {
      std::rethrow_exception(eptr_);
    }
  }

  /**
   * Explicitly mark the future as completed with the output value. Optionally,
   * the storages for all tensors in IValue can be passed as well. The DataPtrs
   * of these storages are used to synchronize CUDA streams. If storages isn't
   * given we will attempt to extract it from the value, if we need to (this
   * happens if a non-empty set of devices was given to the constructor). Thus
   * one only needs to provide storages when 1) they cannot be extracted through
   * IValue::getSubValues() or through pickling in case of Python object; or
   * when 2) customized storage extraction is more efficient.
   */
  using WeakStorage = c10::weak_intrusive_ptr<c10::StorageImpl>;
  void markCompleted(
      IValue value,
      c10::optional<std::vector<WeakStorage>> storages = c10::nullopt) {
    // Start by performing all steps that can throw, before setting any field.
    // Do this before even acquiring the mutex, because extractStorages might
    // acquire the GIL, which could lead to a lock inversion with our mutex.
    // See https://github.com/pytorch/pytorch/issues/58239.
    std::vector<WeakStorage> actualStorages;
    std::vector<c10::Device> usedDevices;
    try {
      // FIXME We should always extract DataPtrs, in order to catch the case of
      // users using CUDA values but forgetting to set devices, which currently
      // leads to a silent synchronization/correctness issue. However, as this
      // might worsen perf in CPU-only cases, we should only do so after careful
      // benchmarks.
      if (impl_.type() != c10::kCPU) {
        actualStorages =
            storages.has_value() ? std::move(*storages) : extractStorages(value);
        usedDevices = getDevicesOfStorages(impl_, actualStorages);
        ensureIsSubsetOfDevices(usedDevices, devices_);
      }
    } catch (const std::exception&) {
      setError(std::current_exception());
      return;
    }

    std::unique_lock<std::mutex> lock(mutex_);
    TORCH_CHECK(
        !completed(),
        "Attempting to mark a completed Future as complete again. Note that "
        "a Future can only be marked completed once.");

    // Only set value_ and completed_ flag once all checks and preparation steps
    // have returned successfully to allow for proper error propagation.
    value_ = std::move(value);
    completed_ = true;

    currentDevice_ = impl_.getDevice();
    storages_ = std::move(actualStorages);
    for (const c10::Device& device : usedDevices) {
      c10::Event event(impl_.type());
      event.record(impl_.getStream(device));
      events_.push_back(std::move(event));
    }

    std::vector<std::function<void(Future&)>> cbs;
    cbs.swap(callbacks_);
    lock.unlock();

    finished_cv_.notify_all();
    for (auto& callback : cbs) {
      invokeCallback(std::move(callback));
    }
  }

  void markCompleted() {
    markCompleted(IValue{});
  }

  void setError(std::exception_ptr eptr) {
    std::unique_lock<std::mutex> lock(mutex_);
    setErrorInternal(std::move(eptr), lock);
  }

  void setErrorIfNeeded(std::exception_ptr eptr) {
    std::unique_lock<std::mutex> lock(mutex_);
    if (completed_) {
      // This should be rare and shouldn't cause log spew. Its important to
      // log errors and thats why we have this log here.
      std::string msg = c10::str(
          "Skipping setting following error on the Future since "
          "it is already marked completed (this is not necessarily "
          "an error):\n",
          tryRetrieveErrorMessageInternal(std::move(eptr)));
      if (eptr_) {
        msg += c10::str(
            ", \nOriginal exception:\n",
            tryRetrieveErrorMessageInternal(eptr_));
      }
      LOG(INFO) << msg;
      return;
    } else {
      setErrorInternal(std::move(eptr), lock);
    }
  }

  // Get the result of the current future.
  IValue value() {
    std::unique_lock<std::mutex> lock(mutex_);
    AT_ASSERT(completed());
    if (eptr_) {
      std::rethrow_exception(eptr_);
    }
    return value_;
  }

  // This accessor should only be used if we know that the future is
  // completed() with no error.
  const IValue& constValue() const {
    std::unique_lock<std::mutex> lock(mutex_);
    AT_ASSERT(completed());
    TORCH_INTERNAL_ASSERT(
      !eptr_,
      "value() accessor should only be used when future is not completed with ",
      "an error, but future had the following error: ",
      tryRetrieveErrorMessageInternal(eptr_)
    );
    return value_;
  }

  // This accessor should only be used if we know that the future is
  // completed() with no error.
  const std::vector<WeakStorage>& storages() const {
    std::unique_lock<std::mutex> lock(mutex_);
    AT_ASSERT(completed());
    AT_ASSERT(!eptr_);
    return storages_;
  }

  /**
   * Add a callback to the future.
   * The callbacks will be executed once the future completes.
   * If the future has already completed,
   * this function will execute the callback immediately.
   */
  template <typename T>
  void addCallback(T callback) {
#if __cpp_lib_is_invocable >= 201703
    static_assert(
        std::is_invocable_r<void, T, Future&>::value,
        "The callback must have signature void(Future&)");
#endif
    std::unique_lock<std::mutex> lock(mutex_);
    if (completed()) {
      lock.unlock();
      invokeCallback(std::move(callback));
      return;
    }
    callbacks_.emplace_back(std::move(callback));
  }

  /**
   * Add a callback to the future, and return another Future to hold the return
   * value of the callback. This is necessary when the callback provider needs
   * to know for sure when the callback has finished.
   */
  template <typename T>
  c10::intrusive_ptr<Future> then(T callback, TypePtr type) {
    using IValueWithStorages = std::tuple<IValue, std::vector<WeakStorage>>;
#if __cpp_lib_is_invocable >= 201703
    static_assert(
        guts::disjunction<
            std::is_invocable_r<IValue, T, Future&>,
            std::is_invocable_r<IValueWithStorages, T, Future&>>::value,
        "The callback must have signature IValue(Future&) or "
        "std::tuple<IValue, std::vector<Storage>>(Future&)");
#endif
    auto childFut = createInstance(::std::move(type));
    addCallback([childFut,
                 cb = std::move(callback)](Future& parentFut) mutable {
      try {
        if constexpr (::std::is_convertible_v<typename c10::invoke_result_t<T &&, Future&>, IValueWithStorages>) {
          auto [ivalue, storages] = cb(parentFut);
          childFut->markCompleted(::std::move(ivalue), ::std::move(storages));
        } else {
          childFut->markCompleted(cb(parentFut));
        }
      } catch (std::exception&) {
        childFut->setError(std::current_exception());
      }
    });
    return childFut;
  }

  template <typename T>
  c10::intrusive_ptr<Future> thenAsync(T callback, TypePtr type) {
#if __cpp_lib_is_invocable >= 201703
    static_assert(
        std::is_invocable_r<c10::intrusive_ptr<Future>, T, Future&>::value,
        "The callback must have signature c10::intrusive_ptr<Future>(Future&)");
#endif
    auto childFut = createInstance(std::move(type));
    addCallback(
        [childFut, cb = std::move(callback)](Future& parentFut) mutable {
          c10::intrusive_ptr<Future> intermediateFut;
          try {
            intermediateFut = cb(parentFut);
          } catch (std::exception&) {
            childFut->setError(std::current_exception());
            return;
          }
          intermediateFut->addCallback(
              [childFut = std::move(childFut)](Future& intermediateFut) {
                if (intermediateFut.hasError()) {
                  childFut->setError(intermediateFut.exception_ptr());
                } else {
                  childFut->markCompleted(
                      intermediateFut.value(), intermediateFut.storages());
                }
              });
        });
    return childFut;
  }

  // Tries to retrieve the error message from std::exception_ptr.
  std::string tryRetrieveErrorMessage() const {
    TORCH_CHECK(hasError(), "No error present on the future.");
    std::unique_lock<std::mutex> lock(mutex_);
    return tryRetrieveErrorMessageInternal(eptr_);
  }

  // Check if the current future has completed
  bool completed() const {
    return completed_;
  }

  bool hasValue() const {
    std::unique_lock<std::mutex> lock(mutex_);
    return completed_ && !eptr_;
  }

  bool hasError() const {
    std::unique_lock<std::mutex> lock(mutex_);
    return eptr_ ? true : false;
  }

  std::exception_ptr exception_ptr() const {
    std::unique_lock<std::mutex> lock(mutex_);
    return eptr_;
  }

  TORCH_API friend std::ostream& operator<<(
      std::ostream& out,
      const Future& v);

  TypePtr elementType() const {
    return type_;
  }

  const std::vector<c10::Device>& devices() const {
    return devices_;
  }

  // This method should be used when one intends to manually create a child
  // future, for example when implementing a customized version of then().
  c10::intrusive_ptr<Future> createInstance(at::TypePtr type) {
    return c10::make_intrusive<Future>(std::move(type), devices_);
  }

 private:

  // This method should always be used when invoking a callback (regardless of
  // how/when that happens) as it will ensure that the proper "environment" is
  // set up before running the callback, as in, it will set up the CUDA streams,
  // synchronize them with the value, and so on (if needed).
  template<typename T>
  void invokeCallback(T callback) {
#if __cpp_lib_is_invocable >= 201703
    static_assert(
        std::is_invocable_r<void, T, Future&>::value,
        "The callback must have signature void(Future&)");
#endif

    c10::OptionalDeviceGuard deviceGuard(currentDevice_);

    std::vector<c10::Stream> streams;
    streams.reserve(devices_.size());
    for (const c10::Device& device : devices_) {
      streams.push_back(impl_.getStreamFromGlobalPool(device));
    }
    c10::MultiStreamGuard streamGuard(streams);
    synchronizeWithCurrentStreams();

    callback(*this);
  }

  // This method should be called before this future's value is used, as it
  // ensures that the CUDA streams that are "current" at the callsite properly
  // synchronize with the value.
  void synchronizeWithCurrentStreams() {
    for (c10::Event& event : events_) {
      event.block(impl_.getStream(event.device()));
    }

    for (const WeakStorage& weak_storage : storages_) {
      c10::intrusive_ptr<c10::StorageImpl> storage = weak_storage.lock();
      if (!storage) {
        continue;
      }
      if (!storage->device().is_cpu()) {
        impl_.recordDataPtrOnStream(
            storage->data_ptr(), impl_.getStream(storage->device()));
      }
    }
  }

  void setErrorInternal(
      std::exception_ptr eptr,
      std::unique_lock<std::mutex>& lock) {
    TORCH_CHECK(
        !eptr_,
        "Error already set on this Future: ",
        tryRetrieveErrorMessageInternal(eptr_),
        ", trying to set error: ",
        tryRetrieveErrorMessageInternal(eptr));
    TORCH_INTERNAL_ASSERT(!completed(), "Future is already marked completed");
    completed_ = true;
    eptr_ = std::move(eptr);

    std::vector<std::function<void(Future&)>> cbs;
    cbs.swap(callbacks_);
    lock.unlock();

    finished_cv_.notify_all();
    for (auto& callback : cbs) {
      invokeCallback(std::move(callback));
    }
  }

  // Tries to retrieve the error message from std::exception_ptr.
  std::string tryRetrieveErrorMessageInternal(std::exception_ptr eptr) const {
    try {
      std::rethrow_exception(std::move(eptr));
    } catch (const std::exception& e) {
      return e.what();
    } catch (...) {
      return "Unknown Exception Type";
    }
  }

  // Defined in ivalue.cpp.
  static std::vector<WeakStorage> extractStorages(
      const at::IValue& value);

  static std::vector<c10::Device> getDevicesOfStorages(
      const c10::impl::VirtualGuardImpl& impl,
      const std::vector<WeakStorage>& storages) {
    c10::DeviceIndex deviceCount = impl.deviceCount();
    std::vector<bool> isDeviceUsed(deviceCount, false);
    for (const WeakStorage& weak_storage : storages) {
      c10::intrusive_ptr<c10::StorageImpl> storage = weak_storage.lock();
      if (!storage) {
        continue;
      }
      c10::Device device = storage->device();
      if (!device.is_cpu()) {
        TORCH_CHECK_VALUE(
            device.type() == impl.type(),
            "Expected all data ptrs to be on a device of type ",
            impl.type(),
            ", got one on device ",
            device);
        isDeviceUsed[device.index()] = true;
      }
    }
    std::vector<c10::Device> devices;
    for (c10::DeviceIndex idx = 0; idx < deviceCount; idx++) {
      if (isDeviceUsed[idx]) {
        devices.emplace_back(impl.type(), idx);
      }
    }
    return devices;
  }

  static std::string formatSetOfDevices(
      const std::vector<c10::Device>& devices) {
    if (devices.empty()) {
      return "(none)";
    }
    std::ostringstream oss;
    oss << devices[0];
    for (const auto idx : c10::irange(1, devices.size())) {
      if (idx == devices.size() - 1) {
        oss << " and ";
      } else {
        oss << ", ";
      }
      oss << devices[idx];
    }
    return oss.str();
  }

  static c10::DeviceType getTypeOfDevices(
      const std::vector<c10::Device>& devices) {
    if (devices.empty()) {
      return c10::kCPU;
    }
    c10::DeviceType deviceType = devices[0].type();
    for (const auto idx : c10::irange(1, devices.size())) {
      TORCH_CHECK_VALUE(
          devices[idx].type() == deviceType,
          "Expected all devices to be of the same type, but got a mismatch between ",
          devices[0],
          " and ",
          devices[idx]);
    }
    return deviceType;
  }

  // We need devices to be sorted in order to use ensureIsSubsetOfDevices.
  static std::vector<c10::Device> sortAndDeduplicateDevices(
      const c10::impl::VirtualGuardImpl& /*impl*/,
      std::vector<c10::Device> devices) {
    std::sort(
      devices.begin(), devices.end(),
      [](const c10::Device& a, const c10::Device& b) { return a.index() < b.index(); });
    // Deduplicate by compacting.
    size_t targetIdx = 0;
    for (const auto sourceIdx : c10::irange(devices.size())) {
      TORCH_CHECK_VALUE(
          devices[sourceIdx].has_index(),
          "Expected devices to have indices, got ", devices[sourceIdx]);
      if (targetIdx > 0 && devices[targetIdx - 1].index() == devices[sourceIdx].index()) {
        // It's a duplicate, skip it.
        continue;
      }
      if (sourceIdx != targetIdx) {
        devices[targetIdx] = devices[sourceIdx];
      }
      targetIdx++;
    }
    // If there were duplicates there's now a gap at the end: trim it. Resizing
    // requires the item type to be default-constructible (which c10::Device is
    // not) because in principle it could be required to create new items. Since
    // we know we'll shrink the vector, we provide a custom dummy value instead.
    devices.resize(targetIdx, c10::Device(c10::kCPU));
    return devices;
  }

  static void ensureIsSubsetOfDevices(
      const std::vector<c10::Device>& subset,
      const std::vector<c10::Device>& superset) {
    // We assume the devices in both vectors have the same consistent type, and
    // their indices are unique and sorted.
    std::vector<c10::Device> excessDevices;
    std::set_difference(
        subset.begin(),
        subset.end(),
        superset.begin(),
        superset.end(),
        std::back_inserter(excessDevices),
        [](const c10::Device& a, const c10::Device& b) { return a.index() < b.index(); });
    TORCH_CHECK_VALUE(
        excessDevices.empty(),
        "The result contained tensors residing on device(s) ",
        formatSetOfDevices(excessDevices),
        " which are not among the expected device(s) ",
        formatSetOfDevices(superset));
  }

  mutable std::mutex mutex_;
  std::atomic_bool completed_ = {false}; // is this future complete
  std::condition_variable finished_cv_;

  IValue value_; // when finished the value
  TypePtr type_;
  std::vector<std::function<void(Future&)>> callbacks_;
  std::exception_ptr eptr_;

  // An upcast pointer to a virtual class which allows us to manipulate events,
  // streams, ... in a generic way, without an explicit dependency on CUDA.
  const c10::impl::VirtualGuardImpl impl_;

  // The device that was current when markCompleted was called, which we'll
  // restore when invoking callbacks. It's optional because we'll only store it
  // if the future completes successfully.
  optional<c10::Device> currentDevice_;

  // The events that correspond to the completion of the async I/O kernels. They
  // are recorded on the appropriate streams when the future is marked completed
  // and can then be queried/waited/blocked on. There is one event for each
  // distinct device on which the value's tensors reside.
  std::vector<c10::Event> events_;

  // A cached version of the storages extracted from the value when the future
  // is first marked completed.
  std::vector<WeakStorage> storages_;

  // The bounding set of devices that this future, and any of its children, is
  // allowed to use. This is a superset of the set of devices used by the events
  // above. We need this to know what streams (for which devices) to set as
  // current when invoking a callback, thus allowing the callback to use devices
  // that the parent future didn't use. This field is set to the value provided
  // in the constructor and will be "inherited" by all child futures.
  const std::vector<c10::Device> devices_;
};

struct C10_EXPORT ivalue::Await final : c10::intrusive_ptr_target {
 private:
  explicit Await(TypePtr elType, std::function<IValue()> fn)
      : elType_(std::move(elType)), type_(AwaitType::create(elType_)), fn_(std::move(fn)) {}

  explicit Await(TypePtr elType) : elType_(std::move(elType)), type_(AwaitType::create(elType_)) { }

  friend c10::intrusive_ptr<Await>;

 public:
  Await(const Await&) = delete;
  Await(Await&&) = delete;
  Await& operator=(const Await&) = delete;
  Await& operator=(Await&&) = delete;

  IValue wait() {
    if (!completed_) {
      TORCH_CHECK(fn_, "Incompleted Await: fn can't be None");
      value_ = fn_();
      completed_ = true;
      args_ = {};
    }
    return value_;
  }

  IValue value() {
    TORCH_CHECK(completed_, "Await must be completed");
    return value_;
  }

  void setFn(std::function<IValue()> fn) {
    fn_ = std::move(fn);
  }

  bool completed() {
    return completed_;
  }

  void markCompleted(IValue value) {
    value_ = std::move(value);
    completed_ = true;
  }

  TORCH_API friend std::ostream& operator<<(
      std::ostream& out,
      const Await& v);

  TypePtr elementType() const {
    return elType_;
  }

  TypePtr type() const {
    return type_;
  }

  void setArgs(std::vector<IValue> args) {
    args_ = std::move(args);
  }

  std::vector<IValue>& args() {
    return args_;
  }

 private:
  TypePtr elType_;
  TypePtr type_;
  std::vector<IValue> args_;
  std::function<IValue()> fn_;
  IValue value_;
  bool completed_{};
};

// Input is a list of Futures with the same target type.
// Output is a Future to the List of completed Futures.
TORCH_API intrusive_ptr<ivalue::Future> collectAll(
    c10::List<c10::intrusive_ptr<ivalue::Future>> srcs);
// Input is a List of Futures with the same target type.
// Output is a Future that will be updated with a seen value.
TORCH_API intrusive_ptr<ivalue::Future> collectAny(
    c10::List<c10::intrusive_ptr<ivalue::Future>> srcs);

// User-defined object.
struct C10_EXPORT ivalue::Object final : c10::intrusive_ptr_target {
 public:
  // In general, class types hold a shared_ptr to its owning CompilationUnit,
  // so that its type and methods do not get deallocated while the class exists.
  // However, the CompilationUnit holds ownership of the type's graphs, so
  // inserting a constant object into a Graph would create a reference cycle if
  // that constant object held a shared_ptr to its CU. For these objects we
  // instatiate them with non-owning references to its CU
  Object(WeakOrStrongTypePtr type, size_t numSlots) : type_(std::move(type)) {
    slots_.resize(numSlots);
  }

  Object(StrongTypePtr type, size_t numSlots)
      : type_(WeakOrStrongTypePtr(std::move(type))) {
    slots_.resize(numSlots);
  }

  static c10::intrusive_ptr<Object> create(
      WeakOrStrongTypePtr type,
      size_t numSlots) {
    return c10::make_intrusive<Object>(std::move(type), numSlots);
  }

  static c10::intrusive_ptr<Object> create(
      StrongTypePtr type,
      size_t numSlots) {
    return c10::make_intrusive<Object>(std::move(type), numSlots);
  }

  static c10::intrusive_ptr<Object> create(ClassTypePtr classType, size_t numSlots);

  /**
   * Slot API.
   *
   * Attributes are stored as a simple vector so that lookups are fast at
   * runtime. A "slot" is just an index into that vector, which can be computed
   * statically if you have access to the class type. Use this API if you are
   * writing compiler stuff.
   */
  void setSlot(size_t slot, IValue v) {
    if (slot >= slots_.size()) {
      // for module types, it is possible that the members of the class have
      // expanded after the object was created. In this case, we expand
      // the slots to the right size
      resizeObject(slot);
    }
    slots_[slot] = std::move(v);
  }

  const IValue& getSlot(size_t slot) const {
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(slot < slots_.size());
    // NOTE: This lookup is fairly hot, so we use unchecked access to the
    // vector.  Errors should still be detectable with ASan.
    return slots_[slot];
  }

  void unsafeRemoveSlot(size_t slot) {
    TORCH_CHECK(slot < slots_.size());
    slots_.erase(slots_.begin() + slot);
  }

  /**
   * Attribute API.
   *
   * Wrappers around the slot stuff so that users can access attributes
   * directly. Use this API if you are a user.
   *
   * Note: Unlike in Python, TorchScript must make a distinction between
   * attributes (which are IValues) and methods (which are Methods). If you
   * want a method, use `obj.type()->getMethod()`
   */
  IValue getAttr(const std::string& name) const;
  void setAttr(const std::string& name, IValue v);
  // Remove attribute by name, caller is responsible for
  // the safety of this operation
  // We didn't remove the attribute in the type because the type
  // might be shared by multiple objects.
  // Therefore after removing attribute, the object is in an inconsistent
  // state where it has more attribute types in its Type than
  // the attribute slots it has, user needs to make sure the object
  // has consistent by removing the attribute in type as well
  void unsafeRemoveAttr(const std::string& name);

  std::string name() const;

  const std::vector<IValue>& slots() const {
    return slots_;
  }
  std::shared_ptr<ClassType> type() const;

  std::shared_ptr<torch::jit::CompilationUnit> compilation_unit() {
    if (type_.holds_strong_ref()) {
      return type_.cu_.getStrongRefOrThrow();
    } else {
      auto weak_ptr = type_.cu_.getWeakRefOrThrow();
      return std::shared_ptr<torch::jit::CompilationUnit>(weak_ptr);
    }
  }

  c10::intrusive_ptr<Object> copy_to_weak_compilation_ref() const;

  void unsafe_make_weak_compilation_ref() {
    type_ = WeakOrStrongTypePtr(type_.asWeakTypePtr());
  }

  c10::intrusive_ptr<Object> copy() const;

  c10::intrusive_ptr<Object> deepcopy() const;

  c10::intrusive_ptr<Object> deepcopy(IValue::HashAliasedIValueMap& memo) const;

  bool is_weak_compilation_ref() const {
    return !type_.holds_strong_ref();
  }

  bool is_empty_strong_compilation_ref() const {
    return type_.holds_empty_strong_ref();
  }

 private:
  void resizeObject(size_t slot);
  WeakOrStrongTypePtr type_;
  std::vector<IValue> slots_;
};

// virtual ivalue PyObjectHolder that hold a py::object, we make this virtual
// because the py::object and refcounting logic should happen in libtorch_python
// see concrete implementation in python_ivalue.h
struct ivalue::PyObjectHolder : c10::intrusive_ptr_target {
 public:
  virtual PyObject* getPyObject() = 0;
  virtual c10::InferredType tryToInferType() = 0;
  virtual IValue toIValue(const TypePtr& type, c10::optional<int32_t> N = c10::nullopt) = 0;
  virtual std::string toStr() = 0;
  virtual std::vector<at::Tensor> extractTensors() = 0;

  ~PyObjectHolder() override = default;
};

struct ivalue::EnumHolder : c10::intrusive_ptr_target {
 public:
  EnumHolder(std::shared_ptr<EnumType> type, std::string name, IValue value)
      : type_(std::move(type)),
        name_(std::move(name)),
        value_(std::move(value)) {}

  bool is(const ivalue::EnumHolder& rhs) {
    return *this == rhs;
  }

  friend bool operator==(
      const ivalue::EnumHolder& lhs,
      const ivalue::EnumHolder& rhs);

  TORCH_API friend std::ostream& operator<<(
      std::ostream& out,
      const EnumHolder& v);

  TORCH_API const std::string qualifiedClassName() const;

  const std::string unqualifiedClassName() const;

  const std::string& name() const {
    return name_;
  }

  const IValue& value() const {
    return value_;
  }

  std::shared_ptr<EnumType> type() const {
    return type_;
  }

 private:
  std::shared_ptr<EnumType> type_;
  std::string name_;
  IValue value_;
};

#undef TORCH_FORALL_TAGS

namespace detail {

struct _guarded_unsigned_long_unique_dummy final {
  _guarded_unsigned_long_unique_dummy(int64_t){};
};
using _guarded_unsigned_long = std::conditional_t<
    std::is_same<unsigned long, uint32_t>::value ||
        std::is_same<unsigned long, uint64_t>::value,
    _guarded_unsigned_long_unique_dummy,
    unsigned long>;

} // namespace detail

inline ivalue::Object& IValue::toObjectRef() const {
  AT_ASSERT(isObject(), "Expected Object but got ", tagKind());
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(payload.u.as_intrusive_ptr != c10::UndefinedTensorImpl::singleton(), "Attempted to create null reference");
  return *static_cast<c10::ivalue::Object*>(payload.u.as_intrusive_ptr);
}

// note: when adding a DEFINE_TO case here you should also add a
// toX method to IValue. These named methods are much more discoverable
// than the to templated function.

#define DEFINE_TO(T, method_name)                          \
  template <>                                              \
  inline T IValue::to<T>()&& {                             \
    return static_cast<T>(std::move(*this).method_name()); \
  }                                                        \
  template <>                                              \
  inline c10::detail::ivalue_to_const_ref_overload_return<T>::type IValue::to<T>() const& { \
    typedef c10::detail::ivalue_to_const_ref_overload_return<T>::type return_type;          \
    return static_cast<return_type>(this->method_name());                                   \
  }

DEFINE_TO(at::Tensor, toTensor)
DEFINE_TO(at::Storage, toStorage)
DEFINE_TO(c10::Stream, toStream)
DEFINE_TO(float, toDouble)
DEFINE_TO(double, toDouble)
DEFINE_TO(c10::complex<double>, toComplexDouble)
DEFINE_TO(unsigned char, toInt)
DEFINE_TO(signed char, toInt)
DEFINE_TO(unsigned short, toInt)
DEFINE_TO(short, toInt)
DEFINE_TO(int, toInt)
DEFINE_TO(uint32_t, toInt)
DEFINE_TO(uint64_t, toInt)
DEFINE_TO(detail::_guarded_unsigned_long, toInt)
DEFINE_TO(int64_t, toInt)
DEFINE_TO(bool, toBool)
DEFINE_TO(c10::intrusive_ptr<caffe2::Blob>, toBlob);
DEFINE_TO(c10::intrusive_ptr<ivalue::ConstantString>, toString)
DEFINE_TO(c10::intrusive_ptr<ivalue::Object>, toObject)
DEFINE_TO(at::Scalar, toScalar)
DEFINE_TO(c10::List<int64_t>, toIntList)
DEFINE_TO(c10::List<double>, toDoubleList)
DEFINE_TO(c10::List<c10::complex<double>>, toComplexDoubleList)
DEFINE_TO(c10::List<bool>, toBoolList)
DEFINE_TO(c10::List<at::Tensor>, toTensorList)
DEFINE_TO(c10::impl::GenericList, toList)
DEFINE_TO(c10::impl::GenericDict, toGenericDict)
DEFINE_TO(c10::intrusive_ptr<ivalue::Tuple>, toTuple)
DEFINE_TO(std::string, toStringRef)
DEFINE_TO(c10::string_view, toStringView)
DEFINE_TO(c10::intrusive_ptr<ivalue::Future>, toFuture)
DEFINE_TO(c10::intrusive_ptr<ivalue::Await>, toAwait)
DEFINE_TO(c10::intrusive_ptr<c10::RRefInterface>, toRRef)
DEFINE_TO(c10::intrusive_ptr<at::Quantizer>, toQuantizer)
DEFINE_TO(IValue, toIValue)
DEFINE_TO(c10::Device, toDevice)
DEFINE_TO(at::ScalarType, toScalarType)
DEFINE_TO(at::Layout, toLayout)
DEFINE_TO(at::MemoryFormat, toMemoryFormat)
DEFINE_TO(at::QScheme, toQScheme)
DEFINE_TO(at::Dimname, toDimname)
DEFINE_TO(at::Generator, toGenerator)
DEFINE_TO(c10::SymInt, toSymInt)
DEFINE_TO(c10::SymFloat, toSymFloat)
DEFINE_TO(c10::SymBool, toSymBool)

template <class T>
struct _fake_type {};

// generic_to<T> converts an IValue from a generic list or generic dict
// to a concrete list/dict type likelike List<T>, Dict<...> or optional<T>.
// Note that in the case of lists, this only works for IValue-based lists,
// i.e. not for int64_t, double, ...
// generic_to<T> is an implementation detail of IValue::to<T> and not
// supposed to be called directly.
// The _fake_type<T> parameter allows us to overload
// based on the return type.
template <class Elem>
// TODO this is deprecated but we don't throw a warning because a lot of ops in
// native_functions.yaml still return std::vector.
// C10_DEPRECATED_MESSAGE("IValues based on std::vector<T> are potentially slow
// and deprecated. Please use torch::List<T> instead.")
std::vector<Elem> generic_to(IValue ivalue, _fake_type<std::vector<Elem>>) {
  // We need to do a deep copy of the vector because there might be other
  // references to this same IValue that also use the list. We can't just
  // move the elements out.
  auto list = std::move(ivalue).to<List<Elem>>();
  std::vector<Elem> result;
  result.reserve(list.size());
  for (Elem v : list) {
    result.push_back(std::move(v));
  }
  return result;
}

template <typename T>
c10::intrusive_ptr<T> IValue::toCustomClass() && {
  static_assert(
      std::is_base_of<torch::CustomClassHolder, T>::value == true,
      "toCustomClass requires that template parameter T must inherit "
      "from torch::CustomClassHolder");
  auto obj = toObject();
  TORCH_CHECK(
      obj->slots().size() == 1,
      "Tried to cast IValue to custom class but it did "
      "not contain a custom class!");
  const auto* expected_type = c10::getCustomClassType<c10::intrusive_ptr<T>>().get();
  ivalue::checkCustomClassType(expected_type, type().get());
  auto userObj =
      c10::static_intrusive_pointer_cast<T>(obj->getSlot(0).toCapsule());
  return userObj;
}

template <typename T>
c10::intrusive_ptr<T> IValue::toCustomClass() const& {
  static_assert(
      std::is_base_of<torch::CustomClassHolder, T>::value == true,
      "toCustomClass requires that template parameter T must inherit "
      "from torch::CustomClassHolder");
  auto obj = toObject();
  TORCH_CHECK(
      obj->slots().size() == 1,
      "Tried to cast IValue to custom class but it did "
      "not contain a custom class!");
  const auto* expected_type = c10::getCustomClassType<c10::intrusive_ptr<T>>().get();
  ivalue::checkCustomClassType(expected_type, type().get());
  auto userObj =
      c10::static_intrusive_pointer_cast<T>(obj->getSlot(0).toCapsule());
  return userObj;
}

template <typename T>
T generic_to(IValue ivalue, _fake_type<T>) {
  using ElemType = typename std::remove_pointer<T>::type::element_type;
  return std::move(ivalue).toCustomClass<ElemType>();
}

template <typename T>
tagged_capsule<T> generic_to(IValue ivalue, _fake_type<tagged_capsule<T>>) {
  return tagged_capsule<T>{std::move(ivalue)};
}

template <typename Elem>
c10::List<Elem> generic_to(IValue ivalue, _fake_type<c10::List<Elem>>) {
  return impl::toTypedList<Elem>(std::move(ivalue).toList());
}

template <typename T>
static T createVectorLikeFromList(const c10::detail::ListImpl* impl) {
  T result;
  result.reserve(impl->list.size());
  for (const auto & i : impl->list) {
    result.push_back(i.to<typename T::value_type>());
  }
  return result;
}

template <typename T>
static std::vector<T> createVectorFromList(const c10::detail::ListImpl* impl) {
  return createVectorLikeFromList<std::vector<T>>(impl);
}

template <typename T>
std::vector<T> createVectorFromList(const c10::List<T>& impl) {
  std::vector<T> result;
  result.reserve(impl.size());
  for (size_t i = 0, N = impl.size(); i < N; ++i) {
    result.push_back(impl[i]);
  }
  return result;
}

template <typename T>
OptionalArray<T> generic_to(IValue ivalue, _fake_type<OptionalArray<T>>) {
  if (ivalue.isNone()) {
    return {};
  }
  return createVectorFromList<T>(
    std::move(ivalue).to<c10::List<T>>()
  );
}

namespace detail {
template <typename Elem, size_t... I>
std::array<Elem, sizeof...(I)> generic_to_array(
    IValue ivalue,
    _fake_type<std::array<Elem, sizeof...(I)>>,
    std::index_sequence<I...>) {
  // We need to do a deep copy of the array because there might be other
  // references to this same IValue that also use the list. We can't just
  // move the elements out.
  auto list = std::move(ivalue).to<List<Elem>>();
  TORCH_CHECK(
      list.size() == sizeof...(I),
      "Tried to convert a List with ",
      list.size(),
      " elements to a fixed-size array of size ",
      sizeof...(I));
  return {list[I]...};
}
} // namespace detail

template <typename Elem, size_t N>
std::array<Elem, N> generic_to(
    IValue ivalue,
    _fake_type<std::array<Elem, N>> ft) {
  return detail::generic_to_array(ivalue, ft, std::make_index_sequence<N>());
}

template <typename Key, typename Value>
c10::Dict<Key, Value> generic_to(
    IValue ivalue,
    _fake_type<c10::Dict<Key, Value>>) {
  return impl::toTypedDict<Key, Value>(std::move(ivalue).toGenericDict());
}

template <typename K, typename V>
C10_DEPRECATED_MESSAGE(
    "IValues based on std::unordered_map are slow and deprecated. Please use c10::Dict<K, V> instead.")
std::unordered_map<K, V> generic_to(
    IValue ivalue,
    _fake_type<std::unordered_map<K, V>>) {
  std::unordered_map<K, V> specialized_dict;

  for (const auto& item : std::move(ivalue).toGenericDict()) {
    specialized_dict[item.key().template to<K>()] = item.value().template to<V>();
  }

  return specialized_dict;
}

template <typename T>
c10::optional<T> generic_to(IValue ivalue, _fake_type<c10::optional<T>>) {
  if (ivalue.isNone()) {
    return c10::nullopt;
  }
  return std::move(ivalue).to<T>();
}

namespace detail {
template <typename Tuple, std::size_t... INDEX>
Tuple generic_to_tuple_impl(
    const ivalue::TupleElements& t,
    std::index_sequence<INDEX...>) {
  return std::make_tuple(
      t[INDEX].to<typename std::tuple_element<INDEX, Tuple>::type>()...);
}
} // namespace detail

template <
    typename... Args,
    typename Indices = std::make_index_sequence<sizeof...(Args)>,
    std::enable_if_t<
        !guts::disjunction<
            std::is_lvalue_reference<Args>...,
            guts::negation<std::is_constructible<IValue, Args>>...>::value,
        std::nullptr_t> = nullptr>
std::tuple<Args...> generic_to(IValue ivalue, _fake_type<std::tuple<Args...>>) {
  const auto& vals = ivalue.toTupleRef().elements();
  TORCH_CHECK(vals.size() == sizeof...(Args));
  return detail::generic_to_tuple_impl<std::tuple<Args...>>(vals, Indices{});
}

template <typename T>
inline T IValue::to() && {
  return generic_to(std::move(*this), _fake_type<T>{});
}

template <>
inline c10::optional<c10::string_view> IValue::to() && {
  // In the default implementation, the IValue is destroyed with std::move.
  // But if the unboxed type is optional<string_view> we cannot destroy
  // the IValue.
  return generic_to(*this, _fake_type<c10::optional<c10::string_view>>{});
}

template <typename T>
inline typename c10::detail::ivalue_to_const_ref_overload_return<T>::type IValue::to() const& {
  return generic_to(*this, _fake_type<T>{});
}

inline c10::List<int64_t> IValue::toIntList() && {
  AT_ASSERT(isIntList(), "Expected IntList but got ", tagKind());
  return c10::List<int64_t>(moveToIntrusivePtr<c10::detail::ListImpl>());
}
inline c10::List<int64_t> IValue::toIntList() const& {
  AT_ASSERT(isIntList(), "Expected IntList but got ", tagKind());
  return c10::List<int64_t>(toIntrusivePtr<c10::detail::ListImpl>());
}
inline std::vector<int64_t> IValue::toIntVector() const {
  AT_ASSERT(isIntList(), "Expected IntList but got ", tagKind());
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
      payload.u.as_intrusive_ptr != c10::UndefinedTensorImpl::singleton(),
      "called toIntVector on null intrusive_ptr IValue");
  return createVectorFromList<int64_t>(
      static_cast<const c10::detail::ListImpl*>(payload.u.as_intrusive_ptr));
}
inline std::vector<c10::SymInt> IValue::toSymIntVector() const {
  AT_ASSERT(isSymIntList() || isIntList(), "Expected SymIntList or IntList but got ", tagKind());
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
      payload.u.as_intrusive_ptr != c10::UndefinedTensorImpl::singleton(),
      "called toSymIntVector on null intrusive_ptr IValue");
  return createVectorFromList<c10::SymInt>(
      static_cast<const c10::detail::ListImpl*>(payload.u.as_intrusive_ptr));
}
inline at::DimVector IValue::toDimVector() const {
  AT_ASSERT(isIntList(), "Expected IntList but got ", tagKind());
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
      payload.u.as_intrusive_ptr != c10::UndefinedTensorImpl::singleton(),
      "called toDimVector on null intrusive_ptr IValue");
  return createVectorLikeFromList<at::DimVector>(
      static_cast<const c10::detail::ListImpl*>(payload.u.as_intrusive_ptr));
}
inline c10::List<double> IValue::toDoubleList() && {
  AT_ASSERT(isDoubleList(), "Expected DoubleList but got ", tagKind());
  return c10::List<double>(moveToIntrusivePtr<c10::detail::ListImpl>());
}
inline c10::List<double> IValue::toDoubleList() const& {
  AT_ASSERT(isDoubleList(), "Expected DoubleList but got ", tagKind());
  return c10::List<double>(toIntrusivePtr<c10::detail::ListImpl>());
}
inline std::vector<double> IValue::toDoubleVector() const {
  AT_ASSERT(isDoubleList(), "Expected DoubleList but got ", tagKind());
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
      payload.u.as_intrusive_ptr != c10::UndefinedTensorImpl::singleton(),
      "called toDoubleVector on null intrusive_ptr IValue");
  return createVectorFromList<double>(
      static_cast<const c10::detail::ListImpl*>(payload.u.as_intrusive_ptr));
}
inline c10::List<c10::complex<double>> IValue::toComplexDoubleList() && {
  AT_ASSERT(isComplexDoubleList(), "Expected ComplexDoubleList but got ", tagKind());
  return c10::List<c10::complex<double>>(moveToIntrusivePtr<c10::detail::ListImpl>());
}
inline c10::List<c10::complex<double>> IValue::toComplexDoubleList() const& {
  AT_ASSERT(isComplexDoubleList(), "Expected ComplexDoubleList but got ", tagKind());
  return c10::List<c10::complex<double>>(toIntrusivePtr<c10::detail::ListImpl>());
}
inline std::vector<c10::complex<double>> IValue::toComplexDoubleVector() const {
  AT_ASSERT(isComplexDoubleList(), "Expected ComplexDoubleList but got ", tagKind());
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
      payload.u.as_intrusive_ptr != c10::UndefinedTensorImpl::singleton(),
      "called toComplexDoubleVector on null intrusive_ptr IValue");
  return createVectorFromList<c10::complex<double>>(
      static_cast<const c10::detail::ListImpl*>(payload.u.as_intrusive_ptr));
}
inline c10::List<bool> IValue::toBoolList() && {
  AT_ASSERT(isBoolList(), "Expected BoolList but got ", tagKind());
  return c10::List<bool>(moveToIntrusivePtr<c10::detail::ListImpl>());
}
inline c10::List<bool> IValue::toBoolList() const& {
  AT_ASSERT(isBoolList(), "Expected BoolList but got ", tagKind());
  return c10::List<bool>(toIntrusivePtr<c10::detail::ListImpl>());
}
inline c10::List<at::Tensor> IValue::toTensorList() && {
  AT_ASSERT(isTensorList(), "Expected TensorList but got ", tagKind());
  return c10::List<at::Tensor>(moveToIntrusivePtr<c10::detail::ListImpl>());
}
inline c10::List<at::Tensor> IValue::toTensorList() const& {
  AT_ASSERT(isTensorList(), "Expected TensorList but got ", tagKind());
  return c10::List<at::Tensor>(toIntrusivePtr<c10::detail::ListImpl>());
}
inline std::vector<at::Tensor> IValue::toTensorVector() const {
  AT_ASSERT(isTensorList(), "Expected TensorList but got ", tagKind());
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
      payload.u.as_intrusive_ptr != c10::UndefinedTensorImpl::singleton(),
      "called toTensorVector on null intrusive_ptr IValue");
  return createVectorFromList<at::Tensor>(
      static_cast<const c10::detail::ListImpl*>(payload.u.as_intrusive_ptr));
}
inline c10::List<c10::optional<at::Tensor>> IValue::toOptionalTensorList() && {
  AT_ASSERT(isOptionalTensorList(), "Expected OptionalTensorList but got ", tagKind());
  return c10::List<c10::optional<at::Tensor>>(moveToIntrusivePtr<c10::detail::ListImpl>());
}
inline c10::List<c10::optional<at::Tensor>> IValue::toOptionalTensorList() const& {
  AT_ASSERT(isOptionalTensorList(), "Expected OptionalTensorList but got ", tagKind());
  return c10::List<c10::optional<at::Tensor>>(toIntrusivePtr<c10::detail::ListImpl>());
}
inline std::vector<c10::optional<at::Tensor>> IValue::toOptionalTensorVector() const {
  AT_ASSERT(isOptionalTensorList(), "Expected OptionalTensorList but got ", tagKind());
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
      payload.u.as_intrusive_ptr != c10::UndefinedTensorImpl::singleton(),
      "called toOptionalTensorVector on null intrusive_ptr IValue");
  return createVectorFromList<c10::optional<at::Tensor>>(
      static_cast<const c10::detail::ListImpl*>(payload.u.as_intrusive_ptr));
}
inline c10::List<IValue> IValue::toList() && {
  AT_ASSERT(isList(), "Expected GenericList but got ", tagKind());
  return c10::List<IValue>(moveToIntrusivePtr<c10::detail::ListImpl>());
}
inline c10::List<IValue> IValue::toList() const& {
  AT_ASSERT(isList(), "Expected GenericList but got ", tagKind());
  return c10::List<IValue>(toIntrusivePtr<c10::detail::ListImpl>());
}
inline c10::ArrayRef<IValue> IValue::toListRef() const {
  AT_ASSERT(isList(), "Expected GenericList but got ", tagKind());
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
      payload.u.as_intrusive_ptr != c10::UndefinedTensorImpl::singleton(),
      "called toListRef on null intrusive_ptr IValue");
  return static_cast<const c10::detail::ListImpl*>(payload.u.as_intrusive_ptr)
      ->list;
}
inline c10::Dict<IValue, IValue> IValue::toGenericDict() && {
  AT_ASSERT(isGenericDict(), "Expected GenericDict but got ", tagKind());
  return c10::Dict<IValue, IValue>(moveToIntrusivePtr<c10::detail::DictImpl>());
}
inline c10::Dict<IValue, IValue> IValue::toGenericDict() const& {
  AT_ASSERT(isGenericDict(), "Expected GenericDict but got ", tagKind());
  return c10::Dict<IValue, IValue>(toIntrusivePtr<c10::detail::DictImpl>());
}
inline c10::intrusive_ptr<ivalue::Tuple> IValue::toTuple() && {
  AT_ASSERT(isTuple(), "Expected Tuple but got ", tagKind());
  return moveToIntrusivePtr<ivalue::Tuple>();
}
inline c10::intrusive_ptr<ivalue::Tuple> IValue::toTuple() const& {
  AT_ASSERT(isTuple(), "Expected Tuple but got ", tagKind());
  return toIntrusivePtr<ivalue::Tuple>();
}
inline ivalue::Tuple& IValue::toTupleRef() const {
  AT_ASSERT(isTuple(), "Expected Tuple but got ", tagKind());
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
      payload.u.as_intrusive_ptr != c10::UndefinedTensorImpl::singleton(),
      "called toTupleRef on null intrusive_ptr IValue");
  return *static_cast<c10::ivalue::Tuple*>(
      payload.u.as_intrusive_ptr);
}

inline IValue::IValue(c10::intrusive_ptr<ivalue::Tuple> v)
    : tag(Tag::Tuple) {
  payload.u.as_intrusive_ptr = null_to_undefined_tensor(v.release());
}
template <
    typename... Args,
    std::enable_if_t<
        !guts::disjunction<
            std::is_lvalue_reference<Args>...,
            guts::negation<std::is_constructible<IValue, Args>>...>::value,
        std::nullptr_t>>
inline IValue::IValue(const std::tuple<Args...>& t)
    : IValue(c10::guts::apply(c10::ivalue::Tuple::create<const Args&...>, t)) {
}

template <
    typename... Args,
    std::enable_if_t<
        !guts::disjunction<
            std::is_lvalue_reference<Args>...,
            guts::negation<std::is_constructible<IValue, Args>>...>::value,
        std::nullptr_t>>
inline IValue::IValue(std::tuple<Args...>&& t)
    : IValue(c10::guts::apply(c10::ivalue::Tuple::create<Args&&...>, std::move(t))) {
}

inline IValue::IValue(c10::intrusive_ptr<ivalue::ConstantString> v)
    : tag(Tag::String) {
  payload.u.as_intrusive_ptr = null_to_undefined_tensor(v.release());
}
inline IValue::IValue(std::string v)
    : IValue(ivalue::ConstantString::create(std::move(v))) {}

inline IValue::IValue(c10::impl::GenericList v)
    : tag(Tag::GenericList) {
  payload.u.as_intrusive_ptr = null_to_undefined_tensor(v.impl_.release());
}

template <class T, IValue::enable_if_list_is_ivalue_constructible<T>>
inline IValue::IValue(c10::List<T>&& v) : IValue(impl::toList<T>(std::move(v))) {}
template <class T, IValue::enable_if_list_is_ivalue_constructible<T>>
inline IValue::IValue(const c10::List<T>& v) : IValue(impl::toList<T>(v)) {}
template <class T, IValue::enable_if_list_is_ivalue_constructible<T>>
inline IValue::IValue(at::ArrayRef<T> v) : IValue(c10::List<T>()) {
  auto list = to<c10::List<T>>();
  list.reserve(v.size());
  for (const auto& e : v) {
    list.push_back(e);
  }
}
template <class T, IValue::enable_if_symint<T>>
inline IValue::IValue(at::ArrayRef<T> v) : IValue() {
  auto vi = c10::asIntArrayRefSlowOpt(v);
  if (vi.has_value()) {
    // This list is entirely integers; ensure it is typed as
    // an IntList so toIntList works
    *this = IValue(*vi);
  } else {
    // This list has SymInts; type it as a SymInt
    *this = IValue(impl::toList<c10::SymInt>(c10::List<c10::SymInt>()));
    auto list = to<c10::List<c10::SymInt>>();
    list.reserve(v.size());
    for (const auto& e : v) {
      list.push_back(e);
    }
  }
}
template <class T, IValue::enable_if_symint<T>>
inline IValue::IValue(at::OptionalArrayRef<T> mb_v) : IValue() {
  if (!mb_v.has_value()) return;
  *this = IValue(*mb_v);
}
template <class T, IValue::enable_if_symint<T>>
inline IValue::IValue(const std::vector<T>& v) : IValue() {
  *this = IValue(at::ArrayRef<T>(v));
}
template <class T, IValue::enable_if_list_is_ivalue_constructible<T>>
inline IValue::IValue(const std::vector<T>& v) : IValue(c10::List<T>()) {
  auto list = to<c10::List<T>>();
  list.reserve(v.size());
  for (const auto& e : v) {
    list.push_back(e);
  }
}
template <class T, IValue::enable_if_list_is_ivalue_constructible<T>>
inline IValue::IValue(c10::OptionalArrayRef<T> v) : IValue() {
  if (v.has_value()) {
    *this = IValue(std::move(*v));
  }
}

template <class T, size_t N>
inline IValue::IValue(std::array<T, N> v) : IValue(c10::List<T>()) {
  auto list = to<c10::List<T>>();
  list.reserve(v.size());
  for (auto& e : v) {
    list.push_back(std::move(e));
  }
}

template <class T, IValue::enable_if_ilist_is_ivalue_constructible<T>>
inline IValue::IValue(c10::IListRef<T> v) : IValue() {
  constexpr bool boxed_type_constructs_ivalue =
      std::is_constructible<IValue, typename c10::IListRef<T>::boxed_type>::value;
  // First, we try to use the boxed value.
  // If we fail (either it's not in the boxed state, or its boxed type
  // can not construct an IValue), we fallback to copying the list.
  if (boxed_type_constructs_ivalue && v.isBoxed()) {
    *this = IValue(impl::toList(v.toBoxed()));
  } else {
    c10::List<T> list;
    list.reserve(v.size());
    for (const auto& t : v) {
      list.push_back(t);
    }
    *this = IValue(impl::toList(std::move(list)));
  }
}

inline IValue::IValue(c10::impl::GenericDict v)
    : tag(Tag::GenericDict) {
  payload.u.as_intrusive_ptr = null_to_undefined_tensor(v.impl_.release());
}
template <class Key, class Value>
inline IValue::IValue(c10::Dict<Key, Value> v)
    : IValue(impl::toGenericDict(std::move(v))) {}

template <class Key, class Value>
inline IValue::IValue(std::unordered_map<Key, Value> v)
    : IValue(Dict<Key, Value>()) {
  auto dict = to<c10::Dict<Key, Value>>();
  dict.reserve(v.size());
  for (auto& e : v) {
    dict.insert(std::move(e.first), std::move(e.second));
  }
}

template <class T, IValue::enable_if_ivalue_constructible<T>>
inline IValue::IValue(c10::optional<T> v) : IValue() {
  if (v.has_value()) {
    *this = IValue(std::move(*v));
  }
}

inline IValue::IValue(c10::nullopt_t) : IValue() {}

inline IValue::IValue(c10::intrusive_ptr<ivalue::Object> v)
    : tag(Tag::Object) {
  payload.u.as_intrusive_ptr = null_to_undefined_tensor(v.release());
}

inline IValue::IValue(c10::intrusive_ptr<ivalue::PyObjectHolder> v)
    : tag(Tag::PyObject) {
  payload.u.as_intrusive_ptr = null_to_undefined_tensor(v.release());
}

inline IValue::IValue(c10::intrusive_ptr<ivalue::EnumHolder> v)
    : tag(Tag::Enum) {
  payload.u.as_intrusive_ptr = null_to_undefined_tensor(v.release());
}

inline IValue IValue::make_capsule(
    intrusive_ptr<torch::CustomClassHolder> blob) {
  IValue iv;
  iv.tag = Tag::Capsule;
  iv.payload.u.as_intrusive_ptr = null_to_undefined_tensor(blob.release());
  return iv;
}

template <
    typename T,
    std::enable_if_t<std::is_base_of<torch::CustomClassHolder, T>::value, int>>
IValue::IValue(c10::intrusive_ptr<T> custom_class) : tag(Tag::Object) {
  auto classType = []() {
    try {
      return c10::getCustomClassType<c10::intrusive_ptr<T>>();
    } catch (const c10::Error&) {
      throw c10::Error(
          "Trying to instantiate a class that isn't a registered custom class: " +
          std::string(c10::util::get_fully_qualified_type_name<T>()),
          "");
    }
  }();
  auto ivalue_obj = c10::ivalue::Object::create(std::move(classType), /* numSlots */1);
  ivalue_obj->setSlot(0, IValue::make_capsule(std::move(custom_class)));
  payload.u.as_intrusive_ptr = null_to_undefined_tensor(ivalue_obj.release());

}

inline IValue::IValue(c10::intrusive_ptr<ivalue::Future> v)
    : tag(Tag::Future) {
  payload.u.as_intrusive_ptr = null_to_undefined_tensor(v.release());
}

inline IValue::IValue(c10::intrusive_ptr<ivalue::Await> v)
    : tag(Tag::Await) {
  payload.u.as_intrusive_ptr = null_to_undefined_tensor(v.release());
}

inline IValue::IValue(c10::intrusive_ptr<c10::RRefInterface> v)
    : tag(Tag::RRef) {
  payload.u.as_intrusive_ptr = null_to_undefined_tensor(v.release());
}

inline IValue::IValue(c10::intrusive_ptr<at::Quantizer> v)
    : tag(Tag::Quantizer) {
  payload.u.as_intrusive_ptr = null_to_undefined_tensor(v.release());
}

template <typename T>
inline IValue::IValue(c10::complex<T> c)
    : tag(Tag::ComplexDouble) {
  auto v = c10::make_intrusive<ivalue::ComplexHolder>(c);
  payload.u.as_intrusive_ptr = v.release();
}

inline const std::string& IValue::toStringRef() const {
  AT_ASSERT(isString(), "Expected String but got ", tagKind());
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
      payload.u.as_intrusive_ptr != c10::UndefinedTensorImpl::singleton(),
      "called toStringRef on null intrusive_ptr IValue");
  return static_cast<const c10::ivalue::ConstantString*>(
             payload.u.as_intrusive_ptr)
      ->string();
}
inline c10::optional<std::reference_wrapper<const std::string>> IValue::
    toOptionalStringRef() const {
  if (isNone()) {
    return c10::nullopt;
  }
  AT_ASSERT(isString(), "Expected optional<string> but got ", tagKind());
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
      payload.u.as_intrusive_ptr != c10::UndefinedTensorImpl::singleton(),
      "called toOptionalStringRef on null intrusive_ptr IValue");
  return std::reference_wrapper<const std::string>(
      static_cast<const c10::ivalue::ConstantString*>(payload.u.as_intrusive_ptr)
          ->string());
}

inline c10::string_view IValue::toStringView() const {
  AT_ASSERT(isString(), "Expected String but got ", tagKind());
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
      payload.u.as_intrusive_ptr != c10::UndefinedTensorImpl::singleton(),
      "called toStringView on null intrusive_ptr IValue");
  return static_cast<const c10::ivalue::ConstantString*>(
        payload.u.as_intrusive_ptr)
    ->string_view();
}

inline PyObject* IValue::toPyObject() const {
  return toPyObjectHolder()->getPyObject();
}

template <typename T>
inline optional<T> IValue::toOptional() {
  if (this->isNone()) {
    return nullopt;
  }
  return this->to<T>();
}

template <typename T>
inline optional<T> IValue::toOptional() const {
  if (this->isNone()) {
    return nullopt;
  }
  return this->to<T>();
}

inline bool IValue::isCustomClass() const {
  return torch::isCustomClass(*this);
}

inline bool IValue::isSameIdentity(const IValue& rhs) const {
  // We choose to not use memcmp for payload check due to potential random
  // padding characters on union type

  // Semantics:
  // 1. Immutable primitive values of the same type (Int, Double, None, Bool,
  // Str) return value equality
  // 2. If it is a tensor type, we need to take undefined tensor into account
  // 3. Undefined_tensor is None and vice versa should be true
  // 4. If it is a reference type (i.e. isIntrusivePtr()), then is True when
  // the pointed-to object is the same.
  // 5. False for all other comparisons.
  if (this->isNone() && rhs.isNone()) {
    return true;
  } else if (this->isBool() && rhs.isBool()) {
    // for bool type, do equality check
    return this->toBool() == rhs.toBool();
  } else if (this->isTensor() && rhs.isTensor()) {
    return this->payload.as_tensor.is_same(rhs.payload.as_tensor);
  } else if (this->isTensor() && rhs.isNone()) {
    // special case: undefined tensor and None are the same identity
    return !this->payload.as_tensor.defined();
  } else if (this->isNone() && rhs.isTensor()) {
    // special case: undefined tensor and None are the same identity
    return !rhs.payload.as_tensor.defined();
  } else if (this->isInt() && rhs.isInt()) {
    return this->toInt() == rhs.toInt();
  } else if (this->isDouble() && rhs.isDouble()) {
    return this->toDouble() == rhs.toDouble();
  } else if (this->isString() && rhs.isString()) {
    return this->toStringRef() == rhs.toStringRef();
  } else {
    // for objects holding in IValue, do shallow compare on pointer address to
    // testify the identity
    return this->isIntrusivePtr() && rhs.isIntrusivePtr() &&
        this->payload.u.as_intrusive_ptr == rhs.payload.u.as_intrusive_ptr;
  }
}

namespace ivalue {
namespace detail {

template <typename T>
IValue from_(T&& x, std::true_type) {
  return IValue(std::forward<T>(x));
}
template <typename T>
IValue from_(c10::intrusive_ptr<T> x, std::false_type) {
  return IValue(std::move(x));
}
template <typename T>
IValue from_(T&& /*x*/, std::false_type) {
  static_assert(
      guts::false_t<T>::value,
      "You are calling from with a type that it doesn't support, and isn't a potential custom class (ie: is an intrusive_ptr)");
  return IValue();
}
} // namespace detail

template <typename T>
IValue from(T&& x) {
  return detail::from_(
      std::forward<T>(x), typename std::is_constructible<IValue, T>::type{});
}

} // namespace ivalue


template <>
struct MaybeOwnedTraits<IValue> {
  using owned_type = IValue;
  using borrow_type = IValue;

  static borrow_type createBorrow(const owned_type& from) {
    if (!from.isPtrType()) {
      return from;
    }
    if (from.isTensor()) {
      return IValue(MaybeOwnedTraits<at::Tensor>::createBorrow(from.toTensor()));
    } else {
      return IValue(from.payload, from.tag);
    }
  }

  static void assignBorrow(borrow_type& lhs, const borrow_type& rhs) {
    lhs.clearToNone();
    if (!rhs.isPtrType()) {
      lhs = rhs;
    } else if (rhs.isTensor()) {
      lhs = IValue(MaybeOwnedTraits<at::Tensor>::createBorrow(rhs.toTensor()));
    } else {
      lhs = IValue(rhs.payload, rhs.tag);
    }
  }

  static void destroyBorrow(borrow_type& toDestroy) {
    toDestroy.clearToNone();
  }

  static const owned_type& referenceFromBorrow(const borrow_type& borrow) {
    return borrow;
  }

  static const owned_type* pointerFromBorrow(const borrow_type& borrow) {
    return &borrow;
  }

  static bool debugBorrowIsValid(const borrow_type&) {
    return true;
  }
};

template <>
struct IValue::TagType<c10::Type> {
  static TORCH_API c10::TypePtr get(const IValue&);
};

template <>
struct IValue::TagType<c10::DynamicType> {
  static TORCH_API c10::TypePtr get(const IValue&);
};

template <typename T>
TypePtr IValue::type() const {
  return IValue::TagType<T>::get(*this);
}

} // namespace c10
