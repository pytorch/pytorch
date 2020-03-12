#pragma once

#include <ATen/core/TensorBody.h>
#include <ATen/core/blob.h>
#include <c10/util/C++17.h>
#include <c10/util/intrusive_ptr.h>
#include <torch/csrc/WindowsTorchApiMacro.h>

namespace torch {
namespace jit {
class CustomClassHolder : public c10::intrusive_ptr_target {};

struct Function;
namespace script {
struct CompilationUnit;
struct Module;
}
} // namespace jit
} // namespace torch
namespace c10 {
template<class Key, class Value> class Dict;
template<class T> class List;
struct IValue;
struct ClassType;
struct Type;
class RRefInterface;
using TypePtr = std::shared_ptr<Type>;
namespace ivalue {
struct Tuple;
struct Future;
struct ConstantString;
struct GenericDict;
struct Object;
struct PyObjectHolder;
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
  _(String) \
  _(Blob) \
  _(GenericList) \
  _(GenericDict) \
  _(Future) \
  _(Device) \
  _(Object) \
  _(PyObject) \
  _(Uninitialized) \
  _(Capsule) \
  _(RRef) \

// [doxygen private]
// These methods are not actually private but we don't want to document them, so
// they are marked `@private`, which hides them on the doxygen documentation for
// this page.


/// IValue (Interpreter Value) is a tagged union over the types supported by the
/// TorchScript interpreter. IValues contain their values as an `IValue::Payload`,
/// which holds primitive types (`int64_t`, `bool`, `double`, `Device`), as
/// values and all other types as a `c10::intrusive_ptr`.
///
/// IValues are used as inputs to and outputs from the TorchScript interpreter.
/// To retrieve the value contained within an IValue, use the `.toX()` methods,
/// where `X` is the type you are trying to get. Note that neither the `.toX()`
/// methods nor the templated `.to<T>` functions do any kind of casting, they
/// only unwrap the contained value. For example:
///
/// \rst
/// .. code-block:: cpp
///
///   // Make the IValue
///   torch::IValue my_ivalue(26);
///   std::cout << my_ivalue << "\n";
///
///   // Unwrap the IValue
///   int64_t my_int = my_ivalue.toInt()
///   std::cout << my_int << "\n";
///
///   // This will throw an error!
///   // `my_ivalue` is tagged as an int and cannot be used as another type
///   torch::Tensor my_tensor = my_ivalue.toTensor()
/// \endrst
struct CAFFE2_API IValue final {
  IValue(const IValue& rhs)
      : IValue(rhs.payload, rhs.tag, rhs.is_intrusive_ptr) {
    if (is_intrusive_ptr) {
      c10::raw::intrusive_ptr::incref(payload.as_intrusive_ptr);
    }
  }
  IValue(IValue&& rhs) noexcept : IValue() {
    swap(rhs);
  }
  /// @private [doxygen private]
  ~IValue() {
    if (is_intrusive_ptr) {
      c10::raw::intrusive_ptr::decref(payload.as_intrusive_ptr);
    }
  }
  IValue& operator=(IValue&& rhs) & noexcept {
    IValue(std::move(rhs)).swap(*this); // this also sets rhs to None
    return *this;
  }
  IValue& operator=(IValue const& rhs) & {
    IValue(rhs).swap(*this);
    return *this;
  }
  void dump() const;

  /// @private [doxygen private]
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

  /// @private [doxygen private]
  size_t use_count() const noexcept {
    if (!is_intrusive_ptr) {
      return 1;
    }

    return c10::raw::intrusive_ptr::use_count(payload.as_intrusive_ptr);
  }

  /// @private [doxygen private]
  void swap(IValue & rhs) noexcept {
    std::swap(payload, rhs.payload);
    std::swap(is_intrusive_ptr, rhs.is_intrusive_ptr);
    std::swap(tag, rhs.tag);
  }

  // Accessors for subtypes are arranged together below
  // While some of these accessors could be generated through templates,
  // we prefer to write them manually for clarity

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
  at::TensorImpl* unsafeToTensorImpl() const {
    return static_cast<at::TensorImpl*>(payload.as_intrusive_ptr);
  }

  const IValue& toIValue() const {
    return *this;
  }
  IValue& toIValue() {
    return *this;
  }

  /// @private [doxygen private]
  IValue(intrusive_ptr<caffe2::Blob> blob)
  : tag(Tag::Blob), is_intrusive_ptr(true) {
    // TODO (after Tensor merge) If we pass in a Blob holding a Tensor, extract
    // and store it as a Tensor instead.
    payload.as_intrusive_ptr = blob.release();
  }

  /// @private [doxygen private]
  bool isBlob() const {
    return Tag::Blob == tag;
  }

  /// @private [doxygen private]
  c10::intrusive_ptr<caffe2::Blob> toBlob() &&;

  /// @private [doxygen private]
  c10::intrusive_ptr<caffe2::Blob> toBlob() const &;

  // Capsule
  IValue(intrusive_ptr<torch::jit::CustomClassHolder> blob);
  bool isCapsule() const {
    return Tag::Capsule == tag;
  }
  c10::intrusive_ptr<torch::jit::CustomClassHolder> toCapsule() &&;
  c10::intrusive_ptr<torch::jit::CustomClassHolder> toCapsule() const &;

  // Tuple
  IValue(c10::intrusive_ptr<ivalue::Tuple> v);

  template <
      typename... Args,
      std::enable_if_t<
          !guts::disjunction<
              std::is_lvalue_reference<Args>...,
              guts::negation<std::is_constructible<IValue, Args>>...>::
              value,
          std::nullptr_t> = nullptr>
  IValue(const std::tuple<Args...>& t);
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

  // RRef
  IValue(c10::intrusive_ptr<c10::RRefInterface> v);
  bool isRRef() const { return Tag::RRef == tag; }
  c10::intrusive_ptr<c10::RRefInterface> toRRef() &&;
  c10::intrusive_ptr<c10::RRefInterface> toRRef() const &;

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
  bool isIntList() const;
  c10::List<int64_t> toIntList() &&;
  c10::List<int64_t> toIntList() const &;
  std::vector<int64_t> toIntVector() const;

  // ConstantString
  IValue(c10::intrusive_ptr<ivalue::ConstantString> v);
  IValue(std::string v);
  IValue(const char* v): IValue(std::string(v)) {}
  bool isString() const { return Tag::String == tag; }
  c10::intrusive_ptr<ivalue::ConstantString> toString() &&;
  c10::intrusive_ptr<ivalue::ConstantString> toString() const &;
  const std::string& toStringRef() const;

  // DoubleList
  bool isDoubleList() const;
  c10::List<double> toDoubleList() &&;
  c10::List<double> toDoubleList() const &;
  std::vector<double> toDoubleVector() const;

  // BoolList
  bool isBoolList() const;
  c10::List<bool> toBoolList() &&;
  c10::List<bool> toBoolList() const &;

  //TensorList
  bool isTensorList() const;
  c10::List<at::Tensor> toTensorList() &&;
  c10::List<at::Tensor> toTensorList() const &;
  std::vector<at::Tensor> toTensorVector() const;

  //GenericList
  IValue(c10::List<IValue> v);
  bool isList() const { return Tag::GenericList == tag; }
  c10::List<IValue> toList() &&;
  c10::List<IValue> toList() const &;
  c10::ArrayRef<IValue> toListRef() const;

  template<class T>
  IValue(c10::List<T> v);
  template<class T>
  IValue(at::ArrayRef<T> v);
  template<class T>
  IValue(const std::vector<T>& v);

  // GenericDict
  IValue(c10::Dict<IValue, IValue> v);
  bool isGenericDict() const { return Tag::GenericDict == tag; }
  c10::Dict<IValue, IValue> toGenericDict() &&;
  c10::Dict<IValue, IValue> toGenericDict() const &;

  template<class Key, class Value>
  IValue(c10::Dict<Key, Value> v);

  template<class Key, class Value>
  /// \cond DOXYGEN_CANNOT_HANDLE_CONSTRUCTORS_WITH_MACROS_SO_EXCLUDE_THIS_LINE_FROM_DOXYGEN
  C10_DEPRECATED_MESSAGE("IValues based on std::unordered_map<K, V> are slow and deprecated. Please use c10::Dict<K, V> instead.")
  /// \endcond
  IValue(std::unordered_map<Key, Value> v);

  template<class T>
  IValue(c10::optional<T> v);
  IValue(c10::nullopt_t);

  // ClassType
  IValue(c10::intrusive_ptr<ivalue::Object> v);
  bool isObject() const { return tag == Tag::Object; }
  c10::intrusive_ptr<ivalue::Object> toObject() &&;
  c10::intrusive_ptr<ivalue::Object> toObject() const & ;
  const ivalue::Object& toObjectRef() const;

  torch::jit::script::Module toModule() const;
  bool isModule() const;

  // PyObject
  IValue(c10::intrusive_ptr<ivalue::PyObjectHolder> v);
  bool isPyObject() const { return tag == Tag::PyObject; }
  c10::intrusive_ptr<ivalue::PyObjectHolder> toPyObjectHolder() &&;
  c10::intrusive_ptr<ivalue::PyObjectHolder> toPyObjectHolder() const &;
  PyObject* toPyObject() const;

  // None
  IValue() : payload{0}, tag(Tag::None), is_intrusive_ptr(false) {}
  bool isNone() const {
    return Tag::None == tag;
  }
  std::string toNone() const {
    AT_ASSERT(isNone());
    return "None";
  }

  static IValue uninitialized() {
    auto i = IValue();
    i.tag = Tag::Uninitialized;
    return i;
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
  IValue(ScalarType t)
  : IValue(static_cast<std::underlying_type<ScalarType>::type>(t)) {}
  at::ScalarType toScalarType() const {
    return static_cast<at::ScalarType>(toInt());
  }

  // Layout
  IValue(Layout l)
  : IValue(static_cast<std::underlying_type<Layout>::type>(l)) {}
  at::Layout toLayout() const {
    return static_cast<at::Layout>(toInt());
  }

  // MemoryFormat
  IValue(MemoryFormat m)
  : IValue(static_cast<std::underlying_type<MemoryFormat>::type>(m)) {}
  at::MemoryFormat toMemoryFormat() const {
    return static_cast<at::MemoryFormat>(toInt());
  }

  // QScheme
  IValue(at::QScheme qscheme)
  : tag(Tag::Int), is_intrusive_ptr(false) {
    payload.as_int = static_cast<int64_t>(qscheme);
  }

  at::QScheme toQScheme() const {
    return static_cast<at::QScheme>(toInt());
  }


  // for debugging
  std::string tagKind() const {
    switch(tag) {
      #define DEFINE_CASE(x) case Tag::x: return #x;
      TORCH_FORALL_TAGS(DEFINE_CASE)
      #undef DEFINE_CASE
    }
    return "InvalidTag(" + c10::guts::to_string(static_cast<int>(tag)) + ")";
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

  /// @private [doxygen private]
  /// this is a shallow comparison of two IValues to test the object identity
  bool isSameIdentity(const IValue& rhs) const;

  // Computes the "official" string representation of an IValue. This produces a
  // TorchScript expression that can be used to recreate an IValue with the same
  // value (e.g. when we are printing constants in the serializer).
  //
  // Callers can use `customFormatter` to override how `repr()` prints out an
  // IValue. This is useful if you have some other environment where you can
  // look up values, and you want to print a reference to that environment (like
  // the serializer's constant table).
  //
  // repr() is not necessarily defined on all objects!
  std::ostream& repr(
      std::ostream& stream,
      std::function<bool(std::ostream&, const IValue& v)> customFormatter)
      const;

  // Computes an "informal" string representation of an IValue. This should be
  // used for debugging, or servicing `print()`-like functions.
  // This is different from `repr()` in that there is no expectation that we can
  // exactly reconstruct an IValue from the output; feel free to use a
  // concise/pretty form
  CAFFE2_API friend std::ostream& operator<<(
      std::ostream& out,
      const IValue& v);

  bool isPtrType() const {
    return is_intrusive_ptr;
  }

  /// @private [doxygen private]
  const void* internalToPointer() const {
    TORCH_INTERNAL_ASSERT(
        isPtrType(), "Can only call internalToPointer() for pointer types");
    return payload.as_intrusive_ptr;
  }

  TypePtr type() const;

  size_t hash() const {
    return payload.as_int;
  }

  // Detection Aliased tensors.
  struct HashIValue {
    size_t operator()(const IValue& val) const {
      if (val.isTensor()) {
        return 0;
      }
      return val.hash();
    }
  };

  struct CompIValues {
    bool operator()(const IValue& lhs, const IValue& rhs) const {
      if (lhs.isTensor() && rhs.isTensor()) {
        return lhs.isAliasOf(rhs);
      }
      return lhs.hash() == rhs.hash();
    }
  };

  using HashAliasedIValues = std::unordered_set<IValue, HashIValue, CompIValues>;

  // Chechs if this and rhs has a subvalues in common.
  // [t1,t2] and [t2, t3] returns true.
  bool overlaps(const IValue& rhs) const;

  // Inserts all subvalues of this in subValues.
  void getSubValues(HashAliasedIValues& subValues) const;

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

  union Payload {
    int64_t as_int;
    double as_double;
    bool as_bool;
    c10::intrusive_ptr_target* as_intrusive_ptr;
    struct {
      DeviceType type;
      DeviceIndex index;
    } as_device;
  };

  IValue(Payload p, Tag t, bool i)
  : payload(p), tag(t), is_intrusive_ptr(i) {}

  Payload payload;
  Tag tag;
  bool is_intrusive_ptr;
  friend struct WeakIValue;
};

struct CAFFE2_API WeakIValue final {
  WeakIValue()
  : payload{0}
  , tag(IValue::Tag::None)
  , is_intrusive_ptr(false) {}

  WeakIValue(const WeakIValue& rhs)
      : payload(rhs.payload),
        tag(rhs.tag),
        is_intrusive_ptr(rhs.is_intrusive_ptr) {
    if (is_intrusive_ptr) {
      c10::raw::weak_intrusive_ptr::incref(payload.as_intrusive_ptr);
    }
  }
  WeakIValue(const IValue& rhs)
      : payload(rhs.payload),
        tag(rhs.tag),
        is_intrusive_ptr(rhs.is_intrusive_ptr) {
    if (is_intrusive_ptr) {
      c10::raw::weak_intrusive_ptr::incref(payload.as_intrusive_ptr);
    }
  }
  WeakIValue(WeakIValue&& rhs) noexcept : WeakIValue() {
    swap(rhs);
  }
  ~WeakIValue() {
    if (is_intrusive_ptr) {
      c10::raw::weak_intrusive_ptr::decref(payload.as_intrusive_ptr);
    }
  }
  WeakIValue & operator=(WeakIValue && rhs) & noexcept {
    WeakIValue(std::move(rhs)).swap(*this); // this also sets rhs to None
    return *this;
  }
  WeakIValue & operator=(WeakIValue const & rhs) & {
    WeakIValue(rhs).swap(*this);
    return *this;
  }
  void swap(WeakIValue & rhs) noexcept {
    std::swap(payload, rhs.payload);
    std::swap(is_intrusive_ptr, rhs.is_intrusive_ptr);
    std::swap(tag, rhs.tag);
  }

  bool isSameIdentity(const WeakIValue& rhs) const {
    return payload.as_int == rhs.payload.as_int && tag == rhs.tag &&
        is_intrusive_ptr == rhs.is_intrusive_ptr;
  }

  IValue lock() const {
    if (!is_intrusive_ptr) {
      return IValue(payload, tag, false);
    }
    auto temp = c10::weak_intrusive_ptr<c10::intrusive_ptr_target>::reclaim(
        payload.as_intrusive_ptr);
    IValue::Payload pl;
    pl.as_intrusive_ptr = temp.lock().release();
    temp.release();
    if (!pl.as_intrusive_ptr) {
      return IValue();
    } else {
      return IValue(pl, tag, true);
    }
  }

  size_t use_count() const noexcept {
    if (!is_intrusive_ptr) {
      return 1;
    }
    auto temp = c10::weak_intrusive_ptr<c10::intrusive_ptr_target>::reclaim(
        payload.as_intrusive_ptr);
    size_t result = temp.use_count();
    temp.release();
    return result;
  }

  size_t weak_use_count() const noexcept {
    if (!is_intrusive_ptr) {
      return 1;
    }
    auto temp = c10::weak_intrusive_ptr<c10::intrusive_ptr_target>::reclaim(
        payload.as_intrusive_ptr);
    size_t result = temp.weak_use_count();
    temp.release();
    return result;
  }
  size_t hash() const {
    return payload.as_int;
  }

private:
  IValue::Payload payload;
  IValue::Tag tag;
  bool is_intrusive_ptr;
};

// An owning pointer to a type. When the type is class type, it requires a pair
// of shared_ptrs to the class type and its owning CU, so that the class type is
// guaranteed to stay alive as long as we hold this object.
struct TORCH_API StrongTypePtr {
  StrongTypePtr(
      std::shared_ptr<torch::jit::script::CompilationUnit> cu,
      std::shared_ptr<Type> type);

  std::shared_ptr<torch::jit::script::CompilationUnit> cu_;
  std::shared_ptr<Type> type_;
};

TORCH_API std::unordered_map<std::string, c10::StrongTypePtr>& getCustomClassTypeMap();

#ifndef C10_MOBILE

template<typename T>
c10::StrongTypePtr getCustomClassType() {
  auto tmap = c10::getCustomClassTypeMap();
  auto res = tmap.find(typeid(T).name());
  if (res == tmap.end()) {
    throw c10::Error("Can't find class id in custom class type map", "");
  }
  return res->second;
}

template<typename T>
inline bool isCustomClassRegistered() {
  auto tmap = c10::getCustomClassTypeMap();
  return tmap.find(typeid(T).name()) != tmap.end();
}

#else  // C10_MOBILE

template<typename T>
c10::StrongTypePtr getCustomClassType() {
  throw c10::Error("Custom class is not supported on mobile.", "");
}

template<typename T>
inline bool isCustomClassRegistered() {
  return false;
}

#endif  // C10_MOBILE

TORCH_API std::unordered_map<std::string, std::function<PyObject*(void*)>>&
getClassConverter();
}

#include <ATen/core/ivalue_inl.h>
