#pragma once

#include <condition_variable>
#include <type_traits>

#include <ATen/core/functional.h>
#include <ATen/core/interned_strings.h>
#include <c10/core/Scalar.h>
#include <c10/core/TensorImpl.h>
#include <c10/core/UndefinedTensorImpl.h>
#include <ATen/core/Dict.h>
#include <ATen/core/List.h>

namespace torch {
namespace jit {
struct Function;
namespace script {
struct CompilationUnit;
}
} // namespace jit
} // namespace torch
namespace c10 {
struct IValue;
struct ClassType;
struct TupleType;

template<class T, class NullType>
c10::intrusive_ptr<T, NullType> IValue::moveToIntrusivePtr() {
  auto t = c10::intrusive_ptr<T, NullType>::reclaim(static_cast<T*>(payload.as_intrusive_ptr));
  clearToNone();
  return t;
}
template<typename T, class NullType>
c10::intrusive_ptr<T, NullType> IValue::toIntrusivePtr() const {
  auto r = c10::intrusive_ptr<T, NullType>::reclaim(static_cast<T*>(payload.as_intrusive_ptr));
  auto p = r;
  r.release();
  return p;
}

inline c10::intrusive_ptr<ivalue::Future> IValue::toFuture() && {
  AT_ASSERT(isFuture(), "Expected Future but got ", tagKind());
  return moveToIntrusivePtr<ivalue::Future>();
}
inline c10::intrusive_ptr<ivalue::Future> IValue::toFuture() const & {
  AT_ASSERT(isFuture(), "Expected Future but got ", tagKind());
  return toIntrusivePtr<ivalue::Future>();
}
inline c10::intrusive_ptr<ivalue::ConstantString> IValue::toString() && {
  AT_ASSERT(isString(), "Expected String but got ", tagKind());
  return moveToIntrusivePtr<ivalue::ConstantString>();
}
inline c10::intrusive_ptr<ivalue::ConstantString> IValue::toString() const & {
  AT_ASSERT(isString(), "Expected String but got ", tagKind());
  return toIntrusivePtr<ivalue::ConstantString>();
}
inline c10::intrusive_ptr<ivalue::Object> IValue::toObject() && {
  AT_ASSERT(isObject(), "Expected Object but got ", tagKind());
  return toIntrusivePtr<ivalue::Object>();
}
inline c10::intrusive_ptr<ivalue::Object> IValue::toObject() const & {
  AT_ASSERT(isObject(), "Expected Object but got ", tagKind());
  return toIntrusivePtr<ivalue::Object>();
}
inline at::Tensor IValue::toTensor() && {
  AT_ASSERT(isTensor(), "Expected Tensor but got ", tagKind());
  return at::Tensor(moveToIntrusivePtr<at::TensorImpl, at::UndefinedTensorImpl>());
}
inline at::Tensor IValue::toTensor() const & {
  AT_ASSERT(isTensor(), "Expected Tensor but got ", tagKind());
  return at::Tensor(toIntrusivePtr<at::TensorImpl, at::UndefinedTensorImpl>());
}
inline c10::intrusive_ptr<caffe2::Blob> IValue::toBlob() && {
  AT_ASSERT(isBlob(), "Expected Blob but got ", tagKind());
  return moveToIntrusivePtr<caffe2::Blob>();
}
inline c10::intrusive_ptr<caffe2::Blob> IValue::toBlob() const & {
  AT_ASSERT(isBlob(), "Expected Blob but got ", tagKind());
  return toIntrusivePtr<caffe2::Blob>();;
}

namespace ivalue {

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

struct Future;

struct CAFFE2_API Tuple : c10::intrusive_ptr_target {
 private:
   std::vector<IValue> elements_;

 public:
  static c10::intrusive_ptr<Tuple> create(std::vector<IValue> elements_, std::shared_ptr<TupleType> type_) {
    return c10::make_intrusive<Tuple>(std::move(elements_), type_);
  }
  C10_DEPRECATED_MESSAGE("Creating tuples without type information is deprecated. Please use Tuple::create(elements, type) instead.")
  static c10::intrusive_ptr<Tuple> create(std::vector<IValue> elements_) {
    return c10::make_intrusive<Tuple>(std::move(elements_), nullptr);
  }

 const std::vector<IValue>& elements() const & {
    return elements_;
  }
  operator const std::vector<IValue>&() const {
    return elements();
  }

  std::vector<IValue>& elements() & {
    return elements_;
  }
  operator std::vector<IValue>&() {
    return elements();
  }

  std::vector<IValue>&& elements() && {
    return std::move(elements_);
  }

  std::shared_ptr<TupleType> type;
 private:
  Tuple(std::vector<IValue> elements, std::shared_ptr<TupleType> type)
    : elements_(std::move(elements)), type(std::move(type)) {}

  friend class c10::intrusive_ptr<Tuple>;
};

struct Object;
}

// Future
struct C10_EXPORT ivalue::Future final : c10::intrusive_ptr_target {
 private:
  c10::intrusive_ptr<Future> intrusive_from_this() {
    c10::raw::intrusive_ptr::incref(this); // we are creating a new pointer
                                           // from a raw `this` pointer
                                           // so we need to bump the refcount
                                           // to account for this ownership
    return c10::intrusive_ptr<Future>::reclaim(this);
  }

 public:
  struct CAFFE2_API FutureError final : public std::exception {
    FutureError(std::string&& error_msg_)
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
    while (!completed_) {
      finished_cv_.wait(lock);
    }
  }

  /**
   * Explicitly mark the future as completed with the output value.
   */
  void markCompleted(IValue value) {
    std::unique_lock<std::mutex> lock(mutex_);
    AT_ASSERT(!completed());
    completed_ = true;
    value_ = std::move(value);

    fireCallbacks();
    finished_cv_.notify_all();
  }

  void markCompleted() {
    markCompleted(IValue {});
  }

  void markCompleted(FutureError&& error_) {
    std::unique_lock<std::mutex> lock(mutex_);
    AT_ASSERT(!completed());
    completed_ = true;
    has_error = true;
    error = std::move(error_);

    fireCallbacks();
    finished_cv_.notify_all();
  }

  // Get the result of the current future.
  IValue value() {
    std::unique_lock<std::mutex> lock(mutex_);
    AT_ASSERT(completed());
    if (has_error) {
      throw error;
    }
    return value_;
  }

  /**
   * Add a callback to the future.
   * The callbacks will be executed once the future completes.
   * If the future has already completed,
   * this function will execute the callback immediately.
   */
  void addCallback(std::function<void(void)> callback) {
    std::unique_lock<std::mutex> lock(mutex_);
    if (completed()) {
      lock.unlock();
      callback();
      return;
    }
    callbacks.push_back(callback);
  }

  // Check if the current future has completed
  bool completed() {
    return completed_;
  }

  CAFFE2_API friend std::ostream& operator<<(
      std::ostream& out,
      const Future& v);

 private:
  void fireCallbacks() {
    AT_ASSERT(completed());
    // There is no need to protect callbacks with the lock.
    // Once completed_ is set to true, no one can add new callback to the list.
    for (auto& callback : callbacks) {
      callback();
    }
    callbacks.clear();
  }

  std::mutex mutex_;
  std::atomic_bool completed_ = {false}; // is this future complete
  std::condition_variable finished_cv_;

  IValue value_; // when finished the value
  std::vector<std::function<void(void)>> callbacks;
  bool has_error = false;
  FutureError error;
};

// User-defined object.
struct C10_EXPORT ivalue::Object final : c10::intrusive_ptr_target {
 public:
  // temporary way to break cyclic dependencies in modules by forcing the deletion
  // of functions when the module object is destructed
  typedef void (*OnDelete)(ivalue::Object*);
  Object(
      StrongTypePtr type,
      size_t numSlots,
      OnDelete on_delete)
      : type_(std::move(type)), on_delete_(on_delete) {
    slots_.resize(numSlots);
  }

  static c10::intrusive_ptr<Object> create(
      StrongTypePtr type,
      size_t numSlots,
      OnDelete on_delete = nullptr) {
    return c10::make_intrusive<Object>(
        std::move(type), numSlots, on_delete);
  }

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
    slots_[slot] = v;
  }

  const IValue& getSlot(size_t slot) const {
    return slots_.at(slot);
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

  std::string name() const;

  const std::vector<IValue>& slots() const {
    return slots_;
  }
  std::shared_ptr<ClassType> type() const {
    return type_.type_;
  }

  std::shared_ptr<torch::jit::script::CompilationUnit> compilation_unit() {
    return type_.cu_;
  }
  // temporarily defined in class_type.cpp to
  // ensure Modules do not leak memory
  ~Object();

 private:
  void resizeObject(size_t slot);
  StrongTypePtr type_;
  std::vector<IValue> slots_;
  OnDelete on_delete_;
};

std::vector<std::pair<IValue, IValue>> iterationOrder(const c10::Dict<IValue, IValue>& dict);

#undef TORCH_FORALL_TAGS

namespace detail {

struct _guarded_unsigned_long_unique_dummy final {
  _guarded_unsigned_long_unique_dummy(int64_t){};
};
using _guarded_unsigned_long = c10::guts::conditional_t<
    std::is_same<unsigned long, uint32_t>::value ||
        std::is_same<unsigned long, uint64_t>::value,
    _guarded_unsigned_long_unique_dummy,
    unsigned long>;

} // namespace detail

inline const ivalue::Object& IValue::toObjectRef() const {
  AT_ASSERT(isObject(), "Expected Object but got ", tagKind());
  return *static_cast<const c10::ivalue::Object*>(payload.as_intrusive_ptr);
}

// note: when adding a DEFINE_TO case here you should also add a
// toX method to IValue. These named methods are much more discoverable
// than the to templated function.

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
DEFINE_TO(float, toDouble)
DEFINE_TO(double, toDouble)
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
DEFINE_TO(c10::List<bool>, toBoolList)
DEFINE_TO(c10::List<at::Tensor>, toTensorList)
DEFINE_TO(c10::impl::GenericList, toGenericList)
DEFINE_TO(c10::impl::GenericDict, toGenericDict)
DEFINE_TO(c10::intrusive_ptr<ivalue::Tuple>, toTuple)
DEFINE_TO(std::string, toStringRef)
DEFINE_TO(c10::intrusive_ptr<ivalue::Future>, toFuture)
DEFINE_TO(IValue, toIValue)
DEFINE_TO(c10::Device, toDevice)
DEFINE_TO(at::ScalarType, toScalarType)
DEFINE_TO(at::Layout, toLayout)
DEFINE_TO(at::MemoryFormat, toMemoryFormat)
DEFINE_TO(at::QScheme, toQScheme)

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
C10_DEPRECATED_MESSAGE("IValues based on std::vector<T> are potentially slow and deprecated. Please use c10::List<T> instead.")
std::vector<Elem> generic_to(
    IValue ivalue,
    _fake_type<std::vector<Elem>>) {
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

template <typename Elem>
c10::List<Elem> generic_to(
    IValue ivalue,
    _fake_type<c10::List<Elem>>) {
  return impl::toTypedList<Elem>(std::move(ivalue).toGenericList());
}

template <typename Key, typename Value>
c10::Dict<Key, Value> generic_to(
    IValue ivalue,
    _fake_type<c10::Dict<Key, Value>>) {
  return impl::toTypedDict<Key, Value>(std::move(ivalue).toGenericDict());
}

template <typename K, typename V>
C10_DEPRECATED_MESSAGE("IValues based on std::unordered_map are slow and deprecated. Please use c10::Dict<K, V> instead.")
std::unordered_map<K, V> generic_to(
    IValue ivalue,
    _fake_type<std::unordered_map<K, V>>) {
  std::unordered_map<K, V> specialized_dict;

  for (auto item : std::move(ivalue).toGenericDict()) {
    specialized_dict[item.key().to<K>()] = item.value().to<V>();
  }

  return specialized_dict;
}

template <typename T>
c10::optional<T> generic_to(
    IValue ivalue,
    _fake_type<c10::optional<T>>) {
  if (ivalue.isNone()) {
    return c10::nullopt;
  }
  return std::move(ivalue).to<T>();
}

template <typename T>
inline T IValue::to() && {
  return generic_to(std::move(*this), _fake_type<T>{});
}

template <typename T>
inline T IValue::to() const& {
  return generic_to(*this, _fake_type<T>{});
}

inline c10::List<int64_t> IValue::toIntList() && {
  AT_ASSERT(isIntList(), "Expected IntList but got ", tagKind());
  return c10::List<int64_t>(moveToIntrusivePtr<c10::detail::ListImpl<int64_t>>());
}
inline c10::List<int64_t> IValue::toIntList() const & {
  AT_ASSERT(isIntList(), "Expected IntList but got ", tagKind());
  return c10::List<int64_t>(toIntrusivePtr<c10::detail::ListImpl<int64_t>>());
}
inline c10::ArrayRef<int64_t> IValue::toIntListRef() const {
  AT_ASSERT(isIntList(), "Expected IntList but got ", tagKind());
  return static_cast<const c10::detail::ListImpl<int64_t>*>(payload.as_intrusive_ptr)->list;
}
inline c10::List<double> IValue::toDoubleList() && {
  AT_ASSERT(isDoubleList(), "Expected DoubleList but got ", tagKind());
  return c10::List<double>(moveToIntrusivePtr<c10::detail::ListImpl<double>>());
}
inline c10::List<double> IValue::toDoubleList() const & {
  AT_ASSERT(isDoubleList(), "Expected DoubleList but got ", tagKind());
  return c10::List<double>(toIntrusivePtr<c10::detail::ListImpl<double>>());
}
inline c10::ArrayRef<double> IValue::toDoubleListRef() const {
  AT_ASSERT(isDoubleList(), "Expected DoubleList but got ", tagKind());
  return static_cast<const c10::detail::ListImpl<double>*>(payload.as_intrusive_ptr)->list;
}
inline c10::List<bool> IValue::toBoolList() && {
  AT_ASSERT(isBoolList(), "Expected BoolList but got ", tagKind());
  return c10::List<bool>(moveToIntrusivePtr<c10::detail::ListImpl<bool>>());
}
inline c10::List<bool> IValue::toBoolList() const & {
  AT_ASSERT(isBoolList(), "Expected BoolList but got ", tagKind());
  return c10::List<bool>(toIntrusivePtr<c10::detail::ListImpl<bool>>());
}
inline c10::List<at::Tensor> IValue::toTensorList() && {
  AT_ASSERT(isTensorList(), "Expected TensorList but got ", tagKind());
  return c10::List<at::Tensor>(moveToIntrusivePtr<c10::detail::ListImpl<at::Tensor>>());
}
inline c10::List<at::Tensor> IValue::toTensorList() const & {
  AT_ASSERT(isTensorList(), "Expected TensorList but got ", tagKind());
  return c10::List<at::Tensor>(toIntrusivePtr<c10::detail::ListImpl<at::Tensor>>());
}
inline c10::ArrayRef<at::Tensor> IValue::toTensorListRef() const {
  AT_ASSERT(isTensorList(), "Expected TensorList but got ", tagKind());
  return static_cast<const c10::detail::ListImpl<at::Tensor>*>(payload.as_intrusive_ptr)->list;
}
inline c10::List<IValue> IValue::toGenericList() && {
  AT_ASSERT(isGenericList(), "Expected GenericList but got ", tagKind());
  return c10::List<IValue>(moveToIntrusivePtr<c10::detail::ListImpl<IValue>>());
}
inline c10::List<IValue> IValue::toGenericList() const & {
  AT_ASSERT(isGenericList(), "Expected GenericList but got ", tagKind());
  return c10::List<IValue>(toIntrusivePtr<c10::detail::ListImpl<IValue>>());
}
inline c10::ArrayRef<IValue> IValue::toGenericListRef() const {
  AT_ASSERT(isGenericList(), "Expected GenericList but got ", tagKind());
  return static_cast<const c10::detail::ListImpl<IValue>*>(payload.as_intrusive_ptr)->list;
}
inline c10::Dict<IValue, IValue> IValue::toGenericDict() && {
  AT_ASSERT(isGenericDict(), "Expected GenericDict but got ", tagKind());
  return c10::Dict<IValue, IValue>(moveToIntrusivePtr<c10::detail::DictImpl>());
}
inline c10::Dict<IValue, IValue> IValue::toGenericDict() const & {
  AT_ASSERT(isGenericDict(), "Expected GenericDict but got ", tagKind());
  return c10::Dict<IValue, IValue>(toIntrusivePtr<c10::detail::DictImpl>());
}
inline c10::intrusive_ptr<ivalue::Tuple> IValue::toTuple() && {
  AT_ASSERT(isTuple(), "Expected Tuple but got ", tagKind());
  return moveToIntrusivePtr<ivalue::Tuple>();
}
inline c10::intrusive_ptr<ivalue::Tuple> IValue::toTuple() const & {
  AT_ASSERT(isTuple(), "Expected Tuple but got ", tagKind());
  return toIntrusivePtr<ivalue::Tuple>();
}

inline IValue::IValue(c10::intrusive_ptr<ivalue::Tuple> v)
: tag(Tag::Tuple), is_intrusive_ptr(true) {
  payload.as_intrusive_ptr = v.release();
}
inline IValue::IValue(c10::List<int64_t> v)
: tag(Tag::IntList), is_intrusive_ptr(true) {
  payload.as_intrusive_ptr = v.impl_.release();
}
inline IValue::IValue(std::vector<int64_t> v)
: IValue(c10::impl::toList(v)) {}
inline IValue::IValue(c10::ArrayRef<int64_t> v)
: IValue(c10::List<int64_t>(v)) {}

inline IValue::IValue(c10::intrusive_ptr<ivalue::ConstantString> v)
: tag(Tag::String), is_intrusive_ptr(true) {
  payload.as_intrusive_ptr = v.release();
}
inline IValue::IValue(std::string v)
: IValue(ivalue::ConstantString::create(std::move(v))) {}

inline IValue::IValue(c10::List<double> v)
: tag(Tag::DoubleList), is_intrusive_ptr(true) {
  payload.as_intrusive_ptr = v.impl_.release();
}
inline IValue::IValue(std::vector<double> v)
: IValue(c10::impl::toList(std::move(v))) {}

inline IValue::IValue(c10::List<bool> v)
: tag(Tag::BoolList), is_intrusive_ptr(true) {
  payload.as_intrusive_ptr = v.impl_.release();
}
inline IValue::IValue(std::vector<bool> v)
: IValue(c10::impl::toList(std::move(v))) {}

inline IValue::IValue(c10::List<at::Tensor> v)
: tag(Tag::TensorList), is_intrusive_ptr(true) {
  payload.as_intrusive_ptr = v.impl_.release();
}
inline IValue::IValue(std::vector<at::Tensor> v)
: IValue(c10::impl::toList(std::move(v))) {}

inline IValue::IValue(c10::impl::GenericList v)
: tag(Tag::GenericList), is_intrusive_ptr(true) {
  payload.as_intrusive_ptr = v.impl_.release();
}

template<class T> inline IValue::IValue(c10::List<T> v)
: IValue(impl::toGenericList<T>(std::move(v))) {
  static_assert(std::is_same<IValue, typename c10::List<T>::StorageT>::value, "Can only use this constructor for generic list types");
}
template<class T> inline IValue::IValue(std::vector<T> v)
: IValue(impl::GenericList()) {
  static_assert(std::is_same<IValue, typename c10::List<T>::StorageT>::value, "Can only use this constructor for generic list types");
  auto list = to<c10::List<T>>();
  list.reserve(v.size());
  for (auto& e : v) {
    list.push_back(std::move(e));
  }
}

inline IValue::IValue(c10::impl::GenericDict v)
: tag(Tag::GenericDict), is_intrusive_ptr(true) {
  payload.as_intrusive_ptr = v.impl_.release();
}
template<class Key, class Value>
inline IValue::IValue(c10::Dict<Key, Value> v)
: IValue(impl::toGenericDict(std::move(v))) {}

template<class Key, class Value> inline IValue::IValue(std::unordered_map<Key, Value> v)
: IValue(Dict<Key, Value>()) {
  auto dict = to<c10::Dict<Key, Value>>();
  dict.reserve(v.size());
  for (auto& e : v) {
    dict.insert(std::move(e.first), std::move(e.second));
  }
}

template<class T> inline IValue::IValue(c10::optional<T> v): IValue() {
  if (v.has_value()) {
    *this = IValue(std::move(*v));
  }
}

inline IValue::IValue(c10::nullopt_t): IValue() {}

inline IValue::IValue(c10::intrusive_ptr<ivalue::Object> v)
: tag(Tag::Object), is_intrusive_ptr(true) {
  payload.as_intrusive_ptr = v.release();
}
inline IValue::IValue(c10::intrusive_ptr<ivalue::Future> v)
: tag(Tag::Future), is_intrusive_ptr(true) {
  payload.as_intrusive_ptr = v.release();
}

inline const std::string& IValue::toStringRef() const {
  return toString()->string();
}

template<typename T>
inline optional<T> IValue::toOptional() {
  if (this->isNone()) {
    return nullopt;
  }
  return this->to<T>();
}

inline bool IValue::isSameIdentity(const IValue& rhs) const {
  // We choose to not use memcmp for payload check due to potential random padding characters on union type

  // Semantics:
  // 1. None is None, False is False, and True is True are all true
  // 2. If it is a tensor type, we need to take undefined tensor into account
  // 3. Undefined_tensor is None and vice versa should be true
  // 4. If it is a reference type (i.e. is_intrusive_ptr), then is is True when the pointed-to object is the same.
  // 5. False for all other comparisons.
  if (this->isNone() && rhs.isNone()) {
    return true;
  } else if (this->isBool() && rhs.isBool()) {
    // for bool type, do equality check
    return this->toBool() == rhs.toBool();
  } else if (this->isTensor() && rhs.isTensor()) {
    // for tensor type, just check the as_intrusive_ptr since is_intrusive_ptr is false for undefined tensor
    return this->payload.as_intrusive_ptr == rhs.payload.as_intrusive_ptr;
  } else if (this->isTensor() && rhs.isNone()) {
    // special case: undefined tensor and None are the same identity
    return !this->is_intrusive_ptr;
  } else if (this->isNone() && rhs.isTensor()) {
    // special case: undefined tensor and None are the same identity
    return !rhs.is_intrusive_ptr;
  } else {
    // for objects holding in IValue, do shallow compare on pointer address to testify the identity
    return this->is_intrusive_ptr && rhs.is_intrusive_ptr
        && this->payload.as_intrusive_ptr == rhs.payload.as_intrusive_ptr;
  }
}

} // namespace c10
