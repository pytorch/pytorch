#pragma once

#include <condition_variable>
#include <type_traits>

#include <ATen/core/functional.h>
#include <ATen/core/interned_strings.h>
#include <c10/core/Scalar.h>
#include <c10/core/TensorImpl.h>
#include <c10/core/UndefinedTensorImpl.h>
#include <ATen/core/Dict.h>

namespace torch {
namespace jit {
struct Function;
} // namespace jit
} // namespace torch
namespace c10 {
struct IValue;
struct ClassType;

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

inline c10::intrusive_ptr<ivalue::Tuple> IValue::toTuple() && {
  AT_ASSERT(isTuple());
  return moveToIntrusivePtr<ivalue::Tuple>();
}
inline c10::intrusive_ptr<ivalue::Tuple> IValue::toTuple() const & {
  AT_ASSERT(isTuple());
  return toIntrusivePtr<ivalue::Tuple>();
}
inline c10::intrusive_ptr<ivalue::Future> IValue::toFuture() && {
  AT_ASSERT(isFuture());
  return moveToIntrusivePtr<ivalue::Future>();
}
inline c10::intrusive_ptr<ivalue::Future> IValue::toFuture() const & {
  AT_ASSERT(isFuture());
  return toIntrusivePtr<ivalue::Future>();
}
inline c10::intrusive_ptr<ivalue::IntList> IValue::toIntList() && {
  AT_ASSERT(isIntList());
  return moveToIntrusivePtr<ivalue::IntList>();
}
inline c10::intrusive_ptr<ivalue::IntList> IValue::toIntList() const & {
  AT_ASSERT(isIntList());
  return toIntrusivePtr<ivalue::IntList>();
}
inline c10::intrusive_ptr<ivalue::ConstantString> IValue::toString() && {
  AT_ASSERT(isString());
  return moveToIntrusivePtr<ivalue::ConstantString>();
}
inline c10::intrusive_ptr<ivalue::ConstantString> IValue::toString() const & {
  AT_ASSERT(isString());
  return toIntrusivePtr<ivalue::ConstantString>();
}
inline c10::intrusive_ptr<ivalue::DoubleList> IValue::toDoubleList() && {
  AT_ASSERT(isDoubleList());
  return moveToIntrusivePtr<ivalue::DoubleList>();
}
inline c10::intrusive_ptr<ivalue::DoubleList> IValue::toDoubleList() const & {
  AT_ASSERT(isDoubleList());
  return toIntrusivePtr<ivalue::DoubleList>();
}
inline c10::intrusive_ptr<ivalue::BoolList> IValue::toBoolList() && {
  AT_ASSERT(isBoolList());
  return moveToIntrusivePtr<ivalue::BoolList>();
}
inline c10::intrusive_ptr<ivalue::BoolList> IValue::toBoolList() const & {
  AT_ASSERT(isBoolList());
  return toIntrusivePtr<ivalue::BoolList>();
}
inline c10::intrusive_ptr<ivalue::TensorList> IValue::toTensorList() && {
  AT_ASSERT(isTensorList());
  return moveToIntrusivePtr<ivalue::TensorList>();
}
inline c10::intrusive_ptr<ivalue::TensorList> IValue::toTensorList() const & {
  AT_ASSERT(isTensorList());
  return toIntrusivePtr<ivalue::TensorList>();
}
inline c10::intrusive_ptr<ivalue::GenericList> IValue::toGenericList() && {
  AT_ASSERT(isGenericList());
  return moveToIntrusivePtr<ivalue::GenericList>();
}
inline c10::intrusive_ptr<ivalue::GenericList> IValue::toGenericList() const & {
  AT_ASSERT(isGenericList());
  return toIntrusivePtr<ivalue::GenericList>();
}
inline c10::intrusive_ptr<ivalue::GenericDict> IValue::toGenericDict() && {
  AT_ASSERT(isGenericDict());
  return moveToIntrusivePtr<ivalue::GenericDict>();
}
inline c10::intrusive_ptr<ivalue::GenericDict> IValue::toGenericDict() const & {
  AT_ASSERT(isGenericDict());
  return toIntrusivePtr<ivalue::GenericDict>();
}
inline c10::intrusive_ptr<ivalue::Object> IValue::toObject() && {
  AT_ASSERT(isObject());
  return toIntrusivePtr<ivalue::Object>();
}
inline c10::intrusive_ptr<ivalue::Object> IValue::toObject() const & {
  AT_ASSERT(isObject());
  return toIntrusivePtr<ivalue::Object>();
}
inline at::Tensor IValue::toTensor() && {
  AT_ASSERT(isTensor());
  return at::Tensor(moveToIntrusivePtr<at::TensorImpl, at::UndefinedTensorImpl>());
}
inline at::Tensor IValue::toTensor() const & {
  AT_ASSERT(isTensor());
  return at::Tensor(toIntrusivePtr<at::TensorImpl, at::UndefinedTensorImpl>());
}
inline c10::intrusive_ptr<caffe2::Blob> IValue::toBlob() && {
  AT_ASSERT(isBlob());
  return moveToIntrusivePtr<caffe2::Blob>();
}
inline c10::intrusive_ptr<caffe2::Blob> IValue::toBlob() const & {
  AT_ASSERT(isBlob());
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

template <typename Elem>
struct CAFFE2_API List : c10::intrusive_ptr_target {
 private:
  std::vector<Elem> elements_;

 public:
  typedef Elem ElemType;

  List(std::vector<Elem> elements_) : elements_(std::move(elements_)) {}
  static c10::intrusive_ptr<List<Elem>> create(std::vector<Elem> elements_) {
    return c10::make_intrusive<List<Elem>>(std::move(elements_));
  }
  const std::vector<Elem>& elements() const & {
    return elements_;
  }
  operator const std::vector<Elem>&() const {
    return elements();
  }

  std::vector<Elem>& elements() & {
    return elements_;
  }
  operator std::vector<Elem>&() {
    return elements();
  }

  std::vector<Elem>&& elements() && {
    return std::move(elements_);
  }
};

struct Future;
struct GenericDict;

struct CAFFE2_API Tuple : public List<IValue> {
  using List<IValue>::List;
  static c10::intrusive_ptr<Tuple> create(std::vector<IValue> elements_) {
    return c10::make_intrusive<Tuple>(std::move(elements_));
  }
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
    if (completed()) {
      return;
    }
    std::condition_variable finished;
    bool fired = false;

    // Add a callback to notify the current thread
    // when the current future completes.
    addCallback([&] {
      std::unique_lock<std::mutex> lock(mutex_);
      finished.notify_all();
      fired = true;
    });

    // The current thread will be blocked unless the above callback is fired.
    std::unique_lock<std::mutex> lock(mutex_);
    while (!fired) {
      finished.wait(lock);
    }

    AT_ASSERT(completed());
  }

  /**
   * Explicitly mark the future as completed with the output value.
   */
  void markCompleted(IValue value) {
    {
      // This is not to protect completed_ but to create a barrier
      // from possible addCallback() calls
      std::unique_lock<std::mutex> lock(mutex_);
      AT_ASSERT(!completed());
      completed_ = true;
      value_ = std::move(value);
    }

    fireCallbacks();
  }

  void markCompleted(FutureError&& error_) {
    {
      // This is not to protect completed_ but to create a barrier
      // from possible addCallback() calls
      std::unique_lock<std::mutex> lock(mutex_);
      AT_ASSERT(!completed());
      completed_ = true;
      has_error = true;
      error = std::move(error_);
    }

    fireCallbacks();
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
  IValue value_; // when finished the value
  std::atomic_bool completed_ = {false}; // is this future complete
  std::vector<std::function<void(void)>> callbacks;
  bool has_error = false;
  FutureError error;
};

// User-defined object.
struct C10_EXPORT ivalue::Object final : c10::intrusive_ptr_target {
 public:
  Object(std::shared_ptr<ClassType> type, size_t numSlots) : type_(std::move(type)) {
    slots_.resize(numSlots);
  }

  static c10::intrusive_ptr<Object> create(
      std::shared_ptr<ClassType> type,
      size_t numSlots) {
    return c10::make_intrusive<Object>(std::move(type), numSlots);
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
    return type_;
  }

 private:
  void resizeObject(size_t slot);
  std::shared_ptr<ClassType> type_;
  std::vector<IValue> slots_;
};

struct C10_EXPORT ivalue::GenericDict : c10::intrusive_ptr_target {
 private:
  c10::impl::GenericDictPtr elements_;

 public:
  GenericDict(c10::impl::GenericDictPtr elements_)
      : elements_(std::move(elements_)) {}
  static c10::intrusive_ptr<GenericDict> create(
      c10::impl::GenericDictPtr elements_) {
    return c10::make_intrusive<GenericDict>(std::move(elements_));
  }
  const c10::impl::GenericDictPtr& elements() const & {
    return elements_;
  }
  c10::impl::GenericDictPtr& elements() & {
    return elements_;
  }

  using IterationOrder = std::vector<std::pair<IValue, IValue>>;
  const IterationOrder iterationOrder() const;
};

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
DEFINE_TO(c10::intrusive_ptr<ivalue::Tuple>, toTuple)
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
DEFINE_TO(c10::intrusive_ptr<ivalue::DoubleList>, toDoubleList)
DEFINE_TO(c10::intrusive_ptr<ivalue::IntList>, toIntList)
DEFINE_TO(c10::intrusive_ptr<ivalue::BoolList>, toBoolList)
DEFINE_TO(c10::intrusive_ptr<ivalue::TensorList>, toTensorList)
DEFINE_TO(c10::intrusive_ptr<ivalue::GenericList>, toGenericList)
DEFINE_TO(c10::intrusive_ptr<ivalue::GenericDict>, toGenericDict)
DEFINE_TO(c10::intrusive_ptr<ivalue::ConstantString>, toString)
DEFINE_TO(c10::intrusive_ptr<ivalue::Object>, toObject)
DEFINE_TO(at::Scalar, toScalar)
DEFINE_TO(std::vector<int64_t>, toIntListRef)
DEFINE_TO(std::vector<double>, toDoubleListRef)
DEFINE_TO(std::vector<bool>, toBoolListRef)
DEFINE_TO(std::vector<at::Tensor>, toTensorListRef)
DEFINE_TO(std::vector<IValue>, toGenericListRef)
DEFINE_TO(std::string, toStringRef)
DEFINE_TO(c10::intrusive_ptr<ivalue::Future>, toFuture)
DEFINE_TO(IValue, toIValue)
DEFINE_TO(c10::Device, toDevice)
DEFINE_TO(at::ScalarType, toScalarType)
DEFINE_TO(at::Layout, toLayout)
DEFINE_TO(at::MemoryFormat, toMemoryFormat)

template <typename T>
struct _fake_type {};

template <typename Elem>
std::vector<Elem> generic_to(
    const IValue* ivalue,
    _fake_type<std::vector<Elem>>) {
  return fmap(ivalue->toGenericListRef(), [](IValue item_ivalue) { return item_ivalue.to<Elem>(); });
}

template <typename K, typename V>
std::unordered_map<K, V> generic_to(
    const IValue* ivalue,
    _fake_type<std::unordered_map<K, V>>) {
  std::unordered_map<K, V> specialized_dict;

  for (auto item : ivalue->toGenericDictRef()) {
    specialized_dict[item.key().to<K>()] = item.value().to<V>();
  }

  return specialized_dict;
}

template <typename T>
inline T IValue::to() && {
  return generic_to(this, _fake_type<T>{});
}

template <typename T>
inline T IValue::to() const& {
  return generic_to(this, _fake_type<T>{});
}

// note: when adding a DEFINE_TO case here you should also add a
// toX method to IValue. These named methods are much more discoverable
// than the to templated function.

inline IValue::IValue(c10::intrusive_ptr<ivalue::Tuple> v)
: tag(Tag::Tuple), is_intrusive_ptr(true) {
  payload.as_intrusive_ptr = v.release();
}

inline IValue::IValue(c10::intrusive_ptr<ivalue::IntList> v)
: tag(Tag::IntList), is_intrusive_ptr(true) {
  payload.as_intrusive_ptr = v.release();
}
inline IValue::IValue(std::vector<int64_t> v)
: IValue(ivalue::IntList::create(std::move(v))) {}

inline IValue::IValue(c10::intrusive_ptr<ivalue::ConstantString> v)
: tag(Tag::String), is_intrusive_ptr(true) {
  payload.as_intrusive_ptr = v.release();
}
inline IValue::IValue(std::string v)
: IValue(ivalue::ConstantString::create(std::move(v))) {}

inline IValue::IValue(c10::intrusive_ptr<ivalue::DoubleList> v)
: tag(Tag::DoubleList), is_intrusive_ptr(true) {
  payload.as_intrusive_ptr = v.release();
}
inline IValue::IValue(std::vector<double> v)
: IValue(ivalue::DoubleList::create(std::move(v))) {}

inline IValue::IValue(c10::intrusive_ptr<ivalue::BoolList> v)
: tag(Tag::BoolList), is_intrusive_ptr(true) {
  payload.as_intrusive_ptr = v.release();
}
inline IValue::IValue(std::vector<bool> v)
: IValue(ivalue::BoolList::create(std::move(v))) {}

inline IValue::IValue(c10::intrusive_ptr<ivalue::TensorList> v)
: tag(Tag::TensorList), is_intrusive_ptr(true) {
  payload.as_intrusive_ptr = v.release();
}
inline IValue::IValue(std::vector<at::Tensor> v)
: IValue(ivalue::TensorList::create(std::move(v))) {}

inline IValue::IValue(c10::intrusive_ptr<ivalue::GenericList> v)
: tag(Tag::GenericList), is_intrusive_ptr(true) {
  payload.as_intrusive_ptr = v.release();
}
inline IValue::IValue(std::vector<IValue> v)
: IValue(ivalue::GenericList::create(std::move(v))) {}

inline IValue::IValue(c10::intrusive_ptr<ivalue::GenericDict> v)
: tag(Tag::GenericDict), is_intrusive_ptr(true) {
  payload.as_intrusive_ptr = v.release();
}
inline IValue::IValue(c10::impl::GenericDictPtr v)
: IValue(ivalue::GenericDict::create(std::move(v))) {}

inline IValue::IValue(c10::intrusive_ptr<ivalue::Object> v)
: tag(Tag::Object), is_intrusive_ptr(true) {
  payload.as_intrusive_ptr = v.release();
}
inline IValue::IValue(c10::intrusive_ptr<ivalue::Future> v)
: tag(Tag::Future), is_intrusive_ptr(true) {
  payload.as_intrusive_ptr = v.release();
}

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

inline const c10::impl::GenericDictPtr& IValue::
    toGenericDictRef() const {
  return toGenericDict()->elements();
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
