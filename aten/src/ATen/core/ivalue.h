#pragma once

#include <condition_variable>
#include <type_traits>

#include <c10/core/Scalar.h>
#include <c10/core/TensorImpl.h>
#include <c10/core/UndefinedTensorImpl.h>
#include <c10/core/blob.h>
#include <c10/util/intrusive_ptr.h>
#include <c10/core/IValue.h>

#include <ATen/core/Tensor.h>

namespace c10 {
struct IValue;

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

struct Future;

struct CAFFE2_API Tuple : public List<IValue> {
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

}

// IValue is the generic tagged union used by the interpreter to hold
// all value types.
// It is a 16-byte object with an 8-byte payload and an 8-byte tag.
// The tag is currently 4 bytes to determine the type, and 1 byte
// to mark whether that type is a subtype of c10::intrusive_ptr_target and needs
// retain/release calls.

struct CAFFE2_API IValue final {
  using Tag = c10::core::C10IValue::Tag;

  IValue(): impl() {}

  IValue(const IValue& rhs) = default;
  IValue(IValue&& rhs) noexcept = default;
  ~IValue() = default;

  IValue & operator=(IValue && rhs) & noexcept = default;
  IValue & operator=(IValue const & rhs) & = default;

  void dump() const;

  bool isAliasOf(const IValue& rhs) const {
    if (impl.tag != rhs.impl.tag) {
      // Trivially don't alias if the type is different
      return false;
    }

    if (!impl.is_intrusive_ptr) {
      // Primitive types don't alias anything
      return false;
    }

    AT_ASSERT(rhs.impl.is_intrusive_ptr);

    // Tensors should be compared based on internal storage
    if (isTensor()) {
      const auto thisTensor = toTensor();
      const auto rhsTensor = rhs.toTensor();
      return thisTensor.is_alias_of(rhsTensor);
    }

    // Other types can be compared by their ptr value
    return impl.payload.as_intrusive_ptr == rhs.impl.payload.as_intrusive_ptr;
  }

  void swap(IValue & rhs) noexcept {
    impl.swap(rhs.impl);
  }

  // Accessors for subtypes are arranged together below
  // While some of these accessors could be generated through templates,
  // we prefer to write them manually for clarity

  // None
  bool isNone() const { return impl.isNone(); }
  std::string toNone() const { return impl.toNone(); }

  // Tensor
  IValue(at::Tensor t): impl(C10Tensor(std::move(t))) {}
  bool isTensor() const { return impl.isTensor(); }
  at::Tensor toTensor() && { return at::Tensor(std::move(impl).toTensor()); }
  at::Tensor toTensor() const & { return at::Tensor(impl.toTensor()); }

  const IValue& toIValue() const {
    return *this;
  }
  IValue& toIValue() {
    return *this;
  }

  IValue(caffe2::Blob blob) : impl(std::move(blob)) {}
  bool isBlob() const { return impl.isBlob(); }
  caffe2::Blob& toBlob() & { return impl.toBlob(); }
  const caffe2::Blob& toBlob() const& { return impl.toBlob(); }

  // Tuple
  IValue(c10::intrusive_ptr<ivalue::Tuple> v)
  : impl(Tag::Tuple, std::move(v)) {}
  bool isTuple() const { return impl.isIntrusivePtr(Tag::Tuple); }
  c10::intrusive_ptr<ivalue::Tuple> toTuple() && {
    return std::move(impl).toIntrusivePtr<ivalue::Tuple>(Tag::Tuple);
  }
  c10::intrusive_ptr<ivalue::Tuple> toTuple() const & {
    return impl.toIntrusivePtr<ivalue::Tuple>(Tag::Tuple);
  }

  // Double
  IValue(double d): impl(d) {}
  bool isDouble() const { return impl.isDouble(); }
  double toDouble() const { return impl.toDouble(); }

  // Int
  IValue(int64_t i): impl(i) {}
  IValue(int32_t i) : impl(i) {}
  bool isInt() const { return impl.isInt(); }
  int64_t toInt() const { return impl.toInt(); }

  // Bool
  IValue(bool b): impl(b) {}
  bool isBool() const { return impl.isBool(); }
  bool toBool() const { return impl.toBool(); }

  // Future
  IValue(c10::intrusive_ptr<ivalue::Future> v)
  : impl(Tag::Future, std::move(v)) {}
  bool isFuture() const { return impl.isIntrusivePtr(Tag::Future); }
  c10::intrusive_ptr<ivalue::Future> toFuture() && {
    return std::move(impl).toIntrusivePtr<ivalue::Future>(Tag::Future);
  }
  c10::intrusive_ptr<ivalue::Future> toFuture() const & {
    return impl.toIntrusivePtr<ivalue::Future>(Tag::Future);
  }

  // IntList
  IValue(c10::intrusive_ptr<ivalue::IntList> v)
  : impl(Tag::IntList, std::move(v)) {}
  IValue(std::vector<int64_t> v) : IValue(ivalue::IntList::create(std::move(v))) {}
  IValue(at::ArrayRef<int64_t> v) : IValue(v.vec()) {}
  bool isIntList() const { return impl.isIntrusivePtr(Tag::IntList); }
  c10::intrusive_ptr<ivalue::IntList> toIntList() && {
    return std::move(impl).toIntrusivePtr<ivalue::IntList>(Tag::IntList);
  }
  c10::intrusive_ptr<ivalue::IntList> toIntList() const & {
    return impl.toIntrusivePtr<ivalue::IntList>(Tag::IntList);
  }

  // DoubleList
  IValue(c10::intrusive_ptr<ivalue::DoubleList> v)
  : impl(Tag::DoubleList, std::move(v)) {}
  IValue(std::vector<double> v) : IValue(ivalue::DoubleList::create(std::move(v))) {}
  IValue(at::ArrayRef<double> v) : IValue(v.vec()) {}
  bool isDoubleList() const { return impl.isIntrusivePtr(Tag::DoubleList); }
  c10::intrusive_ptr<ivalue::DoubleList> toDoubleList() && {
    return std::move(impl).toIntrusivePtr<ivalue::DoubleList>(Tag::DoubleList);
  }
  c10::intrusive_ptr<ivalue::DoubleList> toDoubleList() const & {
    return impl.toIntrusivePtr<ivalue::DoubleList>(Tag::DoubleList);
  }

  // BoolList
  IValue(c10::intrusive_ptr<ivalue::BoolList> v)
  : impl(Tag::BoolList, std::move(v)) {}
  IValue(std::vector<bool> v) : IValue(ivalue::BoolList::create(std::move(v))) {}
  IValue(at::ArrayRef<bool> v) : IValue(v.vec()) {}
  bool isBoolList() const { return impl.isIntrusivePtr(Tag::BoolList); }
  c10::intrusive_ptr<ivalue::BoolList> toBoolList() && {
    return std::move(impl).toIntrusivePtr<ivalue::BoolList>(Tag::BoolList);
  }
  c10::intrusive_ptr<ivalue::BoolList> toBoolList() const & {
    return impl.toIntrusivePtr<ivalue::BoolList>(Tag::BoolList);
  }

  //TensorList
  IValue(c10::intrusive_ptr<ivalue::TensorList> v)
  : impl(Tag::TensorList, std::move(v)) {}
  IValue(std::vector<at::Tensor> v): IValue(ivalue::TensorList::create(std::move(v))) {}
  IValue(at::ArrayRef<at::Tensor> v) : IValue(v.vec()) {}
  bool isTensorList() const { return impl.isIntrusivePtr(Tag::TensorList); }
  c10::intrusive_ptr<ivalue::TensorList> toTensorList() && {
    return std::move(impl).toIntrusivePtr<ivalue::TensorList>(Tag::TensorList);
  }
  c10::intrusive_ptr<ivalue::TensorList> toTensorList() const & {
    return impl.toIntrusivePtr<ivalue::TensorList>(Tag::TensorList);
  }

  //GenericList
  IValue(c10::intrusive_ptr<ivalue::GenericList> v)
  : impl(Tag::GenericList, std::move(v)) {}
  IValue(std::vector<IValue> v): IValue(ivalue::GenericList::create(std::move(v))) {}
  IValue(at::ArrayRef<IValue> v) : IValue(v.vec()) {}
  bool isGenericList() const { return impl.isIntrusivePtr(Tag::GenericList); }
  c10::intrusive_ptr<ivalue::GenericList> toGenericList() && {
    return std::move(impl).toIntrusivePtr<ivalue::GenericList>(Tag::GenericList);
  }
  c10::intrusive_ptr<ivalue::GenericList> toGenericList() const & {
    return impl.toIntrusivePtr<ivalue::GenericList>(Tag::GenericList);
  }

  // ConstantString
  IValue(c10::intrusive_ptr<ivalue::ConstantString> v)
  : impl(Tag::String, std::move(v)) {}
  IValue(std::string v) : IValue(ivalue::ConstantString::create(std::move(v))) {}
  bool isString() const { return impl.isIntrusivePtr(Tag::String); }
  c10::intrusive_ptr<ivalue::ConstantString> toString() && {
    return std::move(impl).toIntrusivePtr<ivalue::ConstantString>(Tag::String);
  }
  c10::intrusive_ptr<ivalue::ConstantString> toString() const & {
    return impl.toIntrusivePtr<ivalue::ConstantString>(Tag::String);
  }

  const std::vector<int64_t>& toIntListRef() const;
  const std::vector<double>& toDoubleListRef() const;
  const std::vector<bool>& toBoolListRef() const;
  const std::vector<at::Tensor>& toTensorListRef() const;
  const std::vector<IValue>& toGenericListRef() const;
  const std::string& toStringRef() const;

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

  // Device
  IValue(c10::Device d) : impl(d) {}
  bool isDevice() const { return impl.isDevice(); }
  c10::Device toDevice() const { return impl.toDevice(); }

  // ScalarType
  at::ScalarType toScalarType() const {
    return static_cast<at::ScalarType>(toInt());
  }

  // Layout
  at::Layout toLayout() const {
    return static_cast<at::Layout>(toInt());
  }

  // for debugging
  std::string tagKind() const {
    return impl.tagKind();
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

  template<typename T>
  optional<T> toOptional();

  // this is a shallow comparison of two IValues to test the object identity
  bool isSameIdentity(IValue& rhs);

  CAFFE2_API friend std::ostream& operator<<(
      std::ostream& out,
      const IValue& v);

  bool isPtrType() const {
    return impl.isPtrType();
  }

 private:
  Tag tag() const {
    return impl.tag;
  }


  c10::core::C10IValue impl;
};

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
DEFINE_TO(c10::intrusive_ptr<ivalue::DoubleList>, toDoubleList)
DEFINE_TO(c10::intrusive_ptr<ivalue::IntList>, toIntList)
DEFINE_TO(c10::intrusive_ptr<ivalue::BoolList>, toBoolList)
DEFINE_TO(c10::intrusive_ptr<ivalue::TensorList>, toTensorList)
DEFINE_TO(c10::intrusive_ptr<ivalue::GenericList>, toGenericList)
DEFINE_TO(c10::intrusive_ptr<ivalue::ConstantString>, toString)
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

// note: when adding a DEFINE_TO case here you should also add a
// toX method to IValue. These named methods are much more discoverable
// than the to templated function.

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

inline bool IValue::isSameIdentity(IValue& rhs) {
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
    return impl.toTensor().impl() == rhs.impl.toTensor().impl();
  } else if (this->isTensor() && rhs.isNone()) {
    // special case: undefined tensor and None are the same identity
    return !impl.toTensor().defined();
  } else if (this->isNone() && rhs.isTensor()) {
    // special case: undefined tensor and None are the same identity
    return !rhs.toTensor().defined();
  } else {
    // for objects holding in IValue, do shallow compare on pointer address to testify the identity
    return impl.isSameIntrusivePtr(rhs.impl);
  }
}


} // namespace c10
