#pragma once

#include <condition_variable>
#include <type_traits>

#include <ATen/core/Dict.h>
#include <ATen/core/List.h>
#include <ATen/core/functional.h>
#include <ATen/core/interned_strings.h>
#include <ATen/core/qualified_name.h>
#include <ATen/core/rref_interface.h>
#include <c10/core/Scalar.h>
#include <c10/core/Stream.h>
#include <c10/core/TensorImpl.h>
#include <c10/core/UndefinedTensorImpl.h>
#include <c10/util/intrusive_ptr.h>
#include <c10/util/hash.h>

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
  return c10::Stream::unpack(payload.u.as_int);
}
inline c10::Stream IValue::toStream() const& {
  return c10::Stream::unpack(payload.u.as_int);
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

namespace ivalue {

void TORCH_API
checkCustomClassType(const Type* expected_type, const Type* actual_type);

template <typename T>
using Shared = c10::intrusive_ptr<T>;

// string
struct TORCH_API ConstantString final : c10::intrusive_ptr_target {
 private:
  const std::string str_;

 public:
  ConstantString(std::string str) : str_(std::move(str)) {}
  static c10::intrusive_ptr<ConstantString> create(std::string str_);
  const std::string& string() const {
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

struct TORCH_API Tuple : c10::intrusive_ptr_target {
 private:
  std::vector<IValue> elements_;
  mutable std::shared_ptr<TupleType>
      type_; // lazily computed for unnamed tuples

 public:
  // named tuples have additional type information, so we
  // directly create them tagged
  static c10::intrusive_ptr<Tuple> createNamed(
      std::vector<IValue> elements_,
      std::shared_ptr<TupleType> type_) {
    return c10::make_intrusive<Tuple>(std::move(elements_), type_);
  }
  static c10::intrusive_ptr<Tuple> create(std::vector<IValue> elements_) {
    return c10::make_intrusive<Tuple>(std::move(elements_));
  }

  template <typename... Args>
  static c10::intrusive_ptr<Tuple> create(Args&&... elements_) {
    return c10::make_intrusive<Tuple>(
        std::vector<IValue>{IValue(std::forward<Args>(elements_))...});
  }

  const std::vector<IValue>& elements() const& {
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
  std::shared_ptr<TupleType> type() const;

  static size_t hash(const Tuple& t) {
    return c10::get_hash(t.elements());
  }

  TORCH_API friend bool operator==(
      const ivalue::Tuple& lhs,
      const ivalue::Tuple& rhs);

 private:
  Tuple(std::vector<IValue> elements, std::shared_ptr<TupleType> type = nullptr)
      : elements_(std::move(elements)), type_(std::move(type)) {}

  friend class c10::intrusive_ptr<Tuple>;
};

struct Object;
struct PyObjectHolder;
struct EnumHolder;
} // namespace ivalue

// Future
struct C10_EXPORT ivalue::Future : c10::intrusive_ptr_target {
 private:
  c10::intrusive_ptr<Future> intrusive_from_this() {
    c10::raw::intrusive_ptr::incref(this); // we are creating a new pointer
                                           // from a raw `this` pointer
                                           // so we need to bump the refcount
                                           // to account for this ownership
    return c10::intrusive_ptr<Future>::reclaim(this);
  }

 public:
  explicit Future(TypePtr type) : type_(type) {}
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
    while (!completed_) {
      finished_cv_.wait(lock);
    }

    if (!eptr_) {
      postWaitHook(value_);
    }
  }

  /**
   * Wait on the future until it completes and throw an
   * exception if an error exists.
   */
  void waitAndThrow() {
    std::unique_lock<std::mutex> lock(mutex_);
    while (!completed_) {
      finished_cv_.wait(lock);
    }

    if (eptr_) {
      std::rethrow_exception(eptr_);
    }

    postWaitHook(value_);
  }

  /**
   * Explicitly mark the future as completed with the output value.
   */
  void markCompleted(IValue value) {
    markCompletedWithDataPtrs(std::move(value));
  }

  /**
   * Explicitly mark the future as completed with the output value and DataPtrs.
   * The data_ptrs contains storage pointers for all tensors in IValue, which
   * will be passed to preMarkCompletedHook. Some subclass, like CUDAFuture,
   * uses these DataPtrs to synchronize CUDA streams. You only need to provide
   * data_ptrs when 1) DataPtrs cannot be extracted through
   * IValue::getSubValues() or 2) customized DataPtrs extraction is more
   * efficient.
   */
  void markCompletedWithDataPtrs(
      IValue value,
      c10::optional<std::vector<std::reference_wrapper<const at::DataPtr>>>
          data_ptrs = c10::nullopt) {
    std::unique_lock<std::mutex> lock(mutex_);
    TORCH_CHECK(
        !completed(),
        "Attempting to mark a completed Future as complete again. Note that "
        "a Future can only be marked completed once.");

    preMarkCompletedHook(value, std::move(data_ptrs));
    // Only set value_ and completed_ flag once preMarkCompletedHook has
    // returned successfully to allow for proper error propagation.
    value_ = std::move(value);
    completed_ = true;

    std::vector<std::function<void(void)>> cbs;
    cbs.swap(callbacks_);
    lock.unlock();

    finished_cv_.notify_all();
    for (auto& callback : cbs) {
      callback();
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
      LOG(INFO)
          << "Skipping setting following error on the Future since "
          << "it is already marked completed (this is not neccessarily an error): "
          << tryRetrieveErrorMessageInternal(eptr);
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
    AT_ASSERT(!eptr_);
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
    callback = wrapCallback(std::move(callback));
    if (completed()) {
      lock.unlock();
      callback();
      return;
    }
    callbacks_.emplace_back(std::move(callback));
  }

  /**
   * Add a callback to the future, and return another Future to hold the return
   * value of the callback. This is necessary when the callback provider needs
   * to know for sure when the callback has finished.
   */
  c10::intrusive_ptr<Future> then(
      std::function<IValue(void)> callback,
      TypePtr type) {
    auto fut = createInstance(std::move(type));
    addCallback(
        [fut, cb = std::move(callback)]() {
          try {
            fut->markCompleted(cb());
          } catch (std::exception&) {
            fut->setError(std::current_exception());
          }
        });
    return fut;
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

  // This method should be overridden by subclasses so that they can produce an
  // instace of their own type.
  virtual c10::intrusive_ptr<Future> createInstance(at::TypePtr type) {
    return c10::make_intrusive<Future>(type);
  }

 protected:

  // This hook will be called by this class (the superclass) when the future is
  // marked completed _with a value_ (hence not in case of error). This is done
  // right away, while the mutex is still held, before any callbacks are run.
  // It allows subclasses to further update their state if they so need. For
  // example the CUDAFuture subclass uses it to determine what devices the value
  // resides on and record an event in those devices' current streams.
  // The data_ptrs field contains storage pointers of all tensors in the value,
  // which is used by the CUDAFuture subclass to synchronize streams.
  virtual void preMarkCompletedHook(
      const at::IValue& value,
      c10::optional<std::vector<std::reference_wrapper<const at::DataPtr>>>
          data_ptrs) {}

  // This hook will be called by the addCallback() and the then() methods before
  // storing the callback for later execution (or before running it inline if
  // the future is already complete). Note that this method could thus be called
  // while the future is _not_ yet complete. By default this method does nothing
  // but subclasses can override this method to add functionality. For example
  // the CUDAFuture subclass ensures the callback runs with CUDA streams which
  // are synchronized with the events recorded in the I/O streams.
  virtual std::function<void(void)> wrapCallback(
      std::function<void(void)> callback) {
    return callback;
  }

  // This hook will be called by this class after a user thread has completed
  // waiting on a successful future. It will thus not be called if the future
  // completes with an error. It will also not be called if the user accesses
  // the future's value without synchronization. Subclasses can override this
  // to add some synchronization to the wait. For example, the CUDAFuture
  // subclass ensures the user's current CUDA streams synchronize with the I/O
  // events stored by the future.
  virtual void postWaitHook(const at::IValue& value) {}

 private:
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

    // Do not call preMarkCompletedHook() here as there isn't any value.

    std::vector<std::function<void(void)>> cbs;
    cbs.swap(callbacks_);
    lock.unlock();

    finished_cv_.notify_all();
    for (auto& callback : cbs) {
      callback();
    }
  }

  // Tries to retrieve the error message from std::exception_ptr.
  std::string tryRetrieveErrorMessageInternal(std::exception_ptr eptr) const {
    try {
      std::rethrow_exception(eptr);
    } catch (const std::exception& e) {
      return e.what();
    } catch (...) {
      return "Unknown Exception Type";
    }
  }

  mutable std::mutex mutex_;
  std::atomic_bool completed_ = {false}; // is this future complete
  std::condition_variable finished_cv_;

  IValue value_; // when finished the value
  TypePtr type_;
  std::vector<std::function<void(void)>> callbacks_;
  std::exception_ptr eptr_;
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
  Object(StrongTypePtr type, size_t numSlots) : type_(std::move(type)) {
    slots_.resize(numSlots);
  }

  static c10::intrusive_ptr<Object> create(
      StrongTypePtr type,
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
    return type_.cu_;
  }

  c10::intrusive_ptr<Object> copy() const;

  c10::intrusive_ptr<Object> deepcopy() const;

  c10::intrusive_ptr<Object> deepcopy(IValue::HashAliasedIValueMap& memo) const;

 private:
  void resizeObject(size_t slot);
  StrongTypePtr type_;
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

  virtual ~PyObjectHolder(){};
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

inline const ivalue::Object& IValue::toObjectRef() const {
  AT_ASSERT(isObject(), "Expected Object but got ", tagKind());
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(payload.u.as_intrusive_ptr != c10::UndefinedTensorImpl::singleton(), "Attempted to create null reference");
  return *static_cast<const c10::ivalue::Object*>(payload.u.as_intrusive_ptr);
}

// note: when adding a DEFINE_TO case here you should also add a
// toX method to IValue. These named methods are much more discoverable
// than the to templated function.

#define DEFINE_TO(T, method_name)          \
  template <>                              \
  inline T IValue::to<T>()&& {             \
    return std::move(*this).method_name(); \
  }                                        \
  template <>                              \
  inline c10::detail::ivalue_to_const_ref_overload_return<T>::type IValue::to<T>() const& { \
    return this->method_name();            \
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
DEFINE_TO(c10::intrusive_ptr<ivalue::Future>, toFuture)
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
  const Type* expected_type = c10::getCustomClassType<c10::intrusive_ptr<T>>().get();
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
  const Type* expected_type = c10::getCustomClassType<c10::intrusive_ptr<T>>().get();
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
static std::vector<T> createVectorFromList(const c10::detail::ListImpl* impl) {
  std::vector<T> result;
  result.reserve(impl->list.size());
  for (size_t i = 0, N = impl->list.size(); i < N; ++i) {
    result.push_back(impl->list[i].to<T>());
  }
  return result;
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
    specialized_dict[item.key().to<K>()] = item.value().to<V>();
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
    const std::vector<IValue>& t,
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
  auto vals = ivalue.toTuple()->elements();
  TORCH_CHECK(vals.size() == sizeof...(Args));
  return detail::generic_to_tuple_impl<std::tuple<Args...>>(vals, Indices{});
}

template <typename T>
inline T IValue::to() && {
  return generic_to(std::move(*this), _fake_type<T>{});
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

inline IValue::IValue(c10::intrusive_ptr<ivalue::Tuple> v)
    : tag(Tag::Tuple), is_intrusive_ptr(true) {
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
    : IValue(
          std::move(c10::guts::apply(c10::ivalue::Tuple::create<const Args&...>, t))) {
}

template <
    typename... Args,
    std::enable_if_t<
        !guts::disjunction<
            std::is_lvalue_reference<Args>...,
            guts::negation<std::is_constructible<IValue, Args>>...>::value,
        std::nullptr_t>>
inline IValue::IValue(std::tuple<Args...>&& t)
    : IValue(
          std::move(c10::guts::apply(c10::ivalue::Tuple::create<Args&&...>, std::move(t)))) {
}

inline IValue::IValue(c10::intrusive_ptr<ivalue::ConstantString> v)
    : tag(Tag::String), is_intrusive_ptr(true) {
  payload.u.as_intrusive_ptr = null_to_undefined_tensor(v.release());
}
inline IValue::IValue(std::string v)
    : IValue(ivalue::ConstantString::create(std::move(v))) {}

inline IValue::IValue(c10::impl::GenericList v)
    : tag(Tag::GenericList), is_intrusive_ptr(true) {
  payload.u.as_intrusive_ptr = null_to_undefined_tensor(v.impl_.release());
}

template <class T, IValue::enable_if_ivalue_constructible<T>>
inline IValue::IValue(c10::List<T>&& v) : IValue(impl::toList<T>(std::move(v))) {}
template <class T, IValue::enable_if_ivalue_constructible<T>>
inline IValue::IValue(const c10::List<T>& v) : IValue(impl::toList<T>(v)) {}
template <class T, IValue::enable_if_ivalue_constructible<T>>
inline IValue::IValue(at::ArrayRef<T> v) : IValue(c10::List<T>()) {
  auto list = to<c10::List<T>>();
  list.reserve(v.size());
  for (const auto& e : v) {
    list.push_back(e);
  }
}
template <class T, IValue::enable_if_ivalue_constructible<T>>
inline IValue::IValue(const std::vector<T>& v) : IValue(c10::List<T>()) {
  auto list = to<c10::List<T>>();
  list.reserve(v.size());
  for (const auto& e : v) {
    list.push_back(e);
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

inline IValue::IValue(c10::impl::GenericDict v)
    : tag(Tag::GenericDict), is_intrusive_ptr(true) {
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
    : tag(Tag::Object), is_intrusive_ptr(true) {
  payload.u.as_intrusive_ptr = null_to_undefined_tensor(v.release());
}

inline IValue::IValue(c10::intrusive_ptr<ivalue::PyObjectHolder> v)
    : tag(Tag::PyObject), is_intrusive_ptr(true) {
  payload.u.as_intrusive_ptr = null_to_undefined_tensor(v.release());
}

inline IValue::IValue(c10::intrusive_ptr<ivalue::EnumHolder> v)
    : tag(Tag::Enum), is_intrusive_ptr(true) {
  payload.u.as_intrusive_ptr = null_to_undefined_tensor(v.release());
}

inline IValue IValue::make_capsule(
    intrusive_ptr<torch::CustomClassHolder> blob) {
  IValue iv;
  iv.tag = Tag::Capsule;
  iv.is_intrusive_ptr = true;
  iv.payload.u.as_intrusive_ptr = null_to_undefined_tensor(blob.release());
  return iv;
}

template <
    typename T,
    std::enable_if_t<std::is_base_of<torch::CustomClassHolder, T>::value, int>>
IValue::IValue(c10::intrusive_ptr<T> custom_class) {
  TypePtr classType = []() {
    try {
      return c10::getCustomClassType<c10::intrusive_ptr<T>>();
    } catch (const c10::Error&) {
      throw c10::Error(
          "Trying to instantiate a class that isn't a registered custom class: " +
          std::string(c10::util::get_fully_qualified_type_name<T>()),
          "");
    }
  }();
  auto ivalue_obj = c10::ivalue::Object::create(
      c10::StrongTypePtr(nullptr, classType), /*num_slots=*/1);
  ivalue_obj->setSlot(0, IValue::make_capsule(std::move(custom_class)));
  payload.u.as_intrusive_ptr = null_to_undefined_tensor(ivalue_obj.release());
  tag = Tag::Object;
  is_intrusive_ptr = true;
}

inline IValue::IValue(c10::intrusive_ptr<ivalue::Future> v)
    : tag(Tag::Future), is_intrusive_ptr(true) {
  payload.u.as_intrusive_ptr = null_to_undefined_tensor(v.release());
}

inline IValue::IValue(c10::intrusive_ptr<c10::RRefInterface> v)
    : tag(Tag::RRef), is_intrusive_ptr(true) {
  payload.u.as_intrusive_ptr = null_to_undefined_tensor(v.release());
}

inline IValue::IValue(c10::intrusive_ptr<at::Quantizer> v)
    : tag(Tag::Quantizer), is_intrusive_ptr(true) {
  payload.u.as_intrusive_ptr = null_to_undefined_tensor(v.release());
}

template <typename T>
inline IValue::IValue(c10::complex<T> c)
    : tag(Tag::ComplexDouble), is_intrusive_ptr(true) {
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
  // 4. If it is a reference type (i.e. is_intrusive_ptr), then is is True when
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
    return this->is_intrusive_ptr && rhs.is_intrusive_ptr &&
        this->payload.u.as_intrusive_ptr == rhs.payload.u.as_intrusive_ptr;
  }
}

namespace ivalue {
namespace detail {

template <typename T>
IValue from_(T&& x, std::true_type) {
  return IValue(std::move(x));
}
template <typename T>
IValue from_(c10::intrusive_ptr<T> x, std::false_type) {
  return IValue(std::move(x));
}
template <typename T>
IValue from_(T&& x, std::false_type) {
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
} // namespace c10
