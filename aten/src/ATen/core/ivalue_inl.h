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
#include <ATen/core/rref_interface.h>

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

// For custom class __init__ registration, we need to pass in a function
// that looks like this: [](IValue x, args...)

// However, kernel_functor.h automatically sets the input types of the function
// by introspecting the types of the functor (which is IValue in this case).
// However, we need the type it binds to be Foo.

// Instead, we pass in a lambda [](ivalue_holder<CurClass> x, args...) from
// which getTypePtr can recover the original class pointer.

template <typename TaggedCapsuleType>
struct tagged_capsule {
  IValue ivalue;
};

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

template<class T, class U>
intrusive_ptr<T> static_intrusive_pointer_cast(intrusive_ptr<U> r) {
  return intrusive_ptr<T>::reclaim(static_cast<T*>(r.release()));
}

template<class T, class U>
intrusive_ptr<T> dynamic_intrusive_pointer_cast(intrusive_ptr<U> r) {
  return intrusive_ptr<T>::reclaim(dynamic_cast<T*>(r.release()));
}

inline c10::intrusive_ptr<ivalue::Future> IValue::toFuture() && {
  AT_ASSERT(isFuture(), "Expected Future but got ", tagKind());
  return moveToIntrusivePtr<ivalue::Future>();
}
inline c10::intrusive_ptr<ivalue::Future> IValue::toFuture() const & {
  AT_ASSERT(isFuture(), "Expected Future but got ", tagKind());
  return toIntrusivePtr<ivalue::Future>();
}
inline c10::intrusive_ptr<c10::RRefInterface> IValue::toRRef() && {
  AT_ASSERT(isRRef(), "Expected RRef but got ", tagKind());
  return moveToIntrusivePtr<c10::RRefInterface>();
}
inline c10::intrusive_ptr<c10::RRefInterface> IValue::toRRef() const & {
  AT_ASSERT(isRRef(), "Expected RRef but got ", tagKind());
  return toIntrusivePtr<c10::RRefInterface>();
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
  return moveToIntrusivePtr<ivalue::Object>();
}
inline c10::intrusive_ptr<ivalue::Object> IValue::toObject() const & {
  AT_ASSERT(isObject(), "Expected Object but got ", tagKind());
  return toIntrusivePtr<ivalue::Object>();
}
inline c10::intrusive_ptr<ivalue::PyObjectHolder> IValue::toPyObjectHolder() && {
  TORCH_INTERNAL_ASSERT(isPyObject(), "Expected PyObject but got", tagKind());
  return moveToIntrusivePtr<ivalue::PyObjectHolder>();
}
inline c10::intrusive_ptr<ivalue::PyObjectHolder> IValue::toPyObjectHolder() const & {
  TORCH_INTERNAL_ASSERT(isPyObject(), "Expected PyObject but got", tagKind());
  return toIntrusivePtr<ivalue::PyObjectHolder>();
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
inline c10::intrusive_ptr<torch::jit::CustomClassHolder> IValue::toCapsule() && {
  TORCH_INTERNAL_ASSERT(isCapsule());
  return moveToIntrusivePtr<torch::jit::CustomClassHolder>();
}
inline c10::intrusive_ptr<torch::jit::CustomClassHolder> IValue::toCapsule() const & {
  TORCH_INTERNAL_ASSERT(isCapsule());
  return toIntrusivePtr<torch::jit::CustomClassHolder>();
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
  mutable std::shared_ptr<TupleType> type_; // lazily computed for unnamed tuples

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
  static c10::intrusive_ptr<Tuple> create(Args... elements_) {
    return c10::make_intrusive<Tuple>(std::vector<IValue>{IValue(elements_)...});
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
  std::shared_ptr<TupleType> type() const;

 private:
  Tuple(std::vector<IValue> elements, std::shared_ptr<TupleType> type = nullptr)
    : elements_(std::move(elements)), type_(std::move(type)) {}

  friend class c10::intrusive_ptr<Tuple>;
};

struct Object;
struct PyObjectHolder;
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
  Future(TypePtr type) : type_(type) {}
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

    std::vector<std::function<void(void)>> cbs;
    cbs.swap(callbacks_);
    lock.unlock();

    finished_cv_.notify_all();
    for (auto& callback : cbs) {
      callback();
    }
  }

  void markCompleted() {
    markCompleted(IValue {});
  }

  void markCompleted(FutureError&& error) {
    std::unique_lock<std::mutex> lock(mutex_);
    AT_ASSERT(!completed());
    completed_ = true;
    has_error_ = true;
    error_ = std::move(error);

    std::vector<std::function<void(void)>> cbs;
    cbs.swap(callbacks_);
    lock.unlock();

    finished_cv_.notify_all();
    for (auto& callback : cbs) {
      callback();
    }
  }

  // Get the result of the current future.
  IValue value() {
    std::unique_lock<std::mutex> lock(mutex_);
    AT_ASSERT(completed());
    if (has_error_) {
      throw error_;
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
    callbacks_.push_back(callback);
  }

  // Check if the current future has completed
  bool completed() const{
    return completed_;
  }

  CAFFE2_API friend std::ostream& operator<<(
      std::ostream& out,
      const Future& v);

  TypePtr type() const {
    return type_;
  }

 private:
  std::mutex mutex_;
  std::atomic_bool completed_ = {false}; // is this future complete
  std::condition_variable finished_cv_;

  IValue value_; // when finished the value
  TypePtr type_;
  std::vector<std::function<void(void)>> callbacks_;
  bool has_error_ = false;
  FutureError error_;
};

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

  std::shared_ptr<torch::jit::script::CompilationUnit> compilation_unit() {
    return type_.cu_;
  }

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
  virtual ~PyObjectHolder() {};
};

std::vector<std::pair<IValue, IValue>> iterationOrder(const c10::Dict<IValue, IValue>& dict);

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
DEFINE_TO(c10::impl::GenericList, toList)
DEFINE_TO(c10::impl::GenericDict, toGenericDict)
DEFINE_TO(c10::intrusive_ptr<ivalue::Tuple>, toTuple)
DEFINE_TO(std::string, toStringRef)
DEFINE_TO(c10::intrusive_ptr<ivalue::Future>, toFuture)
DEFINE_TO(c10::intrusive_ptr<c10::RRefInterface>, toRRef)
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
// TODO this is deprecated but we don't throw a warning because a lot of ops in native_functions.yaml still return std::vector.
//C10_DEPRECATED_MESSAGE("IValues based on std::vector<T> are potentially slow and deprecated. Please use torch::List<T> instead.")
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

template <typename T>
T generic_to(
    IValue ivalue,
    _fake_type<T>) {
    using ElemType = typename std::remove_pointer<T>::type::element_type;
    auto obj = std::move(ivalue).toObject();
    auto capsule = obj->getSlot(0);
    return c10::static_intrusive_pointer_cast<ElemType>(capsule.toCapsule());
}

template <typename T>
tagged_capsule<T> generic_to(
    IValue ivalue,
    _fake_type<tagged_capsule<T>>) {
    return tagged_capsule<T>{std::move(ivalue)};
}

template <typename Elem>
c10::List<Elem> generic_to(
    IValue ivalue,
    _fake_type<c10::List<Elem>>) {
  return impl::toTypedList<Elem>(std::move(ivalue).toList());
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
  TORCH_CHECK(list.size() == sizeof...(I), "Tried to convert a List with ", list.size()," elements to a fixed-size array of size ", sizeof...(I));
  return {list[I]...};
}
}

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
C10_DEPRECATED_MESSAGE("IValues based on std::unordered_map are slow and deprecated. Please use c10::Dict<K, V> instead.")
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
c10::optional<T> generic_to(
    IValue ivalue,
    _fake_type<c10::optional<T>>) {
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
}

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
inline T IValue::to() const& {
  return generic_to(*this, _fake_type<T>{});
}

template<typename T>
static std::vector<T> createVectorFromList(const c10::detail::ListImpl* impl) {
  std::vector<T> result;
  result.reserve(impl->list.size());
  for (size_t i = 0, N = impl->list.size(); i < N; ++i) {
    result.push_back(impl->list[i].to<T>());
  }
  return result;
}

inline c10::List<int64_t> IValue::toIntList() && {
  AT_ASSERT(isIntList(), "Expected IntList but got ", tagKind());
  return c10::List<int64_t>(moveToIntrusivePtr<c10::detail::ListImpl>());
}
inline c10::List<int64_t> IValue::toIntList() const & {
  AT_ASSERT(isIntList(), "Expected IntList but got ", tagKind());
  return c10::List<int64_t>(toIntrusivePtr<c10::detail::ListImpl>());
}
inline std::vector<int64_t> IValue::toIntVector() const {
  AT_ASSERT(isIntList(), "Expected IntList but got ", tagKind());
  return createVectorFromList<int64_t>(static_cast<const c10::detail::ListImpl*>(payload.as_intrusive_ptr));
}
inline c10::List<double> IValue::toDoubleList() && {
  AT_ASSERT(isDoubleList(), "Expected DoubleList but got ", tagKind());
  return c10::List<double>(moveToIntrusivePtr<c10::detail::ListImpl>());
}
inline c10::List<double> IValue::toDoubleList() const & {
  AT_ASSERT(isDoubleList(), "Expected DoubleList but got ", tagKind());
  return c10::List<double>(toIntrusivePtr<c10::detail::ListImpl>());
}
inline std::vector<double> IValue::toDoubleVector() const {
  AT_ASSERT(isDoubleList(), "Expected DoubleList but got ", tagKind());
  return createVectorFromList<double>(static_cast<const c10::detail::ListImpl*>(payload.as_intrusive_ptr));
}
inline c10::List<bool> IValue::toBoolList() && {
  AT_ASSERT(isBoolList(), "Expected BoolList but got ", tagKind());
  return c10::List<bool>(moveToIntrusivePtr<c10::detail::ListImpl>());
}
inline c10::List<bool> IValue::toBoolList() const & {
  AT_ASSERT(isBoolList(), "Expected BoolList but got ", tagKind());
  return c10::List<bool>(toIntrusivePtr<c10::detail::ListImpl>());
}
inline c10::List<at::Tensor> IValue::toTensorList() && {
  AT_ASSERT(isTensorList(), "Expected TensorList but got ", tagKind());
  return c10::List<at::Tensor>(moveToIntrusivePtr<c10::detail::ListImpl>());
}
inline c10::List<at::Tensor> IValue::toTensorList() const & {
  AT_ASSERT(isTensorList(), "Expected TensorList but got ", tagKind());
  return c10::List<at::Tensor>(toIntrusivePtr<c10::detail::ListImpl>());
}
inline std::vector<at::Tensor> IValue::toTensorVector() const {
  AT_ASSERT(isTensorList(), "Expected TensorList but got ", tagKind());
  return createVectorFromList<at::Tensor>(static_cast<const c10::detail::ListImpl*>(payload.as_intrusive_ptr));
}
inline c10::List<IValue> IValue::toList() && {
  AT_ASSERT(isList(), "Expected GenericList but got ", tagKind());
  return c10::List<IValue>(moveToIntrusivePtr<c10::detail::ListImpl>());
}
inline c10::List<IValue> IValue::toList() const & {
  AT_ASSERT(isList(), "Expected GenericList but got ", tagKind());
  return c10::List<IValue>(toIntrusivePtr<c10::detail::ListImpl>());
}
inline c10::ArrayRef<IValue> IValue::toListRef() const {
  AT_ASSERT(isList(), "Expected GenericList but got ", tagKind());
  return static_cast<const c10::detail::ListImpl*>(payload.as_intrusive_ptr)->list;
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
template <
    typename... Args,
    std::enable_if_t<
        !guts::disjunction<
            std::is_lvalue_reference<Args>...,
            guts::negation<std::is_constructible<IValue, Args>>...>::value,
        std::nullptr_t>>
inline IValue::IValue(const std::tuple<Args...>& t)
    : IValue(
          std::move(c10::guts::apply(c10::ivalue::Tuple::create<Args...>, t))) {
}

inline IValue::IValue(c10::intrusive_ptr<ivalue::ConstantString> v)
: tag(Tag::String), is_intrusive_ptr(true) {
  payload.as_intrusive_ptr = v.release();
}
inline IValue::IValue(std::string v)
: IValue(ivalue::ConstantString::create(std::move(v))) {}


inline IValue::IValue(c10::impl::GenericList v)
: tag(Tag::GenericList), is_intrusive_ptr(true) {
  payload.as_intrusive_ptr = v.impl_.release();
}

template<class T> inline IValue::IValue(c10::List<T> v)
: IValue(impl::toList<T>(std::move(v))) {}
template<class T> inline IValue::IValue(at::ArrayRef<T> v)
: IValue(c10::List<T>()) {
  auto list = to<c10::List<T>>();
  list.reserve(v.size());
  for (const auto& e : v) {
    list.push_back(e);
  }
}
template<class T> inline IValue::IValue(const std::vector<T>& v)
: IValue(c10::List<T>()) {
  auto list = to<c10::List<T>>();
  list.reserve(v.size());
  for (const auto& e : v) {
    list.push_back(e);
  }
}
template<class T, size_t N> inline IValue::IValue(std::array<T, N> v)
: IValue(c10::List<T>()) {
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
inline IValue::IValue(c10::intrusive_ptr<ivalue::PyObjectHolder> v)
: tag(Tag::PyObject), is_intrusive_ptr(true) {
  payload.as_intrusive_ptr = v.release();
}
inline IValue::IValue(c10::intrusive_ptr<torch::jit::CustomClassHolder> v)
: tag(Tag::Capsule), is_intrusive_ptr(true) {
  payload.as_intrusive_ptr = v.release();
}
inline IValue::IValue(c10::intrusive_ptr<ivalue::Future> v)
: tag(Tag::Future), is_intrusive_ptr(true) {
  payload.as_intrusive_ptr = v.release();
}

inline IValue::IValue(c10::intrusive_ptr<c10::RRefInterface> v)
: tag(Tag::RRef), is_intrusive_ptr(true) {
  payload.as_intrusive_ptr = v.release();
}
inline const std::string& IValue::toStringRef() const {
  return toString()->string();
}

inline PyObject* IValue::toPyObject() const {
  return toPyObjectHolder()->getPyObject();
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
  // 1. Immutable primitive values of the same type (Int, Double, None, Bool, Str) return value equality
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
  } else if (this->isInt() && rhs.isInt()) {
    return this->toInt() == rhs.toInt();
  } else if (this->isDouble() && rhs.isDouble()) {
    return this->toDouble() == rhs.toDouble();
  } else if (this->isString() && rhs.isString()) {
    return this->toStringRef() == rhs.toStringRef();
  } else {
    // for objects holding in IValue, do shallow compare on pointer address to testify the identity
    return this->is_intrusive_ptr && rhs.is_intrusive_ptr
        && this->payload.as_intrusive_ptr == rhs.payload.as_intrusive_ptr;
  }
}

namespace ivalue {
namespace detail {
// This code allows us to template on a function based on whether IValue has a
// constructor for it. Specifically, has_constructor<T>{} inherits from std::true_type if
// IValue(T) compiles, and inherits from std::false_type if IValue(T) doesn't.
// We use it for calling the IValue constructor for `from` if it exists, and otherwise
// attempt to use our custom class code.
template<class> struct type_sink { typedef void type; };
template<class T> using type_sink_t = typename type_sink<T>::type;
template<class T, class=void> struct has_constructor : std::false_type {}; \
template<class T> struct has_constructor<
  T,
  type_sink_t< decltype( IValue(std::declval<T>())) >
>: std::true_type {};

template <typename T>
IValue from_(T x, std::true_type) {
  return IValue(std::move(x));
}
template <typename T>
IValue from_(c10::intrusive_ptr<T> x, std::false_type) {
  using inputType = c10::intrusive_ptr<T>;
  if (!isCustomClassRegistered<inputType>()) {
    throw c10::Error("Trying to return a class that we don't support and isn't a registered custom class.", "");
  }
  auto res = getCustomClassType<inputType>();
  auto retObject = ivalue::Object::create(
    StrongTypePtr(
      std::shared_ptr<torch::jit::script::CompilationUnit>(),
      std::move(res)),
    1);
  auto objPtr = c10::static_intrusive_pointer_cast<torch::jit::CustomClassHolder>(std::move(x));

  retObject->setSlot(0, IValue(std::move(objPtr)));
  auto resIVal = IValue(std::move(retObject));
  return resIVal;
}
template <typename T>
IValue from_(T x, std::false_type) {
  static_assert(guts::false_t<T>::value, "You are calling from with a type that it doesn't support, and isn't a potential custom class (ie: is an intrusive_ptr)");
  return IValue();
}
}

template <typename T>
IValue from(T x) {
  return detail::from_(std::move(x), detail::has_constructor<T>{});
}

}
} // namespace c10
