#pragma once

#include <ATen/ATen.h>
/**
 * WARNING: EValue is a class used by Executorch, for its boxed operators. It
 * contains similar logic as `IValue` in PyTorch, by providing APIs to convert
 * boxed values to unboxed values.
 *
 * It's mirroring a fbcode internal source file
 * [`EValue.h`](https://www.internalfb.com/code/fbsource/xplat/executorch/core/values/Evalue.h).
 *
 * The reason why we are mirroring this class, is to make sure we have CI job
 * coverage on torchgen logic, given that torchgen is used for both Executorch
 * and PyTorch.
 *
 * If any of the logic here needs to be changed, please update fbcode version of
 * `Evalue.h` as well. These two versions will be merged as soon as Executorch
 * is in OSS (hopefully by Q2 2023).
 */
namespace torch {
namespace executor {

#define ET_CHECK_MSG TORCH_CHECK_MSG
#define EXECUTORCH_FORALL_TAGS(_) \
  _(None)                         \
  _(Tensor)                       \
  _(String)                       \
  _(Double)                       \
  _(Int)                          \
  _(Bool)                         \
  _(ListBool)                     \
  _(ListDouble)                   \
  _(ListInt)                      \
  _(ListTensor)                   \
  _(ListScalar)                   \
  _(ListOptionalTensor)

enum class Tag : uint32_t {
#define DEFINE_TAG(x) x,
  EXECUTORCH_FORALL_TAGS(DEFINE_TAG)
#undef DEFINE_TAG
};

struct EValue;

template <typename T>
struct evalue_to_const_ref_overload_return {
  using type = T;
};

template <>
struct evalue_to_const_ref_overload_return<at::Tensor> {
  using type = const at::Tensor&;
};

template <typename T>
struct evalue_to_ref_overload_return {
  using type = T;
};

template <>
struct evalue_to_ref_overload_return<at::Tensor> {
  using type = at::Tensor&;
};

/*
 * Helper class used to correlate EValues in the executor table, with the
 * unwrapped list of the proper type. Because values in the runtime's values
 * table can change during execution, we cannot statically allocate list of
 * objects at deserialization. Imagine the serialized list says index 0 in the
 * value table is element 2 in the list, but during execution the value in
 * element 2 changes (in the case of tensor this means the TensorImpl* stored in
 * the tensor changes). To solve this instead they must be created dynamically
 * whenever they are used.
 */
template <typename T>
class EValObjectList {
 public:
  EValObjectList() = default;
  /*
   * Wrapped_vals is a list of pointers into the values table of the runtime
   * whose destinations correlate with the elements of the list, unwrapped_vals
   * is a container of the same size whose serves as memory to construct the
   * unwrapped vals.
   */
  EValObjectList(EValue** wrapped_vals, T* unwrapped_vals, int size)
      : wrapped_vals_(wrapped_vals, size), unwrapped_vals_(unwrapped_vals) {}
  /*
   * Constructs and returns the list of T specified by the EValue pointers
   */
  at::ArrayRef<T> get() const;

 private:
  // Source of truth for the list
  at::ArrayRef<EValue*> wrapped_vals_;
  // Same size as wrapped_vals
  mutable T* unwrapped_vals_;
};

// Aggregate typing system similar to IValue only slimmed down with less
// functionality, no dependencies on atomic, and fewer supported types to better
// suit embedded systems (ie no intrusive ptr)
struct EValue {
  union Payload {
    // When in ATen mode at::Tensor is not trivially copyable, this nested union
    // lets us handle tensor as a special case while leaving the rest of the
    // fields in a simple state instead of requiring a switch on tag everywhere.
    union TriviallyCopyablePayload {
      TriviallyCopyablePayload() : as_int(0) {}
      // Scalar supported through these 3 types
      int64_t as_int;
      double as_double;
      bool as_bool;
      // TODO(jakeszwe): convert back to pointers to optimize size of this
      // struct
      at::ArrayRef<char> as_string;
      at::ArrayRef<int64_t> as_int_list;
      at::ArrayRef<double> as_double_list;
      at::ArrayRef<bool> as_bool_list;
      EValObjectList<at::Tensor> as_tensor_list;
      EValObjectList<std::optional<at::Tensor>> as_list_optional_tensor;
    } copyable_union;

    // Since a Tensor just holds a TensorImpl*, there's no value to use Tensor*
    // here.
    at::Tensor as_tensor;

    Payload() {}
    ~Payload() {}
  };

  // Data storage and type tag
  Payload payload;
  Tag tag;

  // Basic ctors and assignments
  EValue(const EValue& rhs) : EValue(rhs.payload, rhs.tag) {}

  EValue(EValue&& rhs) noexcept : tag(rhs.tag) {
    moveFrom(std::move(rhs));
  }

  EValue& operator=(EValue&& rhs) & noexcept {
    if (&rhs == this) {
      return *this;
    }

    destroy();
    moveFrom(std::move(rhs));
    return *this;
  }

  EValue& operator=(EValue const& rhs) & {
    // Define copy assignment through copy ctor and move assignment
    *this = EValue(rhs);
    return *this;
  }

  ~EValue() {
    destroy();
  }

  /****** None Type ******/
  EValue() : tag(Tag::None) {
    payload.copyable_union.as_int = 0;
  }

  bool isNone() const {
    return tag == Tag::None;
  }

  /****** Int Type ******/
  /*implicit*/ EValue(int64_t i) : tag(Tag::Int) {
    payload.copyable_union.as_int = i;
  }

  bool isInt() const {
    return tag == Tag::Int;
  }

  int64_t toInt() const {
    ET_CHECK_MSG(isInt(), "EValue is not an int.");
    return payload.copyable_union.as_int;
  }

  /****** Double Type ******/
  /*implicit*/ EValue(double d) : tag(Tag::Double) {
    payload.copyable_union.as_double = d;
  }

  bool isDouble() const {
    return tag == Tag::Double;
  }

  double toDouble() const {
    ET_CHECK_MSG(isDouble(), "EValue is not a Double.");
    return payload.copyable_union.as_double;
  }

  /****** Bool Type ******/
  /*implicit*/ EValue(bool b) : tag(Tag::Bool) {
    payload.copyable_union.as_bool = b;
  }

  bool isBool() const {
    return tag == Tag::Bool;
  }

  bool toBool() const {
    ET_CHECK_MSG(isBool(), "EValue is not a Bool.");
    return payload.copyable_union.as_bool;
  }

  /****** Scalar Type ******/
  /// Construct an EValue using the implicit value of a Scalar.
  /*implicit*/ EValue(at::Scalar s) {
    if (s.isIntegral(false)) {
      tag = Tag::Int;
      payload.copyable_union.as_int = s.to<int64_t>();
    } else if (s.isFloatingPoint()) {
      tag = Tag::Double;
      payload.copyable_union.as_double = s.to<double>();
    } else if (s.isBoolean()) {
      tag = Tag::Bool;
      payload.copyable_union.as_bool = s.to<bool>();
    } else {
      ET_CHECK_MSG(false, "Scalar passed to EValue is not initialized.");
    }
  }

  bool isScalar() const {
    return tag == Tag::Int || tag == Tag::Double || tag == Tag::Bool;
  }

  at::Scalar toScalar() const {
    // Convert from implicit value to Scalar using implicit constructors.

    if (isDouble()) {
      return toDouble();
    } else if (isInt()) {
      return toInt();
    } else if (isBool()) {
      return toBool();
    } else {
      ET_CHECK_MSG(false, "EValue is not a Scalar.");
      return c10::Scalar();
    }
  }

  /****** Tensor Type ******/
  /*implicit*/ EValue(at::Tensor t) : tag(Tag::Tensor) {
    // When built in aten mode, at::Tensor has a non trivial constructor
    // destructor, so regular assignment to a union field is UB. Instead we must
    // go through placement new (which causes a refcount bump).
    new (&payload.as_tensor) at::Tensor(t);
  }

  bool isTensor() const {
    return tag == Tag::Tensor;
  }

  at::Tensor toTensor() && {
    ET_CHECK_MSG(isTensor(), "EValue is not a Tensor.");
    return std::move(payload.as_tensor);
  }

  at::Tensor& toTensor() & {
    ET_CHECK_MSG(isTensor(), "EValue is not a Tensor.");
    return payload.as_tensor;
  }

  const at::Tensor& toTensor() const& {
    ET_CHECK_MSG(isTensor(), "EValue is not a Tensor.");
    return payload.as_tensor;
  }

  /****** String Type ******/
  /*implicit*/ EValue(const char* s, size_t size) : tag(Tag::String) {
    payload.copyable_union.as_string = at::ArrayRef<char>(s, size);
  }

  bool isString() const {
    return tag == Tag::String;
  }

  at::string_view toString() const {
    ET_CHECK_MSG(isString(), "EValue is not a String.");
    return at::string_view(
        payload.copyable_union.as_string.data(),
        payload.copyable_union.as_string.size());
  }

  /****** Int List Type ******/
  /*implicit*/ EValue(at::ArrayRef<int64_t> i) : tag(Tag::ListInt) {
    payload.copyable_union.as_int_list = i;
  }

  bool isIntList() const {
    return tag == Tag::ListInt;
  }

  at::ArrayRef<int64_t> toIntList() const {
    ET_CHECK_MSG(isIntList(), "EValue is not an Int List.");
    return payload.copyable_union.as_int_list;
  }

  /****** Bool List Type ******/
  /*implicit*/ EValue(at::ArrayRef<bool> b) : tag(Tag::ListBool) {
    payload.copyable_union.as_bool_list = b;
  }

  bool isBoolList() const {
    return tag == Tag::ListBool;
  }

  at::ArrayRef<bool> toBoolList() const {
    ET_CHECK_MSG(isBoolList(), "EValue is not a Bool List.");
    return payload.copyable_union.as_bool_list;
  }

  /****** Double List Type ******/
  /*implicit*/ EValue(at::ArrayRef<double> d) : tag(Tag::ListDouble) {
    payload.copyable_union.as_double_list = d;
  }

  bool isDoubleList() const {
    return tag == Tag::ListDouble;
  }

  at::ArrayRef<double> toDoubleList() const {
    ET_CHECK_MSG(isDoubleList(), "EValue is not a Double List.");
    return payload.copyable_union.as_double_list;
  }

  /****** Tensor List Type ******/
  /*implicit*/ EValue(EValObjectList<at::Tensor> t) : tag(Tag::ListTensor) {
    payload.copyable_union.as_tensor_list = t;
  }

  bool isTensorList() const {
    return tag == Tag::ListTensor;
  }

  at::ArrayRef<at::Tensor> toTensorList() const {
    ET_CHECK_MSG(isTensorList(), "EValue is not a Tensor List.");
    return payload.copyable_union.as_tensor_list.get();
  }

  /****** List Optional Tensor Type ******/
  /*implicit*/ EValue(EValObjectList<std::optional<at::Tensor>> t)
      : tag(Tag::ListOptionalTensor) {
    payload.copyable_union.as_list_optional_tensor = t;
  }

  bool isListOptionalTensor() const {
    return tag == Tag::ListOptionalTensor;
  }

  at::ArrayRef<std::optional<at::Tensor>> toListOptionalTensor() {
    return payload.copyable_union.as_list_optional_tensor.get();
  }

  /****** ScalarType Type ******/
  at::ScalarType toScalarType() const {
    ET_CHECK_MSG(isInt(), "EValue is not a ScalarType.");
    return static_cast<at::ScalarType>(payload.copyable_union.as_int);
  }

  /****** MemoryFormat Type ******/
  at::MemoryFormat toMemoryFormat() const {
    ET_CHECK_MSG(isInt(), "EValue is not a MemoryFormat.");
    return static_cast<at::MemoryFormat>(payload.copyable_union.as_int);
  }

  template <typename T>
  T to() &&;

  template <typename T>
  typename evalue_to_ref_overload_return<T>::type to() &;

  /**
   * Converts the EValue to an optional object that can represent both T and
   * an uninitialized state.
   */
  template <typename T>
  inline std::optional<T> toOptional() {
    if (this->isNone()) {
      return std::nullopt;
    }
    return this->to<T>();
  }

 private:
  // Pre cond: the payload value has had its destructor called
  void clearToNone() noexcept {
    payload.copyable_union.as_int = 0;
    tag = Tag::None;
  }

  // Shared move logic
  void moveFrom(EValue&& rhs) noexcept {
    if (rhs.isTensor()) {
      new (&payload.as_tensor) at::Tensor(std::move(rhs.payload.as_tensor));
      rhs.payload.as_tensor.~Tensor();
    } else {
      payload.copyable_union = rhs.payload.copyable_union;
    }
    tag = rhs.tag;
    rhs.clearToNone();
  }

  // Destructs stored tensor if there is one
  void destroy() {
    // Necessary for ATen tensor to refcount decrement the intrusive_ptr to
    // tensorimpl that got a refcount increment when we placed it in the evalue,
    // no-op if executorch tensor #ifdef could have a
    // minor performance bump for a code maintainability hit
    if (isTensor()) {
      payload.as_tensor.~Tensor();
    } else if (isTensorList()) {
      for (auto& tensor : toTensorList()) {
        tensor.~Tensor();
      }
    } else if (isListOptionalTensor()) {
      for (auto& optional_tensor : toListOptionalTensor()) {
        optional_tensor.~optional();
      }
    }
  }

  EValue(const Payload& p, Tag t) : tag(t) {
    if (isTensor()) {
      new (&payload.as_tensor) at::Tensor(p.as_tensor);
    } else {
      payload.copyable_union = p.copyable_union;
    }
  }
};

#define EVALUE_DEFINE_TO(T, method_name)                           \
  template <>                                                      \
  inline evalue_to_ref_overload_return<T>::type EValue::to<T>()& { \
    return static_cast<T>(this->method_name());                    \
  }

template <>
inline at::Tensor& EValue::to<at::Tensor>() & {
  return this->toTensor();
}

EVALUE_DEFINE_TO(at::Scalar, toScalar)
EVALUE_DEFINE_TO(int64_t, toInt)
EVALUE_DEFINE_TO(bool, toBool)
EVALUE_DEFINE_TO(double, toDouble)
EVALUE_DEFINE_TO(at::string_view, toString)
EVALUE_DEFINE_TO(at::ScalarType, toScalarType)
EVALUE_DEFINE_TO(at::MemoryFormat, toMemoryFormat)
EVALUE_DEFINE_TO(std::optional<at::Tensor>, toOptional<at::Tensor>)
EVALUE_DEFINE_TO(at::ArrayRef<int64_t>, toIntList)
EVALUE_DEFINE_TO(
    std::optional<at::ArrayRef<int64_t>>,
    toOptional<at::ArrayRef<int64_t>>)
EVALUE_DEFINE_TO(
    std::optional<at::ArrayRef<double>>,
    toOptional<at::ArrayRef<double>>)
EVALUE_DEFINE_TO(at::ArrayRef<std::optional<at::Tensor>>, toListOptionalTensor)
EVALUE_DEFINE_TO(at::ArrayRef<double>, toDoubleList)
#undef EVALUE_DEFINE_TO

template <typename T>
at::ArrayRef<T> EValObjectList<T>::get() const {
  for (size_t i = 0; i < wrapped_vals_.size(); i++) {
    unwrapped_vals_[i] = wrapped_vals_[i]->template to<T>();
  }
  return at::ArrayRef<T>{unwrapped_vals_, wrapped_vals_.size()};
}

} // namespace executor
} // namespace torch
