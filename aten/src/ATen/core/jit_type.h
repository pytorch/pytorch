#pragma once

#include <ATen/core/TensorBody.h>
#include <ATen/core/functional.h>
#include <ATen/core/interned_strings.h>
#include <ATen/core/ivalue.h>
#include <ATen/core/qualified_name.h>
#include <c10/util/TypeList.h>

#include <c10/util/Optional.h>

#include <iostream>
#include <memory>
#include <type_traits>
#include <array>

struct ClassType;
namespace torch {
namespace jit {
struct Function;
namespace script {
struct CompilationUnit;
}
} // namespace jit
} // namespace torch

namespace c10 {

struct FunctionSchema;
using OptNameList = c10::optional<std::vector<std::string>>;

#define C10_FORALL_TYPES(_) \
  _(AnyType)                \
  _(TensorType)             \
  _(TupleType)              \
  _(ListType)               \
  _(DictType)               \
  _(NumberType)             \
  _(FloatType)              \
  _(FutureType)             \
  _(IntType)                \
  _(NoneType)               \
  _(StringType)             \
  _(GeneratorType)          \
  _(BoolType)               \
  _(OptionalType)           \
  _(VarType)                \
  _(DeviceObjType)          \
  _(FunctionType)           \
  _(ClassType)              \
  _(CapsuleType)            \
  _(InterfaceType)

enum class TypeKind {
#define DEFINE_TYPE(T) T,
  C10_FORALL_TYPES(DEFINE_TYPE)
#undef DEFINE_TYPE
};

CAFFE2_API const char* typeKindToString(TypeKind kind);

struct Type;
using TypePtr = std::shared_ptr<Type>;

struct CAFFE2_API Type : std::enable_shared_from_this<Type> {
 private:
  TypeKind kind_;

 protected:
  Type(TypeKind kind) : kind_(kind) {}

 public:
  virtual bool operator==(const Type& rhs) const = 0;

  // subtyping relation. By default, we return true for the case
  // when the type is exactly equal or if this <: T where rhs = Optional[T]

  // if this returns false and the why_not stream is non-null, it contains
  // additional details that describe why this is not a subtype of 'rhs'.
  // This additional information should only contain details that are not obvious
  // from the python_str() that describes the type. For instance it is clear that `int <: str` is false
  // but not clear why `Foo <: InterfaceBar` might be false.
  virtual bool isSubtypeOfExt(const TypePtr rhs, std::ostream* why_not) const;
  bool isSubtypeOf(const TypePtr rhs) const {
    return isSubtypeOfExt(rhs, nullptr);
  }

  // How this type will appear in FunctionSchema declarations
  virtual std::string str() const = 0;

  // How this type will appear as if it were a type annotation in Python
  // which is sometimes different than how it appears in declarations (e.g.
  // int[] vs List[int])
  virtual std::string python_str() const {
    return str();
  }

  TypeKind kind() const {
    return kind_;
  }

  virtual bool requires_grad() const {
    for (const auto& ct : containedTypes()) {
      if (ct->requires_grad()) {
        return true;
      }
    }
    return false;
  }

  // Dynamically cast this object to the subclass indicated by the
  // template variable, returning nullptr if the cast is invalid.
  template <typename T>
  std::shared_ptr<T> cast() {
    if (T::Kind == kind()) {
      return std::static_pointer_cast<T>(shared_from_this());
    }
    return nullptr;
  }
  template <typename T>
  std::shared_ptr<const T> cast() const {
    if (T::Kind == kind()) {
      return std::static_pointer_cast<const T>(shared_from_this());
    }
    return nullptr;
  }
  template <typename T>
  std::shared_ptr<T> expect() {
    auto r = cast<T>();
    AT_ASSERT(r);
    return r;
  }
  template <typename T>
  std::shared_ptr<const T> expect() const {
    auto r = cast<const T>();
    AT_ASSERT(r);
    return r;
  }
  virtual ~Type() = default;
  virtual bool hasFreeVariables() const {
    return false;
  }
  // list of types this type contains, e.g. for a List then element type of a
  // list for a tuple, the types of the tuple elements
  virtual at::ArrayRef<TypePtr> containedTypes() const {
    return {};
  }
  // create a new version of this type, replacing its contained types with
  // contained_types
  TypePtr withContained(std::vector<TypePtr> contained_types) {
    auto current_contained = containedTypes();
    AT_ASSERT(current_contained.size() == contained_types.size());
    if (current_contained.equals(contained_types)) {
      return shared_from_this();
    }
    return createWithContained(std::move(contained_types));
  }
  // per-type constructor, you only need to override this if the
  // containedTypes() is not empty
  virtual TypePtr createWithContained(
      std::vector<TypePtr> contained_types) const {
    AT_ERROR(
        "type with contained types did not overload createWithContained: ",
        str());
  }
};

struct AnyType;
using AnyTypePtr = std::shared_ptr<AnyType>;
// Any is the top of the type hierarchy, all other types are subtypes
// T <: Any, forall T
struct CAFFE2_API AnyType : public Type {
  static AnyTypePtr create() {
    return AnyTypePtr(
        new AnyType()); // NOLINT(modernize-make-shared)
  }
  bool operator==(const Type& rhs) const override {
    return rhs.kind() == kind();
  }
  std::string str() const override {
    return "Any";
  }
  static const TypeKind Kind = TypeKind::AnyType;
  // global singleton
  static AnyTypePtr get();

 private:
  AnyType() : Type(TypeKind::AnyType) {}
};

inline std::string toString(TypePtr typePtr) {
  return typePtr->str();
}

inline bool operator!=(const Type& lhs, const Type& rhs) {
  return !(lhs == rhs);
}

// common base for all types that have a single sub element
// e.g. Future[T], Option[T], List[T]
template <TypeKind K, typename T>
struct SingleElementType : public Type {
  static const TypeKind Kind = K;

  TypePtr getElementType() const {
    return elem;
  }

  bool hasFreeVariables() const override {
    return getElementType()->hasFreeVariables();
  }

  at::ArrayRef<TypePtr> containedTypes() const override {
    return elem;
  }

  bool operator==(const Type& rhs) const override {
    if (auto rhs_ = rhs.cast<T>()) {
      return *getElementType() == *rhs_->getElementType();
    }
    return false;
  }

 protected:
  SingleElementType(TypePtr elem) : Type(Kind), elem(std::move(elem)) {}

 private:
  TypePtr elem;
};

struct OptionalType;
using OptionalTypePtr = std::shared_ptr<OptionalType>;
// This type represents an optional type, for each element type.
// Optional[T] can accept both T and None(nullopt in C++)
// Subtype hierarchy for Optional:
// 1. Optional[T] <: Optional[R] iff T <: R
// 2. T <: Optional[R] if T <: R
// 3. None <: Optional[T] for all T
struct CAFFE2_API OptionalType
    : public SingleElementType<TypeKind::OptionalType, OptionalType> {
  static OptionalTypePtr create(TypePtr element) {
    // Optional is a union of [None, T], so Optional[[Optional[T]]] ->
    // Optional[T]
    if (auto opt_ptr = element->cast<OptionalType>()) {
      return opt_ptr;
    }
    return OptionalTypePtr(
        new OptionalType(std::move(element))); // NOLINT(modernize-make-shared)
  }

  std::string str() const override {
    std::stringstream ss;
    ss << getElementType()->str() << "?";
    return ss.str();
  }
  std::string python_str() const override {
    std::stringstream ss;
    ss << "Optional[" << getElementType()->python_str() << "]";
    return ss.str();
  }

  TypePtr createWithContained(
      std::vector<TypePtr> contained_types) const override {
    AT_ASSERT(contained_types.size() == 1);
    return create(contained_types[0]);
  }

  bool isSubtypeOfExt(const TypePtr rhs, std::ostream* why_not) const override {
    if (Type::isSubtypeOfExt(rhs, why_not)) {
      return true;
    }
    if (auto rhs_ = rhs->cast<OptionalType>()) {
      return getElementType()->isSubtypeOfExt(rhs_->getElementType(), why_not);
    }
    return false;
  }
  // common cast Optional[Tensor] for undefined tensor type
  static OptionalTypePtr ofTensor();

 private:
  OptionalType(TypePtr elem) : SingleElementType(elem) {}
};

template <typename T>
inline c10::optional<T> merge_primitive(
    const c10::optional<T>& a,
    const c10::optional<T>& b) {
  if (a.has_value() && b.has_value() && a.value() == b.value()) {
    return a;
  }
  return c10::optional<T>{};
}

// `VaryingShape` tracks if individual dimensions or a rank vary across
// profiled runs. A *varying* or *dynamic* dimension is expressed as
// an empty c10::optional in `sizes_`. If a rank is dynamic, the entire
// `sizes_` becomes the empty optional.
struct CAFFE2_API VaryingShape {
  using ListOfOptionalInts = std::vector<c10::optional<int64_t>>;
  VaryingShape(const std::vector<int64_t>& vec)
      : VaryingShape(ListOfOptionalInts(vec.begin(), vec.end())) {}

  VaryingShape(c10::ArrayRef<int64_t> vec)
      : VaryingShape(ListOfOptionalInts(vec.begin(), vec.end())){}

  VaryingShape(c10::optional<size_t> size = c10::nullopt) : dims_(c10::nullopt) {
    if (size) {
      dims_ = ListOfOptionalInts(*size);
    }
  }

  VaryingShape(ListOfOptionalInts dims)
  : dims_(std::move(dims)) {}

  VaryingShape(size_t size) : VaryingShape(c10::optional<size_t>(size)) {}

  bool operator==(const VaryingShape& other) const {
    return dims_ == other.dims_;
  }

  const c10::optional<int64_t>& operator[](int i) const {
    if (!dims_) {
      throw std::runtime_error("Rank isn't fixed");
    }
    return (*dims_).at(i);
  }

  c10::optional<size_t> size() const {
    if (!dims_) {
      return c10::nullopt;
    }
    const auto& dims = dims_.value();
    return dims.size();
  }

  const c10::optional<ListOfOptionalInts>& sizes() const {
    return dims_;
  }

  VaryingShape merge(const VaryingShape& other) const;

  c10::optional<std::vector<int64_t>> concrete_sizes() const {
    if (!dims_) {
      return c10::nullopt;
    }
    std::vector<int64_t> sizes;
    for (auto d : *dims_) {
      if (!d) {
        return c10::nullopt;
      }
      sizes.push_back(d.value());
    }
    return sizes;
  }

  bool isComplete() const {
    if (!dims_) {
      return false;
    }
    for (auto d : *dims_) {
      if(!d) {
        return false;
      }
    }
    return true;
  }

 private:
  c10::optional<ListOfOptionalInts> dims_;
};

using VaryingStrides = VaryingShape;

struct TensorType;
using TensorTypePtr = std::shared_ptr<TensorType>;
// This type represents a single Tensor with a specific size
struct CAFFE2_API TensorType : public Type {
  static TensorTypePtr create(const at::Tensor& t) {
    return TensorTypePtr(new TensorType(t));
  }

  static TensorTypePtr create(
      c10::optional<at::ScalarType> scalar_type,
      c10::optional<Device> device,
      const VaryingShape& sizes,
      const VaryingStrides& strides,
      c10::optional<bool> requires_grad,
      c10::optional<bool> autograd_zero=c10::nullopt) {
    return TensorTypePtr(new TensorType(
        scalar_type, device, sizes, strides, requires_grad));
  }

  static TensorTypePtr create(
      c10::optional<at::ScalarType> scalar_type,
      c10::optional<Device> device,
      c10::optional<size_t> dim,
      c10::optional<bool> requires_grad) {
    return TensorType::create(
        scalar_type,
        device,
        VaryingShape(dim),
        VaryingShape(dim),
        requires_grad);
  }

  // overloaded create variadic template argument as it could not distinguish
  // initializer list
  static TensorTypePtr createContiguous(
      at::ScalarType scalar_type,
      at::Device device,
      at::IntArrayRef sizes) {
    return create(
        scalar_type,
        device,
        VaryingShape(sizes),
        VaryingShape(contiguousStridesOf(sizes)),
        c10::nullopt);
  }
  static TensorTypePtr create(
      at::ScalarType scalar_type,
      at::Device device,
      at::IntArrayRef sizes,
      at::IntArrayRef strides) {
    return create(
        scalar_type,
        device,
        VaryingShape(sizes),
        c10::VaryingShape(strides),
        c10::nullopt);
  }
  static TypePtr fromNumberType(TypePtr typ);
  static TypePtr fromBoolType();

  c10::optional<size_t> dim() const {
    return sizes().size();
  }

  const VaryingShape& sizes() const {
    return sizes_;
  }
  const VaryingStrides& strides() const {
    return strides_;
  }
  c10::optional<at::Device> device() const {
    return device_;
  }
  c10::optional<at::ScalarType> scalarType() const {
    return scalar_type_;
  }
  c10::optional<bool> requiresGrad() const {
    return requires_grad_;
  }
  bool requires_grad() const override {
    return requires_grad_ ? *requires_grad_ : true;
  }


  bool operator==(const Type& rhs) const override {
    if (rhs.kind() != kind()) {
      return false;
    }

    auto rt = rhs.expect<TensorType>();
    return scalar_type_ == rt->scalarType() && sizes() == rt->sizes() &&
        strides() == rt->strides() && device() == rt->device() &&
        requiresGrad() == rt->requiresGrad() && autogradZero() == rt->autogradZero();
  }
  bool isSubtypeOfExt(const TypePtr rhs, std::ostream* why_not) const override;

  std::string str() const override;

  c10::optional<size_t> numel() const {
    size_t prod = 1;
    const auto& shape = sizes();

    for (size_t i = 0; i < shape.size(); i++) {
      if (!shape[i]) {
        return c10::optional<size_t>{};
      }
      prod *= shape[i].value();
    }
    return prod;
  }

  TensorTypePtr withRequiresGrad(c10::optional<bool> s) {
    auto copy = clone();
    copy->requires_grad_ = s;
    return copy;
  }

  TensorTypePtr withScalarType(c10::optional<ScalarType> st) {
    auto copy = clone();
    copy->scalar_type_ = st;
    return copy;
  }


  TensorTypePtr withDim(c10::optional<size_t> d) {
    auto copy = clone();
    copy->sizes_ = VaryingShape(d);
    copy->strides_ = VaryingShape(d);
    return copy;
  }

  TensorTypePtr withSizesStrides(
      at::IntArrayRef sizes,
      at::IntArrayRef strides) const {
    auto cloned = clone();
    cloned->sizes_ = VaryingShape(sizes);
    cloned->strides_ = VaryingStrides(strides);
    return cloned;
  }

  TensorTypePtr withSizes(at::IntArrayRef sizes) const {
    return withSizesStrides(
        sizes, contiguousStridesOf(sizes));
  }

  TensorTypePtr dimensionedOnly() const {
    auto copy = clone();
    copy->sizes_ = VaryingShape(sizes().size());
    copy->strides_ = VaryingShape(sizes().size());
    return copy;
  }

  TensorTypePtr contiguous() const {
    auto cloned = clone();
    if (auto concrete_sizes = sizes().concrete_sizes()) {
      cloned->strides_ = VaryingShape(contiguousStridesOf(*concrete_sizes));
    } else  {
      cloned->strides_ = VaryingShape(sizes().size());
    }
    return cloned;
  }

  TensorTypePtr merge(TensorTypePtr other) const {
    auto scalar_type = merge_primitive(scalarType(), other->scalarType());
    auto dev = merge_primitive(device(), other->device());
    auto sz = sizes().merge(other->sizes());
    auto srs = strides().merge(other->strides());
    auto gr = merge_primitive(requiresGrad(), other->requiresGrad());
    auto zero = merge_primitive(autogradZero(), other->autogradZero());
    return TensorType::create(scalar_type, dev, sz, srs, gr, zero);
  }
  // is all information about the type specified except for autograd?
  // This replaces the notion of a 'CompleteTensorType' that used to exist
  // in the type-hierarchy. Excluding require_grad and autogradZero allows
  // this to match the old behavior.
  bool isComplete() const {
    return scalar_type_ && device_ && sizes_.isComplete() && strides_.isComplete();
  }

  TensorTypePtr withAutogradZero() {
    auto r = clone();
    r->autograd_zero_ = true;
    return r;
  }

  c10::optional<bool> autogradZero() const {
    return autograd_zero_;
  }

  static TensorTypePtr get();

  static const TypeKind Kind = TypeKind::TensorType;

 private:
  TensorType(const at::Tensor& tensor)
      : Type(TypeKind::TensorType),
        scalar_type_(tensor.scalar_type()),
        device_(tensor.device()),
        sizes_(tensor.sizes().size()),
        strides_(tensor.sizes().size()),
        requires_grad_(tensor.requires_grad()) {
          if (!tensor.is_mkldnn() && !tensor.is_sparse()) {
            sizes_ = tensor.sizes().vec();
            strides_ = tensor.strides().vec();
          }
        }
  TensorType(
      c10::optional<at::ScalarType> scalar_type,
      c10::optional<Device> device,
      const VaryingShape& sizes,
      const VaryingStrides& strides,
      c10::optional<bool> requires_grad,
      c10::optional<bool> autograd_zero=c10::nullopt)
      : Type(TypeKind::TensorType),
        scalar_type_(scalar_type),
        device_(device),
        sizes_(sizes),
        strides_(strides),
        requires_grad_(requires_grad),
        autograd_zero_(autograd_zero) {}

  TensorTypePtr clone() const {
    return TensorTypePtr(new TensorType(
        scalar_type_, device_, sizes_, strides_, requires_grad_, autograd_zero_));
  }

  static std::vector<int64_t> contiguousStridesOf(at::IntArrayRef sizes) {
    std::vector<int64_t> strides(sizes.size());
    if (sizes.empty()) // zero-dim case
      return strides;
    strides.back() = 1;
    for (size_t i = strides.size() - 1; i > 0; i--) {
      strides[i - 1] = strides[i] * sizes[i];
    }
    return strides;
  }

  c10::optional<at::ScalarType> scalar_type_;
  c10::optional<at::Device> device_;
  VaryingShape sizes_;
  VaryingStrides strides_;
  c10::optional<bool> requires_grad_;
  // we exploit the fact certain tensors must be zero in the autograd to
  // optimize gradient computation. If true, this means that this tensor
  // must only contain zeros. Normally this will be nullopt, meaning
  // the tensor may or may not contain only zeros. If false,
  // this means the tensor must have some non-zero elements.
  c10::optional<bool> autograd_zero_;
};

struct ListType;
using ListTypePtr = std::shared_ptr<ListType>;
struct CAFFE2_API ListType
    : public SingleElementType<TypeKind::ListType, ListType> {
  // It's not exactly a singleton, but there should be exactly one instance of
  // List[T] for every T
  friend struct Type;
  template <typename... T>
  static ListTypePtr create(T&&... all) {
    return ListTypePtr(
        new ListType(std::forward<T>(all)...)); // NOLINT(modernize-make-shared)
  }

  std::string str() const override {
    std::stringstream ss;
    ss << getElementType()->str() << "[]";
    return ss.str();
  }
  std::string python_str() const override {
    std::stringstream ss;
    ss << "List[" << getElementType()->python_str() << "]";
    return ss.str();
  }
  TypePtr createWithContained(
      std::vector<TypePtr> contained_types) const override {
    return create(contained_types.at(0));
  }
  // common cast List[Tensor]
  static ListTypePtr ofTensors();
  static ListTypePtr ofInts();
  static ListTypePtr ofFloats();
  static ListTypePtr ofBools();

 private:
  ListType(TypePtr elem) : SingleElementType(elem) {}
};

struct DictType;
using DictTypePtr = std::shared_ptr<DictType>;
struct CAFFE2_API DictType : public Type {
  friend struct Type;
  static const TypeKind Kind = TypeKind::DictType;

  static DictTypePtr create(TypePtr key, TypePtr value) {
    switch (key->kind()) {
      case TypeKind::AnyType:
      case TypeKind::IntType:
      case TypeKind::FloatType:
      case TypeKind::StringType:
      case TypeKind::TensorType:
        return DictTypePtr(new DictType(key, value));
      default:
        AT_ERROR(
            "Cannot create dict for key type '",
            key->str(),
            "', only int, float, Tensor and string keys are supported");
    }
  }

  // aligned with the format in FunctionSchema
  std::string str() const override {
    std::stringstream ss;
    ss << "Dict(" << getKeyType()->str() << ", " << getValueType()->str()
       << ")";
    return ss.str();
  }

  std::string python_str() const override {
    std::stringstream ss;
    ss << "Dict[" << getKeyType()->python_str() << ", "
       << getValueType()->python_str() << "]";
    return ss.str();
  }

  TypePtr createWithContained(
      std::vector<TypePtr> contained_types) const override {
    if (contained_types.size() != 2) {
      throw std::runtime_error("Expected 2 contained types");
    }
    return create(contained_types.at(0), contained_types.at(1));
  }

  TypePtr getKeyType() const {
    return types.at(0);
  }

  TypePtr getValueType() const {
    return types.at(1);
  }

  bool hasFreeVariables() const override {
    return has_free_variables;
  }

  at::ArrayRef<TypePtr> containedTypes() const override {
    return types;
  }

  bool operator==(const Type& rhs) const override {
    if (auto dict_rhs = rhs.cast<DictType>()) {
      return *getKeyType() == *(dict_rhs->getKeyType()) &&
          *getValueType() == *(dict_rhs->getValueType());
    }
    return false;
  }

 private:
  DictType(TypePtr key, TypePtr value)
      : Type(TypeKind::DictType),
        types({key, value}),
        has_free_variables(
            key->hasFreeVariables() || value->hasFreeVariables()) {}
  std::vector<TypePtr> types;
  bool has_free_variables;
};

struct FutureType;
using FutureTypePtr = std::shared_ptr<FutureType>;

struct CAFFE2_API FutureType
    : public SingleElementType<TypeKind::FutureType, FutureType> {
  friend struct Type;
  template <typename... T>
  static FutureTypePtr create(TypePtr elem) {
    return FutureTypePtr(
        new FutureType(std::move(elem))); // NOLINT(modernize-make-shared)
  }

  std::string str() const override {
    std::stringstream ss;
    ss << "Future(" << getElementType()->str() << ")";
    return ss.str();
  }
  std::string python_str() const override {
    std::stringstream ss;
    ss << "Future[" << getElementType()->python_str() << "]";
    return ss.str();
  }
  TypePtr createWithContained(
      std::vector<TypePtr> contained_types) const override {
    return create(contained_types.at(0));
  }

 private:
  FutureType(TypePtr elem) : SingleElementType(elem) {}
};

using ::torch::jit::Function;
struct NamedType;
using NamedTypePtr = std::shared_ptr<NamedType>;

struct CAFFE2_API NamedType : public Type {
  NamedType(TypeKind tk, c10::optional<QualifiedName> name)
      : Type(tk), name_(std::move(name)) {}

  // Fully qualified name of type
  // Looks like: "foo.bar.Baz".
  const c10::optional<QualifiedName>& name() const {
    return name_;
  }
private:
  c10::optional<QualifiedName> name_;
};

struct TupleType;
using TupleTypePtr = std::shared_ptr<TupleType>;
using NameList = std::vector<std::string>;
// This type represents a Tuple
struct CAFFE2_API TupleType : public NamedType {
  static std::shared_ptr<FunctionSchema> namedTupleSchemaFromNamesAndTypes(
      c10::QualifiedName,
      std::vector<std::string>,
      std::vector<TypePtr>);

  static TupleTypePtr create(
      std::vector<TypePtr> types,
      c10::optional<c10::QualifiedName> name = c10::nullopt,
      std::shared_ptr<FunctionSchema> schema = nullptr) {
    return TupleTypePtr(new TupleType(
        std::move(types),
        std::move(name),
        std::move(schema))); // NOLINT(modernize-make-shared)
  }

  at::ArrayRef<TypePtr> elements() const {
    return elements_;
  }

  bool operator==(const Type& rhs) const override;
  bool isSubtypeOfExt(const TypePtr rhs_, std::ostream* why_not) const override;

  std::string str() const override;
  std::string python_str() const override;
  bool hasFreeVariables() const override {
    return has_free_variables_;
  }
  at::ArrayRef<TypePtr> containedTypes() const override {
    return elements_;
  }
  TypePtr createWithContained(
      std::vector<TypePtr> contained_types) const override {
    return create(std::move(contained_types));
  }
  const std::shared_ptr<FunctionSchema>& schema() const {
    return schema_;
  }

  static const TypeKind Kind = TypeKind::TupleType;

 private:
  TupleType(
      std::vector<TypePtr> elements_,
      c10::optional<c10::QualifiedName> name,
      std::shared_ptr<FunctionSchema> schema);

  bool compare(
      const Type& rhs,
      std::function<bool(const TypePtr, const TypePtr)> fn) const {
    if (rhs.kind() != kind()) {
      return false;
    }

    const auto& l_elements = elements();
    const auto& r_elements = rhs.cast<TupleType>()->elements();
    if (l_elements.size() != r_elements.size())
      return false;
    for (size_t i = 0; i < l_elements.size(); ++i) {
      if (!fn(l_elements[i], r_elements[i]))
        return false;
    }
    return true;
  }

  std::vector<TypePtr> elements_;
  bool has_free_variables_;
  std::shared_ptr<FunctionSchema> schema_;
};

struct NumberType;
using NumberTypePtr = std::shared_ptr<NumberType>;
// This type represents a Python number
// Subtype hierarchy for Number Types (NumberType as the base type):
// IntType <: NumberType
// FloatType <: NumberType
struct CAFFE2_API NumberType : public Type {
  static NumberTypePtr create() {
    return NumberTypePtr(new NumberType()); // NOLINT(modernize-make-shared)
  }
  bool operator==(const Type& rhs) const override {
    return rhs.kind() == kind();
  }
  std::string str() const override {
    return "Scalar"; // match what PythonArgParser says for clarity
  }
  std::string python_str() const override {
    return "number"; // technically not a valid python type, but
                     // we need to use it when parsing back in annotations
                     // for implicit conversions
  }
  static const TypeKind Kind = TypeKind::NumberType;
  // global singleton
  static NumberTypePtr get();

 protected:
  NumberType(TypeKind kind = TypeKind::NumberType) : Type(kind) {}
};

struct FloatType;
using FloatTypePtr = std::shared_ptr<FloatType>;
// This type represents a Python float number
struct CAFFE2_API FloatType : public NumberType {
  static FloatTypePtr create() {
    return FloatTypePtr(new FloatType()); // NOLINT(modernize-make-shared)
  }
  bool operator==(const Type& rhs) const override {
    return rhs.kind() == kind();
  }
  std::string str() const override {
    return "float";
  }
  std::string python_str() const override {
    return "float";
  }
  bool isSubtypeOfExt(const TypePtr rhs, std::ostream* why_not) const override {
    return rhs->kind() == TypeKind::NumberType || NumberType::isSubtypeOfExt(rhs, why_not);
  }
  static const TypeKind Kind = TypeKind::FloatType;
  // global singleton
  static FloatTypePtr get();

 private:
  FloatType() : NumberType(TypeKind::FloatType) {}
};

struct IntType;
using IntTypePtr = std::shared_ptr<IntType>;
// This type represents a Python int number
struct CAFFE2_API IntType : public NumberType {
  static IntTypePtr create() {
    return IntTypePtr(new IntType()); // NOLINT(modernize-make-shared)
  }
  bool operator==(const Type& rhs) const override {
    return rhs.kind() == kind();
  }
  std::string str() const override {
    return "int";
  }
  std::string python_str() const override {
    return "int";
  }
  bool isSubtypeOfExt(const TypePtr rhs, std::ostream* why_not) const override {
    return rhs->kind() == TypeKind::NumberType || NumberType::isSubtypeOfExt(rhs, why_not);
  }
  static const TypeKind Kind = TypeKind::IntType;
  // global singleton
  static IntTypePtr get();

 private:
  IntType() : NumberType(TypeKind::IntType) {}
};

struct BoolType;
using BoolTypePtr = std::shared_ptr<BoolType>;
// This node represents a Python bool value
struct CAFFE2_API BoolType : public Type {
  static BoolTypePtr create() {
    return BoolTypePtr(new BoolType());
  }
  bool operator==(const Type& rhs) const override {
    return rhs.kind() == kind();
  }
  std::string str() const override {
    return "bool";
  }
  static const TypeKind Kind = TypeKind::BoolType;
  // global singleton
  static BoolTypePtr get();

 private:
  BoolType() : Type(TypeKind::BoolType) {}
};

struct StringType;
using StringTypePtr = std::shared_ptr<StringType>;
// This type represents a Python string
struct CAFFE2_API StringType : public Type {
  static StringTypePtr create() {
    return StringTypePtr(new StringType()); // NOLINT(modernize-make-shared)
  }
  bool operator==(const Type& rhs) const override {
    return rhs.kind() == kind();
  }
  std::string str() const override {
    // we only use "str" (not "string") in both FunctionSchema and script
    return python_str();
  }
  std::string python_str() const override {
    return "str";
  }
  static const TypeKind Kind = TypeKind::StringType;
  // global singleton
  static StringTypePtr get();

 private:
  StringType() : Type(TypeKind::StringType) {}
};

struct FunctionType;
using FunctionTypePtr = std::shared_ptr<FunctionType>;
using ::torch::jit::Function;
struct CAFFE2_API FunctionType : public NamedType {
  static FunctionTypePtr create(Function* function) {
    return FunctionTypePtr(
        new FunctionType(function)); // NOLINT(modernize-make-shared)
  }
  bool operator==(const Type& rhs) const override {
    if (auto func_type = rhs.cast<FunctionType>()) {
      return func_type->function_ == function_;
    }

    return false;
  }
  std::string str() const override {
    return "Function";
  }
  std::string python_str() const override {
    throw "Function";
  }
  Function* function() const {
    return function_;
  }
  static const TypeKind Kind = TypeKind::FunctionType;

 private:
  FunctionType(Function* function);
  Function* function_;
};

struct NoneType;
using NoneTypePtr = std::shared_ptr<NoneType>;
// This type represents a Python None
struct CAFFE2_API NoneType : public Type {
  static NoneTypePtr create() {
    return NoneTypePtr(new NoneType()); // NOLINT(modernize-make-shared)
  }
  bool operator==(const Type& rhs) const override {
    return rhs.kind() == kind();
  }
  std::string str() const override {
    return "None";
  }
  bool isSubtypeOfExt(const TypePtr rhs, std::ostream *why_not) const override {
    if (rhs->kind() == OptionalType::Kind) {
      return true;
    }
    return Type::isSubtypeOfExt(rhs, why_not);
  }
  static const TypeKind Kind = TypeKind::NoneType;
  // global singleton
  static NoneTypePtr get();

 private:
  NoneType() : Type(TypeKind::NoneType) {}
};

struct GeneratorType;
using GeneratorTypePtr = std::shared_ptr<GeneratorType>;
// This type represents a Generator
struct CAFFE2_API GeneratorType : public Type {
  static GeneratorTypePtr create() {
    return GeneratorTypePtr(
        new GeneratorType()); // NOLINT(modernize-make-shared)
  }
  bool operator==(const Type& rhs) const override {
    return rhs.kind() == kind();
  }
  std::string str() const override {
    return "Generator";
  }
  static const TypeKind Kind = TypeKind::GeneratorType;
  // global singleton
  static GeneratorTypePtr get();

 private:
  GeneratorType() : Type(TypeKind::GeneratorType) {}
};

struct DeviceObjType;
using DeviceObjTypePtr = std::shared_ptr<DeviceObjType>;
// This type represents a Generator
struct CAFFE2_API DeviceObjType : public Type {
  static DeviceObjTypePtr create() {
    return DeviceObjTypePtr(
        new DeviceObjType()); // NOLINT(modernize-make-shared)
  }
  bool operator==(const Type& rhs) const override {
    return rhs.kind() == kind();
  }
  std::string str() const override {
    return "Device";
  }
  static const TypeKind Kind = TypeKind::DeviceObjType;
  // global singleton
  static DeviceObjTypePtr get();

 private:
  DeviceObjType() : Type(TypeKind::DeviceObjType) {}
};

struct VarType;
using VarTypePtr = std::shared_ptr<VarType>;
// This type represents a type variable, used in FunctionSchema
struct VarType : public Type {
  static VarTypePtr create(std::string name_) {
    return VarTypePtr(new VarType(std::move(name_)));
  }
  bool operator==(const Type& rhs) const override {
    return rhs.kind() == kind();
  }
  std::string str() const override {
    return name();
  }
  const std::string& name() const {
    return name_;
  }
  bool hasFreeVariables() const override {
    return true;
  }
  static const TypeKind Kind = TypeKind::VarType;

 private:
  VarType(std::string name_)
      : Type(TypeKind::VarType), name_(std::move(name_)) {}
  std::string name_;
};

struct CapsuleType;
using CapsuleTypePtr = std::shared_ptr<CapsuleType>;
// This type represents a Python Capsule
struct CAFFE2_API CapsuleType : public Type {
  static CapsuleTypePtr create() {
    return CapsuleTypePtr(new CapsuleType()); // NOLINT(modernize-make-shared)
  }
  bool operator==(const Type& rhs) const override {
    return rhs.kind() == kind();
  }
  std::string str() const override {
    return "Capsule";
  }
  static const TypeKind Kind = TypeKind::CapsuleType;
  // global singleton
  static CapsuleTypePtr get();
private:
  CapsuleType()
  : Type(TypeKind::CapsuleType) {}
};

CAFFE2_API std::ostream& operator<<(std::ostream& out, const Type& t);
CAFFE2_API std::ostream& operator<<(std::ostream& out, const VaryingShape& t);
// what is the type, ignoring extra size/shape information?
// e.g. Tensor(2x3) -> Dynamic, and Tuple(Tensor(2x3),...) -> Tuple(Dynamic,...)

inline TypePtr unshapedType(const TypePtr& type) {
  if (type->isSubtypeOf(TensorType::get())) {
    return TensorType::get();
  }
  return type->withContained(fmap(type->containedTypes(), unshapedType));
}

inline TypePtr TensorType::fromNumberType(TypePtr typ) {
  if (typ->isSubtypeOf(IntType::get())) {
    return TensorType::createContiguous(at::kLong, at::kCPU, {});
  } else if (typ->isSubtypeOf(FloatType::get())) {
    return TensorType::createContiguous(at::kFloat, at::kCPU, {});
  } else if (typ->isSubtypeOf(BoolType::get())) {
    return TensorType::createContiguous(at::kLong, at::kCPU, {});
  }
  TORCH_CHECK(false, "Unknown number type: ", typ->str());
}
inline TypePtr TensorType::fromBoolType() {
  return TensorType::createContiguous(at::kLong, at::kCPU, {});
}

inline c10::optional<c10::ScalarType> tryScalarTypeFromJitType(const c10::TypePtr & type) {
  if (type == FloatType::get()) {
    return at::ScalarType::Double;
  } else if (type == IntType::get()) {
    return at::ScalarType::Long;
  } else if (type == BoolType::get()) {
    return at::ScalarType::Bool;
  }
  return c10::nullopt;
}

inline at::ScalarType scalarTypeFromJitType(const c10::TypePtr& type) {
  auto result = tryScalarTypeFromJitType(type);
  AT_ASSERTM(
      result,
      "Add new condition, expected Float, Int, or Bool but got",
      type->str());
  return *result;
}

// Attempt to find the correct supertype of t1 and t2. If none is found then
// nullopt will be returned. If t1 == t2, or t1 is a type refinement of t2,
// then t2 will be returned (and vice versa).
// Two different tensortypes will return dynamic.
// Currently we chose not to support returning a NumberType for a float & int
// input because of a lack of operator support for NumberType
CAFFE2_API c10::optional<TypePtr> unifyTypes(
    const TypePtr& t1,
    const TypePtr& t2);

namespace detail {
template <typename T>
struct getTypePtr_ final {
  static TypePtr call() {
    if (!isCustomClassRegistered<T>()) {
      throw c10::Error("Type could not be converted to any of the known types.", "");
    }
    auto res = getCustomClassType<T>();
    return std::dynamic_pointer_cast<Type>(res.type_);
  }
};

template <>
struct getTypePtr_<at::Tensor> final {
  static TypePtr call() {
    return TensorType::get();
  }
};
template <>
struct getTypePtr_<double> final {
  static TypePtr call() {
    return FloatType::get();
  }
};
template <>
struct getTypePtr_<int64_t> final {
  static TypePtr call() {
    return IntType::get();
  }
};
template <>
struct getTypePtr_<bool> final {
  static TypePtr call() {
    return BoolType::get();
  }
};
template <>
struct getTypePtr_<at::Scalar> final {
  static TypePtr call() {
    return NumberType::get();
  }
};
template <>
struct getTypePtr_<at::Generator*> final {
  static TypePtr call() {
    return OptionalType::create(GeneratorType::get());
  }
};
template <>
struct getTypePtr_<std::string> final {
  static TypePtr call() {
    return StringType::get();
  }
};
template <class T>
struct getTypePtr_<std::vector<T>> final {
  static TypePtr call() {
    static auto type = ListType::create(getTypePtr_<T>::call());
    return type;
  }
};
template <class T>
struct getTypePtr_<c10::ArrayRef<T>> final {
  static TypePtr call() {
    static auto type = ListType::create(getTypePtr_<T>::call());
    return type;
  }
};
template <class T>
struct getTypePtr_<c10::List<T>> final {
  static TypePtr call() {
    static auto type = ListType::create(getTypePtr_<T>::call());
    return type;
  }
};
template <class T, size_t N>
struct getTypePtr_<std::array<T, N>> final {
  static TypePtr call() {
    static auto type = ListType::create(getTypePtr_<T>::call());
    return type;
  }
};
template <class K, class V>
struct getTypePtr_<std::unordered_map<K, V>> final {
  static TypePtr call() {
    static auto type =
        DictType::create(getTypePtr_<K>::call(), getTypePtr_<V>::call());
    return type;
  }
};
template <class K, class V>
struct getTypePtr_<c10::Dict<K, V>> final {
  static TypePtr call() {
    static auto type =
        DictType::create(getTypePtr_<K>::call(), getTypePtr_<V>::call());
    return type;
  }
};
template <class T>
struct getTypePtr_<at::optional<T>> final {
  static TypePtr call() {
    static auto type = OptionalType::create(getTypePtr_<T>::call());
    return type;
  }
};
} // namespace detail
template <class T>
inline TypePtr getTypePtr() {
  // TODO: static_assert that a templated function exists, and throw a friendy
  // error message if not
  return detail::getTypePtr_<T>::call();
}

CAFFE2_API TypePtr incompleteInferTypeFrom(const IValue& value);
CAFFE2_API TypePtr attemptToRecoverType(const IValue& input_ivalue);
CAFFE2_API bool isSubvalueOf(const IValue& input_ivalue, TypePtr type);

using TypeEnv = std::unordered_map<std::string, TypePtr>;
struct MatchTypeReturn {
  MatchTypeReturn(std::string reason) : reason_(std::move(reason)) {}
  static MatchTypeReturn Success() {
    return MatchTypeReturn();
  }
  bool success() const {
    return !reason_.has_value();
  }
  const std::string& reason() const {
    return reason_.value();
  }

 private:
  MatchTypeReturn()
  : reason_(c10::nullopt) {}
  c10::optional<std::string> reason_; // is there is no match, this contains the reason
};

// attempt to match the type variables in formal to actual, adding them to type_env.
// If no match is possible this returns a MatchTypeReturn with r.success() == false
// and a r.reason() that describes why it could not match.
// note: It is possible to successfully match a formal, but for type variables
// in the formal to still not be defined. In particular, None matches Optional[T]
// but does not define the value of T.
CAFFE2_API MatchTypeReturn
matchTypeVariables(TypePtr formal, TypePtr actual, TypeEnv& type_env);

// replace type variables appearing in `type` with the values in
// `type_env`. Returns nullptr if a variable used in `type`
// does not appear in `type_env`
CAFFE2_API TypePtr tryEvalTypeVariables(TypePtr type, TypeEnv& type_env);

/**
 * User Defined Types
 */

struct ClassType;
using ClassTypePtr = std::shared_ptr<ClassType>;
using ::torch::jit::script::CompilationUnit;

// This represents a class in TorchScript.
struct CAFFE2_API ClassType : public NamedType {
  // Create a class type with name `name` and its methods stored in `cu`.
  static ClassTypePtr create(
      c10::optional<QualifiedName> qualifiedName,
      std::weak_ptr<CompilationUnit> cu,
      bool is_module = false);

  bool operator==(const Type& rhs) const override {
    if (auto user_rhs = rhs.cast<ClassType>()) {
      const auto& lhs_name = name().value();
      const auto& rhs_name = user_rhs->name().value();

      return lhs_name == rhs_name;
    }
    return false;
  }

  std::string str() const override {
    const auto& n = name().value();
    return std::string("ClassType<") + n.name() + ">";
  }

  std::string python_str() const override {
    const auto& n = name().value();
    return n.qualifiedName();
  }

  TypePtr getAttribute(const std::string& name) const {
    AT_ASSERT(attributeNames_.size() == attributeTypes_.size());
    size_t pos = 0;
    for (const auto& attr : attributeNames_) {
      if (name == attr) {
        break;
      }
      ++pos;
    }

    if (pos >= attributeNames_.size()) {
      return nullptr;
    }
    return attributeTypes_[pos];
  }

  const TypePtr& getAttribute(size_t slot) const {
    AT_ASSERT(attributeNames_.size() == attributeTypes_.size());
    AT_ASSERT(slot < attributeTypes_.size());
    return attributeTypes_[slot];
  }

  const std::string& getAttributeName(size_t slot) const {
    AT_ASSERT(attributeNames_.size() == attributeTypes_.size());
    AT_ASSERT(slot < attributeTypes_.size());
    return attributeNames_[slot];
  }

  Function* getMethod(const std::string& name) const;
  const std::vector<Function*>& methods() const;
  void addMethod(Function* method) {
    methods_.push_back(method);
  }

  std::shared_ptr<CompilationUnit> compilation_unit();
  std::shared_ptr<const CompilationUnit> compilation_unit() const;

  size_t numAttributes() const {
    AT_ASSERT(attributeNames_.size() == attributeTypes_.size());
    return attributeNames_.size();
  }

  // Attributes are stored in a specific slot at runtime for effiency.
  // When emitting instructions we specify the slot so that attribute access is
  // a constant lookup
  c10::optional<size_t> findAttributeSlot(const std::string& name) const {
    AT_ASSERT(attributeNames_.size() == attributeTypes_.size());
    size_t slot = 0;
    for (const auto& attr : attributeNames_) {
      if (name == attr) {
        return slot;
      }
      slot++;
    }
    return c10::nullopt;
  }
  size_t getAttributeSlot(const std::string& name) const {
    if (auto r = findAttributeSlot(name)) {
      return *r;
    }
    TORCH_CHECK(
        false,
        python_str(),
        " does not have a field with the name '",
        name,
        "'");
  }

  bool hasAttribute(const std::string& name) const {
    return std::find_if(
               attributeNames_.cbegin(),
               attributeNames_.cend(),
               [&](const std::string& attr) { return attr == name; }) !=
        attributeNames_.cend();
  }

  size_t addAttribute(
      const std::string& name,
      TypePtr type,
      bool is_parameter = false);

  at::ArrayRef<std::string> attributeNames() const {
    return attributeNames_;
  }

  at::ArrayRef<TypePtr> containedTypes() const override {
    return attributeTypes_;
  }

  // generate a refined version of this class.
  // It has the same name but the slot Types are subtypes of
  // the original slots. It is only valid to refine a class type in a context
  // where it is know that there are not assignments to the objects slots
  // that would invalidate the refinement.
  // These variants are not registered in the global class table.
  ClassTypePtr refine(at::ArrayRef<TypePtr> refined_slots) const;

  TypePtr createWithContained(std::vector<TypePtr> contained_types) const override {
    auto ptr = ClassType::create(name(), compilation_unit_);
    AT_ASSERT(numAttributes() == contained_types.size());
    for(size_t i = 0; i < attributeNames_.size(); ++i) {
      AT_ASSERT(attributeTypes_[i]->isSubtypeOf(contained_types[i]));
      ptr->addAttribute(attributeNames_[i], contained_types[i]);
    }
    // Copy methods over
    for (const auto& method : methods()) {
      ptr->addMethod(method);
    }
    return ptr;
  }

  bool is_module() const {
    return bool(parameterSlots_);
  }
  bool is_parameter(size_t slot) const {
    TORCH_INTERNAL_ASSERT(
        is_module(), "asking for parameterSlots of non-Module");
    return parameterSlots_->at(slot);
  }

  bool isSubtypeOfExt(const TypePtr rhs, std::ostream* why_not) const override;
  static const TypeKind Kind = TypeKind::ClassType;

 private:
  ClassType(
      c10::optional<QualifiedName> name,
      std::weak_ptr<CompilationUnit> cu,
      bool is_module);

  // Mapping of attribute names -> their type.
  // NOTE: this does not contain methods, which are stored in the module
  // TODO: once modules support arbitrary ivalue attributes, we don't need this
  // anymore.
  // TODO: This is better represented as an OrderedDict, but alas it is not yet
  // available from c10
  std::vector<std::string> attributeNames_;
  std::vector<TypePtr> attributeTypes_;
  // Holds method attributes
  std::weak_ptr<CompilationUnit> compilation_unit_;

  // if present, this class inherits from torch.nn.Module
  // and these are the indices of the attributes which are parameters
  std::shared_ptr<std::vector<bool>> parameterSlots_;

  // List of methods associated with this class.
  std::vector<Function*> methods_;

};


struct InterfaceType;
using InterfaceTypePtr = std::shared_ptr<InterfaceType>;
using ::torch::jit::script::CompilationUnit;
using ::torch::jit::Function;

// Interfaces are a list of abstract methods that a class might meet.
// If a class provides those methods, it implicitly meets the interface.
struct CAFFE2_API InterfaceType : public NamedType {
  static InterfaceTypePtr create(
      QualifiedName qualifiedName);

  bool operator==(const Type& rhs) const override {
    if (auto user_rhs = rhs.cast<InterfaceType>()) {
      return name() == user_rhs->name();
    }
    return false;
  }

  std::string str() const override {
    return std::string("InterfaceType<") + name()->name() + ">";
  }

  std::string python_str() const override {
    return name()->qualifiedName();
  }

  bool isSubtypeOfExt(const TypePtr rhs, std::ostream* why_not) const override;

  // try to find a method of this interface,
  // returns nullptr if not found.
  const FunctionSchema* getMethod(const std::string& name) const;
  void addMethod(FunctionSchema schema);
  const std::vector<FunctionSchema>& methods() {
    return *methods_;
  }
  static const TypeKind Kind = TypeKind::InterfaceType;
  ~InterfaceType() override;
 private:
  InterfaceType(QualifiedName name);

  // shared_ptr so that this header does not have to depend on
  // FunctionSchema.h
  std::shared_ptr<std::vector<FunctionSchema>> methods_;
};

} // namespace c10
