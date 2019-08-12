#pragma once

#include <ATen/core/Tensor.h>
#include <ATen/core/functional.h>
#include <ATen/core/interned_strings.h>
#include <ATen/core/ivalue.h>
#include <ATen/core/qualified_name.h>
#include <c10/util/TypeList.h>

#include <c10/util/Optional.h>

#include <iostream>
#include <memory>
#include <type_traits>

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
  _(TensorType)             \
  _(DimensionedTensorType)  \
  _(CompleteTensorType)     \
  _(AutogradZeroTensorType) \
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
  _(ProfiledTensorType)     \
  _(DeviceObjType)          \
  _(FunctionType)           \
  _(ClassType)              \
  _(CapsuleType)

enum class TypeKind {
#define DEFINE_TYPE(T) T,
  C10_FORALL_TYPES(DEFINE_TYPE)
#undef DEFINE_TYPE
};

CAFFE2_API const char* typeKindToString(TypeKind kind);

#define DEFINE_IS_SUBCLASS(_kind)                       \
  bool isSubclass(const TypeKind kind) const override { \
    return kind == TypeKind::_kind;                     \
  }

struct Type;
using TypePtr = std::shared_ptr<Type>;

struct CAFFE2_API Type : std::enable_shared_from_this<Type> {
 private:
  TypeKind kind_;
  template <typename T>
  static std::shared_ptr<T> sliceType(std::shared_ptr<const T> ptr) {
    auto result = std::make_shared<typename std::remove_const<T>::type>(*ptr);
    // XXX: the line above will correctly slice the struct, and make its runtype
    // type exactly equal to T. However, kind_ is a field of Type, so it will
    // simply be copied, and we need to fix it in here to match the dynamic
    // type.
    result->kind_ = T::Kind;
    return result;
  }

 protected:
  Type(TypeKind kind) : kind_(kind) {}

 public:
  virtual bool operator==(const Type& rhs) const = 0;

  // subtyping relation. By default, we return true for the case
  // when the type is exactly equal or if this <: T where rhs = Optional[T]
  virtual bool isSubtypeOf(const TypePtr rhs) const;

  // If this class can be cast to the kind passed in
  // This removes the need for RTTI
  virtual bool isSubclass(const TypeKind kind) const = 0;

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
  // NOTE: if the cast succeeds, but the casted kind is not the
  // run-time kind of the type, we also slice the structure, so
  // that assignments of those types to values don't accidentally
  // inherit more detailed information from subclasses.
  template <typename T>
  std::shared_ptr<T> cast() {
    std::shared_ptr<T> r = nullptr;
    if (isSubclass(T::Kind)) {
      r = std::static_pointer_cast<T>(shared_from_this());
    }
    if (!r || T::Kind == kind()) {
      return r;
    } else {
      return sliceType<T>(r);
    }
  }
  template <typename T>
  std::shared_ptr<const T> cast() const {
    std::shared_ptr<const T> r = nullptr;
    if (isSubclass(T::Kind)) {
      r = std::static_pointer_cast<const T>(shared_from_this());
    }
    if (!r || T::Kind == kind()) {
      return r;
    } else {
      return sliceType<T>(r);
    }
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
// 1. Optional[T] isSubtypeOf Optional[R] iff T isSubtypeOf R
// 2. T isSubtypeOf Optional[R] if T isSubtypeOf R
// Note: NoneType is NOT a subtype of any optional.
// instead NoneType is convertable in schema matching to any Optional[T]
// it is handled this way because it is not possible to match None to
// Optional[T] and extract T. Intead, we always create a None constant
// instruction with a particular type: v: Optional[int] = None()
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
  DEFINE_IS_SUBCLASS(OptionalType);

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

  bool isSubtypeOf(const TypePtr rhs) const override {
    if (Type::isSubtypeOf(rhs)) {
      return true;
    }
    if (auto rhs_ = rhs->cast<OptionalType>()) {
      return getElementType()->isSubtypeOf(rhs_->getElementType());
    }
    return false;
  }
  // common cast Optional[Tensor] for undefined tensor type
  static OptionalTypePtr ofTensor();

 private:
  OptionalType(TypePtr elem) : SingleElementType(elem) {}
};

struct TensorType;
using TensorTypePtr = std::shared_ptr<TensorType>;
// This type represents a single Tensor, with an unknown shape.
// Subtype hierarchy for Tensor Types (TensorType as the base type):
// CompleteTensorType <: DimensionedTensorType <: TensorType
// AutogradZeroTensorType <: TensorType
struct CAFFE2_API TensorType : public Type {
  static TensorTypePtr create() {
    return TensorTypePtr(new TensorType()); // NOLINT(modernize-make-shared)
  }
  DEFINE_IS_SUBCLASS(TensorType);

  bool requires_grad() const override {
    return true;
  }

  bool operator==(const Type& rhs) const override {
    return rhs.kind() == kind();
  }
  std::string str() const override {
    return "Tensor";
  }
  static const TypeKind Kind = TypeKind::TensorType;
  // global singleton
  static TensorTypePtr get();

 protected:
  TensorType(TypeKind kind = TypeKind::TensorType) : Type(kind) {}
};

struct AutogradZeroTensorType;
using AutogradZeroTensorTypePtr = std::shared_ptr<AutogradZeroTensorType>;
// This type represents an undefined tensor.
struct CAFFE2_API AutogradZeroTensorType : public TensorType {
  static AutogradZeroTensorTypePtr create() {
    return AutogradZeroTensorTypePtr(
        new AutogradZeroTensorType()); // NOLINT(modernize-make-shared)
  }

  DEFINE_IS_SUBCLASS(AutogradZeroTensorType);

  bool requires_grad() const override {
    return false;
  }

  bool operator==(const Type& rhs) const override {
    return rhs.kind() == kind();
  }
  bool isSubtypeOf(const TypePtr rhs) const override {
    return rhs->kind() == TypeKind::TensorType ||
        rhs->kind() == TypeKind::AutogradZeroTensorType ||
        TensorType::isSubtypeOf(rhs);
  }
  std::string str() const override {
    return "AutogradZeroTensor";
  }

  static const TypeKind Kind = TypeKind::AutogradZeroTensorType;
  // global singleton
  static AutogradZeroTensorTypePtr get();

 protected:
  AutogradZeroTensorType() : TensorType(TypeKind::AutogradZeroTensorType) {}
};

struct DimensionedTensorType;
using DimensionedTensorTypePtr = std::shared_ptr<DimensionedTensorType>;
// This type represents a single Tensor with a specific size
struct CAFFE2_API DimensionedTensorType : public TensorType {
  template <typename... T>
  static DimensionedTensorTypePtr create(T&&... all) {
    return DimensionedTensorTypePtr(new DimensionedTensorType(
        std::forward<T>(all)...)); // NOLINT(modernize-make-shared)
  }

  at::ScalarType scalarType() const {
    return scalar_type_;
  }
  at::Device device() const {
    return device_;
  }
  int64_t dim() const {
    return dim_;
  }
  bool requires_grad() const override {
    return requires_grad_;
  }

  DimensionedTensorTypePtr toScalarType(at::ScalarType type) {
    auto t = DimensionedTensorType::create(*this);
    t->scalar_type_ = type;
    return t;
  }
  DimensionedTensorTypePtr withDim(size_t new_dim) {
    auto t = DimensionedTensorType::create(*this);
    t->dim_ = new_dim;
    return t;
  }
  DimensionedTensorTypePtr withRequiresGrad(bool req) {
    auto t = DimensionedTensorType::create(*this);
    t->requires_grad_ = req;
    return t;
  }

  bool operator==(const Type& rhs) const override {
    if (rhs.kind() != TypeKind::DimensionedTensorType)
      return false;
    auto rt = rhs.expect<DimensionedTensorType>();
    return scalarType() == rt->scalarType() && device() == rt->device() &&
        dim() == rt->dim();
  }
  bool isSubtypeOf(const TypePtr rhs) const override {
    return rhs->kind() == TypeKind::TensorType ||
        (rhs->kind() == TypeKind::DimensionedTensorType &&
         Type::isSubtypeOf(rhs)) ||
        TensorType::isSubtypeOf(rhs);
  }
  bool isSubclass(const TypeKind kind) const override {
    return kind == TypeKind::TensorType ||
        kind == TypeKind::DimensionedTensorType;
  }
  std::string str() const override {
    // str is used for user-facing error messages, where we
    // don't want to reveal underlying size information.
    return "Tensor";
  }

  static const TypeKind Kind = TypeKind::DimensionedTensorType;

 protected:
  DimensionedTensorType(
      const at::Tensor& tensor,
      TypeKind kind = TypeKind::DimensionedTensorType)
      : DimensionedTensorType(
            tensor.scalar_type(),
            tensor.device(),
            tensor.dim(),
            tensor.is_variable() && tensor.requires_grad(),
            kind) {}
  DimensionedTensorType(
      at::ScalarType scalar_type,
      at::Device device,
      int64_t dim,
      bool requires_grad = true,
      TypeKind kind = TypeKind::DimensionedTensorType)
      : TensorType(kind),
        scalar_type_(scalar_type),
        requires_grad_(at::isFloatingType(scalar_type) && requires_grad),
        device_(device),
        dim_(dim) {}

  at::ScalarType scalar_type_;
  bool requires_grad_;
  at::Device device_;
  int64_t dim_;
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
  VaryingShape(const IntArrayRef& ref)
      : size_(ref.size()), dims_(ref.begin(), ref.end()) {}

  VaryingShape(const std::vector<int64_t>& vec)
      : size_(vec.size()), dims_(vec.begin(), vec.end()) {}

  VaryingShape(c10::optional<size_t> size)
      : size_(size), dims_(size ? size.value() : 0) {}

  bool operator==(const VaryingShape& other) const {
    return size_ == other.size_ && dims_ == dims_;
  }

  const c10::optional<int64_t>& operator[](int i) const {
    if (!size_) {
      throw std::runtime_error("Rank isn't fixed");
    }
    return dims_[i];
  }

  c10::optional<size_t> size() const {
    AT_ASSERT(!size_ || size_ == dims_.size());
    return size_;
  }

  const std::vector<c10::optional<int64_t>>& sizes() const { return dims_; }

  VaryingShape merge(const VaryingShape& other) const;

  c10::optional<std::vector<int64_t>> concrete_sizes() const {
    c10::optional<std::vector<int64_t>> empty{};
    std::vector<int64_t> sizes;
    if (!size_) {
      return empty;
    }
    for (auto d : dims_) {
      if (!d) {
        return empty;
      }
      sizes.push_back(d.value());
    }

    return sizes;
  }

 private:
  c10::optional<size_t> size_;
  std::vector<c10::optional<int64_t>> dims_;
};

using VaryingStrides = VaryingShape;

struct ProfiledTensorType;
using ProfiledTensorTypePtr = std::shared_ptr<ProfiledTensorType>;
// This type represents a single Tensor with a specific size
struct CAFFE2_API ProfiledTensorType : public TensorType {
  static ProfiledTensorTypePtr create(const at::Tensor& t) {
    return ProfiledTensorTypePtr(new ProfiledTensorType(t));
  }

  static ProfiledTensorTypePtr create(const TypePtr& tptr) {
    if (auto dtt = tptr->cast<DimensionedTensorType>()) {
      at::VaryingShape vshape(c10::optional<size_t>(dtt->dim()));
      return ProfiledTensorType::create(
          {dtt->scalarType()},
          {dtt->device()},
          vshape,
          vshape,
          {dtt->requires_grad()});
    }

    if (auto ptt = tptr->cast<ProfiledTensorType>()) {
      return ptt;
    }

    if (tptr->isSubclass(TypeKind::TensorType)) {
      c10::optional<size_t> sz;
      return ProfiledTensorType::create(
          {}, {}, VaryingShape{sz}, VaryingShape{sz}, {});
    }

    TORCH_INTERNAL_ASSERT(false, "Expected a tensor type");
  }

  static ProfiledTensorTypePtr create(
      c10::optional<at::ScalarType> scalar_type,
      c10::optional<Device> device,
      const VaryingShape& sizes,
      const VaryingStrides& strides,
      c10::optional<bool> requires_grad) {
    return ProfiledTensorTypePtr(new ProfiledTensorType(
        scalar_type, device, sizes, strides, requires_grad));
  }

  static ProfiledTensorTypePtr create(ProfiledTensorTypePtr pttp) {
    return ProfiledTensorTypePtr(new ProfiledTensorType(
        pttp->scalarType(),
        pttp->device(),
        pttp->sizes(),
        pttp->strides(),
        pttp->requiresGrad()));
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
    return requires_grad_ ? *requires_grad_ : false;
  }


  bool operator==(const Type& rhs) const override {
    if (rhs.kind() != kind()) {
      return false;
    }

    auto rt = rhs.expect<ProfiledTensorType>();
    return scalar_type_ == rt->scalarType() && sizes() == rt->sizes() &&
        strides() == rt->strides() && device() == rt->device();
  }
  bool isSubtypeOf(const TypePtr rhs) const override {
    if (rhs->kind() == TypeKind::ProfiledTensorType)
      return *expect<ProfiledTensorType>() == *rhs;
    return rhs->kind() == TypeKind::TensorType || TensorType::isSubtypeOf(rhs);
  }
  bool isSubclass(const TypeKind kind) const override {
    return kind == TypeKind::TensorType || kind == TypeKind::ProfiledTensorType;
  }
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

  ProfiledTensorTypePtr merge(ProfiledTensorTypePtr other) {
    auto scalar_type = merge_primitive(scalarType(), other->scalarType());
    auto dev = merge_primitive(device(), other->device());
    auto sz = sizes().merge(other->sizes());
    auto srs = strides().merge(other->strides());
    auto gr = merge_primitive(requiresGrad(), other->requiresGrad());
    return ProfiledTensorType::create(scalar_type, dev, sz, srs, gr);
  }

  static const TypeKind Kind = TypeKind::ProfiledTensorType;

 private:
  ProfiledTensorType(const at::Tensor& tensor)
      : TensorType(),
        scalar_type_(tensor.scalar_type()),
        device_(tensor.device()),
        sizes_(tensor.sizes().vec()),
        strides_(tensor.strides().vec()),
        requires_grad_(tensor.requires_grad()) {}
  ProfiledTensorType(
      c10::optional<at::ScalarType> scalar_type,
      c10::optional<Device> device,
      const VaryingShape& sizes,
      const VaryingStrides& strides,
      c10::optional<bool> requires_grad)
      : TensorType(),
        scalar_type_(scalar_type),
        device_(device),
        sizes_(sizes),
        strides_(strides),
        requires_grad_(requires_grad) {}

  c10::optional<at::ScalarType> scalar_type_;
  c10::optional<at::Device> device_;
  VaryingShape sizes_;
  VaryingStrides strides_;
  c10::optional<bool> requires_grad_;
};

struct CompleteTensorType;
using CompleteTensorTypePtr = std::shared_ptr<CompleteTensorType>;
// This type represents a single Tensor with a specific size
struct CAFFE2_API CompleteTensorType : public DimensionedTensorType {
  template <typename... T>
  static CompleteTensorTypePtr create(T&&... all) {
    return CompleteTensorTypePtr(new CompleteTensorType(
        std::forward<T>(all)...)); // NOLINT(modernize-make-shared)
  }

  // overloaded create variadic template argument as it could not distinguish
  // initializer list
  static CompleteTensorTypePtr create(
      at::ScalarType scalar_type,
      at::Device device,
      at::IntArrayRef sizes) {
    return CompleteTensorTypePtr(new CompleteTensorType(
        scalar_type, device, sizes)); // NOLINT(modernize-make-shared)
  }
  static CompleteTensorTypePtr create(
      at::ScalarType scalar_type,
      at::Device device,
      at::IntArrayRef sizes,
      at::IntArrayRef strides) {
    return CompleteTensorTypePtr(new CompleteTensorType(
        scalar_type, device, sizes, strides)); // NOLINT(modernize-make-shared)
  }

  const std::vector<int64_t>& sizes() const {
    return sizes_;
  }
  const std::vector<int64_t>& strides() const {
    return strides_;
  }

  TypePtr withSizesStrides(at::IntArrayRef sizes, at::IntArrayRef strides)
      const {
    return CompleteTensorType::create(scalar_type_, device_, sizes, strides);
  }

  TypePtr withSizes(at::IntArrayRef sizes) const {
    return withSizesStrides(
        sizes, CompleteTensorType::contiguousStridesOf(sizes));
  }

  CompleteTensorTypePtr contiguous() const {
    auto t = CompleteTensorType::create(*this);
    t->strides_ = CompleteTensorType::contiguousStridesOf(sizes_);
    return t;
  }

  CompleteTensorTypePtr toScalarType(at::ScalarType type) {
    auto t = CompleteTensorType::create(*this);
    t->scalar_type_ = type;
    return t;
  }

  bool operator==(const Type& rhs) const override {
    if (rhs.kind() != kind()) {
      return false;
    }

    auto rt = rhs.expect<CompleteTensorType>();
    return scalarType() == rt->scalarType() && sizes() == rt->sizes() &&
        strides() == rt->strides() && device() == rt->device();
  }
  bool isSubtypeOf(const TypePtr rhs) const override {
    if (rhs->kind() == TypeKind::DimensionedTensorType)
      return *expect<DimensionedTensorType>() == *rhs;
    return rhs->kind() == TypeKind::TensorType || TensorType::isSubtypeOf(rhs);
  }
  bool isSubclass(const TypeKind kind) const override {
    return kind == TypeKind::TensorType ||
        kind == TypeKind::DimensionedTensorType ||
        kind == TypeKind::CompleteTensorType;
  }
  std::string str() const override {
    // str is used for user-facing error messages, where we
    // don't want to reveal underlying size information.
    return "Tensor";
  }
  size_t numel() const {
    size_t prod = 1;
    for (auto s : sizes()) {
      prod *= s;
    }
    return prod;
  }

  static const TypeKind Kind = TypeKind::CompleteTensorType;

  static TypePtr fromNumberType(TypePtr typ);
  static TypePtr fromBoolType();

 private:
  CompleteTensorType(const at::Tensor& tensor)
      : DimensionedTensorType(tensor, TypeKind::CompleteTensorType),
        sizes_(tensor.sizes().vec()),
        strides_(tensor.strides().vec()) {}
  CompleteTensorType(
      at::ScalarType scalar_type,
      at::Device device,
      at::IntArrayRef sizes,
      bool requires_grad = true)
      : CompleteTensorType(
            scalar_type,
            device,
            sizes,
            CompleteTensorType::contiguousStridesOf(sizes),
            requires_grad) {}
  CompleteTensorType(
      at::ScalarType scalar_type,
      at::Device device,
      at::IntArrayRef sizes,
      at::IntArrayRef strides,
      bool requires_grad = true)
      : DimensionedTensorType(
            scalar_type,
            device,
            sizes.size(),
            requires_grad,
            TypeKind::CompleteTensorType),
        sizes_(sizes.vec()),
        strides_(strides.vec()) {}

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

  std::vector<int64_t> sizes_;
  std::vector<int64_t> strides_;
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
  DEFINE_IS_SUBCLASS(ListType);
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

  DEFINE_IS_SUBCLASS(DictType);

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

  DEFINE_IS_SUBCLASS(FutureType);

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
  NamedType(TypeKind tk, c10::optional<c10::QualifiedName> qualifiedName)
      : Type(tk), name_(std::move(qualifiedName)) {}

  const c10::optional<QualifiedName>& name() const {
    return name_;
  }

 protected:
  // Fully qualified name of type (note that this has to be globally unique).
  // Looks like: "foo.bar.Baz".
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
  DEFINE_IS_SUBCLASS(TupleType);
  at::ArrayRef<TypePtr> elements() const {
    return elements_;
  }
  bool operator==(const Type& rhs) const override;
  bool isSubtypeOf(const TypePtr rhs_) const override;

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
  DEFINE_IS_SUBCLASS(NumberType);
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
  DEFINE_IS_SUBCLASS(FloatType);
  bool operator==(const Type& rhs) const override {
    return rhs.kind() == kind();
  }
  std::string str() const override {
    return "float";
  }
  std::string python_str() const override {
    return "float";
  }
  bool isSubtypeOf(const TypePtr rhs) const override {
    return rhs->kind() == TypeKind::NumberType || NumberType::isSubtypeOf(rhs);
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
  DEFINE_IS_SUBCLASS(IntType);
  bool operator==(const Type& rhs) const override {
    return rhs.kind() == kind();
  }
  std::string str() const override {
    return "int";
  }
  std::string python_str() const override {
    return "int";
  }
  bool isSubtypeOf(const TypePtr rhs) const override {
    return rhs->kind() == TypeKind::NumberType || NumberType::isSubtypeOf(rhs);
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
  DEFINE_IS_SUBCLASS(BoolType);
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
  DEFINE_IS_SUBCLASS(StringType);
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
struct CAFFE2_API FunctionType : public Type {
  static FunctionTypePtr create(Function* function) {
    return FunctionTypePtr(
        new FunctionType(function)); // NOLINT(modernize-make-shared)
  }
  DEFINE_IS_SUBCLASS(FunctionType);
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
  FunctionType(Function* function)
      : Type(TypeKind::FunctionType), function_(function) {}

  Function* function_;
};

struct NoneType;
using NoneTypePtr = std::shared_ptr<NoneType>;
// This type represents a Python None
struct CAFFE2_API NoneType : public Type {
  static NoneTypePtr create() {
    return NoneTypePtr(new NoneType()); // NOLINT(modernize-make-shared)
  }
  DEFINE_IS_SUBCLASS(NoneType);
  bool operator==(const Type& rhs) const override {
    return rhs.kind() == kind();
  }
  bool isSubtypeOf(const TypePtr rhs) const override {
    return rhs->kind() == TypeKind::NoneType;
  }
  std::string str() const override {
    return "None";
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
  DEFINE_IS_SUBCLASS(GeneratorType);
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
  DEFINE_IS_SUBCLASS(DeviceObjType);
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
  DEFINE_IS_SUBCLASS(VarType);
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
  DEFINE_IS_SUBCLASS(CapsuleType);
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
  if (type->kind() == TypeKind::DimensionedTensorType ||
      type->kind() == TypeKind::CompleteTensorType) {
    return TensorType::get();
  }
  return type->withContained(fmap(type->containedTypes(), unshapedType));
}

inline TypePtr CompleteTensorType::fromNumberType(TypePtr typ) {
  if (typ->isSubtypeOf(IntType::get())) {
    return CompleteTensorType::create(at::kLong, at::kCPU, {});
  } else if (typ->isSubtypeOf(FloatType::get())) {
    return CompleteTensorType::create(at::kFloat, at::kCPU, {});
  } else if (typ->isSubtypeOf(BoolType::get())) {
    return CompleteTensorType::create(at::kLong, at::kCPU, {});
  }
  AT_ERROR("unknown number type", typ->str());
}

inline TypePtr CompleteTensorType::fromBoolType() {
  return CompleteTensorType::create(at::kLong, at::kCPU, {});
}

inline at::ScalarType scalarTypeFromJitType(const c10::TypePtr& type) {
  if (type == FloatType::get()) {
    return at::ScalarType::Double;
  } else if (type == IntType::get()) {
    return at::ScalarType::Long;
  } else if (type == BoolType::get()) {
    return at::ScalarType::Bool;
  }
  AT_ASSERTM(
      0,
      "Add new condition, expected Float, Int, or Bool but got",
      type->str());
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
  MatchTypeReturn(TypePtr type) : type(type) {}
  MatchTypeReturn(std::string errMsg) : errMsg(std::move(errMsg)) {}

  c10::optional<TypePtr> type; // nullopt if there is no match
  std::string errMsg; // is there is no match, this contains the reason
};

CAFFE2_API MatchTypeReturn
matchTypeVariables(TypePtr formal, TypePtr actual, TypeEnv& type_env);

CAFFE2_API TypePtr evalTypeVariables(TypePtr type, TypeEnv& type_env);

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

  DEFINE_IS_SUBCLASS(ClassType);
  bool operator==(const Type& rhs) const override {
    if (auto user_rhs = rhs.cast<ClassType>()) {
      return name()->qualifiedName() == user_rhs->name()->qualifiedName();
    }
    return false;
  }

  std::string str() const override {
    return std::string("ClassType<") + name()->name() + ">";
  }

  std::string python_str() const override {
    return name()->qualifiedName();
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
    auto ptr = ClassType::create(name_, compilation_unit_);
    AT_ASSERT(numAttributes() == contained_types.size());
    for(size_t i = 0; i < attributeNames_.size(); ++i) {
      AT_ASSERT(attributeTypes_[i]->isSubtypeOf(contained_types[i]));
      ptr->addAttribute(attributeNames_[i], contained_types[i]);
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

} // namespace c10
