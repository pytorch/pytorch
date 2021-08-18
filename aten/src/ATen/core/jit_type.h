#pragma once

#include <ATen/core/jit_type_base.h>
#include <ATen/core/TensorBody.h>
#include <ATen/core/functional.h>
#include <ATen/core/interned_strings.h>
#include <ATen/core/qualified_name.h>
#include <ATen/core/ivalue.h>
#include <c10/util/TypeList.h>
#include <c10/util/Optional.h>

#include <iostream>
#include <memory>
#include <type_traits>
#include <array>

struct ClassType;
namespace torch {
namespace jit {
struct CompilationUnit;
struct Function;
} // namespace jit
} // namespace torch

namespace c10 {

struct IValue;
struct FunctionSchema;
struct NamedType;
using OptNameList = c10::optional<std::vector<std::string>>;

struct AnyType;
using AnyTypePtr = std::shared_ptr<AnyType>;
// Any is the top of the type hierarchy, all other types are subtypes
// T <: Any, forall T
struct TORCH_API AnyType : public Type {
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
// e.g. Future[T], Optional[T], List[T]
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
  SingleElementType(TypePtr elem) : Type(Kind), elem(std::move(elem)) {
    if (!this->elem) {
      throw std::runtime_error(c10::str(
            "Can not create ", typeKindToString(Kind), " with None type"));
    }
  }

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
struct TORCH_API OptionalType
    : public SingleElementType<TypeKind::OptionalType, OptionalType> {
  static OptionalTypePtr create(TypePtr element) {
    TORCH_INTERNAL_ASSERT(element, "OptionalType requires valid TypePtr");
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

  TypePtr createWithContained(
      std::vector<TypePtr> contained_types) const override {
    AT_ASSERT(contained_types.size() == 1);
    return create(contained_types[0]);
  }

  bool isSubtypeOfExt(const TypePtr& rhs, std::ostream* why_not) const override {
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

  std::string annotation_str_impl(TypePrinter printer = nullptr) const override {
    std::stringstream ss;
    ss << "Optional[" << getElementType()->annotation_str(printer) << "]";
    return ss.str();
  }
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

// If we see `a + b + c`  and know that a, b, and c are the same size and have
// two dimensions (WxH), then we can generate a fused kernel for them. That
// fused kernel would likely have indexing math to handling both the W and H
// dimensions. However, if we knew the WxH dimensions were contiguous, we can
// pretend like we only have a single dimension, simplifying the indexing logic.
// This can be performed even if the dimensions are transposed,
// as long as a, b, and c are transposed in the same way.
// We'd like to have the compiler be able to do this dimensionality reduction,
// but simply knowing sizes is not enough.
// We can extend profiling to also record stride information.
// Rather than recording specific strides,
// we can simply order the strides from smallest to largest with
// `stride_indices` A contiguity marker on the smallest stride (c0) indicates
// the stride is precisely 1, otherwise a contiguity marker means that $stride_n
// = size_{n-1}*stride_{n-1}$
struct TORCH_API Stride {
  Stride() {}
  Stride(
      const c10::optional<size_t>& stride_index,
      const c10::optional<bool>& contiguous,
      const c10::optional<size_t>& stride)
      : stride_index_(stride_index), contiguous_(contiguous), stride_(stride) {}

  bool operator==(const Stride& b) const {
    return stride_index_ == b.stride_index_ && contiguous_ == b.contiguous_ &&
        stride_ == b.stride_;
  }

  bool isComplete() const {
    return stride_index_ && contiguous_ && stride_;
  }

  c10::optional<size_t> stride_index_;
  c10::optional<bool> contiguous_;
  c10::optional<size_t> stride_;
};

template <>
inline c10::optional<Stride> merge_primitive(
    const c10::optional<Stride>& a,
    const c10::optional<Stride>& b) {
  c10::optional<Stride> left = a;
  c10::optional<Stride> right = b;
  if (!left.has_value()) {
    left = {Stride()};
  }
  if (!right.has_value()) {
    right = {Stride()};
  }

  auto merged_index =
      merge_primitive(left->stride_index_, right->stride_index_);
  auto merged_cont = merge_primitive(left->contiguous_, right->contiguous_);
  auto merged_stride = merge_primitive(left->stride_, right->stride_);
  auto r = Stride(merged_index, merged_cont, merged_stride);
  // normalize
  if (!r.stride_index_.has_value() && !r.contiguous_.has_value() &&
      !r.stride_.has_value()) {
    return c10::optional<Stride>{};
  }

  return r;
}

struct TORCH_API ShapeSymbol {
  // needed for use in `std::map`
  ShapeSymbol() : value_(-1) {}
  // is this symbol a fixed/static dimension
  bool is_static() const {
    return value_ >= 0;
  };
  bool operator==(const ShapeSymbol& b) const {
    return value_ == b.value_;
  }
  bool operator<(const ShapeSymbol& b) const {
    return value_ < b.value_;
  }

  static ShapeSymbol fromStaticSize(int64_t val) {
    return ShapeSymbol(val);
  }
  int64_t static_size() const {
    TORCH_CHECK(is_static());
    return value_;
  };

  int64_t value() const {
    return value_;
  };

  static ShapeSymbol newSymbol() {
    return fromStaticSize(-static_cast<int64_t>(++num_symbols));
  };
  friend TORCH_API std::ostream& operator<<(
      std::ostream& os,
      const ShapeSymbol& s);

 private:
  ShapeSymbol(int64_t val) : value_(val) {}
  int64_t value_;
  static std::atomic<size_t> num_symbols;
};

inline ShapeSymbol merge_primitive(
    const ShapeSymbol& a,
    const ShapeSymbol& b) {
  if (a.is_static() && b.is_static() && a == b) {
    return a;
  }
  return ShapeSymbol::newSymbol();
}

// Shape of a Tensor represented with ShapeSymbol's. Unranked, ranked unknown
// dims, partially known and fully known shapes are all supported.
struct TORCH_API SymbolicShape {
  // Unranked shape constructor.
  SymbolicShape() : dims_(c10::nullopt) {}

  // Known rank but unknown dimentions.
  SymbolicShape(c10::optional<size_t> rank) : dims_(c10::nullopt) {
    if(!rank) {
      return;
    }

    std::vector<ShapeSymbol> shape_symbols;
    shape_symbols.reserve(*rank);
    for(size_t i = 0; i < *rank; ++i) {
      shape_symbols.push_back(ShapeSymbol::newSymbol());
    }
    dims_ = shape_symbols;
  }

  // Mix of known and unknown ranks
  SymbolicShape(const std::vector<c10::optional<int64_t>>& dims) {
    std::vector<ShapeSymbol> shape_symbols;
    shape_symbols.reserve(dims.size());
    for(c10::optional<int64_t> dim: dims) {
      if(!dim) {
        shape_symbols.push_back(ShapeSymbol::newSymbol());
      } else {
        shape_symbols.push_back(ShapeSymbol::fromStaticSize(*dim));
      }
    }
    dims_ = shape_symbols;
  }

  void dump() const;

  SymbolicShape(std::vector<ShapeSymbol> dims) : dims_(std::move(dims)) {}

  SymbolicShape(c10::IntArrayRef dims) {
    std::vector<ShapeSymbol> shape_symbols;
    shape_symbols.reserve(dims.size());
    for(int64_t dim : dims) {
      shape_symbols.push_back(ShapeSymbol::fromStaticSize(dim));
    }
    dims_ = shape_symbols;
  }

  ShapeSymbol operator[](size_t i) const {
    if (!dims_) {
      throw std::runtime_error("Rank isn't fixed");
    }
    return (*dims_).at(i);
  }

  ShapeSymbol at(size_t i) const {
    if (!dims_) {
      throw std::runtime_error("Rank isn't fixed");
    }
    return (*dims_).at(i);
  }

  // Returns rank or nullopt in case of unranked shape.
  c10::optional<size_t> rank() const {
    if(!dims_) {
      return c10::nullopt;
    }
    return dims_->size();
  }

  c10::optional<std::vector<ShapeSymbol>> sizes() const {
    return dims_;
  }

  // Checks whether the shape is fully defined/complete, ie. rank and sizes
  // of every dimension are known.
  bool isComplete() const {
    if(!dims_) {
      return false;
    }
    for(auto d : *dims_) {
      if(!d.is_static()) {
        return false;
      }
    }
    return true;
  }

  // Create new SymbolicShape that is result of merging self and another
  // SymbolicShape. Only dimensions that are static and equal will be
  // preserved.
  // If either of two shapes are of unknown rank or they have unmatching rank,
  // result will be unranked.
  SymbolicShape merge(const SymbolicShape& other) const;

  private:
    c10::optional<std::vector<ShapeSymbol>> dims_;
};

namespace detail {
inline bool isComplete(const Stride& s) {
  return s.isComplete();
}

template<typename T>
inline bool isComplete(const T& t) {
  return true;
}
}

template <typename T>
struct VaryingShape {
  using ListOfOptionalElements = std::vector<c10::optional<T>>;
  VaryingShape(const std::vector<T>& vec)
      : VaryingShape(ListOfOptionalElements(vec.begin(), vec.end())) {}

  VaryingShape(c10::ArrayRef<T> vec)
      : VaryingShape(ListOfOptionalElements(vec.begin(), vec.end())) {}

  VaryingShape(c10::optional<size_t> size = c10::nullopt) : dims_(c10::nullopt) {
    if (size) {
      dims_ = ListOfOptionalElements(*size);
    }
  }

  VaryingShape(ListOfOptionalElements dims) : dims_(std::move(dims)) {}

  VaryingShape(size_t size) : VaryingShape(c10::optional<size_t>(size)) {}

  bool operator==(const VaryingShape& other) const {
    return dims_ == other.dims_;
  }

  const c10::optional<T> &operator[](size_t i) const {
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

  const c10::optional<ListOfOptionalElements>& sizes() const {
    return dims_;
  }

  TORCH_API VaryingShape merge(const VaryingShape& other) const;

  c10::optional<std::vector<T>> concrete_sizes() const {
    if (!dims_) {
      return c10::nullopt;
    }
    std::vector<T> sizes;
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
      if (!d || !detail::isComplete(*d)) {
        return false;
      }
    }
    return true;
  }

 private:
  c10::optional<ListOfOptionalElements> dims_;
};

struct TensorType;
using TensorTypePtr = std::shared_ptr<TensorType>;
// This type represents a single Tensor with a specific size
struct TORCH_API TensorType : public Type {
  static TensorTypePtr create(const at::Tensor& t);

  // used by TensorType::create(size_t dim) which in turn used by
  // shape_analysis.cpp
  static TensorTypePtr create(
      c10::optional<at::ScalarType> scalar_type,
      c10::optional<Device> device,
      const VaryingShape<int64_t>& sizes,
      const VaryingShape<int64_t>& strides,
      c10::optional<bool> requires_grad,
      c10::optional<bool> undefined = false,
      bool tensor_contiguity = false);

  static TensorTypePtr create(
      c10::optional<at::ScalarType> scalar_type,
      c10::optional<Device> device,
      const SymbolicShape& sizes,
      const VaryingShape<Stride>& stride_,
      c10::optional<bool> requires_grad,
      c10::optional<bool> undefined = false);

  static TensorTypePtr create(
      c10::optional<at::ScalarType> scalar_type,
      c10::optional<Device> device,
      c10::optional<size_t> dim,
      c10::optional<bool> requires_grad);

  // overloaded create variadic template argument as it could not distinguish
  // initializer list
  static TensorTypePtr createContiguous(
      at::ScalarType scalar_type,
      at::Device device,
      at::IntArrayRef sizes);

  static TypePtr fromNumberType(TypePtr typ);
  static TypePtr fromBoolType();

  c10::optional<size_t> dim() const {
    return sizes().size();
  }

  VaryingShape<int64_t> sizes() const;

  VaryingShape<int64_t> strides() const;

  const VaryingShape<Stride>& stride_properties() const {
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

  bool operator==(const Type& rhs) const override;
  bool isSubtypeOfExt(const TypePtr& rhs, std::ostream* why_not) const override;

  std::string str() const override;

  std::string repr_str() const override {
    return str() + (isInferredType() ? " (inferred)" : "");
  }

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
    // withDim is only used by the legacy executor
    // that only cares about the rank, so create dummy symbols)) :
    copy->sizes_ = SymbolicShape(d);
    copy->strides_ = VaryingShape<Stride>(d);
    return copy;
  }

  TensorTypePtr withSizesStrides(
      at::IntArrayRef sizes,
      at::IntArrayRef strides) const {
    auto cloned = clone();
    auto ssizes = SymbolicShape(sizes);
    cloned->sizes_ = ssizes;
    cloned->strides_ = computeStrideProps(sizes, strides);
    return cloned;
  }

  TensorTypePtr withSymbolicShapes(SymbolicShape ssizes) const {
    auto cloned = clone();
    cloned->sizes_ = std::move(ssizes);
    return cloned;
  }

  TensorTypePtr withSizes(at::IntArrayRef sizes) const {
    return withSizesStrides(
        sizes, contiguousStridesOf(sizes));
  }

  TensorTypePtr dimensionedOnly() const {
    auto copy = clone();
    copy->sizes_ = SymbolicShape(sizes().size());
    copy->strides_ = VaryingShape<Stride>(sizes().size());
    return copy;
  }

  TensorTypePtr contiguous() const {
    auto cloned = clone();
    TORCH_INTERNAL_ASSERT(sizes().concrete_sizes().has_value());
    auto strides = computeStrideProps(
        *sizes().concrete_sizes(),
        contiguousStridesOf(*sizes().concrete_sizes()));
    cloned->strides_ = strides;
    return cloned;
  }

  const SymbolicShape& symbolic_sizes() const;

  TensorTypePtr merge(const TensorType& other, bool merge_sizes = true) const;

  bool matchTensor(const at::Tensor& t);

  // is all information about the type specified except for autograd?
  // This replaces the notion of a 'CompleteTensorType' that used to exist
  // in the type-hierarchy. Excluding require_grad and undefined allows
  // this to match the old behavior.
  bool isComplete() const {
    return scalar_type_ && device_ && sizes_.isComplete() && strides_.isComplete();
  }

  bool isInferredType() const {
    return is_inferred_;
  }

  static TensorTypePtr getInferred() {
    static auto valueInferred = TensorType::create(
        /*scalar_type=*/{},
        /*device=*/{},
        /*sizes=*/SymbolicShape(),
        /*stride=*/VaryingShape<Stride>{},
        /*requires_grad=*/{},
        /*undefined=*/false);
    valueInferred->is_inferred_ = true;
    return valueInferred;
  }

  // this property is used by GuardElimination
  // please see `checkInputs` for more details
  bool isSummarized() const {
    return !(isComplete() && requiresGrad().has_value() &&
             undefined().has_value());
  }

  TensorTypePtr withUndefined() {
    auto r = clone();
    r->undefined_ = true;
    return r;
  }

  TensorTypePtr withPossiblyUndefined() {
    auto r = clone();
    r->undefined_ = c10::nullopt;
    return r;
  }

  c10::optional<bool> undefined() const { return undefined_; }

  static TensorTypePtr get();

  static const TypeKind Kind = TypeKind::TensorType;

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

 private:
  TensorType(
      c10::optional<at::ScalarType> scalar_type,
      c10::optional<Device> device,
      const SymbolicShape& sizes,
      const VaryingShape<Stride>& strides,
      c10::optional<bool> requires_grad,
      c10::optional<bool> undefined = false);

  TensorTypePtr clone() const {
    return TensorTypePtr(new TensorType(
        scalar_type_, device_, sizes_, strides_, requires_grad_, undefined_));
  }

  static VaryingShape<Stride> computeStrideProps(
      at::IntArrayRef sizes,
      at::IntArrayRef strides,
      bool tensor_contiguity = false);

  c10::optional<at::ScalarType> scalar_type_;
  c10::optional<at::Device> device_;
  SymbolicShape sizes_;
  VaryingShape<Stride> strides_;
  c10::optional<bool> requires_grad_;
  // we exploit the fact certain tensors must be zero in the autograd to
  // optimize gradient computation. Such zero tensors are currently implemented
  // with `UndefinedTensorImpl.` They can be handled only by special operators
  // (e.g. `AutogradAdd`) and their `Tensor::defined()` property returns false.
  // Normally, `undefined_` is set to false, unless a type was created
  // with `withUndefined`
  // This will also mean that `undefined` tensors will fail
  // `subtypeOf(TensorType::get())` check
  // undefined_ may become `c10::nullopt` if the tensor was observed to be both
  // defined and undefined. However, no tensor type starts out with
  // `undefined_` set to `c10::nullopt`
  c10::optional<bool> undefined_;
  // Represents whether or not this type was inferred.
  bool is_inferred_ = false;
};

struct ListType;
using ListTypePtr = std::shared_ptr<ListType>;
struct TORCH_API ListType
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
  TypePtr createWithContained(
      std::vector<TypePtr> contained_types) const override {
    return create(contained_types.at(0));
  }

  bool isSubtypeOfExt(const TypePtr& rhs, std::ostream* why_not) const override;

  // common cast List[Tensor]
  static ListTypePtr ofTensors();
  static ListTypePtr ofOptionalTensors();
  static ListTypePtr ofInts();
  static ListTypePtr ofFloats();
  static ListTypePtr ofComplexDoubles();
  static ListTypePtr ofBools();
  static ListTypePtr ofStrings();

 private:
  ListType(TypePtr elem) : SingleElementType(elem) {}

  std::string annotation_str_impl(TypePrinter printer = nullptr) const override {
    std::stringstream ss;
    ss << "List[" << getElementType()->annotation_str(printer) << "]";
    return ss.str();
  }
};

struct DictType;
using DictTypePtr = std::shared_ptr<DictType>;
struct TORCH_API DictType : public Type {
  friend struct Type;
  static const TypeKind Kind = TypeKind::DictType;

  static DictTypePtr create(TypePtr key, TypePtr value) {
    switch (key->kind()) {
      case TypeKind::AnyType:
      case TypeKind::IntType:
      case TypeKind::BoolType:
      case TypeKind::FloatType:
      case TypeKind::ComplexType:
      case TypeKind::StringType:
      case TypeKind::TensorType:
        return DictTypePtr(new DictType(key, value));
      default:
        AT_ERROR(
            "Cannot create dict for key type '",
            key->str(),
            "', only int, float, complex, Tensor and string keys are supported");
    }
  }

  // aligned with the format in FunctionSchema
  std::string str() const override {
    std::stringstream ss;
    ss << "Dict(" << getKeyType()->str() << ", " << getValueType()->str()
       << ")";
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

  std::string annotation_str_impl(TypePrinter printer = nullptr) const override {
    std::stringstream ss;
    ss << "Dict[" << getKeyType()->annotation_str(printer) << ", "
       << getValueType()->annotation_str(printer) << "]";
    return ss.str();
  }

  std::vector<TypePtr> types;
  bool has_free_variables;
};

struct FutureType;
using FutureTypePtr = std::shared_ptr<FutureType>;

struct TORCH_API FutureType
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
  TypePtr createWithContained(
      std::vector<TypePtr> contained_types) const override {
    return create(contained_types.at(0));
  }

  bool isSubtypeOfExt(const TypePtr& rhs, std::ostream* why_not) const override {
    if (Type::isSubtypeOfExt(rhs, why_not)) {
      return true;
    }
    if (auto rhs_ = rhs->cast<FutureType>()) {
      return getElementType()->isSubtypeOfExt(rhs_->getElementType(), why_not);
    }
    return false;
  }

 private:
  FutureType(TypePtr elem) : SingleElementType(elem) {}

  std::string annotation_str_impl(TypePrinter printer = nullptr) const override {
    std::stringstream ss;
    ss << "Future[" << getElementType()->annotation_str(printer) << "]";
    return ss.str();
  }
};

struct RRefType;
using RRefTypePtr = std::shared_ptr<RRefType>;

struct TORCH_API RRefType
    : public SingleElementType<TypeKind::RRefType, RRefType> {
  friend struct Type;
  template <typename... T>
  static RRefTypePtr create(TypePtr elem) {
    return RRefTypePtr(
        new RRefType(std::move(elem))); // NOLINT(modernize-make-shared)
  }

  std::string str() const override {
    std::stringstream ss;
    ss << "RRef(" << getElementType()->str() << ")";
    return ss.str();
  }
  TypePtr createWithContained(
      std::vector<TypePtr> contained_types) const override {
    return create(contained_types.at(0));
  }

 private:
  RRefType(TypePtr elem) : SingleElementType(elem) {}

  std::string annotation_str_impl(TypePrinter printer = nullptr) const override {
    std::stringstream ss;
    ss << "RRef[" << getElementType()->annotation_str(printer) << "]";
    return ss.str();
  }
};


struct NamedType;
using NamedTypePtr = std::shared_ptr<NamedType>;
using ConstNamedTypePtr = std::shared_ptr<const NamedType>;

struct TORCH_API NamedType : public Type {
  NamedType(TypeKind tk, c10::optional<QualifiedName> name)
      : Type(tk), name_(std::move(name)) {
    TORCH_INTERNAL_ASSERT(
        tk == TypeKind::TupleType || tk == TypeKind::FunctionType ||
        tk == TypeKind::ClassType || tk == TypeKind::InterfaceType ||
        tk == TypeKind::EnumType,
        "If you add a new kind of NamedType, ",
        "please update the cast<NamedType> specialization and this assert");
  }

  // Fully qualified name of type
  // Looks like: "foo.bar.Baz".
  const c10::optional<QualifiedName>& name() const {
    return name_;
  }
private:
  c10::optional<QualifiedName> name_;
};

// Any should never appear in a named type like a class, namedtuple or
// interface. If it does, then dynamic type information will be lost in the
// Pickler, leading to hard-to-track-down bugs that will only occur
// after saving or loading a model. This is because we rely on the
// static types in named types to reconstruct type tags of loaded
// values. Lifting this restriction requires solving the serialization
// problem first.
TORCH_API void checkNoAny(
    const Type& base,
    const char* what,
    const std::string& attrname,
    const TypePtr& attrtype);

struct TupleType;
using TupleTypePtr = std::shared_ptr<TupleType>;
using NameList = std::vector<std::string>;
// This type represents a Tuple
struct TORCH_API TupleType : public NamedType {

  static TupleTypePtr createNamed(const c10::optional<c10::QualifiedName>& name,
      const std::vector<std::string>& field_names,
      const std::vector<TypePtr>& field_types,
      std::vector<IValue>& field_defaults);

  static TupleTypePtr createNamed(const c10::optional<c10::QualifiedName>& name,
      const std::vector<std::string>& field_names,
      const std::vector<TypePtr>& field_types);

  static TupleTypePtr create(
      std::vector<TypePtr> types) {
    return TupleTypePtr(new TupleType(
        std::move(types),
        c10::nullopt,
        nullptr)); // NOLINT(modernize-make-shared)
  }
  static TupleTypePtr create() {
    return create({});
  }

  at::ArrayRef<TypePtr> elements() const {
    return elements_;
  }

  bool operator==(const Type& rhs) const override;
  bool isSubtypeOfExt(const TypePtr& rhs_, std::ostream* why_not) const override;

  std::string str() const override;
  bool hasFreeVariables() const override {
    return has_free_variables_;
  }
  at::ArrayRef<TypePtr> containedTypes() const override {
    return elements_;
  }
  TypePtr createWithContained(
      std::vector<TypePtr> contained_types) const override {
    return std::shared_ptr<TupleType>(
        new TupleType(std::move(contained_types), name(), schema()));
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
    const auto& r_elements = rhs.castRaw<TupleType>()->elements();
    if (l_elements.size() != r_elements.size())
      return false;
    for (size_t i = 0; i < l_elements.size(); ++i) {
      if (!fn(l_elements[i], r_elements[i]))
        return false;
    }
    return true;
  }

  std::string annotation_str_impl(TypePrinter printer = nullptr) const override;

  std::vector<TypePtr> elements_;
  bool has_free_variables_;
  std::shared_ptr<FunctionSchema> schema_;
};

struct EnumType;
using EnumTypePtr = std::shared_ptr<EnumType>;
using EnumNameValue = std::pair<std::string, IValue>;
struct TORCH_API EnumType : public NamedType {
  friend struct Type;
  static const TypeKind Kind = TypeKind::EnumType;

  static EnumTypePtr create(
      const c10::QualifiedName& qualified_class_name,
      TypePtr value, std::vector<EnumNameValue> enum_names_values, std::weak_ptr<::torch::jit::CompilationUnit> cu) {
    switch (value->kind()) {
      case TypeKind::IntType:
      case TypeKind::FloatType:
      case TypeKind::StringType:
        return EnumTypePtr(new EnumType(qualified_class_name, std::move(value), std::move(enum_names_values), std::move(cu)));
      default:
        AT_ERROR(
            "Cannot create Enum with value type '",
            value->str(),
            "', only int, float and string are supported");
    }
  }

  std::string str() const override {
    return "Enum<" + annotation_str() + ">";
  }

  std::string repr_str() const override {
    return str();
  }

  TypePtr getValueType() const {
    return value_type_;
  }

  bool operator==(const Type& rhs) const override {
    if (auto enum_rhs = rhs.cast<EnumType>()) {
      return name().value() == enum_rhs->name().value() &&
          *getValueType() == *(enum_rhs->getValueType()) &&
          this->compilation_unit() == enum_rhs->compilation_unit();
    }
    return false;
  }

  bool isSubtypeOfExt(const TypePtr& rhs, std::ostream* why_not) const override;

  std::shared_ptr<const ::torch::jit::CompilationUnit> compilation_unit() const {
    auto cu = cu_.lock();
    return cu;
  }

  const QualifiedName qualifiedClassName() const {
    return name().value();
  }

  at::ArrayRef<TypePtr> containedTypes() const override {
    return value_type_;
  }

  const at::ArrayRef<EnumNameValue> enumNamesValues() const {
    return enum_names_values_;
  }

 private:
  EnumType(
      c10::QualifiedName qualified_class_name,
      TypePtr value_type,
      std::vector<EnumNameValue> enum_names_values,
      std::weak_ptr<torch::jit::CompilationUnit> cu)
      : NamedType(TypeKind::EnumType, std::move(qualified_class_name)),
        value_type_(std::move(value_type)),
        enum_names_values_(std::move(enum_names_values)),
        cu_(cu) {}

  std::string annotation_str_impl(TypePrinter printer = nullptr) const override {
    const auto& n = name().value();
    return n.qualifiedName();
  }

  TypePtr value_type_;
  std::vector<EnumNameValue> enum_names_values_;
  std::weak_ptr<::torch::jit::CompilationUnit> cu_;
};


// the common supertype of all Enums, only used in operator registraion.
// EnumType <: AnyEnumType for all Enums
struct AnyEnumType;
using AnyEnumTypePtr = std::shared_ptr<AnyEnumType>;
struct TORCH_API AnyEnumType : public Type {
  bool operator==(const Type& rhs) const override {
    return rhs.kind() == kind();
  }
  std::string str() const override {
    return "AnyEnumType";
  }
  static const TypeKind Kind = TypeKind::AnyEnumType;
  // global singleton
  static AnyEnumTypePtr get();
private:
  AnyEnumType()
  : Type(TypeKind::AnyEnumType) {}
};


struct NumberType;
using NumberTypePtr = std::shared_ptr<NumberType>;
// This type represents a Python number
// Subtype hierarchy for Number Types (NumberType as the base type):
// IntType <: NumberType
// FloatType <: NumberType
// ComplexType <:NumberType
struct TORCH_API NumberType : public Type {
  bool operator==(const Type& rhs) const override {
    return rhs.kind() == kind();
  }
  std::string str() const override {
    return "Scalar"; // match what PythonArgParser says for clarity
  }
  static const TypeKind Kind = TypeKind::NumberType;
  // global singleton
  static NumberTypePtr get();

 protected:
  NumberType(TypeKind kind = TypeKind::NumberType) : Type(kind) {}

  std::string annotation_str_impl(TypePrinter printer = nullptr) const override {
    return "number"; // technically not a valid python type, but
                     // we need to use it when parsing back in annotations
                     // for implicit conversions
  }
};

struct FloatType;
using FloatTypePtr = std::shared_ptr<FloatType>;
// This type represents a Python float number
struct TORCH_API FloatType : public NumberType {
  bool operator==(const Type& rhs) const override {
    return rhs.kind() == kind();
  }
  std::string str() const override {
    return "float";
  }
  bool isSubtypeOfExt(const TypePtr& rhs, std::ostream* why_not) const override {
    return rhs->kind() == TypeKind::NumberType || NumberType::isSubtypeOfExt(rhs, why_not);
  }
  static const TypeKind Kind = TypeKind::FloatType;
  // global singleton
  static FloatTypePtr get();

 private:
  FloatType() : NumberType(TypeKind::FloatType) {}
  std::string annotation_str_impl(TypePrinter printer = nullptr) const override {
    return "float";
  }
};

struct ComplexType;
using ComplexTypePtr = std::shared_ptr<ComplexType>;
// This type represents a Python float number
struct TORCH_API ComplexType : public NumberType {
  bool operator==(const Type& rhs) const override {
    return rhs.kind() == kind();
  }
  std::string str() const override {
    return "complex";
  }
  bool isSubtypeOfExt(const TypePtr& rhs, std::ostream* why_not) const override {
    return rhs->kind() == TypeKind::NumberType || NumberType::isSubtypeOfExt(rhs, why_not);
  }
  static const TypeKind Kind = TypeKind::ComplexType;
  // global singleton
  static ComplexTypePtr get();

 private:
  ComplexType() : NumberType(TypeKind::ComplexType) {}
  std::string annotation_str_impl(TypePrinter printer = nullptr) const override {
    return "complex";
  }
};

struct IntType;
using IntTypePtr = std::shared_ptr<IntType>;
// This type represents a Python int number
struct TORCH_API IntType : public NumberType {
  bool operator==(const Type& rhs) const override {
    return rhs.kind() == kind();
  }
  std::string str() const override {
    return "int";
  }
  bool isSubtypeOfExt(const TypePtr& rhs, std::ostream* why_not) const override {
    return rhs->kind() == TypeKind::NumberType || NumberType::isSubtypeOfExt(rhs, why_not);
  }
  static const TypeKind Kind = TypeKind::IntType;
  // global singleton
  static IntTypePtr get();

 private:
  IntType() : NumberType(TypeKind::IntType) {}
  std::string annotation_str_impl(TypePrinter printer = nullptr) const override {
    return "int";
  }
};

struct BoolType;
using BoolTypePtr = std::shared_ptr<BoolType>;
// This node represents a Python bool value
struct TORCH_API BoolType : public Type {
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
struct TORCH_API StringType : public Type {
  bool operator==(const Type& rhs) const override {
    return rhs.kind() == kind();
  }
  std::string str() const override {
    // we only use "str" (not "string") in both FunctionSchema and script
    return annotation_str();
  }
  std::string annotation_str_impl(TypePrinter printer = nullptr) const override {
    return "str";
  }
  static const TypeKind Kind = TypeKind::StringType;
  // global singleton
  static StringTypePtr get();

 private:
  StringType() : Type(TypeKind::StringType) {}
};

struct StorageType;
using StorageTypePtr = std::shared_ptr<StorageType>;
struct TORCH_API StorageType : public Type {
  bool operator==(const Type& rhs) const override {
    return rhs.kind() == kind();
  }
  std::string str() const override {
    return annotation_str();
  }
  std::string annotation_str_impl(TypePrinter printer = nullptr) const override {
    return "Storage";
  }
  static const TypeKind Kind = TypeKind::StorageType;
  // global singleton
  static StorageTypePtr get();

 private:
  StorageType() : Type(TypeKind::StorageType) {}
};

struct FunctionType;
using FunctionTypePtr = std::shared_ptr<FunctionType>;
struct TORCH_API FunctionType : public NamedType {
  static FunctionTypePtr create(torch::jit::Function* function) {
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
  torch::jit::Function* function() const {
    return function_;
  }
  static const TypeKind Kind = TypeKind::FunctionType;

 private:
  FunctionType(torch::jit::Function* function);
  std::string annotation_str_impl(TypePrinter printer = nullptr) const override {
    const auto& n = name().value();
    return n.qualifiedName();
  }
  torch::jit::Function* function_;
};

struct NoneType;
using NoneTypePtr = std::shared_ptr<NoneType>;
// This type represents a Python None
struct TORCH_API NoneType : public Type {
  bool operator==(const Type& rhs) const override {
    return rhs.kind() == kind();
  }
  std::string str() const override {
    return "NoneType";
  }
  bool isSubtypeOfExt(const TypePtr& rhs, std::ostream *why_not) const override {
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
struct TORCH_API GeneratorType : public Type {
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

struct QuantizerType;
using QuantizerTypePtr = std::shared_ptr<QuantizerType>;
// This type represents a Quantizer
struct TORCH_API QuantizerType : public Type {
  bool operator==(const Type& rhs) const override {
    return rhs.kind() == kind();
  }
  std::string str() const override {
    return "Quantizer";
  }
  static const TypeKind Kind = TypeKind::QuantizerType;
  // global singleton
  static QuantizerTypePtr get();

 private:
  QuantizerType() : Type(TypeKind::QuantizerType) {}
};

struct QSchemeType;
using QSchemeTypePtr = std::shared_ptr<QSchemeType>;
// This type represents a QScheme
struct TORCH_API QSchemeType : public Type {
  bool operator==(const Type& rhs) const override {
    return rhs.kind() == kind();
  }
  std::string str() const override {
    return "QScheme";
  }
  static const TypeKind Kind = TypeKind::QSchemeType;
  // global singleton
  static QSchemeTypePtr get();

 private:
  QSchemeType() : Type(TypeKind::QSchemeType) {}
};

struct DeviceObjType;
using DeviceObjTypePtr = std::shared_ptr<DeviceObjType>;
// This type represents a Device
struct TORCH_API DeviceObjType : public Type {
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

struct StreamObjType;
using StreamObjTypePtr = std::shared_ptr<StreamObjType>;
// This type represents a Generator
struct TORCH_API StreamObjType : public Type {
  bool operator==(const Type& rhs) const override {
    return rhs.kind() == kind();
  }
  std::string str() const override {
    return "Stream";
  }
  static const TypeKind Kind = TypeKind::StreamObjType;
  // global singleton
  static StreamObjTypePtr get();

private:
  StreamObjType() : Type(TypeKind::StreamObjType) {}
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
// This type represents a Python Capsule.
// It does not appear in the IR and is only used during runtime
struct TORCH_API CapsuleType : public Type {
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

struct PyObjectType;
using PyObjectTypePtr = std::shared_ptr<PyObjectType>;
// This type represents a PyObject Type
struct TORCH_API PyObjectType : public Type {
  bool operator==(const Type& rhs) const override {
    return rhs.kind() == kind();
  }
  std::string str() const override {
    return "PyObject";
  }
  static const TypeKind Kind = TypeKind::PyObjectType;
  // global singleton
  static PyObjectTypePtr get();
private:
  PyObjectType()
  : Type(TypeKind::PyObjectType) {}
};

enum class TypeVerbosity {
  None,
  Type,
  TypeAndStride,
  Full,
  Symbolic,
  Default = Full,
};

TORCH_API TypeVerbosity type_verbosity();

TORCH_API std::ostream& operator<<(std::ostream& out, const Type& t);
template <typename T>
TORCH_API std::ostream& operator<<(
    std::ostream& out,
    const VaryingShape<T>& t);
TORCH_API std::ostream& operator<<(std::ostream& os, const SymbolicShape& s);
TORCH_API std::ostream& operator<<(std::ostream& os, const ShapeSymbol& s);
TORCH_API std::ostream& operator<<(std::ostream& os, const Stride& s);
// what is the type, ignoring extra size/shape information?
// e.g. Tensor(2x3) -> Dynamic, and Tuple(Tensor(2x3),...) -> Tuple(Dynamic,...)

// xxx: be careful with calls because this can be very slow. If calling this on a graph
// use `EraseShapeInformation` in shape_analysis.h
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
    return TensorType::createContiguous(at::kDouble, at::kCPU, {});
  } else if (typ->isSubtypeOf(BoolType::get())) {
    return TensorType::createContiguous(at::kBool, at::kCPU, {});
  } else if (typ->kind() == NumberType::Kind) {
    return TensorType::create(c10::nullopt, at::kCPU, {}, c10::nullopt);
  }
  TORCH_CHECK(false, "Unknown number type: ", typ->str());
}
inline TypePtr TensorType::fromBoolType() {
  return TensorType::createContiguous(at::kBool, at::kCPU, {});
}

inline c10::optional<c10::ScalarType> tryScalarTypeFromJitType(const c10::TypePtr & type) {
  if (type == FloatType::get()) {
    return at::typeMetaToScalarType(c10::get_default_dtype());
  } else if (type == IntType::get()) {
    return at::ScalarType::Long;
  } else if (type == BoolType::get()) {
    return at::ScalarType::Bool;
  }
  return c10::nullopt;
}

inline at::ScalarType scalarTypeFromJitType(const c10::TypePtr& type) {
  auto result = tryScalarTypeFromJitType(type);
  TORCH_CHECK(
      result,
      "Add new condition, expected Float, Complex, Int, or Bool but got",
      type->str());
  return *result;
}

// Attempt to find the correct supertype of t1 and t2. If none is found then
// nullopt will be returned if default_to_any is false, and Any will be returned
// if it is true. If t1 == t2, or t1 is a type refinement of t2,
// then t2 will be returned (and vice versa).
// Two different tensortypes will return dynamic.
// Currently we chose not to support returning a NumberType for a float & int
// input because of a lack of operator support for NumberType.
// If `type_hint` is an `InterfaceType`, then we can use that as a
// potential supertype for `ClassType`s in the list. Otherwise, we have
// no way to find and use some common interface type
TORCH_API c10::optional<TypePtr> unifyTypes(
    const TypePtr& t1,
    const TypePtr& t2,
    bool default_to_any = false,
    TypePtr type_hint=nullptr);

TORCH_API c10::optional<TypePtr> unifyTypeList(
    at::ArrayRef<TypePtr> elements,
    std::ostream& why_not,
    bool default_to_any=false,
    TypePtr type_hint=nullptr);

namespace detail {
template <typename T>
struct getTypePtr_ final {
  static TypePtr call() {
    TypePtr res = []() {
      try {
        return getCustomClassType<T>();
      } catch(const c10::Error&) {
        TORCH_CHECK(
            false,
            "Type ",
            c10::util::get_fully_qualified_type_name<T>(),
            " could not be converted to any of the known types."
        );
      }
    }();
    return std::dynamic_pointer_cast<Type>(std::move(res));
  }
};

template <>
struct getTypePtr_<at::IValue> final {
  static TypePtr call() {
    return AnyType::get();
  }
};

template <>
struct getTypePtr_<at::Tensor> final {
  static TypePtr call() {
    return TensorType::get();
  }
};
template <>
struct getTypePtr_<c10::Storage> final {
  static TypePtr call() {
    return StorageType::get();
  }
};
template <>
struct getTypePtr_<c10::Stream> final {
  static TypePtr call() {
    return StreamObjType::get();
  }
};
template <>
struct getTypePtr_<double> final {
  static TypePtr call() {
    return FloatType::get();
  }
};
template <>
struct getTypePtr_<c10::complex<double>> final {
  static TypePtr call() {
    return ComplexType::get();
  }
};
template <>
struct getTypePtr_<int64_t> final {
  static TypePtr call() {
    return IntType::get();
  }
};
template <>
struct getTypePtr_<c10::ScalarType> final {
  static TypePtr call() {
    return IntType::get();
  }
};
template <>
struct getTypePtr_<c10::Device> final {
  static TypePtr call() {
    return DeviceObjType::get();
  }
};
template <>
struct getTypePtr_<c10::Layout> final {
  static TypePtr call() {
    return IntType::get();
  }
};
template <>
struct getTypePtr_<c10::MemoryFormat> final {
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
struct getTypePtr_<c10::QScheme> final {
  static TypePtr call() {
    return QSchemeType::get();
  }
};
template <>
struct getTypePtr_<at::Generator> final {
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
template <>
struct getTypePtr_<c10::string_view> final {
  static TypePtr call() {
    return StringType::get();
  }
};
template <>
struct getTypePtr_<at::Dimname> final {
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
template <class... Contained>
struct getTypePtr_<std::tuple<Contained...>> final {
  static TypePtr call() {
    std::vector<TypePtr> contained_types = {
      (getTypePtr_<Contained>::call())...
    };
    return TupleType::create(std::move(contained_types));
  }
};
template <>
struct getTypePtr_<void> final {
  static TypePtr call() {
    return NoneType::get();
  }
};
} // namespace detail
template <class T>
inline TypePtr getTypePtr() {
  // TODO: static_assert that a templated function exists, and throw a friendly
  // error message if not
  return detail::getTypePtr_<T>::call();
}

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
TORCH_API MatchTypeReturn
matchTypeVariables(TypePtr formal, TypePtr actual, TypeEnv& type_env);

// replace type variables appearing in `type` with the values in
// `type_env`. Returns nullptr if a variable used in `type`
// does not appear in `type_env`
TORCH_API TypePtr tryEvalTypeVariables(TypePtr type, TypeEnv& type_env);

TORCH_API bool elementTypeCanBeInferredFromMembers(const TypePtr& elem_type);

// This enumerator represents the 'kind' of an attribute - a buffer, a paramter, or neither.
// This state is mutually exclusive. Buffers and Parameters can only appear on modules.
enum class AttributeKind {
  BUFFER,
  PARAMETER,
  REGULAR_ATTRIBUTE
};

// This structure represents all notional booking entities in a class attribute: name, kind (see: AttributeKind), and type (see: TypePtr).
// Note: This structure does not represent the value of the attribute.
struct TORCH_API ClassAttribute {
  public:
  ClassAttribute(AttributeKind kind,
  TypePtr attributeType,
  std::string attributeName) :
    kind_(kind),
    attributeType_(attributeType),
    attributeName_(std::move(attributeName)) {}

  AttributeKind getKind() const {
    return kind_;
  }

  TypePtr getType() const {
    return attributeType_;
  }

  const std::string& getName() const {
    return attributeName_;
  }

  private:
  AttributeKind kind_;
  TypePtr attributeType_;
  std::string attributeName_;
};

/**
 * User Defined Types
 */

struct ClassType;
using ClassTypePtr = std::shared_ptr<ClassType>;
using ::torch::jit::CompilationUnit;

// This represents a class in TorchScript.
struct TORCH_API ClassType : public NamedType {
  // This represents an attribute of a class; a name associated with an attribute, and a
  // getter and (optional) setter for that attribute.
  struct Property {
    std::string name;
    torch::jit::Function* getter;
    torch::jit::Function* setter;
  };

  // Create a class type with name `name` and its methods stored in `cu`.
  static ClassTypePtr create(
      c10::optional<QualifiedName> qualifiedName,
      std::weak_ptr<CompilationUnit> cu,
      bool is_module = false,
      std::string doc_string = "",
      std::vector<std::string> unresolved_class_attributes = {});

  bool operator==(const Type& rhs) const override {
    if (auto user_rhs = rhs.cast<ClassType>()) {
      const auto& lhs_name = name().value();
      const auto& rhs_name = user_rhs->name().value();

      return lhs_name == rhs_name &&
          this->compilation_unit() == user_rhs->compilation_unit();
    }
    return false;
  }

  std::string str() const override {
     return annotation_str();
  }

  std::string repr_str() const override {
    std::stringstream ss;
    ss << str()
       << " (of Python compilation unit at: " << compilation_unit().get() << ")";
    return ss.str();
  }

  const std::vector<torch::jit::Function*>& methods() const;

  TypePtr findAttribute(const std::string& name) const {
    size_t pos = 0;
    for (const auto& attr : attributes_) {
      if (name == attr.getName()) {
        break;
      }
      ++pos;
    }

    if (pos >= attributes_.size()) {
      return nullptr;
    }
    return attributes_[pos].getType();
  }

  TypePtr getAttribute(const std::string& name) const {
    auto type = findAttribute(name);
    TORCH_CHECK(
        type,
        repr_str(),
        " does not have an attribute with name '",
        name,
        "'");
    return type;
  }

  size_t numAttributes() const {
    return attributes_.size();
  }

  const TypePtr getAttribute(size_t slot) const {
    AT_ASSERT(slot < attributes_.size());
    return attributes_.at(slot).getType();
  }

  const std::string getAttributeName(size_t slot) const {
    AT_ASSERT(slot < attributes_.size());
    return attributes_[slot].getName();
  }

  void checkNotExist(const std::string& name, const std::string& what) const;

  // Attributes are stored in a specific slot at runtime for effiency.
  // When emitting instructions we specify the slot so that attribute access is
  // a constant lookup
  c10::optional<size_t> findAttributeSlot(const std::string& name) const {
    size_t slot = 0;
    for (const auto& attr : attributes_) {
      if (name.compare(attr.getName()) == 0) {
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
        repr_str(),
        " does not have an attribute with name '",
        name,
        "'");
  }

  bool hasAttribute(const std::string& name) const {
    return std::find_if(
               attributes_.cbegin(),
               attributes_.cend(),
               [&](const ClassAttribute& attr) { return attr.getName() == name; }) !=
        attributes_.cend();
  }

  bool isUnresolvedClassAttribute(const std::string& name) const;

  at::ArrayRef<TypePtr> containedTypes() const override {
    return attributeTypes_;
  }

  size_t addAttribute(
      const std::string& name,
      const TypePtr& type,
      bool is_parameter = false,
      bool is_buffer = false);

  // [Internal Only] Remove attribute from the ClassType,
  // caller is responsible to make sure the modification is safe:
  // it is unsafe to having existing allocations
  // of this object around anymore, and any code that works on
  // the attribute is now invalid. Only newly created code is
  // valid again.
  void unsafeRemoveAttribute(const std::string& name);

  // [Internal Only] Change the type of an attribute of the ClassType,
  // The caller is responsible to make sure the modification is safe:
  // it is unsafe to maintain uses of the old type of the attribute,
  // and any code that works on the attribute is now invalid.
  // Only newly created code is valid again.
  void unsafeChangeAttributeType(const std::string& name, TypePtr new_ty);

  // Add attribute \p NAME if it doesn't exist or verify that it has a
  // compatible type otherwise.
  size_t addOrCheckAttribute(
      const std::string& name,
      TypePtr ty,
      bool is_parameter = false,
      bool is_buffer = false) {
    auto slot_idx = findAttributeSlot(name);
    if (!slot_idx) {
      return addAttribute(name, ty, is_parameter, is_buffer);
    }

    TORCH_CHECK(
        is_parameter == this->is_parameter(*slot_idx),
        "Parameter field mismatch for the field '",
        name,
        "'");
    TypePtr atype = getAttribute(*slot_idx);
    TORCH_CHECK(
      ty->isSubtypeOf(atype),
      ty->repr_str(),
      " is not compatible with the type ",
      atype->repr_str(),
      " for the field '",
      name,
      "'");
    return *slot_idx;
  }

  // Get the property with the given \p name, if it exists on the class.
  c10::optional<ClassType::Property> getProperty(const std::string& name);
  // Add a property named \p name with \p getter and \p setter as its getter and setter.
  void addProperty(const std::string& name, torch::jit::Function* getter, torch::jit::Function* setter);
  // Get a list of all properties.
  const std::vector<Property>& properties() const {
    return properties_;
  }

  bool hasConstant(const std::string& name) const {
    return std::find_if(
               constantNames_.cbegin(),
               constantNames_.cend(),
               [&](const std::string& constant) { return constant == name; }) !=
        constantNames_.cend();
  }

  size_t addConstant(const std::string& name, const IValue& value);

  c10::optional<size_t> findConstantSlot(const std::string& name) const {
    TORCH_CHECK(constantNames_.size() == constantValues_.size());
    size_t slot = 0;
    for (const auto& constant : constantNames_) {
      if (name == constant) {
        return slot;
      }
      slot++;
    }
    return c10::nullopt;
  }

  size_t getConstantSlot(const std::string& name) const {
    if (auto r = findConstantSlot(name)) {
      return *r;
    }
    TORCH_CHECK(
        false,
        repr_str(),
        " does not have constant field with the name '",
        name,
        "'");
  }

  const std::string& getConstantName(size_t slot) const {
    TORCH_CHECK(constantNames_.size() == constantValues_.size());
    TORCH_CHECK(slot < constantNames_.size());
    return constantNames_[slot];
  }

  const std::string& doc_string() const {
    return doc_string_;
  }

  IValue getConstant(const std::string& name) const;

  IValue getConstant(size_t slot) const;

  c10::optional<IValue> findConstant(const std::string& name) const;

  size_t numConstants() const {
    TORCH_INTERNAL_ASSERT(constantNames_.size() == constantValues_.size());
    return constantNames_.size();
  }

  at::ArrayRef<std::string> constantNames() const {
    return constantNames_;
  }

  at::ArrayRef<IValue> constantValues() const {
    return constantValues_;
  }

  // [Internal Only] Remove constant from the ClassType
  // caller is responsible to make sure the modification is safe:
  // it is unsafe to having existing allocations
  // of this object around anymore, and any code that works on
  // the attribute is now invalid. Only newly created code is
  // valid again.
  void unsafeRemoveConstant(const std::string& name);

  TypePtr createWithContained(std::vector<TypePtr> contained_types) const override {
    auto ptr = ClassType::create(name(), compilation_unit_, is_module());
    AT_ASSERT(numAttributes() == contained_types.size());
    for(size_t i = 0; i < attributes_.size(); ++i) {
      AT_ASSERT(attributes_[i].getType()->isSubtypeOf(contained_types[i]));
      ptr->addAttribute(attributes_[i].getName(), contained_types[i]);
    }
    // Copy methods over
    for (const auto& method : methods()) {
      ptr->addMethod(method);
    }
    return ptr;
  }

  bool is_module() const override {
    return isModule_;
  }

  const std::vector<ClassAttribute>& getAttributes() const {
    return attributes_;
  }

  bool is_parameter(size_t slot) const {
    TORCH_INTERNAL_ASSERT(
        is_module(), "asking for parameterSlots of non-Module");
    return attributes_.at(slot).getKind() == AttributeKind::PARAMETER;
  }

  bool is_buffer(size_t slot) const {
    TORCH_INTERNAL_ASSERT(
        is_module(), "asking for bufferWrittenSlots of non-Module");
    return attributes_.at(slot).getKind() == AttributeKind::BUFFER;
  }

  void addForwardPreHook(torch::jit::Function* pre_hook_ptr);
  void addForwardHook(torch::jit::Function* hook_ptr);
  torch::jit::Function* findForwardPreHook(const std::string& name) const;
  torch::jit::Function* findForwardHook(const std::string& name) const;
  const std::vector<torch::jit::Function*>& getForwardHooks() const;
  const std::vector<torch::jit::Function*>& getForwardPreHooks() const;

  void checkForwardPreHookSchema(
      int pre_hook_idx,
      const FunctionSchema& pre_hook_schema) const;
  void checkForwardHookSchema(
      int hook_idx,
      const FunctionSchema& hook_schema) const;

  void addMethod(torch::jit::Function* method);
  torch::jit::Function* findMethod(const std::string& name) const;
  torch::jit::Function& getMethod(const std::string& name) const;
  torch::jit::Function* findHook(const std::string& name) const;
  torch::jit::Function& getHook(const std::string& name) const;
  bool hasMethod(const std::string& name) const;

  torch::jit::Function* findStaticMethod(const std::string& name) const;
  void addStaticMethod(torch::jit::Function* method);

  // [Internal Only] Remove method from the ClassType
  // caller is responsible to make sure the modification is safe:
  // it is unsafe to having existing allocations
  // of this object around anymore, and any code that works on
  // the attribute is now invalid. Only newly created code is
  // valid again.
  // Note this method is intended for freezing only.
  void unsafeRemoveMethod(const std::string& name);

  std::shared_ptr<CompilationUnit> compilation_unit();

  std::shared_ptr<const CompilationUnit> compilation_unit() const;

  // generate a refined version of this class.
  // It has the same name but the slot Types are subtypes of
  // the original slots. It is only valid to refine a class type in a context
  // where it is know that there are not assignments to the objects slots
  // that would invalidate the refinement.
  // These variants are not registered in the global class table.
  ClassTypePtr refine(at::ArrayRef<TypePtr> refined_slots) const;

  bool isSubtypeOfExt(const TypePtr& rhs, std::ostream* why_not) const override;

  static const TypeKind Kind = TypeKind::ClassType;

 private:
  ClassType(
      c10::optional<QualifiedName> name,
      std::weak_ptr<CompilationUnit> cu,
      bool is_module = false,
      std::string doc_string = "",
      std::vector<std::string> unresolved_class_attributes = {});

  std::string annotation_str_impl(TypePrinter printer = nullptr) const override {
    const auto& n = name().value();
    return n.qualifiedName();
  }

  void addAttribute(ClassAttribute classAttribute);
  std::string getForwardPreHookErrorMessage(int pre_hook_idx) const;
  std::string getForwardHookErrorMessage(int hook_idx) const;

  // Mapping of attribute names -> their type.
  // NOTE: this does not contain methods, which are stored in the module
  // TODO: once modules support arbitrary ivalue attributes, we don't need this
  // anymore.
  // TODO: This is better represented as an OrderedDict, but alas it is not yet
  // available from c10

  // Mapping of constant names -> their value.
  std::vector<std::string> constantNames_;
  std::vector<IValue> constantValues_;
  // Holds method attributes
  std::weak_ptr<CompilationUnit> compilation_unit_;

  // Holds all atrributes, attribute details are found on ClassAttribute
  std::vector<ClassAttribute> attributes_;
  // Construct mirroring attributes_, only around due to the fact that `containedTypes()` method returns an ArrayRef.
  // Never fill this without using the appropriate provideNewClassAttribute method
  std::vector<TypePtr> attributeTypes_;

  // List of methods associated with this class.
  std::vector<torch::jit::Function*> methods_;
  std::vector<torch::jit::Function*> staticmethods_;

  // List of hooks to be run before/after forward.
  std::vector<torch::jit::Function*> forward_hooks_;
  std::vector<torch::jit::Function*> forward_pre_hooks_;

  // List of properties exposed by this class.
  std::vector<Property> properties_;

  bool isModule_ = false;

  // Doc string of class.
  std::string doc_string_ = "";

  // For error reporting accesses to class level attributes.
  std::vector<std::string> unresolved_class_attributes_;
};

struct InterfaceType;
using InterfaceTypePtr = std::shared_ptr<InterfaceType>;
using ::torch::jit::CompilationUnit;

// Interfaces are a list of abstract methods that a class might meet.
// If a class provides those methods, it implicitly meets the interface.

// Subtype relations for Interface with ClassType:
// lhs (ClassType or InterfaceType) is a subtype of rhs if:
// 1. lhs methods are a superset of rhs methods
// 2. if rhs is module interface, the lhs must be module interface or module itself
struct TORCH_API InterfaceType : public NamedType {
  static InterfaceTypePtr create(
      QualifiedName qualifiedName, bool is_module=false);

  bool operator==(const Type& rhs) const override {
    if (auto user_rhs = rhs.cast<InterfaceType>()) {
      return isSubTypeImpl(*this, *user_rhs, nullptr) &&
          isSubTypeImpl(*user_rhs, *this, nullptr);
    }
    return false;
  }

  std::string str() const override {
    return std::string("InterfaceType<") + name()->name() + ">";
  }

  bool isSubtypeOfExt(const TypePtr& rhs, std::ostream* why_not) const override;

  // try to find a method of this interface,
  // returns nullptr if not found.
  const FunctionSchema* getMethod(const std::string& name) const;
  void addMethod(FunctionSchema schema);
  const std::vector<FunctionSchema>& methods() {
    return *methods_;
  }

  bool is_module() const override{
    return is_module_;
  }
  static const TypeKind Kind = TypeKind::InterfaceType;
  ~InterfaceType() override;
 private:
  InterfaceType(QualifiedName name, bool is_module);
  static bool isSubTypeImpl(
      const InterfaceType& lhs,
      const InterfaceType& rhs,
      std::ostream* why_not);

  std::string annotation_str_impl(TypePrinter printer = nullptr) const override {
    return name()->qualifiedName();
  }

  // shared_ptr so that this header does not have to depend on
  // FunctionSchema.h
  std::shared_ptr<std::vector<FunctionSchema>> methods_;
  // flag to distinguish if it's an interface type from a module or not
  bool is_module_;
};

template <TypeKind K>
struct EnumerationType : public Type {
static const TypeKind Kind = K;

bool operator==(const Type& rhs) const override {
  return rhs.kind() == kind();
}

protected:
EnumerationType() : Type(Kind) {}
};

struct LayoutType;
using LayoutTypePtr = std::shared_ptr<LayoutType>;
// This type represents a Generator
struct TORCH_API LayoutType : public EnumerationType<TypeKind::LayoutType> {
std::string str() const override {
return "Layout";
}
static const TypeKind Kind = TypeKind::LayoutType;
// global singleton
static LayoutTypePtr get();

private:
LayoutType() : EnumerationType() {}
};

struct ScalarTypeType;
using ScalarTypeTypePtr = std::shared_ptr<ScalarTypeType>;
// This type represents a Generator
struct TORCH_API ScalarTypeType : public EnumerationType<TypeKind::ScalarTypeType> {
std::string str() const override {
return "ScalarType";
}
static const TypeKind Kind = TypeKind::ScalarTypeType;
// global singleton
static ScalarTypeTypePtr get();

private:
ScalarTypeType() : EnumerationType() {}
};

// the common supertype of all lists,
// List[T] <: AnyList for all T
struct AnyListType;
using AnyListTypePtr = std::shared_ptr<AnyListType>;
struct TORCH_API AnyListType : public Type {
  bool operator==(const Type& rhs) const override {
    return rhs.kind() == kind();
  }
  std::string str() const override {
    return "list";
  }
  static const TypeKind Kind = TypeKind::AnyListType;
  // global singleton
  static AnyListTypePtr get();
private:
  AnyListType()
  : Type(TypeKind::AnyListType) {}
};

// the common supertype of all tuples,
// Tuple[T...] <: AnyTuple for all T
struct AnyTupleType;
using AnyTupleTypePtr = std::shared_ptr<AnyTupleType>;
struct TORCH_API AnyTupleType : public Type {
  bool operator==(const Type& rhs) const override {
    return rhs.kind() == kind();
  }

  std::string str() const override {
    return "tuple";
  }
  static const TypeKind Kind = TypeKind::AnyTupleType;

  // global singleton
  static AnyTupleTypePtr get();
private:
  AnyTupleType()
  : Type(TypeKind::AnyTupleType) {}
};

// the common supertype of all classes,
// ClassType <: AnyClassType for all classes
struct AnyClassType;
using AnyClassTypePtr = std::shared_ptr<AnyClassType>;
struct TORCH_API AnyClassType : public Type {
  bool operator==(const Type& rhs) const override {
    return rhs.kind() == kind();
  }
  std::string str() const override {
    return "AnyClassType";
  }
  static const TypeKind Kind = TypeKind::AnyClassType;
  // global singleton
  static AnyClassTypePtr get();
private:
  AnyClassType()
  : Type(TypeKind::AnyClassType) {}
};

inline bool IValue::isDoubleList() const {
  // note: avoids calling type() to avoid extra referencing counting for the returned type.
  return isList() && static_cast<detail::ListImpl*>(payload.u.as_intrusive_ptr)->elementType->kind() == FloatType::Kind;
}

inline bool IValue::isComplexDoubleList() const {
  // note: avoids calling type() to avoid extra referencing counting for the returned type.
  return isList() && static_cast<detail::ListImpl*>(payload.u.as_intrusive_ptr)->elementType->kind() == ComplexType::Kind;
}

inline bool IValue::isTensorList() const {
  return isList() && static_cast<detail::ListImpl*>(payload.u.as_intrusive_ptr)->elementType->kind() == TensorType::Kind;
}

inline bool IValue::isIntList() const {
  return isList() && static_cast<detail::ListImpl*>(payload.u.as_intrusive_ptr)->elementType->kind() == IntType::Kind;
}

inline bool IValue::isBoolList() const {
  return isList() && static_cast<detail::ListImpl*>(payload.u.as_intrusive_ptr)->elementType->kind() == BoolType::Kind;
}

template<>
inline std::shared_ptr<NamedType> Type::cast() {
  if (kind() == TypeKind::TupleType || kind() == TypeKind::FunctionType ||
      kind() == TypeKind::ClassType || kind() == TypeKind::InterfaceType) {
    return std::static_pointer_cast<NamedType>(shared_from_this());
  }
  return nullptr;
}

template<>
inline std::shared_ptr<const NamedType> Type::cast<NamedType>() const {
  if (kind() == TypeKind::TupleType || kind() == TypeKind::FunctionType ||
      kind() == TypeKind::ClassType || kind() == TypeKind::InterfaceType) {
    return std::static_pointer_cast<const NamedType>(shared_from_this());
  }
  return nullptr;
}

// Used as a return type when inferring the IValue type of a Python object.
struct InferredType {
  /* implicit */ InferredType(TypePtr type) : type_(std::move(type)) {}
  /* implicit */ InferredType(std::string reason)
      : type_(nullptr), reason_(std::move(reason)) {}
  TypePtr type() const {
    TORCH_INTERNAL_ASSERT(type_);
    return type_;
  }
  bool success() const {
    return type_ != nullptr;
  }
  const std::string& reason() const {
    TORCH_INTERNAL_ASSERT(!type_);
    return reason_;
  }

private:
  TypePtr type_;
  std::string reason_;
};

} // namespace c10
