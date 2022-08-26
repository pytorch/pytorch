#pragma once

#include <ATen/core/custom_class.h>
#include <ATen/core/jit_type_base.h>
#include <ATen/core/TensorBody.h>
#include <ATen/core/functional.h>
#include <ATen/core/symbol.h>
#include <ATen/core/type_factory.h>
#include <ATen/core/qualified_name.h>
#include <c10/util/TypeList.h>
#include <c10/util/Optional.h>

#include <array>
#include <memory>
#include <ostream>
#include <sstream>
#include <type_traits>

namespace torch {
namespace jit {
struct Function;
} // namespace jit
} // namespace torch

namespace c10 {

template<class Key, class Value>
class Dict;
struct IValue;
struct FunctionSchema;
struct NamedType;
using OptNameList = c10::optional<std::vector<std::string>>;

void standardizeVectorForUnion(std::vector<TypePtr>& reference, std::vector<TypePtr>* to_fill);
void standardizeVectorForUnion(std::vector<TypePtr>* to_flatten);

inline bool is_contiguous_strides(
    const IntArrayRef sizes,
    const IntArrayRef strides) {
  int n_dim = static_cast<int>(sizes.size());
  if (n_dim == 0) {
    return true;
  }

  if (strides[n_dim - 1] != 1) {
    return false;
  }

  for (int i = n_dim - 2; i >= 0; i--) {
    if (strides[i] != strides[i + 1] * sizes[i + 1]) {
      return false;
    }
  }
  return true;
}

struct AnyType;
using AnyTypePtr = SingletonTypePtr<AnyType>;
// Any is the top of the type hierarchy, all other types are subtypes
// T <: Any, forall T
struct TORCH_API AnyType : public Type {
  bool equals(const Type& rhs) const override {
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

inline std::string toString(const Type& type) {
  return type.str();
}

// Shim for compatibility with code that uses TypePtr.
inline std::string toString(const TypePtr& typePtr) {
  return toString(*typePtr);
}

inline bool operator!=(const Type& lhs, const Type& rhs) {
  return !(lhs == rhs);
}

// common base for all types that have a single sub element
// e.g. Future[T], Optional[T], List[T]
template <TypeKind K, typename T>
struct SingleElementType : public SharedType {
  static const TypeKind Kind = K;

  const TypePtr& getElementType() const {
    return elem;
  }

  bool hasFreeVariables() const override {
    return getElementType()->hasFreeVariables();
  }

  at::ArrayRef<TypePtr> containedTypes() const override {
    return elem;
  }

  bool equals(const Type& rhs) const override {
    if (auto rhs_ = rhs.cast<T>()) {
      return *getElementType() == *rhs_->getElementType();
    }
    return false;
  }

 protected:
  SingleElementType(TypePtr elem) : SharedType(Kind), elem(std::move(elem)) {
    if (!this->elem) {
      throw std::runtime_error(c10::str(
            "Can not create ", typeKindToString(Kind), " with None type"));
    }
  }

 private:
  TypePtr elem;
};

struct UnionType;
using UnionTypePtr = std::shared_ptr<UnionType>;
struct TORCH_API UnionType : public SharedType {
  friend struct Type;

  static const TypeKind Kind = TypeKind::UnionType;

  bool isSubtypeOfExt(const Type& rhs_, std::ostream* why_not) const override;

  std::string str() const override;

  static UnionTypePtr create(std::vector<TypePtr> reference);

  bool equals(const Type& rhs) const override;

  bool isUnionType() const override {
    return true;
  }

  at::ArrayRef<TypePtr> containedTypes() const override {
    return types_;
  }

  // For testing purposes only
  at::ArrayRef<TypePtr> getTypes() const {
    return types_;
  }

  TypePtr createWithContained(std::vector<TypePtr> contained_types) const override {
    return create(std::move(contained_types));
  }

  bool canHoldType(const Type& type) const;

  bool hasFreeVariables() const override {
    return has_free_variables_;
  }

  c10::optional<TypePtr> toOptional() const;

  c10::optional<TypePtr> subtractTypeSet(std::vector<TypePtr>& to_subtract) const;

 protected:
    explicit UnionType(std::vector<TypePtr> types, TypeKind kind=TypeKind::UnionType);
    std::string annotation_str_impl(TypePrinter printer = nullptr) const override;
    std::string unionStr(
        TypePrinter printer = nullptr,
        bool is_annotation_str = false) const;
    // NOLINTNEXTLINE(cppcoreguidelines-non-private-member-variables-in-classes)
    bool has_free_variables_;
    // NOLINTNEXTLINE(cppcoreguidelines-non-private-member-variables-in-classes)
    std::vector<TypePtr> types_;
    // NOLINTNEXTLINE(cppcoreguidelines-non-private-member-variables-in-classes)
    bool can_hold_none_;

};

struct OptionalType;
using OptionalTypePtr = std::shared_ptr<OptionalType>;
// This type represents an optional type. There is one `Optional` for
// each element type. `Optional[T]` can accept both `T` and
// `None`(`c10::nullopt` in C++)
// Subtype hierarchy for Optional:
//     - Optional[T] <: Optional[R] iff T <: R
//     - T <: Optional[R] if T <: R
//     - None <: Optional[T] for all T
//     - Optional[T] == Union[T, None] for all T
struct TORCH_API OptionalType : public UnionType {
  static OptionalTypePtr create(TypePtr contained);

  static const TypeKind Kind = TypeKind::OptionalType;

  friend struct Type;

  bool equals(const Type& rhs) const override;

  const TypePtr& getElementType() const {
    return contained_;
  }

  at::ArrayRef<TypePtr> containedTypes() const override {
    return contained_;
  }

  std::string str() const override {
    std::stringstream ss;
    ss << getElementType()->str() << "?";
    return ss.str();
  }

  TypePtr createWithContained(
      std::vector<TypePtr> contained_types) const override {
    AT_ASSERT(contained_types.size() == 1);
    return create(std::move(contained_types[0]));
  }

  bool isSubtypeOfExt(const Type& rhs, std::ostream* why_not) const override;

  bool isUnionType() const override {
    return true;
  }

  // common cast Optional[Tensor] for undefined tensor type
  static TypePtr ofTensor();
  //
  // global singleton
  static TypePtr get(TypePtr inner);

 private:
  explicit OptionalType(TypePtr contained);

  TypePtr contained_;

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
      c10::optional<bool> contiguous,
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

  c10::optional<std::vector<bool>> symbolicDims() const {
    if (!dims_) {
      return c10::nullopt;
    }
    auto symbolic_dims = std::vector<bool>();
    for (const ShapeSymbol& s : *dims_) {
      symbolic_dims.push_back(!s.is_static());
    }
    return symbolic_dims;
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

  friend bool operator==(const SymbolicShape& lhs, const SymbolicShape& rhs) {
    return lhs.dims_ == rhs.dims_;
  }

  friend bool operator!=(const SymbolicShape& lhs, const SymbolicShape& rhs) {
    return !(lhs == rhs);
  }

  private:
    c10::optional<std::vector<ShapeSymbol>> dims_;
};

namespace detail {
inline bool isComplete(const Stride& s) {
  return s.isComplete();
}

template<typename T>
inline bool isComplete(const T& /*t*/) {
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
// TODO: investigate making this SingletonOrSharedTypePtr<TensorType>
using TensorTypePtr = std::shared_ptr<TensorType>;
// This type represents a single Tensor with a specific size
struct TORCH_API TensorType : public SharedType {
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

  static TypePtr fromNumberType(const Type& typ);
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

  bool equals(const Type& rhs) const override;
  bool isSubtypeOfExt(const Type& rhs, std::ostream* why_not) const override;

  std::string str() const override;

  std::string repr_str() const override {
    if (isInferredType()) {
      return str() + " (inferred)";
    } else {
      return str();
    }
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

  TensorTypePtr withStrides(VaryingShape<Stride> sstrides) const {
    auto cloned = clone();
    cloned->strides_ = sstrides;
    return cloned;
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

  TensorTypePtr withDevice(const c10::optional<at::Device> device) const {
    auto copy = clone();
    copy->device_ = device;
    return copy;
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

  static const TensorTypePtr& get();

  static const TypeKind Kind = TypeKind::TensorType;

  static std::vector<int64_t> contiguousStridesOf(
      at::IntArrayRef in_sizes,
      at::MemoryFormat memory_format = MemoryFormat::Contiguous) {
    auto contiguous_fn = [](const at::IntArrayRef& sizes,
                            const std::vector<int64_t>& dim_order) {
      std::vector<int64_t> strides(sizes.size());
      if (sizes.empty()) // zero-dim case
        return strides;

      strides[dim_order[0]] = 1;
      for (size_t i = 1; i < dim_order.size(); i++) {
        auto cur_dim = dim_order[i];
        auto pre_dim = dim_order[i - 1];
        strides[cur_dim] = strides[pre_dim] * sizes[pre_dim];
      }
      return strides;
    };

    std::vector<int64_t> dim_order(in_sizes.size());
    if (memory_format == MemoryFormat::ChannelsLast) {
      dim_order = {1, 3, 2, 0};
    } else if (memory_format == MemoryFormat::ChannelsLast3d) {
      dim_order = {1, 4, 3, 2, 0};
    } else {
      auto ndims = in_sizes.size();
      for (size_t i = 0; i < ndims; i++) {
        dim_order[i] = ndims - i - 1; // Reverse
      }
    }
    return contiguous_fn(in_sizes, dim_order);
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
    return create(std::move(contained_types.at(0)));
  }

  bool isSubtypeOfExt(const Type& rhs, std::ostream* why_not) const override;

  // global singleton
  // Given an inner type T and an identifier,
  // this function wil return the global singleton type pointer
  // the type List<T>.
  // The extra "identifier" argument is needed beccause we have multiple container types
  // that all re-use this function (List<T>, array<T, N>, etc.)
  static TypePtr get(std::string identifier, TypePtr inner);

  // common cast List[Tensor]
  static ListTypePtr ofTensors();
  static ListTypePtr ofOptionalTensors();
  static ListTypePtr ofInts();
  static ListTypePtr ofFloats();
  static ListTypePtr ofComplexDoubles();
  static ListTypePtr ofBools();
  static ListTypePtr ofStrings();

 private:
  ListType(TypePtr elem) : SingleElementType(std::move(elem)) {}

  std::string annotation_str_impl(TypePrinter printer = nullptr) const override {
    std::stringstream ss;
    ss << "List[" << getElementType()->annotation_str(printer) << "]";
    return ss.str();
  }
};

struct DictType;
using DictTypePtr = std::shared_ptr<DictType>;
struct TORCH_API DictType : public SharedType {
  friend struct Type;
  static const TypeKind Kind = TypeKind::DictType;

  static DictTypePtr create(TypePtr key, TypePtr value) {
    auto kind = key->kind();
    if (auto dyn = key->castRaw<DynamicType>()) {
      kind = dyn->dynamicKind();
    }
    switch (kind) {
      case TypeKind::AnyType:
      case TypeKind::IntType:
      case TypeKind::BoolType:
      case TypeKind::FloatType:
      case TypeKind::ComplexType:
      case TypeKind::StringType:
      case TypeKind::TensorType:
      case TypeKind::DeviceObjType:
        return DictTypePtr(new DictType(std::move(key), std::move(value)));
      default:
        AT_ERROR(
            "Cannot create dict for key type '",
            key->str(),
            "', only int, float, complex, Tensor, device and string keys are supported");
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
    return create(std::move(contained_types.at(0)), std::move(contained_types.at(1)));
  }

  const TypePtr& getKeyType() const {
    return types.at(0);
  }

  const TypePtr& getValueType() const {
    return types.at(1);
  }

  bool hasFreeVariables() const override {
    return has_free_variables;
  }

  at::ArrayRef<TypePtr> containedTypes() const override {
    return types;
  }

  bool equals(const Type& rhs) const override {
    if (auto* dict_rhs = rhs.castRaw<DictType>()) {
      return *getKeyType() == *(dict_rhs->getKeyType()) &&
          *getValueType() == *(dict_rhs->getValueType());
    }
    return false;
  }

  // global singleton
  // Given an inner type T and an identifier,
  // this function wil return the global singleton type pointer
  // the type List<T>.
  // The extra "identifier" argument is needed beccause we have multiple container types
  // that all re-use this function (Dict<K, V> and unordered_map<K, V>)
  static TypePtr get(std::string identifier, TypePtr key, TypePtr val);

 private:
  DictType(TypePtr key, TypePtr value)
      : SharedType(TypeKind::DictType),
        has_free_variables(
            key->hasFreeVariables() || value->hasFreeVariables()) {
    types.reserve(2);
    types.push_back(std::move(key));
    types.push_back(std::move(value));
  }

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
    return create(std::move(contained_types.at(0)));
  }

  bool isSubtypeOfExt(const Type& rhs, std::ostream* why_not) const override {
    if (Type::isSubtypeOfExt(rhs, why_not)) {
      return true;
    }
    if (auto rhs_ = rhs.castRaw<FutureType>()) {
      return getElementType()->isSubtypeOfExt(*rhs_->getElementType(), why_not);
    }
    return false;
  }

 private:
  FutureType(TypePtr elem) : SingleElementType(std::move(elem)) {}

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
    return create(std::move(contained_types.at(0)));
  }

 private:
  RRefType(TypePtr elem) : SingleElementType(std::move(elem)) {}

  std::string annotation_str_impl(TypePrinter printer = nullptr) const override {
    std::stringstream ss;
    ss << "RRef[" << getElementType()->annotation_str(printer) << "]";
    return ss.str();
  }
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

  static TupleTypePtr createNamed(const c10::optional<c10::QualifiedName>& name,
      const std::vector<c10::string_view>& field_names,
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

  bool equals(const Type& rhs) const override;
  bool isSubtypeOfExt(const Type& rhs_, std::ostream* why_not) const override;

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
  c10::optional<std::vector<c10::string_view>> names() const;

  static const TypeKind Kind = TypeKind::TupleType;

 private:
  template <typename S>
  static TupleTypePtr createWithSpec(
      const c10::optional<c10::QualifiedName>& name,
      const std::vector<S>& field_names,
      const std::vector<TypePtr>& field_types,
      std::vector<IValue>& field_defaults);

  TupleType(
      std::vector<TypePtr> elements_,
      c10::optional<c10::QualifiedName> name,
      std::shared_ptr<FunctionSchema> schema);

  bool compare(
      const Type& rhs,
      std::function<bool(const Type&, const Type&)> fn) const {
    if (rhs.kind() != kind()) {
      return false;
    }

    const auto& l_elements = elements();
    const auto& r_elements = rhs.castRaw<TupleType>()->elements();
    if (l_elements.size() != r_elements.size())
      return false;
    for (size_t i = 0; i < l_elements.size(); ++i) {
      if (!fn(*l_elements[i], *r_elements[i]))
        return false;
    }
    return true;
  }

  std::string annotation_str_impl(TypePrinter printer = nullptr) const override;

  std::vector<TypePtr> elements_;
  bool has_free_variables_;
  std::shared_ptr<FunctionSchema> schema_;
};

// the common supertype of all Enums, only used in operator registraion.
// EnumType <: AnyEnumType for all Enums
struct AnyEnumType;
using AnyEnumTypePtr = SingletonTypePtr<AnyEnumType>;
struct TORCH_API AnyEnumType final : public Type {
  bool equals(const Type& rhs) const override {
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
using NumberTypePtr = SingletonTypePtr<NumberType>;
// This type represents a Python number
// Subtype hierarchy for Number Types (NumberType as the base type):
// IntType <: NumberType
// FloatType <: NumberType
// ComplexType <:NumberType
//
// WARNING: if you add a new subtype of NumberType that is not
// represented by a global singleton, you need to change NumberTypePtr
// to a SingletonOrSharedTypePtr and deal with NumberType needing to
// both inherit and not inherit from SharedType!
struct TORCH_API NumberType : public Type {
  bool equals(const Type& rhs) const override;

  bool isSubtypeOfExt(const Type& rhs, std::ostream* why_not) const override;

  std::string str() const override {
    return "Scalar"; // match what PythonArgParser says for clarity
  }
  static const TypeKind Kind = TypeKind::NumberType;
  // global singleton
  static NumberTypePtr get();

 protected:
  NumberType(TypeKind kind = TypeKind::NumberType) : Type(kind) {}

  std::string annotation_str_impl(TypePrinter printer = nullptr) const override {
    (void)printer; // Suppress unused variable warning
    return "number"; // technically not a valid python type, but
                     // we need to use it when parsing back in annotations
                     // for implicit conversions
  }
};

struct FloatType;
using FloatTypePtr = SingletonTypePtr<FloatType>;
// This type represents a Python float number
struct TORCH_API FloatType : public NumberType {
  bool equals(const Type& rhs) const override {
    return rhs.kind() == kind();
  }
  std::string str() const override {
    return "float";
  }
  bool isSubtypeOfExt(const Type& rhs, std::ostream* why_not) const override {
    // NOLINTNEXTLINE(bugprone-parent-virtual-call)
    return rhs.kind() == TypeKind::NumberType || Type::isSubtypeOfExt(rhs, why_not);
  }
  static const TypeKind Kind = TypeKind::FloatType;
  // global singleton
  static FloatTypePtr get();

 private:
  FloatType() : NumberType(TypeKind::FloatType) {}
  std::string annotation_str_impl(TypePrinter printer = nullptr) const override {
    (void)printer; // Suppress unused variable warning
    return "float";
  }
};

struct ComplexType;
using ComplexTypePtr = SingletonTypePtr<ComplexType>;
// This type represents a Python float number
struct TORCH_API ComplexType : public NumberType {
  bool equals(const Type& rhs) const override {
    return rhs.kind() == kind();
  }
  std::string str() const override {
    return "complex";
  }
  bool isSubtypeOfExt(const Type& rhs, std::ostream* why_not) const override {
    // NOLINTNEXTLINE(bugprone-parent-virtual-call)
    return rhs.kind() == TypeKind::NumberType || Type::isSubtypeOfExt(rhs, why_not);
  }
  static const TypeKind Kind = TypeKind::ComplexType;
  // global singleton
  static ComplexTypePtr get();

 private:
  ComplexType() : NumberType(TypeKind::ComplexType) {}
  std::string annotation_str_impl(TypePrinter printer = nullptr) const override {
    (void)printer; // Suppress unused variable warning
    return "complex";
  }
};

// We need to introduce `SymIntType` to represent the `SymInt` type
// used in function schemas e.g. `aten::narrow_copy(... SymInt length)
// `SymInt` will be used to enable tracing arithmetic operations on
// dimension values. Please see [SymInt.h] for more information
struct SymIntType;
using SymIntTypePtr = SingletonTypePtr<SymIntType>;
struct TORCH_API SymIntType : public Type {
  bool equals(const Type& rhs) const override {
    return rhs.kind() == kind();
  }
  std::string str() const override {
    return "SymInt";
  }
  std::string annotation_str_impl(TypePrinter printer = nullptr) const override {
    // TODO: will become a Union[SymIntNodeImpl|int] in the near future
    return "int";
  }
  static const TypeKind Kind = TypeKind::SymIntType;
  // global singleton
  static SymIntTypePtr get();

 private:
  SymIntType() : Type(TypeKind::SymIntType) {}
};

struct IntType;
using IntTypePtr = SingletonTypePtr<IntType>;
// This type represents a Python int number
struct TORCH_API IntType : public NumberType {
  bool equals(const Type& rhs) const override {
    return rhs.kind() == kind();
  }
  std::string str() const override {
    return "int";
  }
  bool isSubtypeOfExt(const Type& rhs, std::ostream* why_not) const override {
    // NOLINTNEXTLINE(bugprone-parent-virtual-call)
    return rhs.kind() == TypeKind::NumberType || Type::isSubtypeOfExt(rhs, why_not);
  }
  static const TypeKind Kind = TypeKind::IntType;
  // global singleton
  static IntTypePtr get();

 private:
  IntType() : NumberType(TypeKind::IntType) {}
  std::string annotation_str_impl(TypePrinter printer = nullptr) const override {
    (void)printer; // Suppress unused variable warning
    return "int";
  }
};

struct BoolType;
using BoolTypePtr = SingletonTypePtr<BoolType>;
// This node represents a Python bool value
struct TORCH_API BoolType : public Type {
  bool equals(const Type& rhs) const override {
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
using StringTypePtr = SingletonTypePtr<StringType>;
// This type represents a Python string
struct TORCH_API StringType : public Type {
  bool equals(const Type& rhs) const override {
    return rhs.kind() == kind();
  }
  std::string str() const override {
    // we only use "str" (not "string") in both FunctionSchema and script
    return annotation_str();
  }
  std::string annotation_str_impl(TypePrinter printer = nullptr) const override {
    (void)printer; // Suppress unused variable warning
    return "str";
  }
  static const TypeKind Kind = TypeKind::StringType;
  // global singleton
  static StringTypePtr get();

 private:
  StringType() : Type(TypeKind::StringType) {}
};

struct StorageType;
using StorageTypePtr = SingletonTypePtr<StorageType>;
struct TORCH_API StorageType : public Type {
  bool equals(const Type& rhs) const override {
    return rhs.kind() == kind();
  }
  std::string str() const override {
    return annotation_str();
  }
  std::string annotation_str_impl(TypePrinter printer = nullptr) const override {
    (void)printer; // Suppress unused variable warning
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
  bool equals(const Type& rhs) const override {
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
    (void)printer; // Suppress unused variable warning
    const auto& n = name().value();
    return n.qualifiedName();
  }
  torch::jit::Function* function_;
};

struct NoneType;
using NoneTypePtr = SingletonTypePtr<NoneType>;
// This type represents a Python None
struct TORCH_API NoneType : public Type {
  bool equals(const Type& rhs) const override {
    return rhs.kind() == kind();
  }
  std::string str() const override {
    return "NoneType";
  }
  bool isSubtypeOfExt(const Type& rhs, std::ostream *why_not) const override;

  static const TypeKind Kind = TypeKind::NoneType;
  // global singleton
  static NoneTypePtr get();

 private:
  NoneType() : Type(TypeKind::NoneType) {}
};

struct GeneratorType;
using GeneratorTypePtr = SingletonTypePtr<GeneratorType>;
// This type represents a Generator
struct TORCH_API GeneratorType : public Type {
  bool equals(const Type& rhs) const override {
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
using QuantizerTypePtr = SingletonTypePtr<QuantizerType>;
// This type represents a Quantizer
struct TORCH_API QuantizerType : public Type {
  bool equals(const Type& rhs) const override {
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
using QSchemeTypePtr = SingletonTypePtr<QSchemeType>;
// This type represents a QScheme
struct TORCH_API QSchemeType : public Type {
  bool equals(const Type& rhs) const override {
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
using DeviceObjTypePtr = SingletonTypePtr<DeviceObjType>;
// This type represents a Device
struct TORCH_API DeviceObjType : public Type {
  bool equals(const Type& rhs) const override {
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
using StreamObjTypePtr = SingletonTypePtr<StreamObjType>;
// This type represents a Generator
struct TORCH_API StreamObjType : public Type {
  bool equals(const Type& rhs) const override {
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
struct VarType : public SharedType {
  static VarTypePtr create(std::string name_) {
    return VarTypePtr(new VarType(std::move(name_)));
  }
  bool equals(const Type& rhs) const override {
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
      : SharedType(TypeKind::VarType), name_(std::move(name_)) {}
  std::string name_;
};

struct CapsuleType;
using CapsuleTypePtr = SingletonTypePtr<CapsuleType>;
// This type represents a Python Capsule.
// It does not appear in the IR and is only used during runtime
struct TORCH_API CapsuleType : public Type {
  bool equals(const Type& rhs) const override {
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
using PyObjectTypePtr = SingletonTypePtr<PyObjectType>;
// This type represents a PyObject Type
struct TORCH_API PyObjectType : public Type {
  bool equals(const Type& rhs) const override {
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

// `unshapedType` is used to remove Tensor subtypes. We treat all Tensor
// subtypes as simply "Tensor"; we also create a new version of any
// container types in which internal Tensors have undergone the same
// operation. This is used for type comparisons between two Tensor types
// (`unshapedType` means that we don't falsely return `false` for e.g.
// Tensors of different dimensions). It's also used in the alias
// analysis pass.
// Be careful with calls because this can be very slow. If calling this
// on a graph, use `EraseShapeInformation` in shape_analysis.h
inline TypePtr unshapedType(const TypePtr& type) {
  if (type->isSubtypeOf(*TensorType::get())) {
    return TensorType::get();
  }
  at::ArrayRef<TypePtr> contained = type->containedTypes();
  if (contained.empty()) {
    return type;
  }
  return type->withContained(fmap(type->containedTypes(), unshapedType));
}

inline TypePtr TensorType::fromNumberType(const Type& typ) {
  if (typ.isSubtypeOf(*IntType::get())) {
    return TensorType::createContiguous(at::kLong, at::kCPU, {});
  } else if (typ.isSubtypeOf(*FloatType::get())) {
    return TensorType::createContiguous(at::kDouble, at::kCPU, {});
  } else if (typ.isSubtypeOf(*BoolType::get())) {
    return TensorType::createContiguous(at::kBool, at::kCPU, {});
  } else if (typ.kind() == NumberType::Kind) {
    return TensorType::create(c10::nullopt, at::kCPU, {}, c10::nullopt);
  }
  TORCH_CHECK(false, "Unknown number type: ", typ.str());
}
inline TypePtr TensorType::fromBoolType() {
  return TensorType::createContiguous(at::kBool, at::kCPU, {});
}

inline c10::optional<c10::ScalarType> tryScalarTypeFromJitType(const Type& type) {
  if (type == *FloatType::get()) {
    return at::typeMetaToScalarType(c10::get_default_dtype());
  } else if (type == *IntType::get()) {
    return at::ScalarType::Long;
  } else if (type == *BoolType::get()) {
    return at::ScalarType::Bool;
  }
  return c10::nullopt;
}

inline at::ScalarType scalarTypeFromJitType(const Type& type) {
  auto result = tryScalarTypeFromJitType(type);
  TORCH_CHECK(
      result,
      "Add new condition, expected Float, Complex, Int, or Bool but got",
      type.str());
  return *result;
}

// Attempt to find the correct supertype of the two types `t1` and `t2`.
// If no supertype is found, then nullopt will be returned if
// `default_to_union` is false, and `Union[t1, t2]` will be returned
// if it is true. If `t1 == t2`, or `t1` is a type refinement of `t2`,
// then `t2` will be returned (and vice versa).
//
// Two different tensortypes will return dynamic.
//
// Currently we chose not to support returning a NumberType for
// two types from the set of {FloatType, IntType, ComplexType}, because
// there is a lack of operator support for NumberType.
//
// If `type_hint` is an `InterfaceType`, then we can use that as a
// potential supertype for `ClassType`s in the list. Otherwise, we have
// no way to find and use some common interface type
TORCH_API c10::optional<TypePtr> unifyTypes(
    const TypePtr& t1,
    const TypePtr& t2,
    bool default_to_union = false,
    TypePtr type_hint = nullptr);

TORCH_API c10::optional<TypePtr> unifyTypeList(
    at::ArrayRef<TypePtr> elements,
    std::ostream& why_not,
    bool default_to_union = false,
    TypePtr type_hint = nullptr);

namespace detail {
template <typename T>
struct getTypePtr_ final {
  static decltype(auto) call() {
    return ([]() {
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
    }());
  }
};

template <>
struct getTypePtr_<at::IValue> final {
  static decltype(auto) call() {
    return AnyType::get();
  }
};

template <>
struct getTypePtr_<at::Tensor> final {
  static decltype(auto) call() {
    return TensorType::get();
  }
};
template <>
struct getTypePtr_<c10::Storage> final {
  static decltype(auto) call() {
    return StorageType::get();
  }
};
template <>
struct getTypePtr_<c10::Stream> final {
  static decltype(auto) call() {
    return StreamObjType::get();
  }
};
template <>
struct getTypePtr_<double> final {
  static decltype(auto) call() {
    return FloatType::get();
  }
};
template <>
struct getTypePtr_<c10::complex<double>> final {
  static decltype(auto) call() {
    return ComplexType::get();
  }
};
template <>
struct getTypePtr_<int64_t> final {
  static decltype(auto) call() {
    return IntType::get();
  }
};

template <>
struct getTypePtr_<SymInt> final {
  static decltype(auto) call() {
    return SymIntType::get();
  }
};
template <>
struct getTypePtr_<c10::Device> final {
  static decltype(auto) call() {
    return DeviceObjType::get();
  }
};
template <>
struct getTypePtr_<bool> final {
  static decltype(auto) call() {
    return BoolType::get();
  }
};
template <>
struct getTypePtr_<at::Scalar> final {
  static decltype(auto) call() {
    return NumberType::get();
  }
};
template <>
struct getTypePtr_<c10::QScheme> final {
  static decltype(auto) call() {
    return QSchemeType::get();
  }
};
template <>
struct getTypePtr_<at::Generator> final {
  static decltype(auto) call() {
    return TypeFactory::create<OptionalType>(
        TypeFactory::get<GeneratorType>());
  }
};
template <>
struct getTypePtr_<std::string> final {
  static decltype(auto) call() {
    return StringType::get();
  }
};
template <>
struct getTypePtr_<c10::string_view> final {
  static decltype(auto) call() {
    return StringType::get();
  }
};
template <>
struct getTypePtr_<at::Dimname> final {
  static decltype(auto) call() {
    return StringType::get();
  }
};
template <class T>
struct getTypePtr_<std::vector<T>> final {
  static const auto& call() {
    static auto inner_type = getTypePtr_<T>::call();
    // The "per vector<T>" static singleton needs to live in a .cpp file,
    // otherwise we'll end up with one singleton instance per shared library.
    static auto type = ListType::get("vector", inner_type);
    return type;
  }
};
template <class T>
struct getTypePtr_<c10::ArrayRef<T>> final {
  static const auto& call() {
    static auto inner_type = getTypePtr_<T>::call();
    // The "per ArrayRef<T>" static singleton needs to live in a .cpp file,
    // otherwise we'll end up with one singleton instance per shared library.
    static auto type = ListType::get("ArrayRef", inner_type);
    return type;
  }
};
template <>
struct getTypePtr_<c10::SymIntArrayRef> final {
  static const auto& call() {
    static auto type = ListType::create(getTypePtr_<c10::SymInt>::call());
    return type;
  }
};
template <class T>
struct getTypePtr_<c10::List<T>> final {
  static const auto& call() {
    static auto inner_type = getTypePtr_<T>::call();
    // The "per List<T>" static singleton needs to live in a .cpp file,
    // otherwise we'll end up with one singleton instance per shared library.
    static auto type = ListType::get("List", inner_type);
    return type;
  }
};
template <class T, size_t N>
struct getTypePtr_<std::array<T, N>> final {
  static const auto& call() {
    static auto inner_type = getTypePtr_<T>::call();
    // The "per array<T, N>" static singleton needs to live in a .cpp file,
    // otherwise we'll end up with one singleton instance per shared library.
    // (Concatenating the length onto the end of the string because we want a unique
    // type_ptr created for every std::array<T, N> type).
    static auto type = ListType::get(std::string("array") + std::to_string(N), inner_type);
    return type;
  }
};
template <class K, class V>
struct getTypePtr_<std::unordered_map<K, V>> final {
  static const auto& call() {
    static auto inner_key_type = getTypePtr_<K>::call();
    static auto inner_val_type = getTypePtr_<V>::call();
    // The "per unordered_map<K, V>" static singleton needs to live in a .cpp file,
    // otherwise we'll end up with one singleton instance per shared library.
    static auto type = DictType::get("unordered_map", inner_key_type, inner_val_type);
    return type;
  }
};
template <class K, class V>
struct getTypePtr_<c10::Dict<K, V>> final {
  static const auto& call() {
    static auto inner_key_type = getTypePtr_<K>::call();
    static auto inner_val_type = getTypePtr_<V>::call();
    // The "per Dict<K, V>" static singleton needs to live in a .cpp file,
    // otherwise we'll end up with one singleton instance per shared library.
    static auto type = DictType::get("Dict", inner_key_type, inner_val_type);
    return type;
  }
};

template <class T>
struct getTypePtr_<at::optional<T>> final {
  static const auto& call() {
    static auto inner_type = getTypePtr_<T>::call();
    // The "per optional<T>" static singleton needs to live in a .cpp file,
    // otherwise we'll end up with one singleton instance per shared library.
    static auto type = OptionalType::get(inner_type);
    return type;
  }
};


template<>
struct getTypePtr_<at::OptionalIntArrayRef> final {
  static const auto& call() {
    static auto type = OptionalType::create(getTypePtr_<IntArrayRef>::call());
    return type;
  }
};

template <class... Contained>
struct getTypePtr_<std::tuple<Contained...>> final {
  static const auto& call() {
    static auto type = ([]() {
      std::vector<TypePtr> contained_types = {
        (getTypePtr_<Contained>::call())...
      };
      return TupleType::create(std::move(contained_types));
    })();
    return type;
  }
};
template <>
struct getTypePtr_<void> final {
  static decltype(auto) call() {
    return NoneType::get();
  }
};
} // namespace detail
template <class T>
inline decltype(auto) getTypePtr() {
  // TODO: static_assert that a templated function exists, and throw a friendly
  // error message if not
  return detail::getTypePtr_<T>::call();
}

template <class T>
inline TypePtr getTypePtrCopy() {
  // TODO: static_assert that a templated function exists, and throw a friendly
  // error message if not
  return getTypePtr<T>();
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
matchTypeVariables(const TypePtr& formal, const TypePtr& actual, TypeEnv& type_env);

// replace type variables appearing in `type` with the values in
// `type_env`. Returns nullptr if a variable used in `type`
// does not appear in `type_env`
TORCH_API TypePtr tryEvalTypeVariables(const TypePtr& type, TypeEnv& type_env);

TORCH_API bool elementTypeCanBeInferredFromMembers(const TypePtr& elem_type);

struct InterfaceType;
using InterfaceTypePtr = std::shared_ptr<InterfaceType>;

// Interfaces are a list of abstract methods that a class might meet.
// If a class provides those methods, it implicitly meets the interface.

// Subtype relations for Interface with ClassType:
// lhs (ClassType or InterfaceType) is a subtype of rhs if:
// 1. lhs methods are a superset of rhs methods
// 2. if rhs is module interface, the lhs must be module interface or module itself
struct TORCH_API InterfaceType : public NamedType {
  static InterfaceTypePtr create(
      QualifiedName qualifiedName, bool is_module=false);

  bool equals(const Type& rhs) const override {
    if (auto user_rhs = rhs.castRaw<InterfaceType>()) {
      return isSubTypeImpl(*this, *user_rhs, nullptr) &&
          isSubTypeImpl(*user_rhs, *this, nullptr);
    }
    return false;
  }

  std::string str() const override {
    return std::string("InterfaceType<") + name()->name() + ">";
  }

  bool isSubtypeOfExt(const Type& rhs, std::ostream* why_not) const override;

  // try to find a method of this interface,
  // returns nullptr if not found.
  const FunctionSchema* getMethod(const std::string& name) const;
  void addMethod(FunctionSchema schema);
  const std::vector<FunctionSchema>& methods() const {
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
    (void)printer; // Suppress unused variable warning
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

bool equals(const Type& rhs) const override {
  return rhs.kind() == kind();
}

protected:
EnumerationType() : Type(Kind) {}
};

// WARNING: These enumeration types below DO NOT actually get parsed out
// from the logical schema strings, instead they are mapped as ints.  To
// observe these types, use real_type() instead of type() on Argument

struct ScalarTypeType;
using ScalarTypeTypePtr = SingletonTypePtr<ScalarTypeType>;
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

struct MemoryFormatType;
using MemoryFormatTypePtr = SingletonTypePtr<MemoryFormatType>;
struct TORCH_API MemoryFormatType : public EnumerationType<TypeKind::MemoryFormatType> {
std::string str() const override {
return "MemoryFormatType";
}
static const TypeKind Kind = TypeKind::MemoryFormatType;
// global singleton
static MemoryFormatTypePtr get();

private:
MemoryFormatType() : EnumerationType() {}
};

struct LayoutType;
using LayoutTypePtr = SingletonTypePtr<LayoutType>;
struct TORCH_API LayoutType : public EnumerationType<TypeKind::LayoutType> {
std::string str() const override {
return "LayoutType";
}
static const TypeKind Kind = TypeKind::LayoutType;
// global singleton
static LayoutTypePtr get();

private:
LayoutType() : EnumerationType() {}
};

namespace detail {
template <>
struct getTypePtr_<c10::ScalarType> final {
  static decltype(auto) call() {
    return ScalarTypeType::get();
  }
};
template <>
struct getTypePtr_<c10::Layout> final {
  static decltype(auto) call() {
    return LayoutType::get();
  }
};
template <>
struct getTypePtr_<c10::MemoryFormat> final {
  static decltype(auto) call() {
    return MemoryFormatType::get();
  }
};
} // namespace detail

// the common supertype of all lists,
// List[T] <: AnyList for all T
struct AnyListType;
using AnyListTypePtr = SingletonTypePtr<AnyListType>;
struct TORCH_API AnyListType : public Type {
  bool equals(const Type& rhs) const override {
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
using AnyTupleTypePtr = SingletonTypePtr<AnyTupleType>;
struct TORCH_API AnyTupleType : public Type {
  bool equals(const Type& rhs) const override {
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
using AnyClassTypePtr = SingletonTypePtr<AnyClassType>;
struct TORCH_API AnyClassType : public Type {
  bool equals(const Type& rhs) const override {
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

template<>
inline typename detail::CastReturnType<NamedType>::type Type::cast<NamedType>() {
  if (kind() == TypeKind::TupleType || kind() == TypeKind::FunctionType ||
      kind() == TypeKind::ClassType || kind() == TypeKind::InterfaceType) {
    return std::static_pointer_cast<NamedType>(static_cast<NamedType *>(this)->shared_from_this());
  }
  return nullptr;
}

template<>
inline typename detail::CastConstReturnType<NamedType>::type Type::cast<NamedType>() const {
  if (kind() == TypeKind::TupleType || kind() == TypeKind::FunctionType ||
      kind() == TypeKind::ClassType || kind() == TypeKind::InterfaceType) {
    return std::static_pointer_cast<const NamedType>(static_cast<const NamedType *>(this)->shared_from_this());
  }
  return nullptr;
}

template<>
inline const NamedType* Type::castRaw<NamedType>() const {
  if (kind() == TypeKind::TupleType || kind() == TypeKind::FunctionType ||
      kind() == TypeKind::ClassType || kind() == TypeKind::InterfaceType) {
    return static_cast<const NamedType*>(this);
  }
  return nullptr;
}

// Used as a return type when inferring the IValue type of a Python object.
struct InferredType {
  /* implicit */ InferredType(TypePtr type) : type_(std::move(type)) {}
  /* implicit */ InferredType(std::string reason)
      : type_(nullptr), reason_(std::move(reason)) {}
  TypePtr type() const {
    TORCH_INTERNAL_ASSERT(
        type_,
        "Tried to get the type from an InferredType but the type is null. ",
        "Reason: ",
        reason_);
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

TORCH_API bool containsAnyType(const TypePtr& type);

} // namespace c10
