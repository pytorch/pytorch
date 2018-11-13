#pragma once

#include <ATen/core/ivalue.h>
#include <ATen/core/interned_strings.h>
#include <ATen/core/functional.h>
#include <ATen/core/Type.h>
#include <ATen/core/TensorMethods.h>

#include <caffe2/core/common.h>

#include <memory>
#include <iostream>
#include <type_traits>

namespace c10 {

#define C10_FORALL_TYPES(_) \
_(DynamicType) \
_(TensorType) \
_(CompleteTensorType) \
_(UndefinedTensorType) \
_(TupleType) \
_(ListType) \
_(NumberType) \
_(FloatType) \
_(FutureType) \
_(IntType) \
_(NoneType) \
_(StringType) \
_(GeneratorType) \
_(BoolType) \
_(OptionalType) \
_(VarType) \

enum class TypeKind {
#define DEFINE_TYPE(T) T,
  C10_FORALL_TYPES(DEFINE_TYPE)
#undef DEFINE_TYPE
};

#define DEFINE_IS_SUBCLASS(_kind) \
  bool isSubclass(const TypeKind kind) const override { \
    return kind == TypeKind::_kind; \
  }

struct Type;
using TypePtr = std::shared_ptr<Type>;

struct CAFFE2_API Type : std::enable_shared_from_this<Type> {
private:
  TypeKind kind_;
  template<typename T>
  static std::shared_ptr<T> sliceType(std::shared_ptr<const T> ptr) {
    auto result = std::make_shared<typename std::remove_const<T>::type>(*ptr);
    // XXX: the line above will correctly slice the struct, and make its runtype
    // type exactly equal to T. However, kind_ is a field of Type, so it will simply
    // be copied, and we need to fix it in here to match the dynamic type.
    result->kind_ = T::Kind;
    return result;
  }

protected:
  Type(TypeKind kind)
    : kind_(kind) {}

public:
  virtual bool operator==(const Type& rhs) const = 0;

  // subtyping relation. By default, we return true for the case
  // when the type is exactly equal
  virtual bool isSubtypeOf(const TypePtr rhs) const {
    return *this == *rhs;
  }

  // If this class can be cast to the kind passed in
  // This removes the need for RTTI
  virtual bool isSubclass(const TypeKind kind) const = 0;

  // How this type will appear in FunctionSchema declarations
  virtual std::string str() const = 0;

  // How this type will appear as if it were a type annotation in Python
  // which is sometimes different than how it appears in declarations (e.g. int[] vs List[int])
  virtual std::string python_str() const {
    return str();
  }

  TypeKind kind() const {
    return kind_;
  }

  virtual bool requires_grad() const { return false; }

  // Dynamically cast this object to the subclass indicated by the
  // template variable, returning nullptr if the cast is invalid.
  // NOTE: if the cast succeeds, but the casted kind is not the
  // run-time kind of the type, we also slice the structure, so
  // that assignments of those types to values don't accidentally
  // inherit more detailed information from subclasses.
  template<typename T>
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
  template<typename T>
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
  template<typename T>
  std::shared_ptr<T> expect() {
    auto r = cast<T>();
    AT_ASSERT(r);
    return r;
  }
  template<typename T>
  std::shared_ptr<const T> expect() const {
    auto r = cast<const T>();
    AT_ASSERT(r);
    return r;
  }
  virtual ~Type() = default;
  virtual bool hasFreeVariables() const {
    return false;
  }
  // list of types this type contains, e.g. for a List then element type of a list
  // for a tuple, the types of the tuple elements
  virtual at::ArrayRef<TypePtr> containedTypes() const {
    return {};
  }
  // create a new version of this type, replacing its contained types with
  // contained_types
  TypePtr withContained(std::vector<TypePtr> contained_types) {
    auto current_contained = containedTypes();
    AT_ASSERT(current_contained.size() == contained_types.size());
    if(current_contained.equals(contained_types)) {
      return shared_from_this();
    }
    return createWithContained(std::move(contained_types));
  }
  // per-type constructor, you only need to override this if the containedTypes()
  // is not empty
  virtual TypePtr createWithContained(std::vector<TypePtr> contained_types) const {
    AT_ERROR("type with contained types did not overload createWithContained: ", str());
  }
};

inline bool operator!=(const Type & lhs, const Type & rhs) {
  return !(lhs == rhs);
}

struct OptionalType;
using OptionalTypePtr = std::shared_ptr<OptionalType>;
// This type represents an optional type, for each element type.
// Optional[T] can accept both T and None(nullopt in C++)
// Subtype hierarchy for Optional:
// 1. Optional[T] isSubtypeOf Optional[R] iff T isSubtypeOf R
// 2. T isSubtypeOf Optional[R] if T isSubtypeOf R
// 3. NoneType isSubtypeOf any Optional Type
struct CAFFE2_API OptionalType: public Type {
  static OptionalTypePtr create(TypePtr element) {
    return OptionalTypePtr(new OptionalType(std::move(element))); // NOLINT(modernize-make-shared)
  }
  DEFINE_IS_SUBCLASS(OptionalType);
  bool operator==(const Type& rhs) const override {
    if(auto rhs_ = rhs.cast<OptionalType>()) {
      return *getElementType() == *rhs_->getElementType();
    }
    return false;
  }
  bool requires_grad() const override {
    return elem->requires_grad();
  }

  bool isSubtypeOf(const TypePtr rhs) const override {
    if(auto rhs_ = rhs->cast<OptionalType>()) {
      return getElementType()->isSubtypeOf(rhs_->getElementType());
    }
    return false;
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
  TypePtr getElementType() const {
    return elem;
  }
  bool hasFreeVariables() const override {
    return has_free_variables_;
  }
  static const TypeKind Kind = TypeKind::OptionalType;
  // common cast Optional[Tensor] for undefined tensor type
  static OptionalTypePtr ofTensor();
private:
  OptionalType(TypePtr elem)
  : Type(TypeKind::OptionalType)
  , elem(std::move(elem))
  , has_free_variables_(getElementType()->hasFreeVariables()) {}
  TypePtr elem;
  bool has_free_variables_;

};

struct DynamicType;
using DynamicTypePtr = std::shared_ptr<DynamicType>;
// This type represents a single Tensor, with an unknown shape.
struct CAFFE2_API DynamicType : public Type {
  static DynamicTypePtr create() {
    return DynamicTypePtr(new DynamicType()); // NOLINT(modernize-make-shared)
  }
  DEFINE_IS_SUBCLASS(DynamicType);

  bool requires_grad() const override { return true; }

  bool operator==(const Type& rhs) const override {
    return rhs.kind() == kind();
  }
  bool isSubtypeOf(const TypePtr rhs) const override {
    if(auto rhs_ = rhs->cast<OptionalType>()) {
      return this->isSubtypeOf(rhs_->getElementType());
    }
    return Type::isSubtypeOf(rhs);
  }
  std::string str() const override {
    return "Tensor";
  }
  static const TypeKind Kind = TypeKind::DynamicType;
  // global singleton
  static DynamicTypePtr get();
protected:
  DynamicType(TypeKind kind=TypeKind::DynamicType)
  : Type(kind) {}
};

struct UndefinedTensorType;
using UndefinedTensorTypePtr = std::shared_ptr<UndefinedTensorType>;
// This type represents an undefined tensor.
struct CAFFE2_API UndefinedTensorType : public DynamicType {
  static UndefinedTensorTypePtr create() {
    return UndefinedTensorTypePtr(new UndefinedTensorType()); // NOLINT(modernize-make-shared)
  }

  DEFINE_IS_SUBCLASS(UndefinedTensorType);

  bool requires_grad() const override { return false; }

  bool operator==(const Type& rhs) const override {
    return rhs.kind() == kind();
  }
  bool isSubtypeOf(const TypePtr rhs) const override {
    return rhs->kind() == TypeKind::DynamicType ||
           rhs->kind() == TypeKind::UndefinedTensorType ||
           DynamicType::isSubtypeOf(rhs);
  }
  std::string str() const override {
    return "UndefinedTensor";
  }

  static const TypeKind Kind = TypeKind::UndefinedTensorType;
  // global singleton
  static UndefinedTensorTypePtr get();
protected:
  UndefinedTensorType(): DynamicType(TypeKind::UndefinedTensorType) {}
};

struct TensorType;
using TensorTypePtr = std::shared_ptr<TensorType>;
// This type represents a single Tensor with a specific size
struct CAFFE2_API TensorType : public DynamicType {
  template<typename ... T>
  static TensorTypePtr create( T&& ... all ) {
    return TensorTypePtr(new TensorType( std::forward<T>(all)... )); // NOLINT(modernize-make-shared)
  }

  at::ScalarType scalarType() const { return scalar_type_; }
  int device() const { return device_; }
  int dim() const { return dim_; }
  bool requires_grad() const override { return requires_grad_; }

  TensorTypePtr toScalarType(at::ScalarType type){
    auto t = TensorType::create(*this);
    t->scalar_type_ = type;
    return t;
  }
  TensorTypePtr withDim(int new_dim) {
    auto t = TensorType::create(*this);
    t->dim_ = new_dim;
    return t;
  }
  TensorTypePtr withRequiresGrad(bool req) {
    auto t = TensorType::create(*this);
    t->requires_grad_ = req;
    return t;
  }

  bool operator==(const Type& rhs) const override {
    if (rhs.kind() != TypeKind::TensorType)
      return false;
    auto rt = rhs.expect<TensorType>();
    return scalarType() == rt->scalarType() &&
           device() == rt->device() &&
           dim() == rt->dim();
  }
  bool isSubtypeOf(const TypePtr rhs) const override {
    return rhs->kind() == TypeKind::DynamicType ||
          (rhs->kind() == TypeKind::TensorType && Type::isSubtypeOf(rhs)) ||
          DynamicType::isSubtypeOf(rhs);
  }
  bool isSubclass(const TypeKind kind) const override {
    return kind == TypeKind::DynamicType ||
        kind == TypeKind::TensorType;
  }
  std::string str() const override {
    // str is used for user-facing error messages, where we
    // don't want to reveal underlying size information.
    return "Tensor";
  }

  static const TypeKind Kind = TypeKind::TensorType;

protected:
  TensorType(const at::Tensor& tensor, TypeKind kind=TypeKind::TensorType)
    : TensorType(tensor.type().scalarType(),
                 tensor.is_cuda() ? tensor.get_device() : -1,
                 tensor.dim(),
                 tensor.is_variable() && tensor.requires_grad(),
                 kind) {}
  TensorType(at::ScalarType scalar_type, int device, int dim, bool requires_grad=true, TypeKind kind=TypeKind::TensorType)
    : DynamicType(kind)
    , scalar_type_(scalar_type)
    , requires_grad_(at::isFloatingType(scalar_type) && requires_grad)
    , device_(device)
    , dim_(dim) {}

  at::ScalarType scalar_type_;
  bool requires_grad_;
  int device_;
  int dim_;
};

struct CompleteTensorType;
using CompleteTensorTypePtr = std::shared_ptr<CompleteTensorType>;
// This type represents a single Tensor with a specific size
struct CAFFE2_API CompleteTensorType : public TensorType {
  template<typename ... T>
  static CompleteTensorTypePtr create( T&& ... all ) {
    return CompleteTensorTypePtr(new CompleteTensorType( std::forward<T>(all)... )); // NOLINT(modernize-make-shared)
  }

  // overloaded create variadic template argument as it could not distinguish initializer list
  static CompleteTensorTypePtr create(at::ScalarType scalar_type, int device, at::IntList sizes) {
    return CompleteTensorTypePtr(new CompleteTensorType(scalar_type, device, sizes)); // NOLINT(modernize-make-shared)
  }
  static CompleteTensorTypePtr create(at::ScalarType scalar_type, int device, at::IntList sizes, at::IntList strides) {
    return CompleteTensorTypePtr(new CompleteTensorType(scalar_type, device, sizes, strides)); // NOLINT(modernize-make-shared)
  }

  const std::vector<int64_t>& sizes() const { return sizes_; }
  const std::vector<int64_t>& strides() const { return strides_; }

  TypePtr withSizesStrides(at::IntList sizes, at::IntList strides) const {
    return CompleteTensorType::create(scalar_type_, device_, sizes, strides);
  }

  TypePtr withSizes(at::IntList sizes) const {
    return withSizesStrides(sizes, CompleteTensorType::contiguousStridesOf(sizes));
  }

  CompleteTensorTypePtr contiguous() const {
    auto t = CompleteTensorType::create(*this);
    t->strides_ = CompleteTensorType::contiguousStridesOf(sizes_);
    return t;
  }

  CompleteTensorTypePtr toScalarType(at::ScalarType type){
    auto t = CompleteTensorType::create(*this);
    t->scalar_type_ = type;
    return t;
  }

  bool operator==(const Type& rhs) const override {
    if(rhs.kind() != kind())
      return false;
    auto rt = rhs.expect<CompleteTensorType>();
    return scalarType() == rt->scalarType() &&
           sizes() == rt->sizes() &&
           strides() == rt->strides() &&
           device() == rt->device();
  }
  bool isSubtypeOf(const TypePtr rhs) const override {
    if (rhs->kind() == TypeKind::TensorType)
      return *expect<TensorType>() ==  *rhs;
    return rhs->kind() == TypeKind::DynamicType ||
           DynamicType::isSubtypeOf(rhs);
  }
  bool isSubclass(const TypeKind kind) const override {
    return kind == TypeKind::DynamicType ||
           kind == TypeKind::TensorType ||
           kind == TypeKind::CompleteTensorType;
  }
  std::string str() const override {
    // str is used for user-facing error messages, where we
    // don't want to reveal underlying size information.
    return "Tensor";
  }
  bool numel() const {
    size_t prod = 1;
    for(auto s : sizes()) {
      prod *= s;
    }
    return prod;
  }

  static const TypeKind Kind = TypeKind::CompleteTensorType;

  static TypePtr fromNumberType(TypePtr typ);
  static TypePtr fromBoolType();

private:
  CompleteTensorType(const at::Tensor& tensor)
    : TensorType(tensor, TypeKind::CompleteTensorType)
    , sizes_(tensor.sizes().vec())
    , strides_(tensor.strides().vec()) {}
  CompleteTensorType(at::ScalarType scalar_type, int device, at::IntList sizes, bool requires_grad=true)
    : CompleteTensorType(scalar_type, device, sizes, CompleteTensorType::contiguousStridesOf(sizes), requires_grad) {}
  CompleteTensorType(at::ScalarType scalar_type, int device, at::IntList sizes, at::IntList strides, bool requires_grad=true)
    : TensorType(scalar_type, device, sizes.size(), requires_grad, TypeKind::CompleteTensorType)
    , sizes_(sizes.vec())
    , strides_(strides.vec()) {}

  static std::vector<int64_t> contiguousStridesOf(at::IntList sizes) {
    std::vector<int64_t> strides(sizes.size());
    if(sizes.empty()) // zero-dim case
      return strides;
    strides.back() = 1;
    for(size_t i = strides.size() - 1; i > 0; i--) {
      strides[i-1] = strides[i] * sizes[i];
    }
    return strides;
  }

  std::vector<int64_t> sizes_;
  std::vector<int64_t> strides_;
};

// common base for all types that have a single sub element
// e.g. Future[T], Option[T], List[T]
template<TypeKind K, typename T>
struct SingleElementType : public Type {
  static const TypeKind Kind = K;
  TypePtr getElementType() const {
    return elem;
  }
  bool hasFreeVariables() const override {
    return has_free_variables_;
  }
  at::ArrayRef<TypePtr> containedTypes() const override {
    return elem;
  }
  bool requires_grad() const override {
    return elem->requires_grad();
  }
  bool operator==(const Type& rhs) const override {
    if(auto rhs_ = rhs.cast<T>()) {
      return *getElementType() == *rhs_->getElementType();
    }
    return false;
  }
protected:
  SingleElementType(TypePtr elem)
  : Type(Kind)
  , elem(std::move(elem))
  , has_free_variables_(getElementType()->hasFreeVariables()) {}
private:
  TypePtr elem;
  bool has_free_variables_;
};

struct ListType;
using ListTypePtr = std::shared_ptr<ListType>;
struct CAFFE2_API ListType : public SingleElementType<TypeKind::ListType, ListType> {
  // It's not exactly a singleton, but there should be exactly once instance of
  // List[T] for every T
  friend struct Type;
  template<typename ... T>
  static ListTypePtr create( T&& ... all ) {
    return ListTypePtr(new ListType( std::forward<T>(all)... )); // NOLINT(modernize-make-shared)
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
  TypePtr createWithContained(std::vector<TypePtr> contained_types) const override {
    return create(contained_types.at(0));
  }
  // common cast List[Tensor]
  static ListTypePtr ofTensors();
  static ListTypePtr ofInts();
  static ListTypePtr ofFloats();
  static ListTypePtr ofBools();
private:
  using SingleElementType::SingleElementType;
};

struct FutureType;
using FutureTypePtr = std::shared_ptr<FutureType>;

struct CAFFE2_API FutureType : public Type {
  friend struct Type;
  template<typename ... T>
  static FutureTypePtr create(TypePtr elem) {
    return FutureTypePtr(new FutureType(std::move(elem))); // NOLINT(modernize-make-shared)
  }

  DEFINE_IS_SUBCLASS(FutureType);

  bool operator==(const Type& rhs) const override {
    if (auto rhs_ = rhs.cast<FutureType>()) {
      return *getElementType() == *rhs_->getElementType();
    }
    return false;
  }
  bool requires_grad() const override {
    return elem->requires_grad();
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
  TypePtr getElementType() const {
    return elem;
  }
  bool hasFreeVariables() const override {
    return has_free_variables_;
  }

  static const TypeKind Kind = TypeKind::FutureType;
private:
  FutureType(TypePtr elem)
  : Type(TypeKind::FutureType)
  , elem(std::move(elem))
  , has_free_variables_(getElementType()->hasFreeVariables()) {}
  TypePtr elem;
  bool has_free_variables_;
};

struct TupleType;
using TupleTypePtr = std::shared_ptr<TupleType>;
// This type represents a Tuple
struct CAFFE2_API TupleType : public Type {
  static TupleTypePtr create(std::vector<TypePtr> types) {
    return TupleTypePtr(new TupleType( std::move(types) )); // NOLINT(modernize-make-shared)
  }
  DEFINE_IS_SUBCLASS(TupleType);
  at::ArrayRef<TypePtr> elements() const {
    return elements_;
  }
  bool operator==(const Type& rhs) const override {
    return compare(rhs, [](const TypePtr a, const TypePtr b) {
      return *a == *b;
    });
  }
  bool isSubtypeOf(const TypePtr rhs) const override {
    // co-variant rules for tuples
    return compare(*rhs, [](const TypePtr a, const TypePtr b) {
      return a->isSubtypeOf(b);
    });
  }
  bool requires_grad() const override {
    return std::any_of(elements_.begin(), elements_.end(),
                       [](const TypePtr& ptr) { return ptr->requires_grad(); });
  }
  std::string str() const override {
    std::stringstream ss;
    ss << "(";
    for(size_t i = 0; i < elements().size(); ++i) {
      if(i > 0)
        ss << ", ";
      ss << elements()[i]->str();
    }
    ss << ")";
    return ss.str();
  }
  std::string python_str() const override {
    std::stringstream ss;
    ss << "Tuple[";
    for(size_t i = 0; i < elements().size(); ++i) {
      if(i > 0)
        ss << ", ";
      ss << elements()[i]->python_str();
    }
    ss << "]";
    return ss.str();
  }
  bool hasFreeVariables() const override {
    return has_free_variables_;
  }

  at::ArrayRef<TypePtr> containedTypes() const override {
    return elements_;
  }
  TypePtr createWithContained(std::vector<TypePtr> contained_types) const override {
    return create(std::move(contained_types));
  }

  static const TypeKind Kind = TypeKind::TupleType;
private:
  TupleType(std::vector<TypePtr> elements_)
  : Type(TypeKind::TupleType)
  , elements_(std::move(elements_)) {
    has_free_variables_ =
        std::any_of(elements_.begin(), elements_.end(), [](TypePtr v) {
          return v->hasFreeVariables();
        });
  }

  bool compare(const Type& rhs, std::function<bool(const TypePtr, const TypePtr)> fn) const {
    if(rhs.kind() != kind())
      return false;
    const auto & l_elements = elements();
    const auto & r_elements = rhs.cast<TupleType>()->elements();
    if(l_elements.size() != r_elements.size())
      return false;
    for(size_t i = 0; i < l_elements.size(); ++i) {
      if(!fn(l_elements[i], r_elements[i]))
        return false;
    }
    return true;
  }
  std::vector<TypePtr> elements_;
  bool has_free_variables_;
};

struct NumberType;
using NumberTypePtr = std::shared_ptr<NumberType>;
// This type represents a Python number
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
  static const TypeKind Kind = TypeKind::NumberType;
  // global singleton
  static NumberTypePtr get();
private:
  NumberType()
  : Type(TypeKind::NumberType) {}
};

struct FloatType;
using FloatTypePtr = std::shared_ptr<FloatType>;
// This type represents a Python float number
struct CAFFE2_API FloatType : public Type {
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
  bool isSubtypeOf(const TypePtr rhs) const override {
    if(auto rhs_ = rhs->cast<OptionalType>()) {
      return this->isSubtypeOf(rhs_->getElementType());
    }
    return *this == *rhs || rhs->kind() == TypeKind::NumberType;
  }
  static const TypeKind Kind = TypeKind::FloatType;
  // global singleton
  static FloatTypePtr get();
private:
  FloatType()
  : Type(TypeKind::FloatType) {}
};

struct IntType;
using IntTypePtr = std::shared_ptr<IntType>;
// This type represents a Python int number
struct CAFFE2_API IntType : public Type {
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
  bool isSubtypeOf(const TypePtr rhs) const override {
    if(auto rhs_ = rhs->cast<OptionalType>()) {
      return this->isSubtypeOf(rhs_->getElementType());
    }
    return *this == *rhs || rhs->kind() == TypeKind::NumberType;
  }
  static const TypeKind Kind = TypeKind::IntType;
  // global singleton
  static IntTypePtr get();
private:
  IntType()
  : Type(TypeKind::IntType) {}
};

struct BoolType;
using BoolTypePtr = std::shared_ptr<BoolType>;
// This node represents a Python bool value
struct CAFFE2_API BoolType : public Type {
  static BoolTypePtr create( ) {
    return BoolTypePtr(new BoolType());
  }
  DEFINE_IS_SUBCLASS(BoolType);
  bool operator==(const Type& rhs) const override {
    return rhs.kind() == kind();
  }
  std::string str() const override {
    return "bool";
  }
  bool isSubtypeOf(const TypePtr rhs) const override {
    return *this == *rhs || rhs->kind() == TypeKind::BoolType;
  }
  static const TypeKind Kind = TypeKind::BoolType;
  // global singleton
  static BoolTypePtr get();
private:
  BoolType()
  : Type(TypeKind::BoolType) {}
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
    return "string";
  }
  bool isSubtypeOf(const TypePtr rhs) const override {
    if(auto rhs_ = rhs->cast<OptionalType>()) {
      return this->isSubtypeOf(rhs_->getElementType());
    }
    return *this == *rhs;
  }
  static const TypeKind Kind = TypeKind::StringType;
  // global singleton
  static StringTypePtr get();
private:
  StringType()
  : Type(TypeKind::StringType) {}
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
    return rhs->kind() == TypeKind::NoneType ||
           rhs->kind() == TypeKind::OptionalType;
  }

  std::string str() const override {
    return "None";
  }
  static const TypeKind Kind = TypeKind::NoneType;
  // global singleton
  static NoneTypePtr get();
private:
  NoneType()
  : Type(TypeKind::NoneType) {}
};

struct GeneratorType;
using GeneratorTypePtr = std::shared_ptr<GeneratorType>;
// This type represents a Generator
struct CAFFE2_API GeneratorType : public Type {
  static GeneratorTypePtr create() {
    return GeneratorTypePtr(new GeneratorType()); // NOLINT(modernize-make-shared)
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
  GeneratorType()
  : Type(TypeKind::GeneratorType) {}
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

CAFFE2_API std::ostream& operator<<(std::ostream & out, const Type & t);
// what is the type, ignoring extra size/shape information?
// e.g. Tensor(2x3) -> Dynamic, and Tuple(Tensor(2x3),...) -> Tuple(Dynamic,...)

inline TypePtr unshapedType(const TypePtr& type) {
  if (type->kind() == TypeKind::TensorType ||
      type->kind() == TypeKind::CompleteTensorType) {
    return DynamicType::get();
  }
  return type->withContained(fmap(type->containedTypes(), unshapedType));
}

inline TypePtr CompleteTensorType::fromNumberType(TypePtr typ) {
  AT_ASSERT(typ->isSubtypeOf(NumberType::get()));
  if (typ->isSubtypeOf(IntType::get())) {
    return CompleteTensorType::create(at::kLong, -1, {});
  } else if (typ->isSubtypeOf(FloatType::get())) {
    return CompleteTensorType::create(at::kFloat, -1, {});
  } else if (typ->isSubtypeOf(BoolType::get())) {
    return CompleteTensorType::create(at::kLong, -1, {});
  }
  AT_ERROR("unknown number type", typ->str());
}

inline TypePtr CompleteTensorType::fromBoolType() {
  return CompleteTensorType::create(at::kLong, -1, {});
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

template <typename T>
TypePtr getTypePtr() {
#define TYPE_STR(Type) #Type, " ",
  AT_ERROR(
      "Type ",
      c10::demangle_type<T>(),
      " could not be converted to any of the known types { ",
      C10_FORALL_TYPES(TYPE_STR) "}");
#undef TYPE_STR
  return nullptr;
}

template<> inline TypePtr getTypePtr<at::Tensor>() { return DynamicType::get(); }
template<> inline TypePtr getTypePtr<double>() { return FloatType::get(); }
template<> inline TypePtr getTypePtr<int64_t>() { return IntType::get(); }
template<> inline TypePtr getTypePtr<bool>() { return BoolType::get(); }
template<> inline TypePtr getTypePtr<at::Scalar>() { return NumberType::get(); }
template<> inline TypePtr getTypePtr<std::string>() { return StringType::get(); }
template<> inline TypePtr getTypePtr<std::vector<at::Tensor>>() { return ListType::ofTensors(); }
template<> inline TypePtr getTypePtr<std::vector<double>>() { return ListType::ofFloats(); }
template<> inline TypePtr getTypePtr<std::vector<int64_t>>() { return ListType::ofInts(); }

CAFFE2_API TypePtr inferTypeFrom(const IValue& value);

struct CAFFE2_API TypeMatchError : public std::exception {
  TypeMatchError(std::string msg_)
  : msg_(std::move(msg_)) {}
  const char * what() const noexcept override {
    return msg_.c_str();
  }
private:
  std::string msg_;
};
using TypeEnv = std::unordered_map<std::string, TypePtr>;
CAFFE2_API TypePtr matchTypeVariables(TypePtr formal, TypePtr actual, TypeEnv & type_env);
CAFFE2_API TypePtr evalTypeVariables(TypePtr type, TypeEnv & type_env);

} // namespace c10
