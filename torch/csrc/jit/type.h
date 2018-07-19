#pragma once

#include "torch/csrc/jit/interned_strings.h"
#include "torch/csrc/assertions.h"

#include <ATen/ATen.h>

#include <memory>
#include <iostream>

namespace torch { namespace jit {

#define TH_FORALL_TYPES(_) \
_(DynamicType) \
_(TensorType) \
_(TupleType) \
_(ListType) \
_(NumberType) \
_(FloatType) \
_(IntType) \

enum class TypeKind {
#define DEFINE_TYPE(T) T,
  TH_FORALL_TYPES(DEFINE_TYPE)
#undef DEFINE_TYPE
};

struct Type;
using TypePtr = std::shared_ptr<Type>;


struct Type : std::enable_shared_from_this<Type> {

private:
  TypeKind kind_;

protected:
  Type(TypeKind kind)
    : kind_(kind) {}

public:
  virtual bool operator==(const Type& rhs) const = 0;

  // subtyping relation. By default, we return true for the case
  // when the type is exactly equal
  virtual bool isSubtypeOf(const Type& rhs) const {
    return *this == rhs;
  }
  // user-friendly form of the type, separate from
  // operator<< which is verbose and unambiguous
  virtual std::string str() const = 0;

  TypeKind kind() const {
    return kind_;
  }

  // Dynamically cast this object to the subclass indicated by the
  // template variable, returning nullptr if the cast is invalid..
  template<typename T>
  T* cast() {
    if (T::Kind == kind())
      return static_cast<T*>(this);
    return nullptr;
  }
  template<typename T>
  const T* cast() const {
    if (T::Kind == kind())
      return static_cast<const T*>(this);
    return nullptr;
  }
  template<typename T>
  T* expect() {
    JIT_ASSERT(T::Kind == kind());
    return static_cast<T*>(this);
  }
  template<typename T>
  const T* expect() const {
    JIT_ASSERT(T::Kind == kind());
    return static_cast<const T*>(this);
  }
  std::shared_ptr<Type> asShared() {
    return shared_from_this();
  }
  virtual ~Type() {}
};

inline bool operator!=(const Type & lhs, const Type & rhs) {
  return !(lhs == rhs);
}

// This node represents a single Tensor value, with an unknown shape.
struct DynamicType : public Type {
  DynamicType()
  : Type(TypeKind::DynamicType) {}
  bool operator==(const Type& rhs) const override {
    return rhs.kind() == kind();
  }
  std::string str() const override {
    return "Tensor";
  }
  static const TypeKind Kind = TypeKind::DynamicType;
  // global singleton
  static TypePtr get();
};

struct TensorType;
using TensorTypePtr = std::shared_ptr<TensorType>;
// This node represents a single Tensor value with a specific size
struct TensorType : public Type {
  friend struct Type;
  TensorType(const at::Tensor& tensor)
    : Type(TypeKind::TensorType)
    , scalar_type_(tensor.type().scalarType())
    , device_(tensor.type().is_cuda() ? tensor.get_device() : -1)
    , sizes_(tensor.sizes())
    , strides_(tensor.strides()) {}
  TensorType(at::ScalarType scalar_type, int device, at::IntList sizes)
    : TensorType(scalar_type, device, sizes, TensorType::contiguousStridesOf(sizes)) {}
  TensorType(at::ScalarType scalar_type, int device, at::IntList sizes, at::IntList strides)
    : Type(TypeKind::TensorType)
    , scalar_type_(scalar_type)
    , device_(device)
    , sizes_(sizes)
    , strides_(strides)
    {}

  static const TypeKind Kind = TypeKind::TensorType;

  at::ScalarType scalarType() const { return scalar_type_; }
  int device() const { return device_; }
  const std::vector<int64_t>& sizes() const { return sizes_; }
  const std::vector<int64_t>& strides() const { return strides_; }

  TypePtr withSizesStrides(at::IntList sizes, at::IntList strides) const {
    return std::make_shared<TensorType>(scalar_type_, device_, sizes, strides);
  }

  TypePtr withSizes(at::IntList sizes) const {
    return withSizesStrides(sizes, TensorType::contiguousStridesOf(sizes));
  }

  TensorTypePtr contiguous() const {
    auto t = std::make_shared<TensorType>(*this);
    t->strides_ = TensorType::contiguousStridesOf(sizes_);
    return t;
  }

  TensorTypePtr toScalarType(at::ScalarType type){
    auto t = std::make_shared<TensorType>(*this);
    t->scalar_type_ = type;
    return t;
  }

  bool operator==(const Type& rhs) const override {
    if(rhs.kind() != kind())
      return false;
    auto rt = rhs.expect<TensorType>();
    return scalarType() == rt->scalarType() &&
           sizes() == rt->sizes() &&
           strides() == rt->strides() &&
           device() == rt->device();
  }
  bool isSubtypeOf(const Type& rhs) const override {
    return *this == rhs || rhs.kind() == TypeKind::DynamicType;
  }
  std::string str() const override {
    // str is used for user-facing error messages, where we
    // don't want to reveal underlying size information.
    return "Tensor";
  }
private:
  static std::vector<int64_t> contiguousStridesOf(at::IntList sizes) {
    std::vector<int64_t> strides(sizes.size());
    if(sizes.size() == 0) // zero-dim case
      return strides;
    strides.back() = 1;
    for(size_t i = strides.size() - 1; i > 0; i--) {
      strides[i-1] = strides[i] * sizes[i];
    }
    return strides;
  }
  at::ScalarType scalar_type_;
  int device_;
  std::vector<int64_t> sizes_;
  std::vector<int64_t> strides_;
};

struct ListType : public Type {
  friend struct Type;
  static const TypeKind Kind = TypeKind::ListType;
  ListType(TypePtr elem)
  : Type(TypeKind::ListType), elem(elem) {}
  bool operator==(const Type& rhs) const override {
    if(auto rhs_ = rhs.cast<ListType>()) {
      return *getElementType() == *rhs_->getElementType();
    }
    return false;
  }
  std::string str() const override {
    std::stringstream ss;
    ss << getElementType()->str() << "[]";
    return ss.str();
  }
  TypePtr getElementType() const {
    return elem;
  }
  // common cast List[Tensor]
  static TypePtr ofTensors();
  static TypePtr ofInts();
private:
  TypePtr elem;
};

struct TupleType : public Type {
  friend struct Type;
  TupleType(std::vector<TypePtr> elements_)
  : Type(TypeKind::TupleType)
  , elements_(std::move(elements_)) {}
  static const TypeKind Kind = TypeKind::TupleType;
  at::ArrayRef<TypePtr> elements() const {
    return elements_;
  }
  bool operator==(const Type& rhs) const override {
    return compare(rhs, [](const Type& a, const Type& b) {
      return a == b;
    });
  }
  bool isSubtypeOf(const Type& rhs) const override {
    // e.g. (Tensor, Tensor, Tensor) <: List[Tensor]
    if(auto lt = rhs.cast<ListType>()) {
      for(auto e : elements()) {
        if(!e->isSubtypeOf(*lt->getElementType()))
          return false;
      }
      return true;
    }
    // co-variant rules for tuples
    return compare(rhs, [](const Type& a, const Type&b) {
      return a.isSubtypeOf(b);
    });
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
private:
  bool compare(const Type& rhs, std::function<bool(const Type&, const Type&)> fn) const {
    if(rhs.kind() != kind())
      return false;
    const auto & l_elements = elements();
    const auto & r_elements = rhs.cast<TupleType>()->elements();
    if(l_elements.size() != r_elements.size())
      return false;
    for(size_t i = 0; i < l_elements.size(); ++i) {
      if(!fn(*l_elements[i], *r_elements[i]))
        return false;
    }
    return true;
  }
  std::vector<TypePtr> elements_;
};

// This node represents a Python number value
struct NumberType : public Type {
  NumberType()
  : Type(TypeKind::NumberType) {}
  bool operator==(const Type& rhs) const override {
    return rhs.kind() == kind();
  }
  std::string str() const override {
    return "Scalar"; // match what PythonArgParser says for clarity
  }
  static const TypeKind Kind = TypeKind::NumberType;
  // global singleton
  static TypePtr get();
};

// This node represents a Python float number value
struct FloatType : public Type {
  FloatType()
  : Type(TypeKind::FloatType) {}
  bool operator==(const Type& rhs) const override {
    return rhs.kind() == kind();
  }
  std::string str() const override {
    return "float";
  }
  bool isSubtypeOf(const Type& rhs) const override {
    return *this == rhs || rhs.kind() == TypeKind::NumberType;
  }
  static const TypeKind Kind = TypeKind::FloatType;
  // global singleton
  static TypePtr get();
};

// This node represents a Python int number value
struct IntType : public Type {
  IntType()
  : Type(TypeKind::IntType) {}
  bool operator==(const Type& rhs) const override {
    return rhs.kind() == kind();
  }
  std::string str() const override {
    return "int";
  }
  bool isSubtypeOf(const Type& rhs) const override {
    return *this == rhs || rhs.kind() == TypeKind::NumberType;
  }
  static const TypeKind Kind = TypeKind::IntType;
  // global singleton
  static TypePtr get();
};


std::ostream& operator<<(std::ostream & out, const Type & t);

}} // namespace torch::jit
