#pragma once

#include "torch/csrc/jit/interned_strings.h"
#include "torch/csrc/jit/generic_if.h"
#include "torch/csrc/assertions.h"

#include <ATen/ATen.h>

#include <memory>
#include <iostream>

namespace torch { namespace jit {

#define TH_FORALL_TYPES(_) \
_(DynamicType) \
_(TensorType) \
_(HandleType) \
_(TupleType)

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
  virtual std::string name() const = 0;
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
  virtual bool operator==(const Type& rhs) const override {
    return rhs.kind() == kind();
  }
  virtual std::string name() const override {
    return "Tensor";
  }
  static const TypeKind Kind = TypeKind::DynamicType;
  // global singleton
  static TypePtr get();
};

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
  const std::vector<std::int64_t>& sizes() const { return sizes_; }
  const std::vector<std::int64_t>& strides() const { return strides_; }

  TypePtr withSizesStrides(at::IntList sizes, at::IntList strides) const {
    return std::make_shared<TensorType>(scalar_type_, device_, sizes, strides);
  }

  TypePtr withSizes(at::IntList sizes) const {
    return withSizesStrides(sizes, TensorType::contiguousStridesOf(sizes));
  }

  TypePtr contiguous() const {
    auto t = std::make_shared<TensorType>(*this);
    t->strides_ = TensorType::contiguousStridesOf(sizes_);
    return t;
  }
  virtual bool operator==(const Type& rhs) const override {
    if(rhs.kind() != kind())
      return false;
    auto rt = rhs.expect<TensorType>();
    return scalarType() == rt->scalarType() &&
           sizes() == rt->sizes() &&
           strides() == rt->strides() &&
           device() == rt->device();
  }
  virtual bool isSubtypeOf(const Type& rhs) const override {
    return *this == rhs || rhs.kind() == TypeKind::DynamicType;
  }
  virtual std::string name() const override {
    std::string retval = std::string(at::toString(scalarType())) + "Tensor[";
    for (size_t i=0; i < sizes_.size(); ++i) {
      retval += std::to_string(sizes_[i]) + (i == sizes_.size() - 1 ? "" : ",");
    }
    retval += "]";
    return retval;
  }
private:
  static std::vector<int64_t> contiguousStridesOf(at::IntList sizes) {
    std::vector<int64_t> strides(sizes.size());
    if(sizes.size() == 0) // zero-dim case
      return strides;
    strides.back() = 1;
    for(std::size_t i = strides.size() - 1; i > 0; i--) {
      strides[i-1] = strides[i] * sizes[i];
    }
    return strides;
  }
  at::ScalarType scalar_type_;
  int device_;
  std::vector<int64_t> sizes_;
  std::vector<int64_t> strides_;
};

// This value represents an opaque handle to external state.
// Operators that produce/consume values of this type agree on
// the format.

/* Example Usage: passing state to opaque autograd Functions:
graph(%1, %8) {
  %2.0, %2.1 = ^AddConstant(2, False)(%1) // first output is Type::Handle, containing ctx
  %4.0, %4.1 = ^Add(False)(%2.1, %1) // first output is Type::Handle, containing ctx
  %6.0, %6.1 = ^Abs()(%4.1) // first output is Type::Handle, containing ctx
  ---------------- stage 1 ----------------
  %13 = AutogradOp[AbsBackward](%6.0, %8) // first argument is Type::Handle, consuming ctx
  %15 = AutogradOp[AddBackward](%4.0, %13.0) // first argument is Type::Handle, consuming ctx
  %18 = AutogradOp[AddConstantBackward](%2.0, %15.1) // first argument is Type::Handle, consuming ctx
  %20 = AutogradOp[N5torch8autograd3AddE](%18.0, %18.0)
  return (%6.0, %20.0);
}
*/
struct HandleType : public Type {
  friend struct Type;
  HandleType()
    : Type(TypeKind::HandleType) {}
  virtual bool operator==(const Type& rhs) const override {
    return rhs.kind() == kind();
  }
  virtual std::string name() const override {
    return "Handle";
  }
  static const TypeKind Kind = TypeKind::HandleType;
  // global singleton
  static TypePtr get();
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
  virtual bool operator==(const Type& rhs) const override {
    return compare(rhs, [](const Type& a, const Type& b) {
      return a == b;
    });
  }
  virtual bool isSubtypeOf(const Type& rhs) const override {
    return compare(rhs, [](const Type& a, const Type&b) {
      return a.isSubtypeOf(b);
    });
  }
  virtual std::string name() const override {
    std::stringstream ss;
    ss << "(";
    for(size_t i = 0; i < elements().size(); ++i) {
      if(i > 0)
        ss << ", ";
      ss << elements()[i]->name();
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



std::ostream& operator<<(std::ostream & out, const Type & t);

}} // namespace torch::jit
