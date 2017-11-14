#pragma once

#include "torch/csrc/jit/interned_strings.h"
#include "torch/csrc/jit/generic_if.h"
#include "torch/csrc/assertions.h"

#include <ATen/ATen.h>

#include <memory>
#include <iostream>

namespace torch { namespace jit {

#define TH_FORALL_TYPES(_) \
_(TensorType) \
_(HandleType)

enum class TypeKind {
#define DEFINE_TYPE(T) T,
  TH_FORALL_TYPES(DEFINE_TYPE)
#undef DEFINE_TYPE
};

struct Type;
using TypePtr = std::shared_ptr<Type>;

struct Type {
private:
  TypeKind kind_;

protected:
  Type(TypeKind kind)
    : kind_(kind) {}

public:
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
  T* expect() {
    JIT_ASSERT(T::Kind == kind());
    return static_cast<T*>(this);
  }
};

// This node represents a single Tensor value
struct TensorType : public Type {
  friend struct Type;
  TensorType(const at::Tensor& tensor)
    : Type(TypeKind::TensorType)
    , scalar_type_(tensor.type().scalarType())
    , device_(tensor.type().isCuda() ? tensor.get_device() : -1)
    , sizes_(tensor.sizes())
    , strides_(tensor.strides()) {}
  TensorType(at::ScalarType scalar_type, int device, std::vector<int64_t> sizes, std::vector<int64_t> strides)
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

  TypePtr withSizesStrides(const std::vector<std::int64_t>& sizes, const std::vector<std::int64_t>& strides) const {
    return std::make_shared<TensorType>(scalar_type_, device_, sizes, strides);
  }

  TypePtr contiguous() const {
    auto t = std::make_shared<TensorType>(*this);
    t->strides_.resize(sizes_.size());
    t->strides_.back() = 1;
    for(size_t i = t->strides_.size() - 1; i > 0; i--) {
      t->strides_[i-1] = t->strides_[i] * t->sizes_[i];
    }
    return t;
  }
private:
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

public:
  static const TypeKind Kind = TypeKind::HandleType;
};

std::ostream& operator<<(std::ostream & out, const Type & t);

// Immutable case
#define TYPE_IF(x,Kind) GENERIC_IF(const,TypeKind::Kind,x,Kind)
#define TYPE_ELSEIF(Kind) GENERIC_ELSEIF(const,TypeKind::Kind,Kind)
#define TYPE_ELSE() GENERIC_ELSE()
#define TYPE_END() GENERIC_END()

}} // namespace torch::jit
