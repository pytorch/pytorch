#pragma once

// ${generated_comment}

#include "ATen/Type.h"

namespace at {

struct AT_API TypeBase : public Type {
  explicit TypeBase(TensorTypeId type_id, bool is_variable, bool is_undefined)
      : Type(type_id, is_variable, is_undefined) {}

  // Make sure overload resolution considers the nullary virtual method.
  // (A single argument overload is generated in the list.)
  bool is_cuda() const override = 0;
  bool is_sparse() const override = 0;
  bool is_distributed() const override = 0;

  Type & toBackend(Backend b) const override;
  Type & toScalarType(ScalarType s) const override;

  Tensor copy(const Tensor & src, bool non_blocking=false) const override;
  Tensor & copy_(Tensor & self, const Tensor & src, bool non_blocking=false) const override;

  Tensor tensorFromBlob(void * data, IntList sizes, const std::function<void(void*)> & deleter=noop_deleter) const override;
  Tensor tensorFromBlob(void * data, IntList sizes, IntList strides, const std::function<void(void*)> & deleter=noop_deleter) const override;
  Tensor tensorWithAllocator(IntList sizes, Allocator* allocator) const override;
  Tensor tensorWithAllocator(IntList sizes, IntList strides, Allocator* allocator) const override;
  Tensor scalarTensor(Scalar s) const override;

  // example
  // virtual Tensor * add(Tensor & a, Tensor & b) = 0;
  ${type_method_declarations}
};

} // namespace at
