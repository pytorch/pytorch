#pragma once

// ${generated_comment}

#include "ATen/ATenGeneral.h"
#include "ATen/Allocator.h"
#include "ATen/Deprecated.h"
#include "ATen/Generator.h"
#include "ATen/Layout.h"
#include "ATen/Scalar.h"
#include "ATen/ScalarType.h"
#include "ATen/SparseTensorRef.h"
#include "ATen/Tensor.h"
#include "ATen/core/ArrayRef.h"
#include "ATen/core/Half.h"
#include "ATen/core/TensorTypeIdRegistration.h"
#include "THNN/Reduction.h"

#include <array>
#include <cstddef>
#include <functional>
#include <limits>
#include <memory>

// To solve the conflict of s_addr in inaddr.h
#ifdef _MSC_VER
#ifdef s_addr
#undef s_addr
#endif
#endif

namespace at {

class Context;
struct Allocator;
struct Generator;
struct Storage;

static inline void noop_deleter(void*) {}

enum class TypeID {
  ${type_ids}
  Undefined,
  NumOptions
};

struct AT_API Type {
  explicit Type(Context* context, TensorTypeId type_id, bool is_variable, bool is_undefined)
      : context(context), type_id_(type_id), is_variable_(is_variable), is_undefined_(is_undefined) {}
  virtual ~Type() {}
  virtual ScalarType scalarType() const = 0;
  virtual Backend backend() const = 0;
  Layout layout() const noexcept { return layout_from_backend(backend()); }
  virtual bool is_cuda() const = 0;
  virtual bool is_sparse() const = 0;
  virtual bool is_distributed() const = 0;
  bool is_variable() const noexcept { return is_variable_; }
  bool is_undefined() const noexcept { return is_undefined_; }
  static void registerCPU(Context * context);
  virtual std::unique_ptr<Storage> storage(bool resizable = false) const = 0;
  virtual std::unique_ptr<Storage> storage(size_t size, bool resizable = false) const = 0;
  virtual std::unique_ptr<Storage> storageFromBlob(void * data, int64_t size, const std::function<void(void*)> & deleter=noop_deleter) const = 0;
  virtual std::unique_ptr<Storage> storageWithAllocator(int64_t size, Allocator* allocator) const = 0;
  virtual std::unique_ptr<Generator> generator() const = 0;
  virtual Tensor unsafeTensorFromTH(void * th_pointer, bool retain) const = 0;
  virtual std::unique_ptr<Storage> unsafeStorageFromTH(void * th_pointer, bool retain) const = 0;
  virtual const char * toString() const = 0;
  virtual size_t elementSizeInBytes() const = 0;
  virtual Type & toBackend(Backend b) const;
  virtual Type & toScalarType(ScalarType s) const;
  Type & toSparse() const {
    return this->toBackend(at::toSparse(this->backend()));
  }
  Type & toDense() const {
    return this->toBackend(at::toDense(this->backend()));
  }
  Type & cpu() const {
    return this->toBackend(at::backendToCPU(this->backend()));
  }
  Type & cuda() const {
    return this->toBackend(at::backendToCUDA(this->backend()));
  }
  Context& get_context() const { return *context; }

  // contiguous IDs for all types in the system
  // for external dispatch
  virtual TypeID ID() const = 0;

  // New-style TensorTypeId that supports open registration.
  TensorTypeId type_id() const { return type_id_; }

  // NB: This will return DeviceType::CPU for Backend::SparseCPU
  DeviceType device_type() const {
    return backendToDeviceType(backend());
  }

  Tensor copy(const Tensor & src, bool non_blocking=false) const;
  Tensor & copy_(Tensor & self, const Tensor & src, bool non_blocking=false) const;
  virtual Tensor & s_copy_(Tensor & self, const Tensor & src, bool non_blocking) const = 0;
  virtual Tensor & _s_copy_from(const Tensor & self, Tensor & dst, bool non_blocking) const = 0;

  Tensor tensorFromBlob(void * data, IntList sizes, const std::function<void(void*)> & deleter=noop_deleter) const;
  Tensor tensorFromBlob(void * data, IntList sizes, IntList strides, const std::function<void(void*)> & deleter=noop_deleter) const;
  Tensor tensorWithAllocator(IntList sizes, Allocator* allocator) const;
  Tensor tensorWithAllocator(IntList sizes, IntList strides, Allocator* allocator) const;
  Tensor scalarTensor(Scalar s) const;

  bool operator==(const Type& other) const;
  bool operator!=(const Type& other) const;

  // example
  // virtual Tensor * add(Tensor & a, Tensor & b) = 0;
  ${type_method_declarations}
protected:
  Context* context;
  TensorTypeId type_id_;
  bool is_variable_;
  bool is_undefined_;

};

inline bool Tensor::is_variable() const noexcept {
  return type().is_variable();
}

inline ScalarType Tensor::dtype() const noexcept {
  return type().scalarType();
}

inline Layout Tensor::layout() const noexcept {
  return type().layout();
}

inline Device Tensor::device() const {
  return Device(type().device_type(), type().is_cuda() ? get_device() : -1);
}

} // namespace at
