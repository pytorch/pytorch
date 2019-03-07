#pragma once

#include <ATen/core/ATenGeneral.h>
#include <c10/core/Allocator.h>
#include <c10/util/Deprecated.h>
#include <ATen/core/Generator.h>
#include <c10/core/Layout.h>
#include <c10/core/Scalar.h>
#include <c10/core/ScalarType.h>
#include <ATen/core/SparseTensorRef.h>
#include <c10/util/ArrayRef.h>
#include <c10/util/Half.h>
#include <c10/core/TensorTypeIdRegistration.h>
#include <ATen/core/Reduction.h>
#include <c10/core/TensorOptions.h>

#include <c10/util/Optional.h>

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

namespace c10 {
struct Storage;
}

namespace at {

class Tensor;
using TensorList = ArrayRef<Tensor>;

class Context;
struct Generator;

static inline void noop_deleter(void*) {}

enum class TypeID {
  ${type_ids}
  CPUComplexFloat,
  CPUComplexDouble,
  CUDAComplexFloat,
  CUDAComplexDouble,
  Undefined,
  NumOptions
};

struct CAFFE2_API Type {
  explicit Type(TensorTypeId type_id, bool is_variable, bool is_undefined)
      : type_id_(type_id), is_variable_(is_variable), is_undefined_(is_undefined) {}

  virtual ~Type() {}
  virtual ScalarType scalarType() const = 0;
  virtual caffe2::TypeMeta typeMeta() const = 0;
  virtual Backend backend() const = 0;
  Layout layout() const noexcept { return layout_from_backend(backend()); }
  virtual bool is_cuda() const = 0;
  virtual bool is_hip() const = 0;
  virtual bool is_sparse() const = 0;
  virtual bool is_distributed() const = 0;
  bool is_variable() const noexcept { return is_variable_; }
  bool is_undefined() const noexcept { return is_undefined_; }
  virtual Allocator * allocator() const = 0;
  virtual Device getDeviceFromPtr(void * data) const = 0;
  virtual Storage storageFromBlob(void * data, int64_t size, const std::function<void(void*)> & deleter=noop_deleter) const = 0;
  virtual Storage storageWithAllocator(int64_t size, Allocator* allocator) const = 0;
  virtual std::unique_ptr<Generator> generator() const = 0;
  virtual Tensor unsafeTensorFromTH(void * th_pointer, bool retain) const = 0;
  virtual Storage unsafeStorageFromTH(void * th_pointer, bool retain) const = 0;
  virtual const char * toString() const = 0;
  virtual size_t elementSizeInBytes() const = 0;
  virtual Type & toBackend(Backend b) const = 0;
  virtual Type & toScalarType(ScalarType s) const = 0;
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
  Type & hip() const {
    return this->toBackend(at::backendToHIP(this->backend()));
  }
  // contiguous IDs for all types in the system
  // for external dispatch
  virtual TypeID ID() const = 0;

  // New-style TensorTypeId that supports open registration.
  TensorTypeId type_id() const { return type_id_; }

  // NB: This will return DeviceType::CPU for Backend::SparseCPU
  DeviceType device_type() const {
    return backendToDeviceType(backend());
  }

  virtual Tensor copy(
      const Tensor& src,
      bool non_blocking = false,
      c10::optional<Device> to_device = {}) const = 0;
  virtual Tensor & copy_(Tensor & self, const Tensor & src, bool non_blocking=false) const = 0;

  virtual void backward(
      Tensor& self,
      c10::optional<Tensor> gradient,
      bool keep_graph,
      bool create_graph) const = 0;
  virtual void set_data(Tensor & self, Tensor new_data) const = 0;

  virtual Tensor tensorFromBlob(void * data, IntArrayRef sizes, const std::function<void(void*)> & deleter=noop_deleter) const = 0;
  virtual Tensor tensorFromBlob(void * data, IntArrayRef sizes, IntArrayRef strides, const std::function<void(void*)> & deleter=noop_deleter) const = 0;
  virtual Tensor tensorWithAllocator(IntArrayRef sizes, Allocator* allocator) const = 0;
  virtual Tensor tensorWithAllocator(IntArrayRef sizes, IntArrayRef strides, Allocator* allocator) const = 0;

  bool operator==(const Type& other) const {
    return this == &other;
  }
  bool operator!=(const Type& other) const {
    return this != &other;
  }

  /// Constructs the `TensorOptions` from a type and a `device_index`.
  TensorOptions options(int16_t device_index = -1) const {
    return TensorOptions().dtype(typeMeta())
                          .device(device_type(), device_index)
                          .layout(layout())
                          .is_variable(is_variable());
  }

  /// Constructs the `TensorOptions` from a type and a Device.  Asserts that
  /// the device type matches the device type of the type.
  TensorOptions options(c10::optional<Device> device_opt) const {
    if (!device_opt.has_value()) {
      return options(-1);
    } else {
      Device device = device_opt.value();
      AT_ASSERT(device.type() == device_type());
      return options(device.index());
    }
  }

  operator TensorOptions() const {
    return options();
  }

  // example
  // virtual Tensor * add(Tensor & a, Tensor & b) = 0;
  ${pure_virtual_type_method_declarations}
protected:
  TensorTypeId type_id_;
  bool is_variable_;
  bool is_undefined_;
};

} // namespace at

#include <ATen/core/Tensor.h>
