#pragma once

// ${generated_comment}

#include <ATen/TypeExtendedInterface.h>

namespace at {

struct CAFFE2_API TypeDefault : public TypeExtendedInterface {
  explicit TypeDefault(TensorTypeId type_id, bool is_variable, bool is_undefined)
      : TypeExtendedInterface(type_id, is_variable, is_undefined) {}

  // Make sure overload resolution considers the nullary virtual method.
  // (A single argument overload is generated in the list.)
  bool is_cuda() const override {
    return backend() == Backend::CUDA || backend() == Backend::SparseCUDA;
  }
  bool is_hip() const override {
    return backend() == Backend::HIP || backend() == Backend::SparseHIP;
  }
  bool is_sparse() const override {
    return backend() == Backend::SparseCPU || backend() == Backend::SparseCUDA || backend() == Backend::SparseHIP;
  }
  bool is_quantized() const override {
    return backend() == Backend::QuantizedCPU;
  }
  bool is_distributed() const override {
    return false;
  }

  Type & toBackend(Backend b) const override;
  Type & toScalarType(ScalarType s) const override;

  void backward(
      Tensor& self,
      c10::optional<Tensor> gradient,
      bool keep_graph,
      bool create_graph) const override;
  void set_data(Tensor & self, Tensor new_data) const override;

  Storage unsafeStorageFromTH(void * th_pointer, bool retain) const override;
  Tensor unsafeTensorFromTH(void * th_pointer, bool retain) const override;

  // example
  // virtual Tensor * add(Tensor & a, Tensor & b) = 0;
  ${type_method_declarations}
};

} // namespace at
