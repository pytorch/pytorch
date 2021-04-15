#pragma once

#include <torch/csrc/utils/python_strings.h>
#include <ATen/Tensor.h>
#include <torch/library.h>

namespace torch {
namespace fx {
struct TORCH_API PythonTensorImpl : public c10::TensorImpl {
  explicit PythonTensorImpl(py::object value): TensorImpl(c10::DispatchKeySet(c10::DispatchKey::PythonKey), caffe2::TypeMeta::Make<float>(),c10::Device(at::kCPU)), value_(value) {

    // asm("int $0x3\n");
    py::object torch_function = PyObject_FastGetAttrString(value.ptr(), "__torch_function__");
    // PyObject_CallFunctionObjArgs(torch_function.ptr(), 0, 0, 0, 0, 0);
  }


  // Returns a reference to BatchDims that represent which dimensions of this
  // tensor are private.

  // Override a bunch of methods inherited from TensorImpl to return error messages.
//   bool is_contiguous_custom(at::MemoryFormat memory_format) const override;
//   void set_size(int64_t dim, int64_t new_size) override;
//   void set_stride(int64_t dim, int64_t new_stride) override;
//   void set_storage_offset(int64_t storage_offset) override;
// #ifdef DEBUG
//   bool has_storage() const override;
// #endif

  py::object value_;
};

void initFx(PyObject* module);

} // namespace fx
} // namespace torch
