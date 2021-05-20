#include <torch/csrc/utils/pybind.h>
#include <ATen/Tensor.h>
#include <torch/library.h>

namespace at {
namespace functorch {
inline at::Tensor getValueFromPyTensor(const py::object& pyTensor) {
  auto out = pyTensor.attr("value").cast<at::Tensor>();
  return out;
}

struct TORCH_API PythonTensorImpl : public c10::TensorImpl {
  explicit PythonTensorImpl(py::object value): TensorImpl(c10::DispatchKeySet(c10::DispatchKey::FuncTorchPython), getValueFromPyTensor(value).dtype(), c10::Device(at::kCPU)), value_(value) {
    set_storage_access_should_throw();
    set_has_contiguity_policy(HasContiguityPolicy::CustomBehavior);
    // asm("int $0x3\n");
    auto tensor = getValueFromPyTensor(value_);

    const auto value_sizes = tensor.sizes();
    const auto value_strides = tensor.strides();
    sizes_and_strides_.resize(tensor.dim());
    for (int64_t dim = 0; dim < tensor.dim(); dim++) {
      sizes_and_strides_.size_at_unchecked(dim) = value_sizes.at(dim);
      sizes_and_strides_.stride_at_unchecked(dim) = value_strides.at(dim);
    }
    refresh_numel();
    refresh_contiguous();
  }
  c10::intrusive_ptr<c10::TensorImpl> shallow_copy_and_detach(
      const c10::VariableVersion& version_counter,
      bool allow_tensor_metadata_change) const;

  c10::intrusive_ptr<c10::TensorImpl> shallow_copy_and_detach(
      c10::VariableVersion&& version_counter,
      bool allow_tensor_metadata_change) const;


  // Returns a reference to BatchDims that represent which dimensions of this
  // tensor are private.

  // Override a bunch of methods inherited from TensorImpl to return error messages.
  bool is_contiguous_custom(at::MemoryFormat memory_format) const override;
  void set_size(int64_t dim, int64_t new_size) override;
  void set_stride(int64_t dim, int64_t new_stride) override;
  void set_storage_offset(int64_t storage_offset) override;
// #ifdef DEBUG
//   bool has_storage() const override;
// #endif

  py::object value_;
};

PythonTensorImpl* getPythonImpl(at::Tensor tensor);

at::Tensor addPythonKey(const py::object& tensor);
bool hasPythonKey(at::Tensor tensor);

py::object removePythonKey(at::Tensor tensor);
}}
