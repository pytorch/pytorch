#include <ATen/functorch/TensorWrapper.h>
#include <torch/csrc/utils/tensor_list.h>

#include <c10/util/irange.h>
#include <torch/csrc/Exceptions.h>
#include <torch/csrc/autograd/python_variable.h>
#include <torch/csrc/utils/python_scalars.h>

using namespace at;

namespace torch::utils {

static PyObject* recursive_to_list(
    const char* data,
    IntArrayRef sizes,
    IntArrayRef strides,
    int64_t dim,
    ScalarType scalarType,
    size_t elementSize) {
  int64_t ndim = static_cast<int64_t>(sizes.size());
  if (dim == ndim) {
    return torch::utils::load_scalar(data, scalarType);
  }
  auto n = sizes[dim];
  auto list = THPObjectPtr(PyList_New(n));
  if (!list)
    throw python_error();
  for (const auto i : c10::irange(n)) {
    PyObject* obj = recursive_to_list(
        data, sizes, strides, dim + 1, scalarType, elementSize);
    if (!obj)
      throw python_error();
    PyList_SET_ITEM(list.get(), i, obj);
    auto advance_data_ptr = strides[dim] * elementSize;
    TORCH_INTERNAL_ASSERT(data || (advance_data_ptr == 0));
    data += advance_data_ptr;
  }
  return list.release();
}

const Tensor& recursive_unwrap(const Tensor& tensor) {
  if (auto* wrapper = at::functorch::maybeGetTensorWrapper(tensor))
    return recursive_unwrap(wrapper->value());
  return tensor;
}

PyObject* tensor_to_list(const Tensor& tensor) {
  {
    py::object pytensor =
        py::reinterpret_steal<py::object>(THPVariable_Wrap(tensor));
    TORCH_CHECK(
        !tensor.unsafeGetTensorImpl()->is_python_dispatch(),
        ".tolist() is not supported for tensor subclasses, got ",
        Py_TYPE(pytensor.ptr())->tp_name);
  }
  // check if it is a grad tracking tensor and unwrap.
  Tensor data = tensor.resolve_conj().resolve_neg();
  data = recursive_unwrap(data);
  if (!data.device().is_cpu()) {
    pybind11::gil_scoped_release no_gil;
    data = data.toBackend(Backend::CPU);
    data = recursive_unwrap(data);
  }
  TORCH_CHECK(
      tensor.numel() == 0 || data.const_data_ptr(),
      "tolist() shouldn't be called on a tensor with unallocated storage");
  return recursive_to_list(
      (const char*)data.const_data_ptr(),
      data.sizes(),
      data.strides(),
      0,
      data.scalar_type(),
      tensor.numel() == 0 ? 0 : data.dtype().itemsize());
}

} // namespace torch::utils
