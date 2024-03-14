#include <torch/csrc/python_headers.h>

#include <torch/csrc/jit/python/pybind_utils.h>
#include <torch/csrc/utils/pybind.h>
#include <pybind11/functional.h>

#include <c10/cuda/MemPool.h>

template <typename T>
using shared_ptr_class_ = py::class_<T, std::shared_ptr<T>>;

void THCPMemPool_init(PyObject* module) {
  auto torch_C_m = py::handle(module).cast<py::module>();
  shared_ptr_class_<::c10::cuda::MemPool>(torch_C_m, "_MemPool")
      .def(py::init<uint64_t, uint64_t, bool>());
  shared_ptr_class_<::c10::cuda::MemPoolContext>(torch_C_m, "_MemPoolContext")
      .def(py::init<c10::cuda::MemPool*>());
}
