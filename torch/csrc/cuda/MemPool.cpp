#include <torch/csrc/python_headers.h>

#include <torch/csrc/jit/python/pybind_utils.h>
#include <torch/csrc/utils/pybind.h>

#include <c10/cuda/CUDACachingAllocator.h>

template <typename T>
using shared_ptr_class_ = py::class_<T, std::shared_ptr<T>>;

void THCPMemPool_init(PyObject* module) {
  auto torch_C_m = py::handle(module).cast<py::module>();
  shared_ptr_class_<::c10::cuda::MemPool>(torch_C_m, "_MemPool")
      .def(py::init<c10::cuda::CUDACachingAllocator::CUDAAllocator*, bool>())
      .def_property_readonly("id", &::c10::cuda::MemPool::id)
      .def_property_readonly("allocator", &::c10::cuda::MemPool::allocator);
  shared_ptr_class_<::c10::cuda::MemPoolContext>(torch_C_m, "_MemPoolContext")
      .def(py::init<c10::cuda::MemPool*>())
      .def_static(
          "active_pool", &::c10::cuda::MemPoolContext::getActiveMemPool);
}
