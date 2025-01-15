#include <torch/csrc/python_headers.h>

#include <torch/csrc/jit/python/pybind_utils.h>
#include <torch/csrc/utils/device_lazy_init.h>
#include <torch/csrc/utils/pybind.h>

#include <c10/cuda/CUDACachingAllocator.h>

template <typename T>
using shared_ptr_class_ = py::class_<T, std::shared_ptr<T>>;

void THCPMemPool_init(PyObject* module) {
  auto torch_C_m = py::handle(module).cast<py::module>();
  shared_ptr_class_<::c10::cuda::MemPool>(torch_C_m, "_MemPool")
      .def(
          py::init([](c10::cuda::CUDACachingAllocator::CUDAAllocator* allocator,
                      bool is_user_created) {
            torch::utils::device_lazy_init(at::kCUDA);
            return std::make_shared<::c10::cuda::MemPool>(
                allocator, is_user_created);
          }))
      .def_property_readonly("id", &::c10::cuda::MemPool::id)
      .def_property_readonly("allocator", &::c10::cuda::MemPool::allocator)
      .def("use_count", &::c10::cuda::MemPool::use_count);
  shared_ptr_class_<::c10::cuda::MemPoolContext>(torch_C_m, "_MemPoolContext")
      .def(py::init<c10::cuda::MemPool*>())
      .def_static(
          "active_pool", &::c10::cuda::MemPoolContext::getActiveMemPool);
}
