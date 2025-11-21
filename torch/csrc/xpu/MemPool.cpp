#include <torch/csrc/python_headers.h>

#include <torch/csrc/jit/python/pybind_utils.h>
#include <torch/csrc/utils/device_lazy_init.h>
#include <torch/csrc/utils/pybind.h>

#include <c10/xpu/XPUCachingAllocator.h>

template <typename T>
using shared_ptr_class_ = py::class_<T, std::shared_ptr<T>>;

void THXPMemPool_init(PyObject* module) {
  auto torch_C_m = py::handle(module).cast<py::module>();
  // Use _XPUMemPool instead of _MemPool to avoid naming conflict with CUDA
  // backend. Python user API remains torch.xpu.MemPool unchanged.
  shared_ptr_class_<::c10::xpu::MemPool>(torch_C_m, "_XPUMemPool")
      .def(py::init([](c10::xpu::XPUCachingAllocator::XPUAllocator* allocator,
                       bool is_user_created,
                       bool use_on_oom) {
        torch::utils::device_lazy_init(at::kXPU);
        return std::make_shared<::c10::xpu::MemPool>(
            allocator, is_user_created, use_on_oom);
      }))
      .def_property_readonly("id", &::c10::xpu::MemPool::id)
      .def_property_readonly("allocator", &::c10::xpu::MemPool::allocator)
      .def("use_count", &::c10::xpu::MemPool::use_count);
}