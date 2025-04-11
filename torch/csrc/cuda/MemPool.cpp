#include <torch/csrc/python_headers.h>

#include <torch/custom_class.h>
#include <torch/csrc/jit/python/pybind_utils.h>
#include <torch/csrc/utils/device_lazy_init.h>
#include <torch/csrc/utils/pybind.h>

#include <c10/cuda/CUDACachingAllocator.h>

template <typename T>
using shared_ptr_class_ = py::class_<T, std::shared_ptr<T>>;

template <typename T>
using intrusive_ptr_class_ = py::class_<T, c10::intrusive_ptr<T>>;

namespace {

static const auto mempool_cls = torch::class_<::c10::cuda::MemPool>("c10", "MemPool");

} // namespace

void THCPMemPool_init(PyObject* module) {
  auto torch_C_m = py::handle(module).cast<py::module>();
  intrusive_ptr_class_<::c10::cuda::MemPool>(torch_C_m, "_MemPool")
      .def(
          py::init([](c10::cuda::CUDACachingAllocator::CUDAAllocator* allocator,
                      bool is_user_created) {
            torch::utils::device_lazy_init(at::kCUDA);
            return c10::make_intrusive<::c10::cuda::MemPool>(
                allocator, is_user_created);
          }))
      .def_property_readonly("id", &::c10::cuda::MemPool::id)
      .def_property_readonly("allocator", &::c10::cuda::MemPool::allocator)
      .def("use_count", &::c10::cuda::MemPool::use_count)
      .def("boxed", [](c10::intrusive_ptr<::c10::cuda::MemPool> self) {
        return torch::jit::toPyObject(c10::IValue(std::move(self)));
      })
      .def_static("unbox", [](py::object obj) {
          auto typePtr = torch::getCustomClass("__torch__.torch.classes.c10.MemPool");
          auto ivalue = torch::jit::toIValue(obj, typePtr);
          return ivalue.toCustomClass<::c10::cuda::MemPool>();
      });
  shared_ptr_class_<::c10::cuda::MemPoolContext>(torch_C_m, "_MemPoolContext")
      .def(py::init<c10::cuda::MemPool*>())
      .def_static(
          "active_pool", &::c10::cuda::MemPoolContext::getActiveMemPool);
}
