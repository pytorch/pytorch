#include <torch/csrc/jit/runtime/static/init.h>
#include <torch/csrc/jit/runtime/static/impl.h>

namespace torch {
namespace jit {

void initStaticRuntimeBindings(PyObject* module) {
  auto m = py::handle(module).cast<py::module>();
  py::class_<StaticRuntime>(m, "StaticRuntime")
      .def(
          "run",
          py::overload_cast<const std::vector<at::Tensor>&>(
              &StaticRuntime::run, py::const_))
      .def(
          "run",
          py::overload_cast<
              const std::vector<c10::IValue>&,
              const std::unordered_map<std::string, c10::IValue>&>(
              &StaticRuntime::run, py::const_));
  m.def(
       "_jit_to_static_runtime",
       [](const std::shared_ptr<torch::jit::Graph>& g) {
         return StaticRuntime(PrepareForStaticRuntime(g));
       })
      .def("_jit_to_static_runtime", [](const torch::jit::Module& m) {
        return StaticRuntime(PrepareForStaticRuntime(m));
      });
}

} // namespace jit
} // namespace torch
