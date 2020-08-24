#include <torch/csrc/jit/runtime/static/init.h>
#include <torch/csrc/jit/runtime/static/impl.h>

namespace torch {
namespace jit {

void initStaticRuntimeBindings(PyObject* module) {
  auto m = py::handle(module).cast<py::module>();
  py::class_<StaticRuntime>(m, "StaticRuntime").def("run", &StaticRuntime::run);
  m.def(
       "_jit_to_static_runtime",
       [](const std::shared_ptr<torch::jit::Graph>& g) {
         return StaticRuntime(g);
       })
      .def("_jit_to_static_runtime", [](const torch::jit::Module& m) {
        return StaticRuntime(m);
      });
}

} // namespace jit
} // namespace torch
