#include <torch/csrc/jit/runtime/static/accelerant.h>
#include <torch/csrc/jit/runtime/static/init.h>

namespace torch {
namespace jit {

void initAccelerantBindings(PyObject* module) {
  auto m = py::handle(module).cast<py::module>();
  py::class_<Accelerant>(m, "Accelerant").def("run", &Accelerant::run);
  m.def(
       "_jit_to_accelerant",
       [](const std::shared_ptr<torch::jit::Graph>& g) {
         return Accelerant(g);
       })
      .def(
          "_jit_to_accelerant",
          [](const torch::jit::Module& m,
             const std::shared_ptr<torch::jit::Graph>& g) {
            return Accelerant(m, g);
          });
}

} // namespace jit
} // namespace torch
