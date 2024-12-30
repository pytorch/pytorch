#include <torch/csrc/inductor/aoti_package/model_package_loader.h>
#include <torch/csrc/inductor/aoti_runner/model_container_runner.h>
#include <torch/csrc/inductor/aoti_runner/model_container_runner_cpu.h>
#ifdef USE_CUDA
#include <torch/csrc/inductor/aoti_runner/model_container_runner_cuda.h>
#endif

#include <torch/csrc/inductor/aoti_runner/pybind.h>
#include <torch/csrc/utils/pybind.h>

namespace torch::inductor {

void initAOTIPackageBindings(PyObject* module) {
  auto rootModule = py::handle(module).cast<py::module>();
  auto m = rootModule.def_submodule("_aoti");

  py::class_<AOTIModelPackageLoader>(m, "AOTIModelPackageLoader")
      .def(py::init<const std::string&, const std::string&>())
      .def(py::init<const std::string&>())
      .def("get_metadata", &AOTIModelPackageLoader::get_metadata)
      .def(
          "run",
          &AOTIModelPackageLoader::run,
          py::arg("inputs"),
          py::arg("stream_handle") = nullptr)
      .def("get_call_spec", &AOTIModelPackageLoader::get_call_spec)
      .def("load_constants", &AOTIModelPackageLoader::load_constants)
      .def("get_constant_fqns", &AOTIModelPackageLoader::get_constant_fqns);
}
} // namespace torch::inductor
