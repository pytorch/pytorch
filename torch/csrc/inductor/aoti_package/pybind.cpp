#include <torch/csrc/inductor/aoti_package/model_package_loader.h>
#include <torch/csrc/inductor/aoti_package/pybind.h>
#ifdef USE_CUDA
#include <torch/csrc/inductor/aoti_runner/model_container_runner_cuda.h>
#endif

#include <c10/core/Device.h>
#include <torch/csrc/autograd/python_variable.h>
#include <torch/csrc/inductor/aoti_runner/pybind.h>

namespace torch::inductor {

class AOTIModelPackageLoaderPybind : public AOTIModelPackageLoader {
 public:
  AOTIModelPackageLoaderPybind(
      const std::string& model_package_path,
      const std::string& model_name,
      const bool run_single_threaded,
      const size_t num_runners,
      const c10::DeviceIndex device_index)
      : AOTIModelPackageLoader(
            model_package_path,
            model_name,
            run_single_threaded,
            num_runners,
            device_index) {}

  py::list boxed_run(py::list& inputs, void* stream_handle = nullptr) {
    std::vector<at::Tensor> input_tensors;
    input_tensors.reserve(inputs.size());
    for (auto& item : inputs) {
      input_tensors.emplace_back(py::cast<at::Tensor>(item));
    }
    // Explicitly clear the passed-in Python list
    inputs.attr("clear")();

    std::vector<at::Tensor> result_tensors = AOTIModelPackageLoader::boxed_run(
        std::move(input_tensors), stream_handle);

    py::list outputs;
    for (const auto& tensor : result_tensors) {
      outputs.append(
          py::reinterpret_steal<py::object>(THPVariable_Wrap(tensor)));
    }
    return outputs;
  }
};

void initAOTIPackageBindings(PyObject* module) {
  auto rootModule = py::handle(module).cast<py::module>();
  auto m = rootModule.def_submodule("_aoti");
  py::class_<AOTIModelPackageLoaderPybind>(m, "AOTIModelPackageLoader")
      .def(py::init<
           const std::string&,
           const std::string&,
           const bool,
           const size_t,
           const c10::DeviceIndex>())
      .def("get_metadata", &AOTIModelPackageLoaderPybind::get_metadata)
      .def(
          "run",
          &AOTIModelPackageLoaderPybind::run,
          py::arg("inputs"),
          py::arg("stream_handle") = nullptr)
      .def(
          "boxed_run",
          &AOTIModelPackageLoaderPybind::boxed_run,
          py::arg("inputs"),
          py::arg("stream_handle") = nullptr)
      .def("get_call_spec", &AOTIModelPackageLoaderPybind::get_call_spec)
      .def(
          "get_constant_fqns", &AOTIModelPackageLoaderPybind::get_constant_fqns)
      .def(
          "load_constants",
          &AOTIModelPackageLoaderPybind::load_constants,
          py::arg("constants_map"),
          py::arg("use_inactive"),
          py::arg("check_full_update"),
          py::arg("user_managed") = false)
      .def(
          "update_constant_buffer",
          &AOTIModelPackageLoaderPybind::update_constant_buffer,
          py::arg("tensor_map"),
          py::arg("use_inactive"),
          py::arg("validate_full_updates"),
          py::arg("user_managed") = false)
      .def_static(
          "load_metadata_from_package",
          &AOTIModelPackageLoaderPybind::load_metadata_from_package,
          py::arg("model_package_path"),
          py::arg("model_name"));
}
} // namespace torch::inductor
