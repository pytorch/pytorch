#include <torch/csrc/inductor/aoti_runner/model_container_runner_cpu.h>
#ifdef USE_CUDA
#include <torch/csrc/inductor/aoti_runner/model_container_runner_cuda.h>
#endif
#ifdef USE_XPU
#include <torch/csrc/inductor/aoti_runner/model_container_runner_xpu.h>
#endif
#ifdef __APPLE__
#include <torch/csrc/inductor/aoti_runner/model_container_runner_mps.h>
#endif
#include <torch/csrc/inductor/aoti_runner/pybind.h>
#include <torch/csrc/inductor/aoti_torch/tensor_converter.h>
#include <torch/csrc/inductor/aoti_torch/utils.h>

#include <torch/csrc/utils/pybind.h>

#include <memory>
#include <utility>

namespace torch::inductor {

namespace {

// Constructs a Runner with the GIL released. py::call_guard cannot do this for
// py::init: class_::init_instance() would then register the new instance
// without the GIL (https://github.com/pybind/pybind11/issues/5473). Releasing
// inside the factory confines the release to the C++ constructor.
template <typename Runner, typename... Args>
auto init_releasing_gil() {
  return py::init([](Args... args) {
    py::gil_scoped_release no_gil;
    return std::make_unique<Runner>(std::forward<Args>(args)...);
  });
}

} // namespace

void initAOTIRunnerBindings(PyObject* module) {
  auto rootModule = py::handle(module).cast<py::module>();
  auto m = rootModule.def_submodule("_aoti");

  // Nothing in the runners needs the caller's GIL: they are the same C++ that
  // runs without an interpreter. Construction, destruction and run() release
  // it because they can block for long. swap_constant_buffer() has to as well:
  // it waits for model_exec_mutex_, which run() holds while code under it may
  // take the GIL (e.g. a Python kernel).
  py::class_<AOTIModelContainerRunnerCpu>(
      m,
      "AOTIModelContainerRunnerCpu",
      py::release_gil_before_calling_cpp_dtor())
      .def(init_releasing_gil<
           AOTIModelContainerRunnerCpu,
           const std::string&,
           int>())
      .def(
          init_releasing_gil<
              AOTIModelContainerRunnerCpu,
              const std::string&,
              size_t,
              std::unordered_map<std::string, at::Tensor>&>(),
          py::arg("model_so_path"),
          py::arg("num_models"),
          py::arg("constants"))
      .def(
          "run",
          &AOTIModelContainerRunnerCpu::run,
          py::arg("inputs"),
          py::arg("stream_handle") = nullptr,
          py::call_guard<py::gil_scoped_release>())
      .def("get_call_spec", &AOTIModelContainerRunnerCpu::get_call_spec)
      .def(
          "get_constant_names_to_original_fqns",
          &AOTIModelContainerRunnerCpu::getConstantNamesToOriginalFQNs)
      .def(
          "get_constant_names_to_dtypes",
          &AOTIModelContainerRunnerCpu::getConstantNamesToDtypes)
      .def(
          "extract_constants_map",
          &AOTIModelContainerRunnerCpu::extract_constants_map)
      .def(
          "update_constant_buffer",
          static_cast<void (AOTIModelContainerRunnerCpu::*)(
              std::unordered_map<std::string, at::Tensor>&, bool, bool, bool)>(
              &AOTIModelContainerRunnerCpu::update_constant_buffer),
          py::arg("tensor_map"),
          py::arg("use_inactive"),
          py::arg("validate_full_updates"),
          py::arg("user_managed") = false)
      .def(
          "swap_constant_buffer",
          &AOTIModelContainerRunnerCpu::swap_constant_buffer,
          py::call_guard<py::gil_scoped_release>())
      .def(
          "free_inactive_constant_buffer",
          &AOTIModelContainerRunnerCpu::free_inactive_constant_buffer)
      .def(
          "did_call_load_constants",
          &AOTIModelContainerRunnerCpu::did_call_load_constants)
      .def(
          "update_constant_buffer_from_blob",
          &AOTIModelContainerRunnerCpu::update_constant_buffer_from_blob,
          py::arg("weights_path"));

#ifdef USE_CUDA
  py::class_<AOTIModelContainerRunnerCuda>(
      m,
      "AOTIModelContainerRunnerCuda",
      py::release_gil_before_calling_cpp_dtor())
      .def(init_releasing_gil<
           AOTIModelContainerRunnerCuda,
           const std::string&,
           int>())
      .def(init_releasing_gil<
           AOTIModelContainerRunnerCuda,
           const std::string&,
           int,
           const std::string&>())
      .def(init_releasing_gil<
           AOTIModelContainerRunnerCuda,
           const std::string&,
           int,
           const std::string&,
           const std::string&>())
      .def(init_releasing_gil<
           AOTIModelContainerRunnerCuda,
           const std::string&,
           int,
           const std::string&,
           const std::string&,
           const bool>())
      .def(
          init_releasing_gil<
              AOTIModelContainerRunnerCuda,
              const std::string&,
              size_t,
              const std::string&,
              const std::string&,
              std::unordered_map<std::string, at::Tensor>&>(),
          py::arg("model_so_path"),
          py::arg("num_models"),
          py::arg("device_str"),
          py::arg("cubin_dir"),
          py::arg("constants"))
      .def(
          "run",
          &AOTIModelContainerRunnerCuda::run,
          py::arg("inputs"),
          py::arg("stream_handle") = nullptr,
          py::call_guard<py::gil_scoped_release>())
      .def("get_call_spec", &AOTIModelContainerRunnerCuda::get_call_spec)
      .def(
          "get_constant_names_to_original_fqns",
          &AOTIModelContainerRunnerCuda::getConstantNamesToOriginalFQNs)
      .def(
          "get_constant_names_to_dtypes",
          &AOTIModelContainerRunnerCuda::getConstantNamesToDtypes)
      .def(
          "extract_constants_map",
          &AOTIModelContainerRunnerCuda::extract_constants_map)
      .def(
          "update_constant_buffer",
          static_cast<void (AOTIModelContainerRunnerCuda::*)(
              std::unordered_map<std::string, at::Tensor>&, bool, bool, bool)>(
              &AOTIModelContainerRunnerCuda::update_constant_buffer),
          py::arg("tensor_map"),
          py::arg("use_inactive"),
          py::arg("validate_full_updates"),
          py::arg("user_managed") = false)
      .def(
          "update_constant_buffer_from_cpu",
          static_cast<void (AOTIModelContainerRunnerCuda::*)(
              std::unordered_map<std::string, at::Tensor>&, bool, bool)>(
              &AOTIModelContainerRunnerCuda::update_constant_buffer_from_cpu),
          py::arg("tensor_map"),
          py::arg("use_inactive"),
          py::arg("validate_full_updates"))
      .def(
          "swap_constant_buffer",
          &AOTIModelContainerRunnerCuda::swap_constant_buffer,
          py::call_guard<py::gil_scoped_release>())
      .def(
          "free_inactive_constant_buffer",
          &AOTIModelContainerRunnerCuda::free_inactive_constant_buffer)
      .def(
          "did_call_load_constants",
          &AOTIModelContainerRunnerCuda::did_call_load_constants)
      .def(
          "update_constant_buffer_from_blob",
          &AOTIModelContainerRunnerCuda::update_constant_buffer_from_blob,
          py::arg("weights_path"));
#endif
#ifdef USE_XPU
  py::class_<AOTIModelContainerRunnerXpu>(
      m,
      "AOTIModelContainerRunnerXpu",
      py::release_gil_before_calling_cpp_dtor())
      .def(init_releasing_gil<
           AOTIModelContainerRunnerXpu,
           const std::string&,
           int>())
      .def(init_releasing_gil<
           AOTIModelContainerRunnerXpu,
           const std::string&,
           int,
           const std::string&>())
      .def(init_releasing_gil<
           AOTIModelContainerRunnerXpu,
           const std::string&,
           int,
           const std::string&,
           const std::string&>())
      .def(
          init_releasing_gil<
              AOTIModelContainerRunnerXpu,
              const std::string&,
              size_t,
              const std::string&,
              const std::string&,
              std::unordered_map<std::string, at::Tensor>&>(),
          py::arg("model_so_path"),
          py::arg("num_models"),
          py::arg("device_str"),
          py::arg("cubin_dir"),
          py::arg("constants"))
      .def(
          "run",
          &AOTIModelContainerRunnerXpu::run,
          py::arg("inputs"),
          py::arg("stream_handle") = nullptr,
          py::call_guard<py::gil_scoped_release>())
      .def("get_call_spec", &AOTIModelContainerRunnerXpu::get_call_spec)
      .def(
          "get_constant_names_to_original_fqns",
          &AOTIModelContainerRunnerXpu::getConstantNamesToOriginalFQNs)
      .def(
          "get_constant_names_to_dtypes",
          &AOTIModelContainerRunnerXpu::getConstantNamesToDtypes)
      .def(
          "extract_constants_map",
          &AOTIModelContainerRunnerXpu::extract_constants_map)
      .def(
          "update_constant_buffer",
          static_cast<void (AOTIModelContainerRunnerXpu::*)(
              std::unordered_map<std::string, at::Tensor>&, bool, bool, bool)>(
              &AOTIModelContainerRunnerXpu::update_constant_buffer),
          py::arg("tensor_map"),
          py::arg("use_inactive"),
          py::arg("validate_full_updates"),
          py::arg("user_managed") = false)
      .def(
          "update_constant_buffer_from_cpu",
          static_cast<void (AOTIModelContainerRunnerXpu::*)(
              std::unordered_map<std::string, at::Tensor>&, bool, bool)>(
              &AOTIModelContainerRunnerXpu::update_constant_buffer_from_cpu),
          py::arg("tensor_map"),
          py::arg("use_inactive"),
          py::arg("validate_full_updates"))
      .def(
          "swap_constant_buffer",
          &AOTIModelContainerRunnerXpu::swap_constant_buffer,
          py::call_guard<py::gil_scoped_release>())
      .def(
          "free_inactive_constant_buffer",
          &AOTIModelContainerRunnerXpu::free_inactive_constant_buffer)
      .def(
          "did_call_load_constants",
          &AOTIModelContainerRunnerXpu::did_call_load_constants)
      .def(
          "update_constant_buffer_from_blob",
          &AOTIModelContainerRunnerXpu::update_constant_buffer_from_blob,
          py::arg("weights_path"));
#endif
#if defined(USE_MPS) && defined(__APPLE__) && \
    !(defined(FBCODE_CAFFE2) || defined(OVRSOURCE))
  py::class_<AOTIModelContainerRunnerMps>(
      m,
      "AOTIModelContainerRunnerMps",
      py::release_gil_before_calling_cpp_dtor())
      .def(init_releasing_gil<
           AOTIModelContainerRunnerMps,
           const std::string&,
           int>())
      .def(
          "run",
          &AOTIModelContainerRunnerMps::run,
          py::arg("inputs"),
          py::arg("stream_handle") = nullptr,
          py::call_guard<py::gil_scoped_release>())
      .def("get_call_spec", &AOTIModelContainerRunnerMps::get_call_spec)
      .def(
          "get_constant_names_to_original_fqns",
          &AOTIModelContainerRunnerMps::getConstantNamesToOriginalFQNs)
      .def(
          "get_constant_names_to_dtypes",
          &AOTIModelContainerRunnerMps::getConstantNamesToDtypes)
      .def(
          "extract_constants_map",
          &AOTIModelContainerRunnerMps::extract_constants_map)
      .def(
          "update_constant_buffer",
          static_cast<void (AOTIModelContainerRunnerMps::*)(
              std::unordered_map<std::string, at::Tensor>&, bool, bool, bool)>(
              &AOTIModelContainerRunnerMps::update_constant_buffer),
          py::arg("tensor_map"),
          py::arg("use_inactive"),
          py::arg("validate_full_updates"),
          py::arg("user_managed") = false)
      .def(
          "update_constant_buffer_from_cpu",
          static_cast<void (AOTIModelContainerRunnerMps::*)(
              std::unordered_map<std::string, at::Tensor>&, bool, bool)>(
              &AOTIModelContainerRunnerMps::update_constant_buffer_from_cpu),
          py::arg("tensor_map"),
          py::arg("use_inactive"),
          py::arg("validate_full_updates"))
      .def(
          "swap_constant_buffer",
          &AOTIModelContainerRunnerMps::swap_constant_buffer,
          py::call_guard<py::gil_scoped_release>())
      .def(
          "free_inactive_constant_buffer",
          &AOTIModelContainerRunnerMps::free_inactive_constant_buffer)
      .def(
          "did_call_load_constants",
          &AOTIModelContainerRunnerMps::did_call_load_constants)
      .def(
          "update_constant_buffer_from_blob",
          &AOTIModelContainerRunnerMps::update_constant_buffer_from_blob,
          py::arg("weights_path"));
#endif

  m.def(
      "unsafe_alloc_void_ptrs_from_tensors",
      [](const std::vector<at::Tensor>& tensors) {
        std::vector<AtenTensorHandle> handles =
            torch::aot_inductor::unsafe_alloc_new_handles_from_tensors(tensors);
        std::vector<void*> result(
            reinterpret_cast<void**>(handles.data()),
            reinterpret_cast<void**>(handles.data()) + handles.size());
        return result;
      });
  m.def("unsafe_alloc_void_ptr_from_tensor", [](at::Tensor& tensor) {
    return reinterpret_cast<void*>(
        torch::aot_inductor::new_tensor_handle(std::move(tensor)));
  });
  m.def(
      "alloc_tensors_by_stealing_from_void_ptrs",
      [](std::vector<void*>& raw_handles) {
        return torch::aot_inductor::alloc_tensors_by_stealing_from_handles(
            reinterpret_cast<AtenTensorHandle*>(raw_handles.data()),
            raw_handles.size());
      });
  m.def("alloc_tensor_by_stealing_from_void_ptr", [](void* raw_handle) {
    return *torch::aot_inductor::tensor_handle_to_tensor_pointer(
        reinterpret_cast<AtenTensorHandle>(raw_handle));
  });
}
} // namespace torch::inductor
