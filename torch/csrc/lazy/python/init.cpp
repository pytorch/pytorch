#include <torch/csrc/lazy/python/init.h>

#include <c10/core/Device.h>
#include <torch/csrc/jit/python/pybind.h>
#include <torch/csrc/lazy/backend/backend_device.h>
#include <torch/csrc/lazy/core/debug_util.h>
#include <torch/csrc/lazy/core/lazy_mode.h>
#include <torch/csrc/lazy/core/lazy_graph_executor.h>
#include <torch/csrc/lazy/core/metrics.h>
#include <torch/csrc/lazy/python/python_util.h>
#include <torch/csrc/lazy/backend/backend_interface.h>
#ifndef FBCODE_CAFFE2
#include <torch/csrc/lazy/ts_backend/ts_backend_impl.h>
#endif // FBCODE_CAFFE2
#include <string>
#include <vector>

namespace torch {
namespace lazy {

// TODO(whc) backend 'device' related APIs are not very clear, this code could be
// simplified but it should probably be done together with designing/refactoring
// the overall approach to get/set of default eager/lazy device types
torch::lazy::BackendDevice GetDeviceOrCurrent(const std::string& device_str) {
  if (device_str.empty()) {
    getBackend()->GetDefaultDeviceType();
    return torch::lazy::BackendDevice();
  }
  return torch::lazy::atenDeviceToBackendDevice(c10::Device(device_str));
}

void initLazyBindings(PyObject* module){
  auto m = py::handle(module).cast<py::module>();
  auto lazy = m.def_submodule("_lazy");
  auto lazy_ts_backend = m.def_submodule("_lazy_ts_backend");

  lazy.def(
      "_mark_step",
      // TODO(whc) this API should probably change from vector<string> to
      // vector<c10::device> but in a separate PR
      [](const std::string& device_str, const std::vector<std::string>& devices,
         bool wait) {
        pybind11::gil_scoped_release no_gil;
        auto backend_device = GetDeviceOrCurrent(device_str);
        torch::lazy::LazyGraphExecutor::Get()->SyncLiveTensorsGraph(&backend_device, devices, wait);
        torch::lazy::LazyGraphExecutor::Get()->MarkStep(backend_device);
      },
      py::arg("device") = "", py::arg("devices"), py::arg("wait") = true);
  lazy.def(
      "_wait_device_ops",
      [](const std::vector<std::string>& devices) {
        pybind11::gil_scoped_release no_gil;
        // TODO: Add support of non-empty devices.
        if (!devices.empty()) {
          LOG(ERROR) << "Non-empty devices are not supported.";
        }
        torch::lazy::LazyGraphExecutor::Get()->WaitDeviceOps({});
      },
      py::arg("devices"));
  lazy.def("_reset_metrics",
        []() { torch::lazy::MetricsArena::Get()->Reset(); });
  lazy.def("_counter_names", []() { return torch::lazy::GetCounterNames(); });

  lazy.def(
    "_lazy_mode_enter",
    [](c10::Device device) {
      pybind11::gil_scoped_release no_gil;
      torch::lazy::LazyModeEnter(device);
    },
    py::arg("device"));
  lazy.def(
    "_lazy_mode_exit",
    [](c10::Device device) {
      pybind11::gil_scoped_release no_gil;
      torch::lazy::LazyModeExit(device);
    },
    py::arg("device"));


  lazy_ts_backend.def(
    "_init",
    []() {
#ifndef FBCODE_CAFFE2
      torch::lazy::InitTorchScriptBackend();
#else
      TORCH_CHECK(false, "TorchScript backend not yet supported in FBCODE builds");
#endif // FBCODE_CAFFE2
    });

#ifndef USE_DEPLOY
  // When libtorch_python is loaded, we register the python frame getter
  // otherwise, debug util simply omits python frames
  // TODO(whc) can we make this work inside torch deploy interpreter?
  // it doesn't work as-is, possibly becuase GetPythonFrames resolves to external
  // cpython rather than embedded cpython
  GetPythonFramesFunction() = GetPythonFrames;
#endif
}

}  // namespace lazy
}  // namespace torch
