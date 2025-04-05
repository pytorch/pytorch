
#include <unordered_map>

#include <pybind11/pybind11.h>

#include <caffe2/serialize/file_adapter.h>
#include <torch/csrc/jit/python/pybind_utils.h>
#include <torch/csrc/utils/pybind.h> // @manual=//caffe2:torch-cpp-cpu

#include "c10/core/Device.h"

#include "torch/csrc/nativert/ModelRunner.h"

namespace py = pybind11;
using namespace torch::nativert;

template <typename T>
using shared_ptr_class_ = py::class_<T, std::shared_ptr<T>>;

namespace torch {
namespace nativert {

void initModelRunnerPybind(py::module& m) {
  py::enum_<ExecutorType>(m, "PyExecutorType")
      .value("INTERPRETER", ExecutorType::INTERPRETER)
      .value("AOTINDUCTOR", ExecutorType::AOTINDUCTOR)
      .value("MTIA", ExecutorType::MTIA)
      .export_values();

  py::class_<Placement>(m, "PyPlacement")
      .def(py::init<>())
      .def(py::init<std::optional<c10::Device>>(), py::arg("defaultDevice"))
      .def(
          py::init<
              const std::unordered_map<c10::Device, c10::Device>&,
              std::optional<c10::Device>>(),
          py::arg("deviceMap"),
          py::arg("defaultDevice") = std::nullopt)
      .def("get_mapped_device", &Placement::getMappedDevice);

  py::class_<BaseRuntimeConfigs>(m, "PyRuntimeConfigs")
      .def(py::init<>())
      .def_readwrite("isDebug", &BaseRuntimeConfigs::isDebug)
      .def_readwrite("validateInputs", &BaseRuntimeConfigs::validateInputs)
      .def_readwrite(
          "enableStaticCPUKernels", &BaseRuntimeConfigs::enableStaticCPUKernels)
      .def_readwrite(
          "deferInitialization", &BaseRuntimeConfigs::deferInitialization)
      .def_readwrite("platformArch", &BaseRuntimeConfigs::platformArch)
      .def_readwrite(
          "maxNumConcurrentThreads",
          &BaseRuntimeConfigs::maxNumConcurrentThreads)
      .def_readwrite("maxParallelOps", &BaseRuntimeConfigs::maxParallelOps);

  shared_ptr_class_<core::ModelRunner>(m, "PyModelRunner")
      .def(
          py::init<
              const std::string&,
              const std::string&,
              ExecutorType,
              const BaseRuntimeConfigs&,
              const Placement&>(),
          py::arg("packagePath"),
          py::arg("modelName"),
          py::arg("executorType"),
          py::arg("runtimeConfigs"),
          py::arg("placement") = Placement())
      .def(py::init<
           std::shared_ptr<caffe2::serialize::PyTorchStreamReader>,
           const std::string&,
           ExecutorType,
           const BaseRuntimeConfigs&,
           std::function<Placement(const Graph& graph)>&>())
      .def(
          "run",
          [](core::ModelRunner& self,
             py::args pyargs,
             const py::kwargs& pykwargs) {
            std::vector<c10::IValue> args;
            for (const auto i : c10::irange(pyargs.size())) {
              auto ivalue =
                  torch::jit::toIValue(pyargs[i], c10::AnyType::get());
              args.push_back(std::move(ivalue));
            }
            std::unordered_map<std::string, c10::IValue> kwargs;
            for (const auto& [key, pyarg] : pykwargs) {
              auto ivalue = torch::jit::toIValue(pyarg, c10::AnyType::get());
              kwargs[py::str(key)] = std::move(ivalue);
            }
            c10::IValue ret = self.run(args, kwargs);
            return torch::jit::createPyObjectForStack({ret});
          })
      .def(
          "__call__",
          [](core::ModelRunner& self,
             py::args pyargs,
             const py::kwargs& pykwargs) {
            std::vector<c10::IValue> args;
            for (const auto i : c10::irange(pyargs.size())) {
              auto ivalue =
                  torch::jit::toIValue(pyargs[i], c10::AnyType::get());
              args.push_back(std::move(ivalue));
            }
            std::unordered_map<std::string, c10::IValue> kwargs;
            for (const auto& [key, pyarg] : pykwargs) {
              auto ivalue = torch::jit::toIValue(pyarg, c10::AnyType::get());
              kwargs[py::str(key)] = std::move(ivalue);
            }
            c10::IValue ret = self.run(args, kwargs);
            return torch::jit::createPyObjectForStack({ret});
          })
      .def(
          "load_sample_inputs",
          [](core::ModelRunner& self,
             const std::string& packagePath,
             const Placement& placement = Placement()) {
            auto reader =
                std::make_shared<caffe2::serialize::PyTorchStreamReader>(
                    std::make_unique<caffe2::serialize::FileAdapter>(
                        packagePath));
            const auto [args, kwargs] = self.loadSampleInputs(reader);
            const auto val = argsToIValue(args, kwargs);
            return torch::jit::createPyObjectForStack({val});
          })
      .def(
          "run_with_flat_inputs_and_outputs",
          [](core::ModelRunner& self, py::args pyargs) {
            std::vector<c10::IValue> args;
            for (const auto i : c10::irange(pyargs.size())) {
              auto ivalue =
                  torch::jit::toIValue(pyargs[i], c10::AnyType::get());
              args.push_back(std::move(ivalue));
            }

            auto rets = self.runWithFlatInputsAndOutputs(std::move(args));
            return torch::jit::createPyObjectForStack(std::move(rets));
          });
}

} // namespace nativert
} // namespace torch

// TODO Remove this once we fully migrate to OSS build.
#ifdef FBCODE_CAFFE2
PYBIND11_MODULE(model_runner_pybind, m) {
  initModelRunnerPybind(m);
}
#endif
