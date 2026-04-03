#include <unordered_map>

#include <torch/csrc/jit/python/pybind_utils.h>
#include <torch/csrc/utils/pybind.h>

#ifdef FBCODE_CAFFE2
#include <torch/nativert/ModelRunner.h>

namespace py = pybind11;

template <typename T>
using shared_ptr_class_ = py::class_<T, std::shared_ptr<T>>;

namespace torch::nativert {

using torch::nativert::detail::argsToIValue;

void initModelRunnerPybind(py::module& m) {
#if !defined(OVRSOURCE)
  shared_ptr_class_<ModelRunner>(m, "PyModelRunner")
      .def(
          py::init<const std::string&, const std::string&>(),
          py::arg("packagePath"),
          py::arg("modelName"))
      .def(
          "run",
          [](torch::nativert::ModelRunner& self,
             py::args pyargs,
             const py::kwargs& pykwargs) {
            std::vector<c10::IValue> args;
            args.reserve(pyargs.size());
            for (const auto i : c10::irange(pyargs.size())) {
              auto ivalue =
                  torch::jit::toIValue(pyargs[i], c10::AnyType::get());
              args.push_back(std::move(ivalue));
            }
            std::unordered_map<std::string, c10::IValue> kwargs;
            kwargs.reserve(pykwargs.size());
            for (const auto& [key, pyarg] : pykwargs) {
              auto ivalue = torch::jit::toIValue(pyarg, c10::AnyType::get());
              kwargs[py::str(key)] = std::move(ivalue);
            }
            c10::IValue ret = self.run(args, kwargs);
            return torch::jit::createPyObjectForStack({ret});
          })
      .def(
          "__call__",
          [](torch::nativert::ModelRunner& self,
             py::args pyargs,
             const py::kwargs& pykwargs) {
            std::vector<c10::IValue> args;
            args.reserve(pyargs.size());
            for (const auto i : c10::irange(pyargs.size())) {
              auto ivalue =
                  torch::jit::toIValue(pyargs[i], c10::AnyType::get());
              args.push_back(std::move(ivalue));
            }
            std::unordered_map<std::string, c10::IValue> kwargs;
            kwargs.reserve(pykwargs.size());
            for (const auto& [key, pyarg] : pykwargs) {
              auto ivalue = torch::jit::toIValue(pyarg, c10::AnyType::get());
              kwargs[py::str(key)] = std::move(ivalue);
            }
            c10::IValue ret = self.run(args, kwargs);
            return torch::jit::createPyObjectForStack({ret});
          })
      .def(
          "run_with_flat_inputs_and_outputs",
          [](torch::nativert::ModelRunner& self, py::args pyargs) {
            std::vector<c10::IValue> args;
            args.reserve(pyargs.size());
            for (const auto i : c10::irange(pyargs.size())) {
              auto ivalue =
                  torch::jit::toIValue(pyargs[i], c10::AnyType::get());
              args.push_back(std::move(ivalue));
            }

            auto rets = self.runWithFlatInputsAndOutputs(std::move(args));
            return torch::jit::createPyObjectForStack(std::move(rets));
          });
#endif // !defined(OVRSOURCE)
}

} // namespace torch::nativert

#else // !FBCODE_CAFFE2

namespace py = pybind11;

namespace torch::nativert {

class StubModelRunner {};

// PyModelRunner is referenced from
// https://github.com/pytorch/benchmark/blob/b8d35ba51a3149b7212888b4010ddee97f19947f/userbenchmark/dynamo/dynamobench/common.py#L45
void initModelRunnerPybind(py::module& m) {
  py::class_<StubModelRunner, std::shared_ptr<StubModelRunner>>(
      m, "PyModelRunner");
}

} // namespace torch::nativert

#endif // FBCODE_CAFFE2
