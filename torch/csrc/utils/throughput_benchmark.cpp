#include <torch/csrc/utils/throughput_benchmark.h>

#include <pybind11/pybind11.h>
#include <torch/csrc/jit/python/pybind_utils.h>

namespace torch {
namespace throughput_benchmark {

std::ostream& operator<<(std::ostream& os, const BenchmarkExecutionStats& value) {
    return os << "Average latency / iter (ms): " << value.latency_avg_ms
              << "\n Total number of iters: " << value.num_iters;
}

void ThroughputBenchmark::addInput(py::args args, py::kwargs kwargs) {
  CHECK(script_module_.initialized() ^ module_.initialized());
  if (script_module_.initialized()) {
    script_module_.addInput(std::move(args), std::move(kwargs));
  } else {
    CHECK(module_.initialized());
    module_.addInput(std::move(args), std::move(kwargs));
  }
}

py::object ThroughputBenchmark::runOnce(py::args&& args, py::kwargs&& kwargs)  {
  CHECK(script_module_.initialized() ^ module_.initialized());
  if (script_module_.initialized()) {
    c10::IValue result;
    {
      pybind11::gil_scoped_release no_gil_guard;
      result = script_module_.runOnce(std::move(args), std::move(kwargs));
    }
    return jit::toPyObject(std::move(result));
  } else {
    CHECK(module_.initialized());
    return module_.runOnce(std::move(args), std::move(kwargs));
  }
}

ThroughputBenchmark::ThroughputBenchmark(
    jit::Module script_module)
    : script_module_(script_module) {}

ThroughputBenchmark::ThroughputBenchmark(
    py::object module)
    : module_(std::move(module)) {}

BenchmarkExecutionStats ThroughputBenchmark::benchmark(
    const BenchmarkConfig& config) const {
  CHECK(script_module_.initialized() ^ module_.initialized());
  // Main benchmark thread doesn't hold the GIL after scheduling worker threads
  // But for now we don't release it as we will be implicitly manipulating with
  // py::object ref. counts in the case of nn.Module benchmarking.
  if (script_module_.initialized()) {
    return script_module_.benchmark(config);
  } else {
    CHECK(module_.initialized());
    TORCH_WARN("Starting benchmark on an nn.Module. This can be slow due "
    "to Python GIL.For proper inference simulation you might want to switch to "
    "a ScriptModule instead");
    return module_.benchmark(config);
  }
}

namespace detail {

template <>
void ScriptModuleBenchmark::runOnce(ScriptModuleInput&& input) const {
  CHECK(initialized_);
  // TODO: provide guarantees that compiler won't optimize this out
  model_.get_method("forward").function()(std::move(input));
}

template <>
ScriptModuleOutput ScriptModuleBenchmark::runOnce(
    py::args&& args,
    py::kwargs&& kwargs) const {
  CHECK(initialized_);
  auto& function = model_.get_method("forward").function();
  ScriptModuleInput stack = jit::createStackForSchema(
      function.getSchema(),
      std::move(args),
      // NOLINTNEXTLINE(performance-move-const-arg)
      std::move(kwargs),
      model_._ivalue());
  return function(std::move(stack));
}

template <>
void ModuleBenchmark::runOnce(ModuleInput&& input) const {
  CHECK(initialized_);
  pybind11::gil_scoped_acquire gil_guard;
  model_(*input.args, **input.kwargs);
}

template <>
ModuleOutput ModuleBenchmark::runOnce(py::args&& args, py::kwargs&& kwargs)
    const {
  CHECK(initialized_);
  pybind11::gil_scoped_acquire gil_guard;
  return model_(*args, **kwargs);
}

template <>
void ScriptModuleBenchmark::addInput(py::args&& args, py::kwargs&& kwargs) {
  jit::Stack stack = jit::createStackForSchema(
      model_.get_method("forward").function().getSchema(),
      std::move(args),
      // NOLINTNEXTLINE(performance-move-const-arg)
      std::move(kwargs),
      model_._ivalue());
  inputs_.emplace_back(std::move(stack));
}

template <>
void ScriptModuleBenchmark::addInput(ScriptModuleInput&& input) {
  input.insert(input.begin(), model_._ivalue());
  inputs_.emplace_back(std::move(input));
}

template <>
void ModuleBenchmark::addInput(py::args&& args, py::kwargs&& kwargs) {
  inputs_.emplace_back(std::move(args), std::move(kwargs));
}

template <>
ModuleInput cloneInput<ModuleInput>(const ModuleInput& input) {
  pybind11::gil_scoped_acquire gil_guard;
  py::args args = input.args;
  py::kwargs kwargs = input.kwargs;
  return {std::move(args), std::move(kwargs)};
}

template <>
ScriptModuleInput cloneInput<ScriptModuleInput>(
    const ScriptModuleInput& input) {
  return input;
}

} // namespace detail

} // namespace throughput_benchmark
} // namepsace torch
