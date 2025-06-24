#pragma once

#include <ATen/core/ivalue.h>
#include <pybind11/pybind11.h>
#include <torch/csrc/jit/api/module.h>
#include <torch/csrc/utils/pybind.h>

#include <torch/csrc/jit/python/pybind_utils.h>

#include <iosfwd>
#include <memory>
#include <string>
#include <vector>

namespace py = pybind11;

namespace torch::throughput_benchmark {

/**
 * The struct is used to provide results of a benchmark to the caller
 * In the future all additional statistics should be added here.
 */
struct BenchmarkExecutionStats {
  float latency_avg_ms{-1};
  int64_t num_iters{-1};
};

std::ostream& operator<<(
    std::ostream& os,
    const BenchmarkExecutionStats& value);

/**
 * Use this struct in order to configure a throughput benchmark run.
 * This struct should include parameters related to threading, batching, number
 * of iterations, warm-up, etc. More configs can be added as needed.
 * General rule here is that only things that c++ must(!) to be aware of should
 * be here. If we can keep other parts in python, we should keep them there.
 * This is typical for things that are not perf critical and don't affect
 * execution statistics benchmark returns.
 */
struct BenchmarkConfig {
 public:
  // Calling threads are those threads that are calling into a module in
  // parallel.
  int num_calling_threads{1};
  // Worker threads are not supported yet. This is just an example that we plan
  // to support some sort of multi-threaded forward calls. We may change this
  // setting in the future to support different intra and inter op parallelism
  // which is not available in PyTorch yet
  int num_worker_threads{1};
  // Warmup iters are used to make sure we run a module a few times before
  // actually measuring things. This way we avoid cold caches and any other
  // similar problems
  int num_warmup_iters{1};
  // Number of iterations the benchmark should run with. This number is separate
  // from the warmup iterations
  int64_t num_iters{100};
  // If set autograd profiler will be enabled. I.e. this variable would be
  // created before the main benchmark loop (but after the warmup):
  // RecordProfile guard(profiler_output_path);
  std::string profiler_output_path;
};

namespace detail {

/**
 * A helper class to abstract out different models we test throughput of
 */
template <class Input, class Output, class Model>
class BenchmarkHelper {
 public:
  BenchmarkHelper();
  explicit BenchmarkHelper(Model model)
      : model_(std::move(model)), initialized_(true) {}

  // This method to be used in benchmark() method
  // Note that there is no result. This way we don't have to call this under GIL
  // even when running in the nn.Module mode. Otherwise destructor of the result
  // would race with Python
  void runOnce(Input&&) const;
  // This method is to be used when calling from Python directly
  Output runOnce(const py::args&, const py::kwargs&) const;
  // Aggregate input in the format Model expects in order to avoid further
  // conversions at the benchmark time
  void addInput(py::args&&, py::kwargs&&);
  void addInput(Input&&);
  BenchmarkExecutionStats benchmark(const BenchmarkConfig& config) const;

  bool initialized() const {
    return initialized_;
  }

  // Destructor doesn't require the GIL because it is going to be executed on
  // the PyThon thread
  std::vector<Input> inputs_;
  Model model_;
  bool initialized_{false};
};

struct C10_HIDDEN ModuleInput {
  ModuleInput(ModuleInput&& other) = default;

  ModuleInput(const ModuleInput&) = delete;
  ModuleInput& operator=(ModuleInput& other) = delete;
  ModuleInput& operator=(ModuleInput&& other) = delete;
  ~ModuleInput() = default;

  ModuleInput(py::args&& args, py::kwargs&& kwargs)
      : args(std::move(args)), kwargs(std::move(kwargs)) {}

  py::args args;
  py::kwargs kwargs;
};
typedef py::object ModuleOutput;
typedef std::vector<at::IValue> ScriptModuleInput;
typedef at::IValue ScriptModuleOutput;

template <class Input>
Input cloneInput(const Input& input);

typedef BenchmarkHelper<ScriptModuleInput, at::IValue, jit::Module>
    ScriptModuleBenchmark;
template <>
inline BenchmarkHelper<ScriptModuleInput, at::IValue, jit::Module>::
    BenchmarkHelper()
    : model_("Module", std::make_shared<jit::CompilationUnit>()),
      initialized_(false) {}
typedef BenchmarkHelper<ModuleInput, py::object, py::object> ModuleBenchmark;
template <>
inline BenchmarkHelper<ModuleInput, py::object, py::object>::BenchmarkHelper()
    : initialized_(false) {}

template <>
void ScriptModuleBenchmark::runOnce(ScriptModuleInput&& input) const;

template <>
ScriptModuleOutput ScriptModuleBenchmark::runOnce(
    const py::args& args,
    const py::kwargs& kwargs) const;

template <>
void ModuleBenchmark::runOnce(ModuleInput&& input) const;

template <>
ModuleOutput ModuleBenchmark::runOnce(
    const py::args& args,
    const py::kwargs& kwargs) const;

template <>
void ScriptModuleBenchmark::addInput(py::args&& args, py::kwargs&& kwargs);
template <>
void ScriptModuleBenchmark::addInput(ScriptModuleInput&& input);

template <>
void ModuleBenchmark::addInput(py::args&& args, py::kwargs&& kwargs);

} // namespace detail

/**
 * This class is a small c++ component responsible for executing a PyTorch
 * module under an inference server like load. It can emulate multiple calling
 * threads to a single module provided. In the future we plan to enhance this
 * component to support inter and intra-op parallelism as well as multiple
 * models running in a single process.
 *
 * For current available configurations refer to the BenchmarkConfig
 * documentation
 *
 * The class supports working with either nn.Module or ScriptModule.
 * Under the hood it just dispatches to corresponding specialization of
 * class BenchmarkHelper<Input, Output, Model>
 */
class C10_HIDDEN ThroughputBenchmark {
 public:
  explicit ThroughputBenchmark(const jit::Module& module);
  explicit ThroughputBenchmark(py::object module);

  // Add one more input example. This input example should be in the exact
  // format the module under test expects. It is responsibility of the module to
  // perform any such format checks, the benchmark doesn't perform any
  // validation of its own
  void addInput(py::args args, py::kwargs kwargs);

  // Equivalent to just running the model directly on the given input
  py::object runOnce(const py::args& args, const py::kwargs& kwargs);

  // The main method of the class allows to perform a multi-threaded benchmark
  // It returns BenchmarkExecutionStats object with a lot of useful statistics
  // about runtime execution. We can enhance this class in the future to provide
  // more information to the user
  BenchmarkExecutionStats benchmark(const BenchmarkConfig& config) const;

 private:
  detail::ScriptModuleBenchmark script_module_;
  detail::ModuleBenchmark module_;
};
} // namespace torch::throughput_benchmark

#include <torch/csrc/utils/throughput_benchmark-inl.h>
