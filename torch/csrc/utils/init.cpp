#include <ATen/core/ivalue.h>
#include <torch/csrc/utils/init.h>
#include <torch/csrc/utils/throughput_benchmark.h>

#include <pybind11/functional.h>
#include <torch/csrc/utils/pybind.h>

namespace torch::throughput_benchmark {

void initThroughputBenchmarkBindings(PyObject* module) {
  auto m = py::handle(module).cast<py::module>();
  using namespace torch::throughput_benchmark;
  py::class_<BenchmarkConfig>(m, "BenchmarkConfig")
      .def(py::init<>())
      .def_readwrite(
          "num_calling_threads", &BenchmarkConfig::num_calling_threads)
      .def_readwrite("num_worker_threads", &BenchmarkConfig::num_worker_threads)
      .def_readwrite("num_warmup_iters", &BenchmarkConfig::num_warmup_iters)
      .def_readwrite("num_iters", &BenchmarkConfig::num_iters)
      .def_readwrite(
          "profiler_output_path", &BenchmarkConfig::profiler_output_path);

  py::class_<BenchmarkExecutionStats>(m, "BenchmarkExecutionStats")
      .def_readonly("latency_avg_ms", &BenchmarkExecutionStats::latency_avg_ms)
      .def_readonly("num_iters", &BenchmarkExecutionStats::num_iters);

  py::class_<ThroughputBenchmark>(m, "ThroughputBenchmark", py::dynamic_attr())
      .def(py::init<jit::Module>())
      .def(py::init<py::object>())
      .def(
          "add_input",
          [](ThroughputBenchmark& self, py::args args, py::kwargs kwargs) {
            self.addInput(std::move(args), std::move(kwargs));
          })
      .def(
          "run_once",
          [](ThroughputBenchmark& self,
             py::args args,
             const py::kwargs& kwargs) {
            // Depending on this being ScriptModule of nn.Module we will release
            // the GIL or not further down in the stack
            return self.runOnce(std::move(args), kwargs);
          })
      .def(
          "benchmark",
          [](ThroughputBenchmark& self, const BenchmarkConfig& config) {
            // The benchmark always runs without the GIL. GIL will be used where
            // needed. This will happen only in the nn.Module mode when
            // manipulating inputs and running actual inference
            pybind11::gil_scoped_release no_gil_guard;
            return self.benchmark(config);
          });
}

} // namespace torch::throughput_benchmark
