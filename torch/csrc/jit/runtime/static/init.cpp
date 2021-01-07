#include <torch/csrc/jit/runtime/static/init.h>
#include <torch/csrc/jit/passes/freeze_module.h>
#include <torch/csrc/jit/runtime/static/fusion.h>
#include <torch/csrc/jit/runtime/static/impl.h>

namespace torch {
namespace jit {

void initStaticRuntimeBindings(PyObject* module) {
  auto m = py::handle(module).cast<py::module>();
  py::class_<StaticRuntime> static_runtime(m, "StaticRuntime");
  py::class_<StaticRuntime::IndividualMetrics>(
      static_runtime, "IndividualMetrics")
      .def_readonly("setup_time", &StaticRuntime::IndividualMetrics::setup_time)
      .def_readonly("total_time", &StaticRuntime::IndividualMetrics::total_time)
      .def_readonly(
          "time_per_node", &StaticRuntime::IndividualMetrics::time_per_node)
      .def_readonly(
          "time_per_node_type",
          &StaticRuntime::IndividualMetrics::time_per_node_type)
      .def_readonly(
          "percent_per_node_type",
          &StaticRuntime::IndividualMetrics::percent_per_node_type)
      .def_readonly(
          "instances_per_node_type",
          &StaticRuntime::IndividualMetrics::instances_per_node_type);
  static_runtime
      .def(
          "run",
          py::overload_cast<const std::vector<at::Tensor>&>(
              &StaticRuntime::run))
      .def(
          "run",
          [](StaticRuntime& self,
             const std::vector<at::Tensor>& args,
             const std::unordered_map<std::string, at::Tensor>& kwargs) {
            std::vector<c10::IValue> arg_ivalues{args.begin(), args.end()};
            std::unordered_map<std::string, c10::IValue> kwarg_ivalues{
                kwargs.begin(), kwargs.end()};
            c10::IValue ret = self.run(arg_ivalues, kwarg_ivalues);
            return toPyObject(ret);
          })
      .def(
          "benchmark",
          [](StaticRuntime& self,
             const std::vector<at::Tensor>& args,
             const std::unordered_map<std::string, at::Tensor>& kwargs,
             const int warmup_runs,
             const int main_runs) {
            std::vector<c10::IValue> arg_ivalues{args.begin(), args.end()};
            std::unordered_map<std::string, c10::IValue> kwarg_ivalues{
                kwargs.begin(), kwargs.end()};
            self.benchmark(arg_ivalues, kwarg_ivalues, warmup_runs, main_runs);
          })
      .def(
          "benchmark_individual_ops",
          [](StaticRuntime& self,
             const std::vector<at::Tensor>& args,
             const std::unordered_map<std::string, at::Tensor>& kwargs,
             const int warmup_runs,
             const int main_runs) {
            std::vector<c10::IValue> arg_ivalues{args.begin(), args.end()};
            std::unordered_map<std::string, c10::IValue> kwarg_ivalues{
                kwargs.begin(), kwargs.end()};
            return self.benchmark_individual_ops(
                arg_ivalues, kwarg_ivalues, warmup_runs, main_runs);
          });
  m.def(
       "_jit_to_static_runtime",
       [](std::shared_ptr<torch::jit::Graph> g) {
         return StaticRuntime(PrepareForStaticRuntime(g));
       })
      .def(
          "_jit_to_static_runtime",
          [](const torch::jit::Module& m) {
            return StaticRuntime(PrepareForStaticRuntime(m));
          })
      .def(
          "_fuse_to_static_runtime",
          [](torch::jit::Module& module) {
            module.eval();
            module = freeze_module(module);

            Method method = module.get_method("forward");
            auto graph = method.graph();
            fuseStaticSubgraphs(graph);
          })
      .def("_fuse_to_static_runtime", [](std::shared_ptr<torch::jit::Graph> g) {
        fuseStaticSubgraphs(g);
      });
}

} // namespace jit
} // namespace torch
