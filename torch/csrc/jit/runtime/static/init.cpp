#include <torch/csrc/jit/runtime/static/init.h>

#include <torch/csrc/jit/passes/freeze_module.h>
#include <torch/csrc/jit/runtime/static/fusion.h>
#include <torch/csrc/jit/runtime/static/impl.h>

// This number is a heuristic determined with pytorch/benchmark
#define DEFAULT_FUSION_SIZE 4

namespace torch {
namespace jit {

void initStaticModuleBindings(PyObject* module) {
  auto m = py::handle(module).cast<py::module>();
  py::class_<StaticModule> static_module(m, "StaticModule");
  py::class_<StaticRuntime::IndividualMetrics>(
      static_module, "IndividualMetrics")
      .def_readonly("setup_time", &StaticRuntime::IndividualMetrics::setup_time)
      .def_readonly(
          "memory_alloc_time",
          &StaticRuntime::IndividualMetrics::memory_alloc_time)
      .def_readonly(
          "memory_dealloc_time",
          &StaticRuntime::IndividualMetrics::memory_dealloc_time)
      .def_readonly(
          "output_dealloc_time",
          &StaticRuntime::IndividualMetrics::output_dealloc_time)
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
  static_module
      .def(
          "__call__",
          py::overload_cast<const std::vector<at::Tensor>&>(
              &StaticModule::operator()))
      .def(
          "__call__",
          [](StaticModule& self,
             const std::vector<at::Tensor>& args,
             const std::unordered_map<std::string, at::Tensor>& kwargs) {
            std::vector<c10::IValue> arg_ivalues{args.begin(), args.end()};
            std::unordered_map<std::string, c10::IValue> kwarg_ivalues{
                kwargs.begin(), kwargs.end()};
            c10::IValue ret = self(arg_ivalues, kwarg_ivalues);
            return toPyObject(ret);
          })
      .def(
          "benchmark",
          [](StaticModule& self,
             const std::vector<at::Tensor>& args,
             const std::unordered_map<std::string, at::Tensor>& kwargs,
             const int warmup_runs,
             const int main_runs) {
            std::vector<c10::IValue> arg_ivalues{args.begin(), args.end()};
            std::unordered_map<std::string, c10::IValue> kwarg_ivalues{
                kwargs.begin(), kwargs.end()};
            self.runtime().benchmark(
                arg_ivalues, kwarg_ivalues, warmup_runs, main_runs);
          })
      .def(
          "benchmark_individual_ops",
          [](StaticModule& self,
             const std::vector<at::Tensor>& args,
             const std::unordered_map<std::string, at::Tensor>& kwargs,
             const int warmup_runs,
             const int main_runs) {
            std::vector<c10::IValue> arg_ivalues{args.begin(), args.end()};
            std::unordered_map<std::string, c10::IValue> kwarg_ivalues{
                kwargs.begin(), kwargs.end()};
            return self.runtime().benchmark_individual_ops(
                arg_ivalues, kwarg_ivalues, warmup_runs, main_runs);
          });
  m.def(
       "_jit_to_static_module",
       [](std::shared_ptr<torch::jit::Graph> g) { return StaticModule(g); })
      .def(
          "_jit_to_static_module",
          [](const torch::jit::Module& module) { return StaticModule(module); })
      .def(
          "_fuse_to_static_module",
          [](torch::jit::Module& module, size_t min_size) {
            module.eval();
            module = freeze_module(module);

            Method method = module.get_method("forward");
            auto graph = method.graph();
            fuseStaticSubgraphs(graph, min_size);
          },
          py::arg("module"),
          py::arg("min_size") = DEFAULT_FUSION_SIZE)
      .def(
          "_fuse_to_static_module",
          [](std::shared_ptr<torch::jit::Graph> g, size_t min_size) {
            fuseStaticSubgraphs(g, min_size);
          },
          py::arg("graph"),
          py::arg("min_size") = DEFAULT_FUSION_SIZE);
}

} // namespace jit
} // namespace torch
