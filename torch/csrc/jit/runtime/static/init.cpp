#include <torch/csrc/jit/runtime/static/init.h>

#include <torch/csrc/jit/passes/freeze_module.h>
#include <torch/csrc/jit/runtime/static/fusion.h>
#include <torch/csrc/jit/runtime/static/impl.h>

#include <utility>

// This number is a heuristic determined with pytorch/benchmark
static constexpr int DEFAULT_FUSION_SIZE = 4;

namespace torch::jit {

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
      .def_readonly(
          "first_iter_time", &StaticRuntime::IndividualMetrics::first_iter_time)
      .def_readonly("total_time", &StaticRuntime::IndividualMetrics::total_time)
      .def_readonly(
          "out_nodes_count", &StaticRuntime::IndividualMetrics::out_nodes_count)
      .def_readonly(
          "total_nodes_count",
          &StaticRuntime::IndividualMetrics::total_nodes_count)
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
          &StaticRuntime::IndividualMetrics::instances_per_node_type)
      .def_readonly("out_nodes", &StaticRuntime::IndividualMetrics::out_nodes);
  static_module
      .def(
          "__call__",
          [](StaticModule& self,
             const py::args& args,
             const py::kwargs& kwargs) {
            std::vector<c10::IValue> arg_ivalues;
            arg_ivalues.reserve(args.size());
            std::unordered_map<std::string, c10::IValue> kwarg_ivalues;
            kwarg_ivalues.reserve(kwargs.size());
            for (const auto& arg : args) {
              auto ivalue = torch::jit::toIValue(arg, c10::AnyType::get());
              arg_ivalues.push_back(std::move(ivalue));
            }
            for (const auto& kv : kwargs) {
              kwarg_ivalues[py::cast<std::string>(kv.first)] =
                  torch::jit::toIValue(kv.second, c10::AnyType::get());
            }
            c10::IValue ret = self(arg_ivalues, kwarg_ivalues);
            return toPyObject(std::move(ret));
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
                {arg_ivalues}, {kwarg_ivalues}, warmup_runs, main_runs);
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
                {arg_ivalues}, {kwarg_ivalues}, warmup_runs, main_runs);
          })
      .def(
          "runAsync",
          [](StaticModule& self,
             const py::tuple& args,
             const py::dict& kwargs) {
            std::vector<c10::IValue> arg_ivalues;
            arg_ivalues.reserve(args.size());
            for (const auto& elem : args) {
              arg_ivalues.push_back(
                  torch::jit::toIValue(elem, c10::AnyType::get()));
            }
            std::unordered_map<std::string, c10::IValue> kwarg_ivalues;
            kwarg_ivalues.reserve(kwargs.size());
            for (const auto& kv : kwargs) {
              kwarg_ivalues[py::cast<std::string>(kv.first)] =
                  torch::jit::toIValue(kv.second, c10::AnyType::get());
            }
            // custom executor for async op execution
            auto task_launcher = [](const std::function<void()>& f) {
              at::launch(f);
            };
            return toPyObject(self.runtime().runAsync(
                arg_ivalues, kwarg_ivalues, task_launcher));
          });
  m.def(
       "_jit_to_static_module",
       [](const std::shared_ptr<torch::jit::Graph>& g) {
         return StaticModule(g);
       })
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
            fuseStaticSubgraphs(std::move(g), min_size);
          },
          py::arg("graph"),
          py::arg("min_size") = DEFAULT_FUSION_SIZE);
}

} // namespace torch::jit
