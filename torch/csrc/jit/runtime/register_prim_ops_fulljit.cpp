#include <aten/src/ATen/Context.h>
#include <torch/csrc/autograd/autograd.h>
#include <torch/csrc/autograd/edge.h>
#include <torch/csrc/autograd/function.h>
#include <torch/csrc/autograd/generated/variable_factories.h>
#include <torch/csrc/autograd/profiler.h>
#include <torch/csrc/autograd/variable.h>
#include <torch/csrc/jit/api/compilation_unit.h>
#include <torch/csrc/jit/codegen/fuser/interface.h>
#include <torch/csrc/jit/frontend/error_report.h>
#include <torch/csrc/jit/ir/ir.h>
#include <torch/csrc/jit/runtime/custom_operator.h>
#include <torch/csrc/jit/runtime/graph_executor.h>
#include <torch/csrc/jit/runtime/jit_exception.h>
#include <torch/csrc/jit/runtime/logging.h>
#include <torch/csrc/jit/runtime/operator.h>
#include <torch/csrc/jit/runtime/print_handler.h>
#include <torch/csrc/jit/runtime/profiling_record.h>
#include <torch/csrc/jit/runtime/vararg_functions.h>
#include <torch/csrc/jit/serialization/pickle.h>

#include <ATen/ExpandUtils.h>
#include <ATen/Parallel.h>
#include <ATen/WrapDimUtils.h>
#include <ATen/core/Dict.h>
#include <ATen/core/ivalue.h>
#include <c10/core/thread_pool.h>
#include <c10/util/SmallVector.h>
#include <c10/util/math_compat.h>

#include <algorithm>
#include <bitset>
#include <cctype>
#include <cmath>
#include <exception>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <ostream>
#include <stdexcept>
#include <string>
#include <typeinfo>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace torch {
namespace jit {

namespace {

c10::AliasAnalysisKind aliasAnalysisFromSchema() {
  return c10::AliasAnalysisKind::FROM_SCHEMA;
}

c10::AliasAnalysisKind aliasAnalysisConservative() {
  return c10::AliasAnalysisKind::CONSERVATIVE;
}

c10::AliasAnalysisKind aliasAnalysisSpecialCase() {
  return c10::AliasAnalysisKind::INTERNAL_SPECIAL_CASE;
}

RegisterOperators reg({
    Operator(
        prim::profile,
        [](const Node* node) -> Operation {
          auto callback = node->cast<ProfileOp>()->getCallback();
          return [callback](Stack& stack) {
            callback(stack);
            return 0;
          };
        },
        aliasAnalysisSpecialCase()),
    Operator(
        prim::CudaFusionGroup,
        [](const Node* node) -> Operation {
          const auto key = registerFusion(node);
          return [key](Stack& stack) {
            RECORD_FUNCTION("CudaFusionGroup", std::vector<c10::IValue>());
            runFusion(key, stack);
            return 0;
          };
        },
        aliasAnalysisSpecialCase()),
    Operator(
        prim::FusionGroup,
        [](const Node* node) -> Operation {
          const auto key = registerFusion(node);
          return [key](Stack& stack) {
            RECORD_FUNCTION("FusionGroup", std::vector<c10::IValue>());
            runFusion(key, stack);
            return 0;
          };
        },
        aliasAnalysisSpecialCase()),
    Operator(
        "prim::Guard(Tensor(a) t) -> Tensor(a)",
        [](Stack& stack) {
          AT_ERROR("Should be replaced by prim::BailOut");
          return 0;
        },
        aliasAnalysisFromSchema()),
    Operator(
        "prim::BailOut(...) -> Tensor(a)",
        [](Stack& /* stack */) {
          AT_ERROR("prim::BailOut not yet implemented"); // NOLINT
          return 0;
        },
        aliasAnalysisFromSchema()),
    Operator(
        "prim::BailoutTemplate() -> int",
        [](Stack& stack) {
          // TODO: today, we put a single bailout template at the front to
          // carry the un-optimized graph for bailout nodes to use. Ideally
          // this should never run, but we haven't written the code to remove
          // it yet.
          // TORCH_INTERNAL_ASSERT(false);

          // Returns an int so that we have an easy way to do graph traversal
          push(stack, 1);
          return 0;
        },
        aliasAnalysisFromSchema()),
    Operator(
        "aten::grad(Tensor[] outputs, Tensor[] inputs, Tensor?[]? grad_outputs=None, bool? retain_graph=None, bool create_graph=False, bool allow_unused=False) -> Tensor?[]",
        [](Stack& stack) {
          bool allow_unused = pop(stack).toBool();
          bool create_graph = pop(stack).toBool();
          auto retain_graph = pop(stack).toOptional<bool>();
          auto grad_outputs = pop(stack);
          auto inputs = pop(stack).toTensorList();
          auto outputs = pop(stack).toTensorList();
          std::vector<torch::autograd::Variable> input_vars(
              inputs.begin(), inputs.end());
          std::vector<torch::autograd::Variable> output_vars(
              outputs.begin(), outputs.end());
          std::vector<torch::autograd::Variable> gradients;

          if (!grad_outputs.isNone()) {
            for (const IValue& v : grad_outputs.toListRef()) {
              gradients.emplace_back(v.isNone() ? at::Tensor() : v.toTensor());
            }
          }

          auto res = torch::autograd::grad(
              output_vars,
              input_vars,
              gradients,
              retain_graph,
              create_graph,
              allow_unused);

          c10::impl::GenericList res_list{OptionalType::ofTensor()};
          for (const at::Tensor& t : res) {
            res_list.emplace_back(t.defined() ? t : IValue());
          }
          push(stack, res_list);
          return 0;
        },
        aliasAnalysisFromSchema()),
    // NB: backward op might write to every input tensors in the graph and it's
    // much more expensive to analayze the leaves and sometimes it might retain
    // the whole gradients in every tensor of the Autograd graph with
    // create_graph=True so we use aliasAnalysisConservative for these two OPs
    Operator(
        "aten::backward(Tensor[](a!) tensors, Tensor?[]? grad_tensors=None, bool? retain_graph=None, bool create_graph=False) -> ()",
        [](Stack& stack) {
          bool create_graph = pop(stack).toBool();
          auto retain_graph = pop(stack).toOptional<bool>();
          auto grad_tensors = pop(stack);
          auto outputs = pop(stack).toTensorList();
          std::vector<torch::autograd::Variable> output_vars(
              outputs.begin(), outputs.end());
          std::vector<torch::autograd::Variable> gradients;

          if (!grad_tensors.isNone()) {
            for (const IValue& v : grad_tensors.toListRef()) {
              gradients.emplace_back(v.isNone() ? at::Tensor() : v.toTensor());
            }
          }

          torch::autograd::backward(
              output_vars, gradients, retain_graph, create_graph);
          return 0;
        },
        aliasAnalysisConservative()),
    Operator(
        "aten::backward(Tensor(a!) self, Tensor? gradient=None, bool? retain_graph=None, bool create_graph=False) -> ()",
        [](Stack& stack) {
          bool create_graph = pop(stack).toBool();
          auto retain_graph = pop(stack).toOptional<bool>();
          IValue gradient_ivalue = pop(stack);
          at::Tensor gradient = gradient_ivalue.isNone()
              ? at::Tensor()
              : gradient_ivalue.toTensor();
          at::Tensor self = pop(stack).toTensor();
          bool keep_graph = retain_graph ? retain_graph.value() : create_graph;
          self.backward(gradient, keep_graph, create_graph);
          return 0;
        },
        aliasAnalysisConservative()),
    Operator(
        "aten::save(t item, str filename) -> ()",
        [](Stack& stack) {
          auto filename = pop(stack).toStringRef();
          auto ivalue = pop(stack);

          // Pickle the tensor
          auto data = jit::pickle_save(ivalue);

          // Write file
          std::fstream output(filename, std::ios::out | std::ios::binary);
          output.write(data.data(), data.size());
          return 0;
        },
        aliasAnalysisFromSchema()),
    Operator(
        "prim::Print(...) -> ()",
        [](Stack& stack) {
          auto num_inputs = pop(stack).toInt();
          std::stringstream ss;
          bool first = true;
          for (const IValue& i : last(stack, num_inputs)) {
            if (!first)
              ss << " ";
            first = false;
            ss << i;
          }
          drop(stack, num_inputs);
          ss << std::endl;
          auto* handler = getPrintHandler();
          TORCH_INTERNAL_ASSERT(handler);
          handler(ss.str());
          return 0;
        },
        aliasAnalysisSpecialCase()),
    Operator(
        "prim::RaiseException(str msg) -> ()",
        [](Stack& stack) {
          throw JITException(pop(stack).toStringRef());
          return 0;
        },
        aliasAnalysisFromSchema()),
    Operator(
        "prim::IgnoredPythonOp(...) -> None",
        [](Stack& stack) {
          throw JITException(
              "This Python function is annotated to be ignored"
              " and cannot be and has not been included in the exported"
              " binary, meaning that it cannot be executed now."
              " Make sure that ignored operations are never executed after"
              " import");
          return 0;
        },
        aliasAnalysisFromSchema()),
});

RegisterOperators logging_operators(
    {Operator(
         "prim::AddStatValue(str key, int val) -> ()",
         [](Stack& stack) {
           auto val = pop(stack).toInt();
           auto key = pop(stack).toString();

           auto schema =
               parseSchema("prim::AddStatValue(str key, int val) -> ()");
           // TODO: remove this custom tracing code once the custom op bugfix
           // lands
           if (jit::tracer::isTracing()) {
             const auto& graph = tracer::getTracingState()->graph;
             Node* node = graph->create(prim::AddStatValue, /*num_outputs=*/0);
             tracer::recordSourceLocation(node);
             node->addInput(insertConstant(*graph, key));
             tracer::addInputs(node, "val", val);
             graph->insertNode(node);
           }
           torch::jit::logging::getLogger()->addStatValue(*key, val);
           return 0;
         },
         aliasAnalysisFromSchema()),
     Operator(
         "prim::TimePoint() -> int",
         [](Stack& stack) {
           auto schema = parseSchema("prim::TimePoint() -> int");
           Node* node = nullptr;
           // TODO: remove this custom tracing code once the custom op bugfix
           // lands
           if (jit::tracer::isTracing()) {
             const auto& graph = tracer::getTracingState()->graph;
             Node* node = graph->create(prim::TimePoint, /*num_outputs=*/0);
             tracer::recordSourceLocation(node);
             graph->insertNode(node);
           }
           auto output = autograd::profiler::getTime();
           push(stack, output);
           if (jit::tracer::isTracing()) {
             jit::tracer::addOutput(node, output);
           }
           return 0;
         },
         aliasAnalysisFromSchema())});

} // namespace
} // namespace jit
} // namespace torch
