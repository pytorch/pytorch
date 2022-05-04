#include <gtest/gtest.h>

#include <torch/csrc/autograd/generated/variable_factories.h>
#include <torch/csrc/jit/frontend/ir_emitter.h>
#include <torch/csrc/jit/ir/alias_analysis.h>
#include <torch/csrc/jit/ir/ir.h>
#include <torch/csrc/jit/ir/irparser.h>
#include <torch/csrc/jit/jit_log.h>
#include <torch/csrc/jit/passes/constant_pooling.h>
#include <torch/csrc/jit/passes/pass_manager.h>
#include <torch/csrc/jit/passes/tensorexpr_fuser.h>
#include <torch/csrc/jit/passes/utils/subgraph_utils.h>
#include <torch/csrc/jit/runtime/custom_operator.h>
#include <torch/csrc/jit/testing/file_check.h>
#include <memory>
#include "c10/util/Exception.h"
#include "c10/util/irange.h"

namespace torch {
namespace jit {

namespace {

// clang-format off
/*
In this test file, we will be writing a pass that fuses
multiple adds into one single fusion group that invokes
`_foreach_add_.Scalar(Tensor(a!)[] self, Scalar scalar) -> ()`
Because this kernel is only defined for cuda tensors,
the pass only operators on cuda tensors, and sets up appropriate
guards to ensure that the runtime tesnors follow the same necessary
properties as the profiled tensors.

Example before Graph
graph(%x.1 : Float(4, 4, strides=[4, 1], requires_grad=0, device=cuda:0),
      %y.1 : Float(1, strides=[1], requires_grad=0, device=cuda:0),
      %z.1 : Float(1, 1, strides=[1, 1], requires_grad=0, device=cuda:0),
      %scalar.1 : int):
  %4 : int = prim::Constant[value=1]()
  %6 : Float(4, 4, strides=[4, 1], requires_grad=0, device=cuda:0) = aten::add(%x.1, %scalar.1, %4)
  %8 : Float(1, strides=[1], requires_grad=0, device=cuda:0) = aten::add(%y.1, %scalar.1, %4)
  %11 : Float(4, 4, strides=[4, 1], requires_grad=0, device=cuda:0) = aten::add(%x.1, %y.1, %4)
  %13 : Float(1, 1, strides=[1, 1], requires_grad=0, device=cuda:0) = aten::add(%z.1, %scalar.1, %4)
  %18 : (Tensor, Tensor, Tensor, Tensor) = prim::TupleConstruct(%6, %8, %13, %11)
  return (%18)
Example After Graph
graph(%x.1 : Float(4, 4, strides=[4, 1], requires_grad=0, device=cuda:0),
      %y.1 : Float(1, strides=[1], requires_grad=0, device=cuda:0),
      %z.1 : Float(1, 1, strides=[1, 1], requires_grad=0, device=cuda:0),
      %scalar.1 : int):
  %4 : int = prim::Constant[value=1]()
  %25 : Float(4, 4, strides=[4, 1], requires_grad=0, device=cuda:0), %26 : Float(1, strides=[1], requires_grad=0, device=cuda:0), %27 : Float(1, 1, strides=[1, 1], requires_grad=0, device=cuda:0), %28 : bool = prim::TypeCheck[types=[Tensor(device=cuda:0), Tensor(device=cuda:0), Tensor(device=cuda:0)]](%x.1, %y.1, %z.1)
  %29 : Float(4, 4, strides=[4, 1], requires_grad=0, device=cuda:0), %30 : Float(1, strides=[1], requires_grad=0, device=cuda:0), %31 : Float(1, 1, strides=[1, 1], requires_grad=0, device=cuda:0) = prim::If(%28)
    block0():
      %20 : Float(4, 4, strides=[4, 1], requires_grad=0, device=cuda:0), %22 : Float(1, strides=[1], requires_grad=0, device=cuda:0), %24 : Float(1, 1, strides=[1, 1], requires_grad=0, device=cuda:0) = prim::FusedAddGraph_0(%25, %scalar.1, %26, %27)
      -> (%20, %22, %24)
    block1():
      %38 : Tensor, %39 : Tensor, %40 : Tensor = prim::FallbackGraph_1(%x.1, %scalar.1, %y.1, %z.1)
      -> (%38, %39, %40)
  %11 : Float(4, 4, strides=[4, 1], requires_grad=0, device=cuda:0) = aten::add(%x.1, %y.1, %4)
  %18 : (Tensor, Tensor, Tensor, Tensor) = prim::TupleConstruct(%29, %30, %31, %11)
  return (%18)
*/
// clang-format on

size_t counter = 0;

Operation createFusedAddGroup(const Node* node) {
  auto num_inputs = node->inputs().size();
  TORCH_INTERNAL_ASSERT(node->input(1)->type()->isSubtypeOf(NumberType::get()));
  return [num_inputs](Stack& stack) {
    counter += 1;
    std::vector<at::Tensor> inputs;
    at::Scalar scalar_arg;
    for (const size_t i : c10::irange(num_inputs)) {
      if (i == 1) {
        scalar_arg = peek(stack, i, num_inputs).toScalar();
      } else {
        inputs.push_back((peek(stack, i, num_inputs)).toTensor());
      }
    }
    drop(stack, num_inputs);
    auto out = at::_foreach_add(inputs, scalar_arg);
    for (const auto& ten : out) {
      stack.push_back(ten);
    }
    return 0;
  };
};

Symbol FusedAddGraphSymbol = Symbol::prim("FusedAddGraph");

// We dont actually need to register a custom operator to do this,
// could just rewrite graph to call _for_each,
// but do it show more complete tutorial
void register_operator() {
  torch::jit::RegisterFusionSubgraphKind(FusedAddGraphSymbol);
  RegisterOperators FusedAddGraph({
      torch::jit::Operator(
          FusedAddGraphSymbol,
          createFusedAddGroup,
          AliasAnalysisKind::PURE_FUNCTION),
  });
}

struct ForEachAddFuser {
  ForEachAddFuser(std::shared_ptr<Graph> graph) : graph_(std::move(graph)) {
    ConstantPooling(graph_);
    alias_db_ = std::make_unique<AliasDb>(graph_);
  }

 public:
  Node* CreateFusedAddSubgraph(Node* n) {
    GRAPH_UPDATE("Creating a fused ::Group node from: ", *n);
    return SubgraphUtils::createSingletonSubgraphAndUpdateAliasing(
        n, FusedAddGraphSymbol, *alias_db_);
  }

  bool checkTensorInputType(const TensorTypePtr& tt) {
    // check input tensor conditions: here, just cuda
    return tt->device()->is_cuda();
  }

  void createFusionGroups(Block* b) {
    // iterate over Nodes in graph, trying horizontall merge add nodes with the
    // same scalar
    Node* curr_fusion_graph = nullptr;
    Value* curr_scalar_arg = nullptr;
    // NB: in this fusion pass, we are just merging nodes horizontally,
    // typically, one might pull in inputs to fusion group to fuse vertically
    // See tensorexpr_fuser.cpp or create_autodiff_subgraphs.cpp
    // for example on vertical fusion

    for (auto it = b->nodes().begin(); it != b->nodes().end();) {
      Node* node = *it;
      for (Block* block : node->blocks()) {
        createFusionGroups(block);
      }
      it++; // advance iterator bc the current node may be destroyed
      if (!node->matches(
              "aten::add.Scalar(Tensor self, Scalar other, Scalar alpha=1) -> Tensor",
              /*const_inputs=*/attr::alpha)) {
        continue;
      }
      // _foreach_add.Scalar(Tensor[] tensors, Scalar scalar) -> Tensor[]
      // does not support alpha
      if (node->get<at::Scalar>(attr::alpha)->toDouble() != 1) {
        continue;
      }
      // only supported on CUDA
      if (!checkTensorInputType(node->input(0)->type()->expect<TensorType>())) {
        continue;
      }
      Value* scalar_arg = node->input(1);
      // if there exists a fusion graph, and we can merge current node after it,
      // and the same scalar arg exists between the two, merge them
      if (curr_fusion_graph && scalar_arg == curr_scalar_arg &&
          alias_db_->moveAfterTopologicallyValid(node, curr_fusion_graph)) {
        SubgraphUtils::mergeNodeIntoSubgraphAndUpdateAliasing(
            node, curr_fusion_graph, *alias_db_);
        continue;
      }
      curr_scalar_arg = scalar_arg;
      curr_fusion_graph = CreateFusedAddSubgraph(node);
      fusion_groups_.push_back(curr_fusion_graph);
    }
  }

  void guardFusionGroups() {
    for (Node* n : fusion_groups_) {
      // the types present in the graph were present during profiling,
      // and are not guaranteed in future executions
      // to do this (cuda-only) optimization, we need to guard that the inputs
      // are cuda If the guards fail, then we re-profile the fused graph nodes
      // NB: if we wanted to guard on complete shapes/strides/etc, we would
      // return profiled_type To guard without strides, we remove strides from
      // the input: profiled_type->withStrides({}); To guard on number of
      // dimensions, we would remove complete sizes
      // profiled_type->withDim(*profiled_type->dim());
      auto tensor_type_converter = [](const TensorTypePtr& profiled_type) {
        TORCH_INTERNAL_ASSERT(profiled_type->device()->has_index());
        return TensorType::get()->withDevice(profiled_type->device());
      };
      // unless you are doing special, use prim::TypeCheck which will
      // ensure the types at runtime match up to the types provided
      Symbol guard_kind = prim::TypeCheck;
      insertTypeGuard(n, tensor_type_converter, guard_kind);
    }
  }

  void run() {
    // create fusion groups
    createFusionGroups(graph_->block());
    // TODO: could unnmerge single-element fusion groups
    guardFusionGroups();
  }

 private:
  std::vector<Node*> fusion_groups_;
  std::shared_ptr<Graph> graph_;
  std::unique_ptr<AliasDb> alias_db_;
};

void FuseForEachAdd(std::shared_ptr<Graph>& graph) {
  // see jit_log.h for more details on logging
  GRAPH_DUMP("In FuseForEachAdd Custom Post Pass: ", graph);
  ForEachAddFuser fuser(graph);
  fuser.run();
  GRAPH_DUMP("After FuseForEachAdd Custom Post Pass: ", graph);
}

} // namespace

TEST(ProfilingCustomPassTest, Basic_CUDA) {
  // only test in profiling
  if (!getExecutorMode()) {
    return;
  }
  register_operator();

  RegisterPass p(FuseForEachAdd);
  std::shared_ptr<Graph> graph = std::make_shared<Graph>();
  parseIR(
      R"IR(
graph(%x.1 : Tensor,
      %y.1 : Tensor,
      %z.1 : Tensor,
      %scalar.1 : int):
  %6 : int = prim::Constant[value=1]()
  %7 : Tensor = aten::add(%x.1, %scalar.1, %6)
  %11 : Tensor = aten::add(%y.1, %scalar.1, %6)
  %15 : Tensor = aten::add(%x.1, %y.1, %6)
  %19 : Tensor = aten::add(%z.1, %scalar.1, %6)
  %20 : (Tensor, Tensor, Tensor, Tensor) = prim::TupleConstruct(%7, %11, %19, %15)
  return (%20)
    )IR",
      &*graph);

  auto cuda_ten1 = at::rand({4, 4}).cuda();
  auto cuda_ten2 = at::rand({1}).cuda();
  auto cuda_ten3 = at::rand({1, 1}).cuda();
  auto scalar_input = 5;
  std::vector<IValue> stack = {cuda_ten1, cuda_ten2, cuda_ten3, scalar_input};
  GraphExecutor executor(graph, "");
  // profiling run
  executor.run(stack);
  stack = {cuda_ten1, cuda_ten2, cuda_ten3, 5};
  // fusion executed
  executor.run(stack);
  auto tup_output = stack.back().toTuple();
  std::vector<at::Tensor> inps = {cuda_ten1, cuda_ten2, cuda_ten3};
  for (size_t i = 0; i < 3; ++i) {
    ASSERT_TRUE(at::allclose(
        tup_output->elements().at(i).toTensor(),
        at::add(inps[i], scalar_input)));
  }
  ASSERT_TRUE(at::allclose(
      tup_output->elements().at(3).toTensor(), at::add(inps[0], inps[1])));

  TORCH_INTERNAL_ASSERT(counter == 1);
  auto last_graph = lastExecutedOptimizedGraph();
  torch::jit::testing::FileCheck().check("FusedAddGraph")->run(*last_graph);
  stack = {cuda_ten1.cpu(), cuda_ten2.cpu(), cuda_ten3.cpu(), scalar_input};
  executor.run(stack);
  // should fail cuda type guard
  TORCH_INTERNAL_ASSERT(counter == 1);
}

} // namespace jit
} // namespace torch
