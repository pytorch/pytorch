#include <gtest/gtest.h>

#include <ATen/core/interned_strings.h>
#include <c10/util/Exception.h>
#include <c10/util/Optional.h>
#include <test/cpp/jit/test_utils.h>
#include <torch/csrc/jit/ir/ir.h>
#include <torch/csrc/jit/ir/ir_views.h>
#include <torch/csrc/jit/ir/irparser.h>
#include <torch/csrc/jit/passes/constant_propagation.h>
#include <torch/csrc/jit/passes/symbolic_shape_runtime_fusion.h>
#include <torch/csrc/jit/passes/utils/subgraph_utils.h>
#include <torch/csrc/jit/runtime/graph_iterator.h>
#include <torch/csrc/jit/runtime/interpreter.h>
#include <torch/csrc/jit/testing/file_check.h>
#include <torch/cuda.h>
#include <unordered_map>

namespace torch {
namespace jit {

namespace {

Node* findNode(std::shared_ptr<Graph>& g, Symbol k) {
  DepthFirstGraphNodeIterator graph_it(g);
  for (auto node = graph_it.next(); node != nullptr; node = graph_it.next()) {
    if (node->kind() == k) {
      return node;
    }
  }
  TORCH_INTERNAL_ASSERT(false, "Couldn't find node");
}

} // namespace

TEST(ShapeAnalysisTest, DynamicShapesFusion) {
  // Test Generalizing shapes to symbolic dimensions, guarding those symbolic
  // dimensions and passing in runtime computed symbolic dimensions via inlined
  // shape functions
  std::shared_ptr<Graph> subgraph = std::make_shared<Graph>();
  const auto graph_string = R"IR(
      graph(%x.1 : Tensor, %y.1 : Tensor, %z: Tensor):
        %11 : int = prim::Constant[value=0]()
        %3 : Tensor = aten::tanh(%x.1)
        %out1.1 : Tensor = aten::erf(%3)
        %out2.1 : Tensor = aten::relu(%y.1)
        %10 : Tensor[] = prim::ListConstruct(%out1.1, %out2.1)
        %25 : Tensor = aten::cat(%10, %11)
        %28 : Tensor = aten::hardswish(%25)
        %29 : Tensor = aten::mul(%28, %z)
        return (%28))IR";
  torch::jit::parseIR(graph_string, subgraph.get());

  /*
  set up fused TensorExprGroup
  */

  std::shared_ptr<Graph> g = std::make_shared<Graph>();
  auto x_inp = g->addInput("x_inp");
  auto y_inp = g->addInput("y_inp");
  auto z_inp = g->addInput("z_inp");
  auto x_type = TensorType::create(at::rand({10, 5}));
  auto y_type = TensorType::create(at::rand({4, 5}));
  auto z_type = TensorType::create(at::rand({1, 1}));
  x_inp->setType(x_type);
  y_inp->setType(y_type);
  z_inp->setType(z_type);
  subgraph->inputs().at(0)->setType(x_type);
  subgraph->inputs().at(1)->setType(y_type);
  subgraph->inputs().at(2)->setType(z_type);
  subgraph->outputs().at(0)->setType(TensorType::create(at::rand({14, 5})));
  auto output = g->insertNode(g->create(prim::TensorExprGroup))->output();
  subgraph->outputs().at(0)->setType(TensorType::create(at::rand({14, 5})));
  output->node()->addInput(x_inp);
  output->node()->addInput(y_inp);
  output->node()->addInput(z_inp);
  output->node()->g_(attr::Subgraph, subgraph);

  auto success = GenerateGuard(output->node());
  TORCH_INTERNAL_ASSERT(success);
  testing::FileCheck()
      .check("TensorExprDynamicGuard")
      ->check_next("prim::If")
      ->check("aten::add")
      ->check("TensorExprGroup")
      ->check_same("symbolic_shape_inputs")
      ->check("block1")
      ->check("aten::cat")
      ->run(*g);

  // clang-format off
  /* Graph Should Look Something like: (note: strides not yet handled)
  graph(%x_inp : Float(10, 5, strides=[5, 1], requires_grad=0, device=cpu),
      %y_inp : Float(4, 5, strides=[5, 1], requires_grad=0, device=cpu),
      %z_inp : Float(1, 1, strides=[1, 1], requires_grad=0, device=cpu)):
  %4 : bool = prim::TensorExprDynamicGuard[types=[Float(SS(-2), SS(-3), strides=[5, 1], requires_grad=0, device=cpu), Float(SS(-4), SS(-3), strides=[5, 1], requires_grad=0, device=cpu), Float(1, 1, strides=[1, 1], requires_grad=0, device=cpu)]](%x_inp, %y_inp, %z_inp)
  %5 : Tensor = prim::If(%4)
    block0():
      %15 : int[] = aten::size(%x_inp)
      %16 : int[] = aten::size(%y_inp)
      %17 : int = prim::Constant[value=1]()
      %18 : int = prim::Constant[value=0]()
      %elem.3 : int = aten::__getitem__(%15, %18) # <string>:40:10
      %elem.5 : int = aten::__getitem__(%15, %17) # <string>:40:10
      %elem.11 : int = aten::__getitem__(%16, %18) # <string>:40:10
      %cat_dim_size.48 : int = aten::add(%elem.3, %elem.11) # <string>:321:29
      %3 : Tensor = prim::TensorExprGroup_0[symbolic_shape_inputs=[-5, -4, -3, -2]](%x_inp, %y_inp, %z_inp, %cat_dim_size.48, %elem.11, %elem.5, %elem.3)
      -> (%3)
    block1():
      // FallbackGraph is inlined
      %14 : Tensor = prim::FallbackGraph_1(%x_inp, %y_inp, %z_inp)
      -> (%14)
  return ()
  with prim::TensorExprGroup_0 = graph(%x.1 : Float(SS(-2), SS(-3), strides=[5, 1], requires_grad=0, device=cpu),
        %y.1 : Float(SS(-4), SS(-3), strides=[5, 1], requires_grad=0, device=cpu),
        %z : Float(1, 1, strides=[1, 1], requires_grad=0, device=cpu),
        %SS_5 : int,
        %SS_4 : int,
        %SS_3 : int,
        %SS_2 : int):
    %3 : int = prim::Constant[value=0]()
    %4 : Tensor(SS(-2), SS(-3)) = aten::tanh(%x.1)
    %5 : Tensor(SS(-2), SS(-3)) = aten::erf(%4)
    %6 : Tensor(SS(-4), SS(-3)) = aten::relu(%y.1)
    %7 : Tensor[] = prim::ListConstruct(%5, %6)
    %8 : Tensor(SS(-5), SS(-3)) = aten::cat(%7, %3)
    %9 : Tensor(SS(-5), SS(-3)) = aten::hardswish(%8)
    %10 : Tensor(SS(-5), SS(-3)) = aten::mul(%9, %z)
    return (%9)
  */
  // clang-format on

  DepthFirstGraphNodeIterator graph_it(g);
  Node* te_group = findNode(g, prim::TensorExprGroup);

  /*
  Test that input to the kernel - (10, 5), (4, 5), (1, 1) - are correctly
  generalized to sym dimensions, and that the output - (10 + 4, 5)
  correctly preserves non-catted dim as sym shape and catted dim as new sym
  shape
  */

  auto tensorexpr_graph = te_group->g(attr::Subgraph);
  auto inp1 = tensorexpr_graph->inputs().at(0)->type()->expect<TensorType>();
  auto inp2 = tensorexpr_graph->inputs().at(1)->type()->expect<TensorType>();
  auto inp3 = tensorexpr_graph->inputs().at(2)->type()->expect<TensorType>();
  auto out = tensorexpr_graph->outputs().at(0)->type()->expect<TensorType>();

  // 1 dims are preserved
  auto inp3_sizes = inp3->sizes().concrete_sizes();
  TORCH_INTERNAL_ASSERT(inp3_sizes);
  TORCH_INTERNAL_ASSERT(
      inp3_sizes->size() == 2 && inp3_sizes->at(0) == 1 &&
      inp3_sizes->at(1) == 1);

  // 5 made into sym shape
  ASSERT_EQ(
      inp1->symbolic_sizes()[1].value(), inp2->symbolic_sizes()[1].value());
  ASSERT_EQ(
      out->symbolic_sizes()[1].value(), inp2->symbolic_sizes()[1].value());

  // 4, 10, 14 are different sym shapes
  ASSERT_NE(
      inp1->symbolic_sizes()[0].value(), inp2->symbolic_sizes()[0].value());
  ASSERT_NE(
      out->symbolic_sizes()[0].value(), inp1->symbolic_sizes()[0].value());
  ASSERT_NE(
      out->symbolic_sizes()[0].value(), inp2->symbolic_sizes()[0].value());

  /*
    Test guard behaves correctly at runtime and symbolic shapes are computed
    correctly. As we don't have have TE Kernel support for dynamic shapes we're
    going to return all of the computed runtime symbolic dimensions as outputs
    of the graph on guard success, and return None on guard failure
  */

  // Setting up guard to return sym shapes on guard success and None on failure
  Node* if_node = findNode(g, prim::If);
  IfView if_v(if_node);
  if_node->eraseOutput(0);
  if_v.thenBlock()->eraseOutput(0);
  if_v.elseBlock()->eraseOutput(0);
  WithInsertPoint guard(if_node);
  auto none_val = g->insertConstant(IValue());

  auto sym_shapes = te_group->is(Symbol::attr("symbolic_shape_inputs"));
  auto offset = te_group->inputs().size() - sym_shapes.size();
  for (size_t i = 0; i < sym_shapes.size(); ++i) {
    if_v.thenBlock()->insertOutput(i, te_group->inputs().at(offset + i));
    if_v.elseBlock()->insertOutput(i, none_val);
    if_node->insertOutput(i)->setType(OptionalType::create(IntType::get()));
  }

  auto new_outputs = g->createTuple(if_node->outputs())->insertAfter(if_node);

  g->registerOutput(new_outputs->output());
  te_group->destroy();
  findNode(g, prim::FallbackGraph)->destroy();

  // Testing bad inputs

  auto first_inp = at::rand({2, 5});
  std::vector<std::vector<at::Tensor>> second_inps = {
      {at::rand({3, 4}), at::rand({1, 1})}, // sym shape mismatch
      {at::rand({5, 2}).transpose(0, 1), at::rand({1, 1})}, // discontiguous
      {at::zeros({2, 5}).to(at::ScalarType::Int),
       at::rand({1, 1})}, // wrong dtype
      {at::rand({2, 5, 1}), at::rand({1, 1})}, // wrong # dims
      {at::rand({2, 5}).requires_grad_(true),
       at::rand({1, 1})}, // requires grad
      {at::rand({2, 5}), at::rand({1, 12})}, // concrete dim mismatch (1)
  };
  if (torch::cuda::is_available()) {
    second_inps.push_back({at::rand({2, 5}).cuda(), at::rand({1, 1})});
  }
  for (const auto& last_inps : second_inps) {
    // todo - reusing interpreter across iters gave error
    Code code(g, "");
    InterpreterState interp(code);
    auto stack = createStack({at::rand({2, 5}), last_inps[0], last_inps[1]});
    interp.run(stack);
    TORCH_INTERNAL_ASSERT(pop(stack).toTuple()->elements().at(0).isNone());
  }

  // Test good inputs
  Code code(g, "");
  InterpreterState interp(code);
  std::vector<at::Tensor> inps = {
      at::rand({2, 5}), at::rand({4, 5}), at::rand({1, 1})};
  Stack stack(inps.begin(), inps.end());
  interp.run(stack);
  auto tuple = pop(stack).toTuple();
  TORCH_INTERNAL_ASSERT(tuple->elements().at(0).isInt());

  // Testing that the sym shape calculation was correct
  for (size_t i = 0; i < sym_shapes.size(); ++i) {
    auto sym_shape = sym_shapes[i];
    auto computed_value = tuple->elements().at(i).toInt();
    if (sym_shape == inp1->symbolic_sizes().at(0).value()) {
      ASSERT_EQ(computed_value, 2);
    } else if (sym_shape == inp1->symbolic_sizes().at(1).value()) {
      ASSERT_EQ(computed_value, 5);
    } else if (sym_shape == inp2->symbolic_sizes().at(0).value()) {
      ASSERT_EQ(computed_value, 4);
    } else if (sym_shape == out->symbolic_sizes().at(0).value()) {
      ASSERT_EQ(computed_value, 6);
    } else {
      TORCH_INTERNAL_ASSERT(false);
    }
  }
}

TEST(ShapeAnalysisTest, MovingConstantOutOfFusionGroups) {
  std::shared_ptr<Graph> subgraph = std::make_shared<Graph>();
  const auto graph_string = R"IR(
      graph(%x.1 : Tensor):
        %none : NoneType = prim::Constant()
        %size1 : int = prim::Constant[value=1]()
        %size10 : int = prim::Constant[value=10]()
        %sizes : int[] = prim::ListConstruct(%size10, %size1)
        %device : Device = prim::Constant[value="cpu"]()
        %10 : Tensor = aten::ones(%sizes, %none, %none, %device, %none)
        %3 : Tensor = aten::tanh(%x.1)
        %29 : Tensor = aten::mul(%3, %10)
        return (%29))IR";
  torch::jit::parseIR(graph_string, subgraph.get());
  ConstantPropagation(subgraph);

  std::shared_ptr<Graph> g = std::make_shared<Graph>();
  auto x_inp = g->addInput("x_inp");
  auto x_type = TensorType::create(at::rand({10, 5}));
  x_inp->setType(x_type);
  subgraph->inputs().at(0)->setType(x_type);
  subgraph->outputs().at(0)->setType(x_type);
  auto output = g->insertNode(g->create(prim::TensorExprGroup))->output();
  output->node()->addInput(x_inp);
  output->node()->g_(attr::Subgraph, subgraph);

  auto success = GenerateGuard(output->node());
  TORCH_INTERNAL_ASSERT(success);

  // Check that the constants have been moved out of the fused graph.
  // This should result in not have any conditionals other than the one
  // checking the result of TensorExprDynamicGuard.
  testing::FileCheck()
      .check("TensorExprDynamicGuard")
      ->check_next("prim::If")
      ->check_not("prim::If") // no other IFs due to constants.
      ->check("TensorExprGroup")
      ->check("block1")
      ->check("FallbackGraph")
      ->run(*g);
}

} // namespace jit
} // namespace torch
