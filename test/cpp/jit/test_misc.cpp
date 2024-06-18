#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <ATen/ATen.h>
#include <ATen/Parallel.h>
#include <ATen/core/interned_strings.h>
#include <ATen/core/ivalue.h>
#include <ATen/core/jit_type_base.h>
#include <test/cpp/jit/test_utils.h>
#include <torch/csrc/jit/passes/remove_mutation.h>
#include <torch/csrc/jit/passes/tensorexpr_fuser.h>
#include <torch/csrc/jit/tensorexpr/kernel.h>

#include <torch/csrc/autograd/engine.h>
#include <torch/csrc/autograd/generated/variable_factories.h>
#include <torch/csrc/autograd/profiler.h>
#include <torch/csrc/autograd/variable.h>
#include <torch/csrc/jit/api/function_impl.h>
#include <torch/csrc/jit/api/module.h>
#include <torch/csrc/jit/codegen/fuser/interface.h>
#include <torch/csrc/jit/frontend/ir_emitter.h>
#include <torch/csrc/jit/frontend/tracer.h>
#include <torch/csrc/jit/ir/alias_analysis.h>
#include <torch/csrc/jit/ir/attributes.h>
#include <torch/csrc/jit/ir/irparser.h>
#include <torch/csrc/jit/ir/scope.h>
#include <torch/csrc/jit/ir/type_hashing.h>
#include <torch/csrc/jit/jit_log.h>
#include <torch/csrc/jit/passes/bailout_graph.h>
#include <torch/csrc/jit/passes/canonicalize.h>
#include <torch/csrc/jit/passes/common_subexpression_elimination.h>
#include <torch/csrc/jit/passes/constant_propagation.h>
#include <torch/csrc/jit/passes/create_autodiff_subgraphs.h>
#include <torch/csrc/jit/passes/dead_code_elimination.h>
#include <torch/csrc/jit/passes/graph_fuser.h>
#include <torch/csrc/jit/passes/guard_elimination.h>
#include <torch/csrc/jit/passes/inline_autodiff_subgraphs.h>
#include <torch/csrc/jit/passes/insert_guards.h>
#include <torch/csrc/jit/passes/liveness.h>
#include <torch/csrc/jit/passes/loop_unrolling.h>
#include <torch/csrc/jit/passes/lower_grad_of.h>
#include <torch/csrc/jit/passes/lower_tuples.h>
#include <torch/csrc/jit/passes/pass_manager.h>
#include <torch/csrc/jit/passes/requires_grad_analysis.h>
#include <torch/csrc/jit/passes/restore_mutation.h>
#include <torch/csrc/jit/passes/shape_analysis.h>
#include <torch/csrc/jit/passes/symbolic_shape_analysis.h>
#include <torch/csrc/jit/passes/utils/subgraph_utils.h>
#include <torch/csrc/jit/runtime/argument_spec.h>
#include <torch/csrc/jit/runtime/autodiff.h>
#include <torch/csrc/jit/runtime/custom_operator.h>
#include <torch/csrc/jit/runtime/decomposition_registry.h>
#include <torch/csrc/jit/runtime/graph_executor.h>
#include <torch/csrc/jit/runtime/interpreter.h>
#include <torch/csrc/jit/runtime/jit_trace.h>
#include <torch/csrc/jit/runtime/profiling_record.h>
#include <torch/csrc/jit/runtime/symbolic_script.h>
#include <torch/csrc/jit/runtime/symbolic_shape_registry.h>
#include <torch/csrc/jit/serialization/import.h>
#include <torch/csrc/jit/testing/file_check.h>
#include <torch/jit.h>
#include <torch/script.h>

#include <onnx/onnx_pb.h>

#include <c10/util/Exception.h>
#include <c10/util/ThreadLocalDebugInfo.h>

#include <torch/csrc/jit/passes/freeze_module.h>
#include <torch/csrc/jit/passes/frozen_graph_optimizations.h>
#include <algorithm>
#include <cstddef>
#include <functional>
#include <iostream>
#include <memory>
#include <set>
#include <stdexcept>
#include <string>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace torch {
namespace jit {
inline c10::AliasAnalysisKind aliasAnalysisFromSchema() {
  return c10::AliasAnalysisKind::FROM_SCHEMA;
}

template <typename T>
std::ostream& operator<<(std::ostream& out, const std::vector<T>& list) {
  size_t i = 0;
  out << "{";
  for (auto&& e : list) {
    if (i++ > 0)
      out << ", ";
    out << e;
  }
  out << "}";
  return out;
}

TEST(InternedStringsTest, Basic) {
  ASSERT_EQ(prim::Param, Symbol::prim("Param"));
  ASSERT_EQ(prim::Return, Symbol::prim("Return"));
  ASSERT_EQ(prim::Return.toUnqualString(), std::string("Return"));
  ASSERT_EQ(prim::Return.toQualString(), std::string("prim::Return"));
  Symbol newsym = Symbol::aten("__NEW_SYMBOL");
  size_t symstart = newsym;
  ASSERT_EQ(newsym.toQualString(), std::string("aten::__NEW_SYMBOL"));
  // TODO: This test is a bit too close to the implementation details.
  ASSERT_EQ(Symbol::aten("What"), symstart + 1);
  ASSERT_EQ(Symbol::aten("What2"), symstart + 2);
  ASSERT_EQ(Symbol::aten("What"), symstart + 1);
  ASSERT_EQ(Symbol::aten("What2"), symstart + 2);
  ASSERT_EQ(Symbol(symstart + 2).toUnqualString(), std::string("What2"));
}

TEST(FromQualStringTest, Basic) {
  ASSERT_EQ(Symbol::fromQualString("prim::Param"), Symbol::prim("Param"));
  ASSERT_EQ(Symbol::fromQualString("aten::mm"), Symbol::aten("mm"));
  ASSERT_EQ(Symbol::fromQualString("onnx::LSTM"), Symbol::onnx("LSTM"));
  ASSERT_EQ(Symbol::fromQualString("attr::value"), Symbol::attr("value"));
  ASSERT_EQ(Symbol::fromQualString("scope::"), Symbol::scope(""));
  ASSERT_EQ(Symbol::fromQualString("::").toUnqualString(), std::string(""));
  ASSERT_EQ(
      Symbol::fromQualString("::").ns().toQualString(),
      std::string("namespaces::"));
  ASSERT_EQ(
      Symbol::fromQualString("new_ns::param").toUnqualString(),
      std::string("param"));
  ASSERT_EQ(
      Symbol::fromQualString("new_ns::param").ns().toUnqualString(),
      std::string("new_ns"));
  ASSERT_EQ(
      Symbol::fromQualString("new_ns::param").ns(),
      Symbol::fromQualString("namespaces::new_ns"));

  auto bad_inputs = {"scope", ":", ""};
  for (auto input : bad_inputs) {
    try {
      Symbol::fromQualString(input);
      ASSERT_TRUE(0);
    } catch (const std::exception& c) {
    }
  }
}

TEST(THNNConvTest, Basic) {
  std::vector<int64_t> input_size = {4, 3, 15, 17}; // B x C x H x W
  std::vector<int64_t> kernel_size = {3, 5};
  std::vector<int64_t> stride = {1, 2};
  std::vector<int64_t> padding = {2, 1};
  constexpr int out_channels = 5;

  // make inputs
  at::Tensor input = torch::randn(input_size);
  at::Tensor weight = torch::randn(
      {out_channels, input_size[1], kernel_size[0], kernel_size[1]});
  at::Tensor bias = torch::randn({out_channels});

  // run forward eagerly
  at::Tensor output = at::_slow_conv2d_forward(
      input, weight, kernel_size, bias, stride, padding);

  // make grad_outputs
  at::Tensor grad_output =
      torch::randn_like(output, at::MemoryFormat::Preserve);

  // run backward eagerly
  at::Tensor grad_input, grad_weight, grad_bias;
  std::tie(grad_input, grad_weight, grad_bias) = at::_slow_conv2d_backward(
      grad_output,
      input,
      weight,
      kernel_size,
      stride,
      padding,
      {true, true, true});

  // make JIT graph
  auto graph = std::make_shared<Graph>();
  auto ksz_val = graph->insertConstant(kernel_size);
  auto kst_val = graph->insertConstant(stride);
  auto pad_val = graph->insertConstant(padding);

  auto inputg = graph->addInput("self");
  auto weightg = graph->addInput("weight");
  auto biasg = graph->addInput("bias");

  Value* conv = graph->insert(
      aten::_slow_conv2d_forward,
      {inputg, weightg, ksz_val, biasg, kst_val, pad_val});
  auto outputs = conv->node()->outputs();
  for (auto output : outputs) {
    graph->registerOutput(output);
  }
  LowerAllTuples(graph);
  graph->lint();

  // differentiate JIT graph
  EliminateDeadCode(graph); // Tracing of some ops depends on the DCE trick
  ConstantPropagation(graph);
  auto grad_spec = differentiate(graph);
  LowerGradOf(*grad_spec.df);

  // prepare JIT inputs / gradients
  tensor_list tensors_in;
  tensors_in.push_back(input);
  tensors_in.push_back(weight);
  tensors_in.push_back(bias);

  tensor_list tensor_grads_in;
  tensor_grads_in.push_back(grad_output);

  // Get outputs from the interpreter
  tensor_list tensors_out, tensor_grads_out;
  std::tie(tensors_out, tensor_grads_out) =
      runGradient(grad_spec, tensors_in, tensor_grads_in);

  // prepare expected structs
  tensor_list expected_tensors_out, expected_tensor_grads_out;
  expected_tensors_out.push_back(output);
  expected_tensor_grads_out.push_back(grad_input);
  expected_tensor_grads_out.push_back(grad_weight);
  expected_tensor_grads_out.push_back(grad_bias);

  // Compare results
  assertAllClose(tensors_out, expected_tensors_out);
  assertAllClose(tensor_grads_out, expected_tensor_grads_out);
}

TEST(ATenNativeBatchNormTest, Basic) {
  // aten::native_batch_norm(Tensor input, Tensor weight, Tensor bias, Tensor
  // running_mean, Tensor running_var, bool training, float momentum, float eps)
  // -> (Tensor, Tensor, Tensor)
  std::vector<int64_t> input_size = {4, 3, 15, 17}; // B x C x H x W
  bool training = true;
  float momentum = 0.9;
  float eps = 1e-5;

  // make inputs
  at::Tensor input = torch::randn(input_size);
  at::Tensor weight = torch::randn({input_size[1]});
  at::Tensor bias = torch::randn({input_size[1]});
  at::Tensor running_mean = torch::randn({input_size[1]});
  at::Tensor running_var = torch::randn({input_size[1]});

  // running_mean and running_var are changed in-place, so clone and send them
  at::Tensor running_mean_eager = running_mean.clone();
  at::Tensor running_var_eager = running_var.clone();
  at::Tensor running_mean_jit = running_mean.clone();
  at::Tensor running_var_jit = running_var.clone();

  // run forward eagerly
  at::Tensor output, savemean, saveinvstd;
  std::tie(output, savemean, saveinvstd) = at::native_batch_norm(
      input,
      weight,
      bias,
      running_mean_eager,
      running_var_eager,
      training,
      momentum,
      eps);

  // make grad_outputs
  at::Tensor grad_output =
      torch::randn_like(output, at::MemoryFormat::Preserve);
  at::Tensor grad_savemean =
      torch::zeros_like(savemean, at::MemoryFormat::Preserve);
  at::Tensor grad_saveinvstd =
      torch::zeros_like(saveinvstd, at::MemoryFormat::Preserve);

  // run backward eagerly
  at::Tensor grad_input, grad_weight, grad_bias;
  // aten::native_batch_norm_backward(Tensor grad_out, Tensor input, Tensor
  // weight, Tensor running_mean, Tensor running_var, Tensor save_mean, Tensor
  // save_invstd, bool train, float eps, bool[3] output_mask) -> (Tensor,
  // Tensor, Tensor)
  std::tie(grad_input, grad_weight, grad_bias) = at::native_batch_norm_backward(
      grad_output,
      input,
      weight,
      running_mean_eager,
      running_var_eager,
      savemean,
      saveinvstd,
      training,
      eps,
      {true, true, true});

  // make JIT graph
  auto graph = std::make_shared<Graph>();
  auto training_val = graph->insertConstant(IValue(training));
  auto momentum_val = graph->insertConstant(IValue(momentum));
  auto eps_val = graph->insertConstant(IValue(eps));

  auto inputg = graph->addInput("self");
  auto weightg = graph->addInput("weight");
  auto biasg = graph->addInput("bias");
  auto running_meang = graph->addInput("running_mean");
  auto running_varg = graph->addInput("running_var");

  Value* bn = graph->insert(
      aten::native_batch_norm,
      {inputg,
       weightg,
       biasg,
       running_meang,
       running_varg,
       training_val,
       momentum_val,
       eps_val});
  auto outputs = bn->node()->outputs();
  for (auto output : outputs) {
    graph->registerOutput(output);
  }
  LowerAllTuples(graph);
  graph->lint();

  // differentiate JIT graph
  EliminateDeadCode(graph); // Tracing of some ops depends on the DCE trick
  ConstantPropagation(graph);
  auto grad_spec = differentiate(graph);
  LowerGradOf(*grad_spec.df);

  // prepare JIT inputs / gradients
  tensor_list tensors_in;
  tensors_in.push_back(input);
  tensors_in.push_back(weight);
  tensors_in.push_back(bias);
  tensors_in.push_back(running_mean_jit);
  tensors_in.push_back(running_var_jit);

  tensor_list tensor_grads_in;
  tensor_grads_in.push_back(grad_output);
  tensor_grads_in.push_back(grad_savemean);
  tensor_grads_in.push_back(grad_saveinvstd);

  // Get outputs from the interpreter
  tensor_list tensors_out, tensor_grads_out;
  std::tie(tensors_out, tensor_grads_out) =
      runGradient(grad_spec, tensors_in, tensor_grads_in);

  // prepare expected structs
  tensor_list expected_tensors_out, expected_tensor_grads_out;
  expected_tensors_out.push_back(output);
  expected_tensors_out.push_back(savemean);
  expected_tensors_out.push_back(saveinvstd);
  expected_tensors_out.push_back(running_mean_eager);
  expected_tensors_out.push_back(running_var_eager);
  expected_tensor_grads_out.push_back(grad_input);
  expected_tensor_grads_out.push_back(grad_weight);
  expected_tensor_grads_out.push_back(grad_bias);

  tensors_out.push_back(running_mean_jit);
  tensors_out.push_back(running_var_jit);

  // Compare results
  assertAllClose(tensors_out, expected_tensors_out);
  assertAllClose(tensor_grads_out, expected_tensor_grads_out);
}

TEST(CustomFusionTest, Basic) {
#if defined(FBCODE_CAFFE2)
  return;
#endif

  auto graph_string = R"IR(
    graph(%0 : Float(2, 3, 4),
          %1 : Float(2, 3, 4)):
      %2 : Tensor = aten::mul(%0, %1)
      %3 : Tensor = aten::mul(%2, %0)
      return (%3))IR";
  auto g = std::make_shared<Graph>();
  torch::jit::parseIR(graph_string, g.get());

  torch::jit::overrideCanFuseOnCPU(true);
  CustomFuseGraph(
      g,
      [](Node* n) { return n->kind() != prim::Param; },
      Symbol::fromQualString("prim::FusionGroup"));
  torch::jit::overrideCanFuseOnCPU(false);

  const auto& nodes = g->nodes();
  auto fusion_group =
      std::find_if(nodes.begin(), nodes.end(), [](const Node* node) {
        return node->kind() == Symbol::fromQualString("prim::FusionGroup");
      });
  AT_ASSERT(fusion_group != nodes.end());

  auto subgraph = fusion_group->g(attr::Subgraph);
  auto hits = 0;
  // two multiplications
  for (const auto& n : subgraph->nodes()) {
    (void)n;
    hits++;
  }
  AT_ASSERT(hits == 2);
}

TEST(CustomFusionTest, NestedBlocks) {
#if defined(FBCODE_CAFFE2)
  return;
#endif

  auto graph_string = R"IR(
  graph(%0 : Float(2, 3, 4),
        %1 : Float(2, 3, 4),
        %2 : Float(2, 3, 4)):
    %3 : int = prim::Constant[value=1]()
    %4 : Tensor = prim::If(%2)
      block0():
        %5 : Tensor = aten::mul(%0, %2)
        %6 : Tensor = aten::mul(%5, %1)
        -> (%6)
      block1():
        %7 : Tensor = aten::add(%0, %2, %3)
        %8 : Tensor = aten::add(%7, %1, %3)
        -> (%8)
    %9 : Tensor = aten::add(%4, %2, %3)
    return (%4))IR";
  auto g = std::make_shared<Graph>();
  torch::jit::parseIR(graph_string, g.get());

  CustomFuseGraph(
      g,
      [](Node* n) { return n->kind() == aten::mul; },
      Symbol::fromQualString("prim::FusionGroup"));

  // Could be done in more efficient ways, but this is only a test.
  std::function<bool(const Block*, Symbol)> dfs = [&](const Block* b,
                                                      Symbol s) {
    for (auto node : b->nodes()) {
      if (node->kind() == s)
        return true;
      for (auto nested_b : node->blocks())
        if (dfs(nested_b, s))
          return true;
    }
    return false;
  };

  AT_ASSERT(dfs(g->block(), Symbol::fromQualString("prim::FusionGroup")));
}

static const auto cf_examples = R"JIT(
  def if_test(a, b):
      # FIXME: use 0 instead of a.
      # c = 0
      c = a
      if bool(a < b):
        c = b
      else:
        c = a
      return c
  def if_one(a, b):
    c = b
    if bool(a < b):
      c = a
    return c
  def while_test(a, i):
    while bool(i < 3):
      a *= a
      i += 1
    return a
)JIT";

TEST(ControlFlowTest, Basic) {
  auto cu = compile(cf_examples);

  auto run = [&](const std::string& name, std::vector<IValue> stack) {
    auto graph = toGraphFunction(cu->get_function(name)).graph();
    Code code(graph, "");
    InterpreterState interp(code);
    interp.run(stack);
    return stack;
  };

  auto L = [](int64_t l) { return IValue(scalar_to_tensor(at::Scalar(l))); };
  auto V = [](IValue t) { return std::move(t).toTensor().item<int64_t>(); };
  auto run_binary = [&](const std::string& name, int64_t a, int64_t b) {
    return V(run(name, {L(a), L(b)})[0]);
  };
  ASSERT_EQ(2, run_binary("if_test", 1, 2));
  ASSERT_EQ(3, run_binary("if_test", 3, 2));
  ASSERT_EQ(2, run_binary("if_one", 2, 3));
  ASSERT_EQ(2, run_binary("if_one", 3, 2));
  ASSERT_EQ(256, run_binary("while_test", 2, 0));
}

#if defined(__has_feature)
#if __has_feature(address_sanitizer)
#define HAS_ASANUBSAN 1
#endif
#endif

#ifndef HAS_ASANUBSAN
// This test fails vptr UBSAN checks

TEST(ProtoTest, Basic) {
  ::ONNX_NAMESPACE::ModelProto proto;
  proto.set_producer_name("foo");
}
#endif

// test a few features that are not directly used in schemas yet
TEST(SchemaParserTest, NestedArrays) {
  // nested arrays
  auto s = parseSchema("at::what(int[][4] foo) -> ()");
  ASSERT_TRUE(s.arguments().at(0).N() == 4);
  ASSERT_TRUE(IntType::get()->isSubtypeOf(*s.arguments()
                                               .at(0)
                                               .type()
                                               ->expectRef<ListType>()
                                               .getElementType()
                                               ->expectRef<ListType>()
                                               .getElementType()));
  auto s2 = parseSchema("at::what(int[][] foo) -> ()");
  ASSERT_TRUE(IntType::get()->isSubtypeOf(*s2.arguments()
                                               .at(0)
                                               .type()
                                               ->expectRef<ListType>()
                                               .getElementType()
                                               ->expectRef<ListType>()
                                               .getElementType()));
}

TEST(SchemaParserTest, OutVariant) {
  auto schema_with_out = parseSchema(
      "at::foo(Tensor self, *, Tensor(a!) f, Tensor(b!) l) -> (Tensor(a!) f, Tensor(b!) l)");
  ASSERT_TRUE(schema_with_out.arguments().at(1).is_out());
  ASSERT_TRUE(schema_with_out.arguments().at(2).is_out());

  auto schema_without_out =
      parseSchema("at::foo(Tensor self, *, int scalar) -> (int)");

  for (const auto& arg : schema_without_out.arguments()) {
    ASSERT_TRUE(!arg.is_out());
  }

  auto schema_with_is_write = parseSchema(
      "aten::ne_.Scalar(Tensor(a!) self, Scalar other) -> (Tensor(a!))");

  for (const auto& arg : schema_with_is_write.arguments()) {
    ASSERT_TRUE(!arg.is_out());
  }
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(SchemaParserTest, NamedReturns) {
  // named returns
  parseSchema("at::what(Tensor! i_will_be_written_to) -> ()");
  auto s3 =
      parseSchema("at::what() -> (Tensor the_return, Tensor the_return2)");
  ASSERT_TRUE(s3.returns().at(0).name() == "the_return");
  ASSERT_TRUE(s3.returns().at(1).name() == "the_return2");
}

TEST(SchemaParserTest, Futures) {
  // futures
  auto s4 = parseSchema("at::what(Future(int) foo) -> ()");
  ASSERT_TRUE(IntType::get()->isSubtypeOf(
      *s4.arguments().at(0).type()->expectRef<FutureType>().getElementType()));
}

TEST(SchemaParserTest, AnnotatedAliasSets) {
  // test tensor with annotated alias sets
  parseSchema("at::what(Tensor(a) foo) -> (Tensor(a))");
}

TEST(SchemaParserTest, TensorListAnnotatedAliasSets) {
  const auto s = parseSchema(
      "at::foo(Tensor(a!) self, Tensor(b!)[] out)"
      " -> ()");
  const AliasInfo* selfAliasInfo = s.arguments().at(0).alias_info();
  const AliasInfo* outAliasInfo = s.arguments().at(1).alias_info();
  ASSERT_TRUE(
      selfAliasInfo->beforeSets() ==
      std::unordered_set<Symbol>{Symbol::fromQualString("alias::a")});
  ASSERT_TRUE(selfAliasInfo->isWrite());

  ASSERT_TRUE(outAliasInfo->isWrite());
  ASSERT_TRUE(outAliasInfo->beforeSets().empty());
  ASSERT_EQ(outAliasInfo->containedTypes().size(), 1);

  auto containedType = outAliasInfo->containedTypes()[0];

  ASSERT_TRUE(containedType.isWrite());
  ASSERT_TRUE(
      containedType.beforeSets() ==
      std::unordered_set<Symbol>{Symbol::fromQualString("alias::b")});
}

TEST(SchemaParserTest, AnnotatedAliasWithoutBeforeSet) {
  EXPECT_THAT(
      []() { parseSchema("at::foo(Tensor(!) self) -> Tensor"); },
      ::testing::Throws<std::runtime_error>(::testing::Property(
          &std::runtime_error::what,
          ::testing::HasSubstr("expected ident but found '!' here"))));
}

TEST(SchemaParserTest, BeforeAfterSets) {
  const auto s = parseSchema(
      "at::what(Tensor(b|c)[](a!) list, Tensor(c) element)"
      " -> (Tensor(b|c)[](a!))");

  // The list itself is annotated with `a`
  const AliasInfo* aliasInfo = s.arguments().at(0).alias_info();
  ASSERT_NE(aliasInfo, nullptr);
  ASSERT_TRUE(
      aliasInfo->beforeSets() ==
      std::unordered_set<Symbol>{Symbol::fromQualString("alias::a")});
  ASSERT_TRUE(aliasInfo->isWrite());

  // Check the contained types
  ASSERT_TRUE(!aliasInfo->containedTypes().empty());
  const auto& containedAliasInfo = aliasInfo->containedTypes()[0];
  const auto expected = std::unordered_set<Symbol>{
      Symbol::fromQualString("alias::b"),
      Symbol::fromQualString("alias::c"),
  };
  ASSERT_TRUE(containedAliasInfo.beforeSets() == expected);
  ASSERT_TRUE(containedAliasInfo.afterSets() == expected);
  ASSERT_FALSE(containedAliasInfo.isWrite());
}

TEST(SchemaParserTest, BeforeAfterSets2) {
  const auto s = parseSchema(
      "at::what(Tensor(b -> b|c)[](a!) list, Tensor(c) element)"
      " -> (Tensor(b|c)[](a!))");

  // The list itself is annotated with `a`
  const AliasInfo* aliasInfo = s.arguments().at(0).alias_info();
  ASSERT_NE(aliasInfo, nullptr);
  ASSERT_EQ(
      aliasInfo->beforeSets(),
      std::unordered_set<Symbol>{Symbol::fromQualString("alias::a")});
  ASSERT_EQ(
      aliasInfo->afterSets(),
      std::unordered_set<Symbol>{Symbol::fromQualString("alias::a")});
  ASSERT_TRUE(aliasInfo->isWrite());
  ASSERT_EQ(aliasInfo->containedTypes().size(), 1);

  // Check the contained types
  ASSERT_TRUE(!aliasInfo->containedTypes().empty());
  const auto& containedAliasInfo = aliasInfo->containedTypes()[0];
  const auto expectedBefore = std::unordered_set<Symbol>{
      Symbol::fromQualString("alias::b"),
  };
  const auto expectedAfter = std::unordered_set<Symbol>{
      Symbol::fromQualString("alias::b"), Symbol::fromQualString("alias::c")};
  ASSERT_TRUE(containedAliasInfo.beforeSets() == expectedBefore);
  ASSERT_TRUE(containedAliasInfo.afterSets() == expectedAfter);
  ASSERT_FALSE(containedAliasInfo.isWrite());
}

TEST(TopologicalIndexTest, Basic) {
  Graph graph;
  auto node1 = graph.create(prim::AutogradZero);
  auto node2 = graph.create(prim::AutogradZero);
  auto node3 = graph.create(prim::AutogradZero);
  auto node4 = graph.create(prim::AutogradZero);

  graph.appendNode(node4);
  graph.prependNode(node1);
  node2->insertAfter(node1);
  node3->insertBefore(node4);

  // nodes should be in numerical order
  ASSERT_TRUE(node1->isBefore(node2));
  ASSERT_TRUE(node1->isBefore(node3));
  ASSERT_TRUE(node1->isBefore(node4));
  ASSERT_TRUE(node2->isAfter(node1));
  ASSERT_TRUE(node2->isBefore(node3));
  ASSERT_TRUE(node2->isBefore(node4));
  ASSERT_FALSE(node3->isBefore(node1));
  ASSERT_FALSE(node3->isBefore(node2));
  ASSERT_FALSE(node3->isAfter(node4));

  // Built up a block structure
  //  node3
  //   /\        ...
  //  A  B     block1
  //      \      ...
  //      C    block2
  auto block1 = node3->addBlock();
  auto A = graph.create(prim::AutogradZero);
  block1->appendNode(A);
  auto B = graph.create(prim::AutogradZero);
  block1->appendNode(B);
  auto block2 = B->addBlock();
  auto C = graph.create(prim::AutogradZero);
  block2->appendNode(C);

  // Check isAfter on different block levels
  ASSERT_TRUE(node1->isBefore(A));
  ASSERT_TRUE(A->isBefore(B));
  ASSERT_TRUE(A->isBefore(C));

  // make sure things don't blow up on deletions
  node2->destroy();
  auto node2p = graph.create(prim::AutogradZero);
  node2p->insertAfter(node1);
  ASSERT_TRUE(node1->isBefore(node2p));
  ASSERT_TRUE(node2p->isBefore(node3));
}

TEST(TopologicalIndexTest, Reindex) {
  // Induce reindexing to test that path
  Graph graph;
  std::map<size_t, Node*> nodes;

  auto anchor = graph.create(prim::AutogradZero);
  graph.appendNode(anchor);
  // Inserting to the same place a lot will trigger reindexing
  for (auto i = 0; i < 100; ++i) {
    auto n = graph.create(prim::AutogradZero);
    n->insertAfter(anchor);
    nodes[i] = n;
  }

  // Nodes should be in reverse order
  for (auto i = 0; i < 100; ++i) {
    for (auto j = i + 1; j < 100; ++j) {
      ASSERT_TRUE(nodes[i]->isAfter(nodes[j]));
    }
  }
}

at::Tensor invokeTestRecordFunction(at::Tensor& t) {
  RECORD_FUNCTION("test", std::vector<c10::IValue>({t}));

  auto t2 = t.pow(2);
  return t2;
}

static const auto invokeTestRecordFunction_JIT = R"JIT(
  def foo(self, t):
    t2 = t.pow(2)
    return t2

  def forward(self, t):
    return self.foo(t)
)JIT";

at::Tensor invokeTestRecordFunctionJIT(at::Tensor& t) {
  RECORD_FUNCTION("test", std::vector<c10::IValue>({t}));

  auto module = std::make_shared<script::Module>(
      "RecordFunctionTestModule", std::make_shared<script::CompilationUnit>());
  module->define(invokeTestRecordFunction_JIT);
  return module->forward({t}).toTensor();
}

using TracedTestValues =
    std::vector<std::tuple<std::string, std::vector<std::vector<int64_t>>>>;

void checkTracedInputs(const TracedTestValues& inputs) {
  bool found_test = false;
  bool found_pow = false;
  bool found_mul = false;
  for (const auto& input : inputs) {
    const auto& fn = std::get<0>(input);
    const auto& sizes = std::get<1>(input);

    if (fn == "test") {
      found_test = true;
      TORCH_CHECK(sizes.size() == 1);
      TORCH_CHECK(sizes[0] == std::vector<int64_t>({1, 2, 3}));
    } else if (fn == "aten::pow") {
      found_pow = true;
      TORCH_CHECK(sizes.size() == 2);
      TORCH_CHECK(sizes[0] == std::vector<int64_t>({1, 2, 3}));
      TORCH_CHECK(sizes[1].empty());
    } else if (fn == "aten::mul") {
      found_mul = true;
      TORCH_CHECK(sizes.size() > 1);
      TORCH_CHECK(sizes[0] == std::vector<int64_t>({1, 2, 3}));
    }
  }
  TORCH_CHECK(found_test);
  TORCH_CHECK(found_pow);
  TORCH_CHECK(found_mul);
}

void checkTracedOutputs(const TracedTestValues& outputs) {
  bool found_test = false;
  bool found_pow = false;
  bool found_mul = false;
  for (const auto& output : outputs) {
    const auto& fn = std::get<0>(output);
    const auto& sizes = std::get<1>(output);

    if (fn == "test") {
      found_test = true;
      TORCH_CHECK(sizes.empty());
    } else if (fn == "aten::pow") {
      found_pow = true;
      TORCH_CHECK(sizes.size() == 1);
      TORCH_CHECK(sizes[0] == std::vector<int64_t>({1, 2, 3}));
    } else if (fn == "aten::mul") {
      found_mul = true;
      TORCH_CHECK(sizes.size() == 1);
      TORCH_CHECK(sizes[0] == std::vector<int64_t>({1, 2, 3}));
    }
  }
  TORCH_CHECK(found_test);
  TORCH_CHECK(found_pow);
  TORCH_CHECK(found_mul);
}

static bool bad_scope = false;
template <RecordScope scope, size_t* cnt>
std::unique_ptr<at::ObserverContext> checkScopeCallback(
    const at::RecordFunction& fn) {
  if (fn.scope() == scope) {
    ++(*cnt);
  } else {
    bad_scope = true;
  }
  return nullptr;
}

template <RecordScope scope, size_t* cnt>
void pushScopedCallback() {
  at::addGlobalCallback(
      at::RecordFunctionCallback(checkScopeCallback<scope, cnt>)
          .scopes({scope}));
}

// These cannot be function-local because that would prohibit them
// from being used as template arguments prior to C++17.
static size_t fun_cnt;
static size_t ts_fun_cnt;
static size_t user_scope_cnt;

void checkScopeCallbacks() {
  static bool found_function_scope;
  static bool found_method_scope;
  static bool found_user_scope;
  found_function_scope = false;
  found_method_scope = false;
  found_user_scope = false;
  at::addGlobalCallback(at::RecordFunctionCallback(
      [](const at::RecordFunction& fn) -> std::unique_ptr<at::ObserverContext> {
        if (fn.scope() == at::RecordScope::FUNCTION &&
            std::string(fn.name()) == "test_function") {
          found_function_scope = true;
        }
        if (fn.scope() == at::RecordScope::TORCHSCRIPT_FUNCTION &&
            std::string(fn.name()) == "test_method") {
          found_method_scope = true;
        }
        if (fn.scope() == at::RecordScope::USER_SCOPE &&
            std::string(fn.name()) == "test_user_scope") {
          found_user_scope = true;
        }
        return nullptr;
      }));

  bad_scope = false;
  fun_cnt = 0;
  pushScopedCallback<at::RecordScope::FUNCTION, &fun_cnt>();
  ts_fun_cnt = 0;
  pushScopedCallback<at::RecordScope::TORCHSCRIPT_FUNCTION, &ts_fun_cnt>();
  user_scope_cnt = 0;
  pushScopedCallback<at::RecordScope::USER_SCOPE, &user_scope_cnt>();

  TORCH_CHECK(at::hasCallbacks());

  {
    RECORD_TORCHSCRIPT_FUNCTION("test_method", {});
    { RECORD_FUNCTION("test_function", {}); }
    { RECORD_USER_SCOPE("test_user_scope"); }
  }

  TORCH_CHECK(!bad_scope);
  TORCH_CHECK(fun_cnt == 1);
  TORCH_CHECK(ts_fun_cnt == 1);
  TORCH_CHECK(user_scope_cnt == 1);

  TORCH_CHECK(found_function_scope);
  TORCH_CHECK(found_method_scope);
  TORCH_CHECK(found_user_scope);
}

static TracedTestValues traced_inputs;
static TracedTestValues traced_outputs;
static std::unordered_set<std::string> ts_input_names;
static std::unordered_set<std::string> ts_output_names;

std::unique_ptr<at::ObserverContext> tracedInputsCallback(
    const RecordFunction& fn) {
  if (fn.scope() == RecordScope::FUNCTION) {
    auto inputs = fn.inputs();
    std::vector<std::vector<int64_t>> sizes;
    for (const auto& input : inputs) {
      if (input.isTensor()) {
        sizes.push_back(input.toTensor().sizes().vec());
      } else if (input.isScalar()) {
        // NOLINTNEXTLINE(modernize-use-emplace)
        sizes.push_back(std::vector<int64_t>());
      }
    }
    traced_inputs.push_back(std::make_tuple(fn.name(), sizes));
  } else if (fn.scope() == RecordScope::TORCHSCRIPT_FUNCTION) {
    ts_input_names.insert(fn.name());
  }
  return nullptr;
}

void tracedOutputsCallback(const RecordFunction& fn, ObserverContext* ctx_ptr) {
  if (fn.scope() == RecordScope::FUNCTION) {
    auto outputs = fn.outputs();
    std::vector<std::vector<int64_t>> sizes;
    for (const auto& output : outputs) {
      if (output.isTensor()) {
        sizes.push_back(output.toTensor().sizes().vec());
      } else if (output.isScalar()) {
        sizes.emplace_back();
      }
    }
    traced_outputs.push_back(std::make_tuple(fn.name(), sizes));
  } else if (fn.scope() == RecordScope::TORCHSCRIPT_FUNCTION) {
    ts_output_names.insert(fn.name());
  }
}

TEST(RecordFunctionTest, TracedTestInputsOutputs) {
  // disabling the inlining of method calls
  GraphOptimizerEnabledGuard opt_guard(false);

  // [(fn, [[sizes], [sizes], ...]), ...]
  addGlobalCallback(
      RecordFunctionCallback(tracedInputsCallback, tracedOutputsCallback)
          .needsInputs(true)
          .needsOutputs(true));

  TracedTestValues eager_inputs, eager_outputs, jit_inputs, jit_outputs;
  {
    auto t = torch::randn({1, 2, 3}, at::kCPU);
    t.set_requires_grad(true);
    auto t2 = invokeTestRecordFunction(t);
    t2.backward(torch::ones_like(t2, at::MemoryFormat::Preserve));
    eager_inputs = traced_inputs;
    eager_outputs = traced_outputs;
    traced_inputs.clear();
    traced_outputs.clear();

    TORCH_CHECK(ts_input_names.empty());
    TORCH_CHECK(ts_output_names.empty());

    t = torch::randn({1, 2, 3}, at::kCPU);
    t.set_requires_grad(true);
    t2 = invokeTestRecordFunctionJIT(t);
    t2.backward(torch::ones_like(t2, at::MemoryFormat::Preserve));
    jit_inputs = traced_inputs;
    jit_outputs = traced_outputs;
    traced_inputs.clear();
    traced_outputs.clear();
  }

  TORCH_CHECK(ts_input_names.find("forward") != ts_input_names.end());
  TORCH_CHECK(ts_input_names.find("foo") != ts_input_names.end());
  TORCH_CHECK(ts_output_names.find("forward") != ts_output_names.end());
  TORCH_CHECK(ts_output_names.find("foo") != ts_output_names.end());

  checkTracedInputs(eager_inputs);
  checkTracedOutputs(eager_outputs);
  checkTracedInputs(jit_inputs);
  checkTracedOutputs(jit_outputs);
  at::clearCallbacks();
}

static int sampled_cb_ctr = 0;
std::unique_ptr<ObserverContext> sampledCallback(const RecordFunction& fn) {
  if (std::string(fn.name()) == "test") {
    ++sampled_cb_ctr;
  }
  return nullptr;
}

static int non_sampled_cb_ctr = 0;
std::unique_ptr<ObserverContext> nonSampledCallback(const RecordFunction& fn) {
  if (std::string(fn.name()) == "test") {
    ++non_sampled_cb_ctr;
  }
  return nullptr;
}

TEST(RecordFunctionTest, SampledCallbacks) {
  // disabling the inlining of method calls
  GraphOptimizerEnabledGuard opt_guard(false);

  // test sampled callbacks
  sampled_cb_ctr = 0;
  auto setup_sampled_callback = [](double sampling_prob) {
    return addGlobalCallback(
        RecordFunctionCallback(sampledCallback).samplingProb(sampling_prob));
  };

  addGlobalCallback(RecordFunctionCallback(nonSampledCallback));

  auto handle = setup_sampled_callback(0.5);

  auto run_test_function = []() {
    auto t = torch::randn({1, 2, 3}, at::kCPU);
    for (auto k = 0; k < 1000; k++) {
      invokeTestRecordFunction(t);
    }
  };

  run_test_function();
  TORCH_CHECK(non_sampled_cb_ctr == 1000);
  TORCH_CHECK(sampled_cb_ctr > 0 && sampled_cb_ctr < 1000);

  sampled_cb_ctr = 0;
  removeCallback(handle);
  handle = setup_sampled_callback(0.0);
  run_test_function();

  TORCH_CHECK(non_sampled_cb_ctr == 2000);
  TORCH_CHECK(sampled_cb_ctr == 0);

  sampled_cb_ctr = 0;
  removeCallback(handle);
  // NOLINTNEXTLINE(clang-analyzer-deadcode.DeadStores)
  handle = setup_sampled_callback(1.0);
  run_test_function();

  TORCH_CHECK(non_sampled_cb_ctr == 3000);
  TORCH_CHECK(sampled_cb_ctr == 1000);
  clearCallbacks();

  // test the scope of the callbacks
  checkScopeCallbacks();
  clearCallbacks();
}

TEST(RecordFunctionTest, RecordFunctionGuard) {
  // disabling the inlining of method calls
  GraphOptimizerEnabledGuard opt_guard(false);

  static std::vector<std::string> fn_names;
  static std::mutex guard_mtx;

  // check record function guard
  addGlobalCallback(RecordFunctionCallback(
      [](const RecordFunction& fn) -> std::unique_ptr<at::ObserverContext> {
        std::lock_guard<std::mutex> lock(guard_mtx);
        // NOLINTNEXTLINE(modernize-use-emplace)
        fn_names.push_back(fn.name());
        return nullptr;
      }));
  {
    RecordFunctionGuard g1(false);
    {
      RECORD_USER_SCOPE("A");
      {
        RecordFunctionGuard g2(true);
        RECORD_USER_SCOPE("B");
        {
          DisableRecordFunctionGuard g3;
          RECORD_USER_SCOPE("C");
        }
      }
      { RECORD_USER_SCOPE("D"); }
    }
  }
  TORCH_CHECK(fn_names.size() == 1);
  TORCH_CHECK(fn_names[0] == "B");
  clearCallbacks();
}

static std::vector<size_t> ids;

template <size_t id>
auto add_remove_test_add_cb() {
  return addGlobalCallback(RecordFunctionCallback(
      [](const RecordFunction& fn) -> std::unique_ptr<at::ObserverContext> {
        ids.push_back(id);
        return nullptr;
      }));
}

TEST(RecordFunctionTest, Callbacks) {
  // disabling the inlining of method calls
  GraphOptimizerEnabledGuard opt_guard(false);

  auto h1 = add_remove_test_add_cb<1>();
  add_remove_test_add_cb<2>();
  auto h3 = add_remove_test_add_cb<3>();

  { RECORD_USER_SCOPE("test"); }

  TORCH_CHECK(ids.size() == 3);
  TORCH_CHECK(std::find(ids.begin(), ids.end(), 1) != ids.end());
  TORCH_CHECK(std::find(ids.begin(), ids.end(), 2) != ids.end());
  TORCH_CHECK(std::find(ids.begin(), ids.end(), 3) != ids.end());

  ids.clear();
  removeCallback(h1);

  { RECORD_USER_SCOPE("test"); }

  TORCH_CHECK(ids.size() == 2);
  TORCH_CHECK(std::find(ids.begin(), ids.end(), 2) != ids.end());
  TORCH_CHECK(std::find(ids.begin(), ids.end(), 3) != ids.end());

  ids.clear();
  removeCallback(h3);

  { RECORD_USER_SCOPE("test"); }

  TORCH_CHECK(ids.size() == 1);
  TORCH_CHECK(std::find(ids.begin(), ids.end(), 2) != ids.end());

  clearCallbacks();

  // thread local / global callbacks

  ids.clear();
  add_remove_test_add_cb<1>();

  { RECORD_USER_SCOPE("test"); }

  TORCH_CHECK(ids.size() == 1);
  TORCH_CHECK(ids[0] == 1);
  ids.clear();

  auto th = std::thread([]() {
    addThreadLocalCallback(RecordFunctionCallback(
        [](const RecordFunction& fn) -> std::unique_ptr<at::ObserverContext> {
          ids.push_back(2);
          return nullptr;
        }));

    { RECORD_USER_SCOPE("test_thread"); }
  });
  th.join();
  TORCH_CHECK(ids.size() == 2);
  TORCH_CHECK(std::find(ids.begin(), ids.end(), 1) != ids.end());
  TORCH_CHECK(std::find(ids.begin(), ids.end(), 2) != ids.end());
  ids.clear();

  { RECORD_USER_SCOPE("test"); }

  TORCH_CHECK(ids.size() == 1);
  TORCH_CHECK(ids[0] == 1);
  ids.clear();

  clearCallbacks();

  // START: thread local / global context check callbacks
  struct TestContext : public ObserverContext {
    int a{0};
    std::string b;
  };
  ids.clear();
  { // START: global test
    addGlobalCallback(RecordFunctionCallback(
        [](const RecordFunction&
           /* unused */) -> std::unique_ptr<at::ObserverContext> {
          auto ctx = std::make_unique<TestContext>();
          ctx->a = 123;
          ctx->b = "test_str";
          ids.push_back(1);
          return ctx;
        },
        [](const RecordFunction& /* unused */, ObserverContext* ctx_ptr) {
          auto ctx = dynamic_cast<TestContext*>(ctx_ptr);
          TORCH_CHECK(ctx != nullptr);
          TORCH_CHECK(ctx->a == 123);
          TORCH_CHECK(ctx->b == "test_str");
        }));

    { RECORD_USER_SCOPE("test"); }

    TORCH_CHECK(ids.size() == 1);
    TORCH_CHECK(ids[0] == 1);
    ids.clear();
  } // END: global test
  { // START: thread local test
    auto ctx_th = std::thread([]() {
      const std::string test_str = "test thread str";
      addThreadLocalCallback(RecordFunctionCallback(
          [](const RecordFunction&
             /* unused */) -> std::unique_ptr<at::ObserverContext> {
            auto ctx = std::make_unique<TestContext>();
            ctx->a = 234;
            ctx->b = "test_thread_str";
            ids.push_back(2);
            return ctx;
          },
          [](const RecordFunction& /* unused */, ObserverContext* ctx_ptr) {
            auto ctx = dynamic_cast<TestContext*>(ctx_ptr);
            TORCH_CHECK(ctx_ptr != nullptr);
            TORCH_CHECK(ctx->a == 234);
            TORCH_CHECK(ctx->b == "test_thread_str");
          }));

      // Will call both global and thread local callbacks.
      { RECORD_USER_SCOPE("test_thread"); }
    });
    ctx_th.join();
    TORCH_CHECK(ids.size() == 2);
    TORCH_CHECK(std::find(ids.begin(), ids.end(), 1) != ids.end());
    TORCH_CHECK(std::find(ids.begin(), ids.end(), 2) != ids.end());
    ids.clear();
  } // END: thread local test

  clearCallbacks();
}

TEST(RecordFunctionTest, ShouldRun) {
  // disabling the inlining of method calls
  GraphOptimizerEnabledGuard opt_guard(false);

  static bool ran = false;
  auto handle = addGlobalCallback(RecordFunctionCallback(
      [](const RecordFunction& fn) -> std::unique_ptr<at::ObserverContext> {
        ran = true;
        return nullptr;
      }));

  { RECORD_USER_SCOPE("test"); }

  EXPECT_TRUE(ran) << "first run didn't happen";
  ran = false;

  disableCallback(handle);

  { RECORD_USER_SCOPE("test"); }

  EXPECT_FALSE(ran) << "second run happened but shouldn't have";
  ran = false;

  reenableCallback(handle);

  { RECORD_USER_SCOPE("test"); }

  EXPECT_TRUE(ran) << "run after re-enable didn't happen";
  ran = false;

  clearCallbacks();
}

TEST(RecordFunctionTest, Basic) {
  // disabling the inlining of method calls
  GraphOptimizerEnabledGuard opt_guard(false);

  static std::string recorded_op;
  static bool has_ids = false;

  // test propagation of TLS callbacks
  std::thread t([]() {
    RecordFunctionGuard enable_rec_fn;
    auto handle = addThreadLocalCallback(RecordFunctionCallback(
        [](const RecordFunction& fn) -> std::unique_ptr<at::ObserverContext> {
          recorded_op = fn.name();
          return nullptr;
        }));
    ThreadLocalState state;
    std::thread t_child([state]() {
      ThreadLocalStateGuard g_tls(state);
      RECORD_USER_SCOPE("test_in_thread");
    });
    t_child.join();
    EXPECT_EQ(recorded_op, "test_in_thread");
    removeCallback(handle);
  });
  t.join();
  clearCallbacks();

  // test set ids
  addGlobalCallback(
      RecordFunctionCallback(
          [](const RecordFunction& fn) -> std::unique_ptr<at::ObserverContext> {
            has_ids = fn.handle() > 0;
            return nullptr;
          })
          .needsIds(true));
  { RECORD_USER_SCOPE("test"); }
  TORCH_CHECK(has_ids);
  clearCallbacks();
  has_ids = false;
  addGlobalCallback(RecordFunctionCallback(
      [](const RecordFunction& fn) -> std::unique_ptr<at::ObserverContext> {
        has_ids = fn.handle() > 0;
        return nullptr;
      }));
  { RECORD_USER_SCOPE("test"); }
  TORCH_CHECK(!has_ids);
  clearCallbacks();
}

TEST(RecordFunctionTest, OperatorNameOverload) {
  static std::set<std::string> operator_names;
  at::addGlobalCallback(at::RecordFunctionCallback(
                            [](const at::RecordFunction& fn)
                                -> std::unique_ptr<at::ObserverContext> {
                              std::optional<c10::OperatorName> op_name =
                                  fn.operator_name();
                              if (op_name.has_value()) {
                                operator_names.insert(c10::toString(*op_name));
                              } else {
                                operator_names.insert("No Operator Name");
                              }
                              return nullptr;
                            })
                            .scopes({at::RecordScope::FUNCTION}));
  auto t = torch::randn({1, 2, 3}, at::kCPU);
  t.set_requires_grad(false);
  auto t2 = t.pow(2);

  at::clearCallbacks();
  EXPECT_TRUE(operator_names.count("No Operator Name") == 0)
      << "Expected that all traced operators had an associated OperatorName object";
  EXPECT_TRUE(operator_names.count("aten::randn") == 1)
      << "Expected aten::randn to have been called and recorded, but it was not";
  EXPECT_TRUE(operator_names.count("aten::pow.Tensor_Scalar") == 1)
      << "Expected aten::pow.Tensor_Scalar to have been called and recorded, but it was not";
}

class TestThreadLocalDebugInfo : public c10::DebugInfoBase {
 public:
  int getModelId() const {
    return model_id_;
  }

  void setModelId(int model_id) {
    model_id_ = model_id;
  }

  // NOLINTNEXTLINE(modernize-use-equals-default)
  virtual ~TestThreadLocalDebugInfo() override {}

 private:
  int model_id_ = 0;
};

void checkDebugInfo(c10::DebugInfoKind kind, int model_id) {
  auto* debug_info = c10::ThreadLocalDebugInfo::get(kind);
  TORCH_CHECK(debug_info != nullptr);
  auto* test_debug_info = dynamic_cast<TestThreadLocalDebugInfo*>(debug_info);
  TORCH_CHECK(test_debug_info != nullptr);
  TORCH_CHECK(test_debug_info->getModelId() == model_id);
}

TEST(ThreadLocalDebugInfoTest, Basic) {
  static std::atomic<bool> done{false};

  TORCH_CHECK(
      c10::ThreadLocalDebugInfo::get(c10::DebugInfoKind::TEST_INFO) == nullptr);
  auto debug_info = std::make_shared<TestThreadLocalDebugInfo>();
  debug_info->setModelId(42);
  {
    c10::DebugInfoGuard guard(c10::DebugInfoKind::TEST_INFO, debug_info);
    checkDebugInfo(c10::DebugInfoKind::TEST_INFO, 42);
  }

  // check that thread local debug info is propagated through fork calls
  TORCH_CHECK(
      c10::ThreadLocalDebugInfo::get(c10::DebugInfoKind::TEST_INFO) == nullptr);
  {
    c10::DebugInfoGuard guard(c10::DebugInfoKind::TEST_INFO, debug_info);
    at::launch([]() {
      checkDebugInfo(c10::DebugInfoKind::TEST_INFO, 42);
      done = true;
    });
  }
  while (!done) {
  }

  // check that thread local debug info is propagated through backward pass
  TORCH_CHECK(
      c10::ThreadLocalDebugInfo::get(c10::DebugInfoKind::TEST_INFO) == nullptr);
  done = false;
  auto handle = addGlobalCallback(RecordFunctionCallback(
      [](const RecordFunction&) -> std::unique_ptr<at::ObserverContext> {
        checkDebugInfo(c10::DebugInfoKind::TEST_INFO, 42);
        done = true;
        return nullptr;
      }));
  {
    c10::DebugInfoGuard guard(c10::DebugInfoKind::TEST_INFO, debug_info);
    auto t = torch::randn({1, 2, 3}, at::kCPU);
    t.set_requires_grad(true);
    auto t2 = t.pow(2);
    t2.backward(torch::ones_like(t2, at::MemoryFormat::Preserve));
  }
  removeCallback(handle);
  TORCH_CHECK(done);

  // check nested debug info
  TORCH_CHECK(
      c10::ThreadLocalDebugInfo::get(c10::DebugInfoKind::TEST_INFO) == nullptr);
  {
    c10::DebugInfoGuard guard(c10::DebugInfoKind::TEST_INFO, debug_info);
    {
      checkDebugInfo(c10::DebugInfoKind::TEST_INFO, 42);
      {
        auto debug_info = std::make_shared<TestThreadLocalDebugInfo>();
        debug_info->setModelId(314);
        c10::DebugInfoGuard guard(c10::DebugInfoKind::TEST_INFO_2, debug_info);
        {
          checkDebugInfo(c10::DebugInfoKind::TEST_INFO, 42);
          checkDebugInfo(c10::DebugInfoKind::TEST_INFO_2, 314);
          done = false;
          at::launch([]() {
            checkDebugInfo(c10::DebugInfoKind::TEST_INFO, 42);
            checkDebugInfo(c10::DebugInfoKind::TEST_INFO_2, 314);
            done = true;
          });
          while (!done) {
          }
        }
      }
    }
  }
}

TEST(TestSymIntArrayRef, BasicConversion) {
  const size_t X = 2, Y = 4, Z = 5;
  std::vector<int64_t> tgt_size_v{2, 4, 5};
  std::vector<c10::SymInt> tgt_size({SymInt(X), SymInt(Y), SymInt(Z)});
  auto a = at::randn({1, 4, 1}, at::kCPU);
  auto b = a.expand_symint(tgt_size);
  auto c = a.expand(tgt_size_v);
  ASSERT_TRUE(torch::allclose(b, c));
}

TEST(TestSymInt, NarrowCopyWithSymbolicInt) {
  static const size_t LENGTH = 5;
  auto a = at::randn({10}, at::kCPU);
  c10::SymInt si(LENGTH);
  auto b = a.narrow_copy_symint(0, 0, si);
  auto c = a.narrow(0, 0, LENGTH);
  ASSERT_TRUE(torch::allclose(b, c));
}

TEST(TestSymInt, NarrowCopy) {
  static const size_t LENGTH = 5;
  auto a = at::randn({10}, at::kCPU);
  auto b = a.narrow_copy(0, 0, LENGTH);
  auto c = a.narrow(0, 0, LENGTH);
  ASSERT_TRUE(torch::allclose(b, c));
}

TEST(TestSymInt, AddSymbolicInt) {
  c10::SymInt a(5);
  c10::SymInt b(3);
  ASSERT_TRUE((a + b).expect_int() == 8);
}

TEST(FallbackGraphsTest, Basic) {
  auto x = at::randn({1}, at::kCPU);
  auto y = at::randn({1}, at::kCPU);
  auto stack = createStack({x.clone(), y.clone()});

  auto graph_string = R"IR(
    graph(%0 : Float(1),
          %1 : Float(1)):
      %2 : Tensor = aten::mul(%0, %1)
      %3 : Tensor = aten::mul(%2, %0)
      return (%3))IR";
  auto graph = std::make_shared<Graph>();
  torch::jit::parseIR(graph_string, graph.get());

  {
    Code code(graph, "");
    InterpreterState interpreter{code};
    interpreter.run(stack);
  }
  at::Tensor et;
  pop(stack, et);
  float ef = et.item<float>();
  {
    EnableProfilingGuard epg;
    GraphFunction f("fallbackGraphs", graph, nullptr);
    for (size_t i = 0; i < getNumProfiledRuns() + 1; i++) {
      stack.emplace_back(x.clone());
      stack.emplace_back(y.clone());
      if (i == getNumProfiledRuns()) {
        // we will be modifying a profiled graph
        // before ProfilingGraphExecutor
        // will optimize it in the next iteration
        auto opt_graph = lastExecutedOptimizedGraph();
        // this is safe to do since we are done profiling
        ProfilingRecord::removeProfileCounter(opt_graph->block());
        replaceBlockWithFallbackGraph(opt_graph->block(), opt_graph->inputs());
        auto it = opt_graph->block()->nodes().begin();
        ASSERT_EQ(it->kind(), prim::FallbackGraph);
        auto fallback = *it++;
        ASSERT_EQ(it, opt_graph->block()->nodes().end());
        ASSERT_TRUE(fallback->hasAttribute(attr::Subgraph));
        testing::FileCheck()
            .check("Tensor = aten::mul")
            ->check("Tensor = aten::mul")
            ->run(*fallback->g(attr::Subgraph));
      }
      f.run(stack);
      at::Tensor at;
      pop(stack, at);
      float af = at.item<float>();
      ASSERT_EQ(af, ef);
    }

    auto opt_graph = lastExecutedOptimizedGraph();
    testing::FileCheck()
        .check("(Tensor) = prim::CallFunction")
        ->run(*opt_graph);
  }
}

// TODO this test wasn't running and is broken.
// TEST(AutogradProfilerTest, Basic) {
//   constexpr int batch_size = 4;
//   constexpr int input_size = 256;
//   constexpr int seq_len = 32;

//   int hidden_size = 2 * input_size;
//   auto input = torch::randn({seq_len, batch_size, input_size}, at::kCPU);
//   auto hx = torch::randn({batch_size, hidden_size}, at::kCPU);
//   auto cx = torch::randn({batch_size, hidden_size}, at::kCPU);
//   auto w_ih = t_def(torch::randn({4 * hidden_size, input_size}, at::kCPU));
//   auto w_hh = t_def(torch::randn({4 * hidden_size, hidden_size}, at::kCPU));

//   std::stringstream ss;
//   {
//     RecordProfile guard(ss);
//     for (size_t i = 0; i < 100; ++i) {
//       std::tie(hx, cx) = lstm(input[0], hx, cx, w_ih, w_hh);
//     }
//   }

//   std::string result = ss.str();
//   size_t count = 0;
//   for (size_t pos = 0; (pos = result.find("tanh", pos)) != std::string::npos;
//        count++, pos++) {
//   }
//   ASSERT_EQ((count, 200);
// }

TEST(NoneSchemaMatchTest, Basic) {
  RegisterOperators reg({
      Operator(
          "prim::test_none() -> int?",
          [](Stack& stack) { push(stack, IValue()); },
          aliasAnalysisFromSchema()),
      Operator(
          "prim::is_none(int? a) -> bool",
          [](Stack& stack) {
            IValue a = pop(stack);
            if (a.isNone()) {
              push(stack, true);
            } else {
              push(stack, false);
            }
          },
          aliasAnalysisFromSchema()),
  });

  // Constant propagation will run test_none and produce a None,
  // testing that its type is set appropriately and schema matching  doesn't
  // fail when running is_none

  auto r = std::make_shared<Graph>();
  auto& g = *r;
  auto opt_int = g.insert(Symbol::fromQualString("prim::test_none"), {});
  auto out_bool = g.insert(Symbol::fromQualString("prim::is_none"), {opt_int});
  g.registerOutput(out_bool);
  ConstantPropagation(r);

  auto nodes = r->block()->nodes();
  // checking that constant propagation ran wo/failure
  AT_ASSERT(std::distance(nodes.begin(), nodes.end()) == 1);
}

static int testPassValue = 0;
void fakePass(std::shared_ptr<Graph>& g) {
  testPassValue++;
  return;
}

RegisterPass p(fakePass);

TEST(PassManagementTest, Basic) {
  std::shared_ptr<Graph> graph = std::make_shared<Graph>();
  parseIR(
      R"IR(
graph(%a):
  return (%a))IR",
      &*graph);

  std::vector<IValue> stack = {IValue(torch::randn({22}, at::kCPU))};
  auto run = [&](std::shared_ptr<Graph>& graph, std::vector<IValue> stack) {
    GraphExecutor executor(graph, "");
    executor.run(stack);
    return stack;
  };
  run(graph, stack);
  // we will not run fusion in simple mode
  if (!getExecutorMode()) {
    AT_ASSERT(testPassValue);
  }
}

static void checkShape(TypePtr typ, std::vector<int64_t> expected) {
  auto ptp = typ->expect<TensorType>();
  ASSERT_EQ(ptp->sizes().concrete_sizes().value(), expected);
}

static void checkShape(
    Node* n,
    std::vector<int64_t> expected,
    bool prev = true) {
  auto profile = (prev) ? n->inputs().at(0)->node() : n;
  checkShape(profile->output()->type(), expected);
}

void count_(
    Block* block,
    const std::function<bool(Node* n)>& pred,
    size_t& count) {
  for (Node* n : block->nodes()) {
    if (pred(n)) {
      count++;
    }

    for (Block* ib : n->blocks()) {
      count_(ib, pred, count);
    }
  }
}

size_t countNodes(
    const std::shared_ptr<Graph>& graph,
    const std::function<bool(Node* n)>& pred) {
  size_t count = 0;
  count_(graph->block(), pred, count);
  return count;
}

bool true_pred(Node* n) {
  return true;
};

bool is_loop(Node* n) {
  return n->kind() == prim::Loop;
};

TEST(LoopPeelerTest, NoInductionVariableUse) {
  // do not use an induction variable explicitly
  static const auto str_func_def = R"JIT(
    def test_peel_n_times():
      sum = 0
      for i in range(10):
        sum += 2
      return sum
    )JIT";

  auto cu = compile(str_func_def);
  auto& f = toGraphFunction(cu->get_function("test_peel_n_times"));
  auto stack = createStack({});
  // peeling loop once
  {
    LoopsPeeler peeler(true_pred, 1);
    auto copy = f.graph()->copy();
    peeler.run(copy);
    int num_loops =
        std::count_if(copy->nodes().begin(), copy->nodes().end(), is_loop);
    ASSERT_EQ(num_loops, 2);
    Code code(copy, "");
    InterpreterState interpreter{code};
    interpreter.run(stack);
    ASSERT_EQ(stack.back().toInt(), 20);
  }

  // test peeling more than one iteration
  {
    LoopsPeeler peeler(true_pred, 3);
    auto copy = f.graph()->copy();
    peeler.run(copy);
    int num_loops =
        std::count_if(copy->nodes().begin(), copy->nodes().end(), is_loop);
    ASSERT_EQ(num_loops, 2);
    Code code(copy, "");
    InterpreterState interpreter{code};
    interpreter.run(stack);
    ASSERT_EQ(stack.back().toInt(), 20);
  }
}

TEST(LoopPeelerTest, YesInductionVariableUse) {
  // uses the induction variable
  static const auto str_func_def = R"JIT(
    def test_peel_n_times():
      sum = 0
      for i in range(10):
        sum += i
      return sum
    )JIT";

  auto cu = compile(str_func_def);
  auto& f = toGraphFunction(cu->get_function("test_peel_n_times"));
  auto stack = createStack({});
  // peeling loop once
  {
    LoopsPeeler peeler(true_pred, 1);
    auto copy = f.graph()->copy();
    peeler.run(copy);
    int num_loops =
        std::count_if(copy->nodes().begin(), copy->nodes().end(), is_loop);
    ASSERT_EQ(num_loops, 2);
    Code code(copy, "");
    InterpreterState interpreter{code};
    interpreter.run(stack);
    ASSERT_EQ(stack.back().toInt(), 45);
  }

  // test peeling more than one iteration
  {
    LoopsPeeler peeler(true_pred, 3);
    auto copy = f.graph()->copy();
    peeler.run(copy);
    int num_loops =
        std::count_if(copy->nodes().begin(), copy->nodes().end(), is_loop);
    ASSERT_EQ(num_loops, 2);
    Code code(copy, "");
    InterpreterState interpreter{code};
    interpreter.run(stack);
    ASSERT_EQ(stack.back().toInt(), 45);
  }
}

TEST(LoopPeelerTest, LoopWithTerminationCondition) {
  // tests with explicit termination conditions
  static const auto str_func_def = R"JIT(
    def test_with_cond_times():
      sum = 0
      i = 0
      while (sum < 2):
        sum += i
        i += 1
      return sum
    )JIT";

  // the peel changes the termination condition to false
  // so the original loop doesn't run
  auto cu = compile(str_func_def);
  auto& f = toGraphFunction(cu->get_function("test_with_cond_times"));
  auto stack = createStack({});
  // peeling 5 iterations should update the termination
  // condition to false
  {
    LoopsPeeler peeler(true_pred, 5);
    auto copy = f.graph()->copy();
    peeler.run(copy);
    int num_loops =
        std::count_if(copy->nodes().begin(), copy->nodes().end(), is_loop);
    ASSERT_EQ(num_loops, 2);
    Code code(copy, "");
    InterpreterState interpreter{code};
    interpreter.run(stack);
    ASSERT_EQ(stack.back().toInt(), 3);
  }

  // the termination condition remains true
  {
    LoopsPeeler peeler(true_pred, 1);
    auto copy = f.graph()->copy();
    peeler.run(copy);
    int num_loops =
        std::count_if(copy->nodes().begin(), copy->nodes().end(), is_loop);
    ASSERT_EQ(num_loops, 2);
    Code code(copy, "");
    InterpreterState interpreter{code};
    interpreter.run(stack);
    ASSERT_EQ(stack.back().toInt(), 3);
  }
}

// tests simple nested loops
TEST(LoopPeelerTest, SimpleNestedLoops) {
  static const auto str_func_def = R"JIT(
    def test_nested_loops():
      sum = 0
      i = 0
      for i in range(10):
        for j in range(10):
          sum += i + j
      return sum
    )JIT";

  auto cu = compile(str_func_def);
  auto& f = toGraphFunction(cu->get_function("test_nested_loops"));
  auto stack = createStack({});

  {
    LoopsPeeler peeler(true_pred, 1);
    auto copy = f.graph()->copy();
    peeler.run(copy);
    ASSERT_EQ(countNodes(copy, is_loop), 5);
    Code code(copy, "");
    InterpreterState interpreter{code};
    interpreter.run(stack);
    ASSERT_EQ(stack.back().toInt(), 900);
  }

  {
    LoopsPeeler peeler(true_pred, 5);
    auto copy = f.graph()->copy();
    peeler.run(copy);
    ASSERT_EQ(countNodes(copy, is_loop), 5);
    Code code(copy, "");
    InterpreterState interpreter{code};
    interpreter.run(stack);
    ASSERT_EQ(stack.back().toInt(), 900);
  }
}

TEST(LoopPeelerTest, SimpleNestedLoops2) {
  static const auto str_func_def = R"JIT(
    def test_nested_loops():
      sum = 0
      i = 0
      for i in range(10):
        j = 0
        while sum < 2:
          sum += i + j
          j += 1
      return sum
    )JIT";

  auto cu = compile(str_func_def);
  auto& f = toGraphFunction(cu->get_function("test_nested_loops"));
  auto stack = createStack({});
  {
    LoopsPeeler peeler(true_pred, 1);
    auto copy = f.graph()->copy();
    peeler.run(copy);
    ASSERT_EQ(countNodes(copy, is_loop), 5);
    Code code(copy, "");
    InterpreterState interpreter{code};
    interpreter.run(stack);
    ASSERT_EQ(stack.back().toInt(), 3);
  }

  {
    LoopsPeeler peeler(true_pred, 5);
    auto copy = f.graph()->copy();
    peeler.run(copy);
    ASSERT_EQ(countNodes(copy, is_loop), 5);
    Code code(copy, "");
    InterpreterState interpreter{code};
    interpreter.run(stack);
    ASSERT_EQ(stack.back().toInt(), 3);
  }
}

TEST(JitTracing, Basic) {
  constexpr int batch_size = 4;
  constexpr int input_size = 256;

  int hidden_size = 2 * input_size;

  auto input = at::randn({batch_size, input_size}, at::kCPU);
  auto hx = at::randn({batch_size, hidden_size}, at::kCPU);
  auto cx = at::randn({batch_size, hidden_size}, at::kCPU);
  auto w_ih = t_def(at::randn({4 * hidden_size, input_size}, at::kCPU));
  auto w_hh = t_def(at::randn({4 * hidden_size, hidden_size}, at::kCPU));

  auto graph = build_lstm();
  auto stack = createStack({input, hx, cx, w_ih, w_hh});
  auto traced = TraceGraph(graph, stack);

  // Check that the inputs of traced graph have the same type as the inputs
  // specified here.
  ASSERT_EQ(*traced->inputs().at(0)->type(), *TensorType::create(input));
  ASSERT_EQ(*traced->inputs().at(1)->type(), *TensorType::create(hx));
  ASSERT_EQ(*traced->inputs().at(2)->type(), *TensorType::create(cx));
  ASSERT_EQ(*traced->inputs().at(3)->type(), *TensorType::create(w_ih));
  ASSERT_EQ(*traced->inputs().at(4)->type(), *TensorType::create(w_hh));

  Tensor prof_out;
  pop(stack, prof_out);

  {
    stack = createStack({input, hx, cx, w_ih, w_hh});
    Code cd(traced, "traced");
    InterpreterState is{cd};
    is.run(stack);
    Tensor traced_out;
    pop(stack, traced_out);
    torch::allclose(prof_out, traced_out);
  }

  {
    stack = createStack({input, hx, cx, w_ih, w_hh});
    Code cd(graph, "graph");
    InterpreterState is{cd};
    is.run(stack);
    Tensor scripted_out;
    pop(stack, scripted_out);
    torch::allclose(prof_out, scripted_out);
  }
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(InsertAndEliminateRedundantGuardsTest, Basic) {
  static const auto basic_example = R"JIT(
  def basic(x, y):
    a = x + y
    b = x * y
    c = x + 1
    d = a - c
    e = b - c
    return d + e
  )JIT";

  auto cu = compile(basic_example);
  auto& fun = toGraphFunction(cu->get_function("basic"));
  auto pr = ProfilingRecord::instrumentGraph(fun.graph());
  auto x = at::randn({2, 3}, at::kCPU);
  auto y = at::randn({2, 3}, at::kCPU);
  auto stack = createStack({x, y});
  // introduce some profiling information
  Code cd(pr->profiled_graph_, "");
  InterpreterState is{cd};
  is.run(stack);
  auto copy = pr->profiled_graph_->copy();
  ProfilingRecord::removeProfileCounter(copy->block());
  InsertGuards(copy);
  auto nodes = copy->block()->nodes();
  auto guard = std::find_if(nodes.begin(), nodes.end(), [](Node* n) {
    return n->kind() == prim::Guard;
  });
  ASSERT_NE(guard, nodes.end());
  ASSERT_EQ(
      guard->input()->type()->expectRef<TensorType>().sizes().size(),
      c10::nullopt);
  checkShape(*guard, {2, 3}, false);
  auto is_guard = [](Node* n) { return n->kind() == prim::Guard; };
  int num_guards = std::count_if(nodes.begin(), nodes.end(), is_guard);
  ASSERT_EQ(num_guards, 12);
  // now eliminate as many guards as possible
  // we should be left with two guards on x and y's defs
  EliminateRedundantGuards(copy);
  num_guards = std::count_if(nodes.begin(), nodes.end(), is_guard);
  ASSERT_EQ(num_guards, 2);
}

TEST(InsertBailOutsTest, Basic) {
  static const auto basic_example = R"JIT(
  def basic_loop(x, y):

      a = x + 1
      b = y + 2
      c = x + y + 3

      for i in range(10):
          a = a + b
          # invariant
          d = b * c
          #
          a = a - d

      e = a + 4
      return e
  )JIT";

  auto cu = compile(basic_example);
  auto& fun = toGraphFunction(cu->get_function("basic_loop"));
  auto pr = ProfilingRecord::instrumentGraph(fun.graph());
  auto x = at::randn({2, 3}, at::kCPU);
  auto y = at::randn({2, 3}, at::kCPU);
  auto stack = createStack({x, y});
  // introduce some profiling information
  Code cd(pr->profiled_graph_, "");
  InterpreterState is{cd};
  is.run(stack);
  auto copy = pr->profiled_graph_->copy();
  ProfilingRecord::removeProfileCounter(copy->block());
  InsertGuards(copy);
  EliminateRedundantGuards(copy);
  auto nodes = copy->block()->nodes();
  auto is_guard = [](Node* n) { return n->kind() == prim::Guard; };
  auto num_guards = std::count_if(nodes.begin(), nodes.end(), is_guard);
  ASSERT_EQ(num_guards, 3);
  InsertBailOuts(copy);
  auto is_bailout = [](Node* n) { return n->kind() == prim::BailOut; };
  auto num_bailouts = std::count_if(nodes.begin(), nodes.end(), is_bailout);
  ASSERT_EQ(num_guards, num_bailouts);
  std::vector<Node*> bailouts(num_bailouts);
  std::copy_if(nodes.begin(), nodes.end(), bailouts.begin(), is_bailout);

  for (auto blo : bailouts) {
    ASSERT_EQ(blo->inputs().at(0)->node()->kind(), prim::BailoutTemplate);
  }
}

TEST(ProfilerTest, Basic) {
  constexpr int batch_size = 4;
  constexpr int input_size = 256;

  int hidden_size = 2 * input_size;

  auto input = at::randn({batch_size, input_size}, at::kCPU);
  auto hx = at::randn({batch_size, hidden_size}, at::kCPU);
  auto cx = at::randn({batch_size, hidden_size}, at::kCPU);
  auto w_ih = t_def(at::randn({4 * hidden_size, input_size}, at::kCPU));
  auto w_hh = t_def(at::randn({4 * hidden_size, hidden_size}, at::kCPU));

  auto g = build_lstm();
  auto stack = createStack({input, hx, cx, w_ih, w_hh});

  auto& opt_graph = *g.get();
  ArgumentSpecCreator arg_spec_creator(opt_graph);
  ArgumentSpec spec =
      arg_spec_creator.create(autograd::GradMode::is_enabled(), stack);
  arg_spec_creator.specializeTypes(opt_graph, spec);
  auto pr = ProfilingRecord::instrumentGraph(g);
  Code cd(pr->profiled_graph_, "");
  InterpreterState is{cd};
  is.run(stack);

  // profiled types are stored as attributes and show up in the dump, e.g.
  // Tensor = prim::profile[profiled_type=Double(4, 256, strides=[256, 1],
  // requires_grad=0, device=cpu)
  testing::FileCheck()
      .check("Tensor = prim::profile[profiled_type")
      ->check_same("256")
      ->run(*pr->profiled_graph_);

  auto begin = pr->profiled_graph_->block()->nodes().begin();
  auto end = pr->profiled_graph_->block()->nodes().end();
  auto mm =
      std::find_if(begin, end, [](Node* n) { return n->kind() == aten::add; });
  ASSERT_NE(mm, end);
  std::vector<int64_t> mm_expected{4, 2048};
  std::vector<int64_t> eltwise{4, 512};
  checkShape(mm->inputs().at(0)->node()->ty(attr::profiled_type), mm_expected);
  auto mul_n =
      std::find_if(begin, end, [](Node* n) { return n->kind() == aten::mul; });
  ASSERT_NE(mul_n, end);
  checkShape(mul_n->inputs().at(0)->node()->ty(attr::profiled_type), eltwise);
  auto tanh_n =
      std::find_if(begin, end, [](Node* n) { return n->kind() == aten::tanh; });
  checkShape(tanh_n->inputs().at(0)->node()->ty(attr::profiled_type), eltwise);
}

TEST(ProfilerTest, OptionalProfiling) {
  auto graph = std::make_shared<Graph>();
  std::unordered_map<std::string, Value*> vmap;
  parseIR(
      R"IR(
graph(%inp : Tensor,
      %weight : Tensor,
      %bias : Tensor?):
  %1 : Tensor = aten::linear(%inp, %weight, %bias)
  return (%1))IR",
      &*graph,
      vmap);

  auto pr = ProfilingRecord::instrumentGraph(graph);
  pr->profiling_count_ = 2;

  auto input = torch::randn({1, 2});
  auto weight = torch::randn({2, 2});
  auto bias = torch::randn({1, 2});

  auto stack = createStack({input, weight, bias});
  Code cd(pr->profiled_graph_, "");
  InterpreterState is{cd};
  is.run(stack);

  testing::FileCheck()
      .check_count("Tensor? = prim::profile[profiled_type", 1, true)
      ->run(*pr->profiled_graph_);

  // make sure we recorded the shape
  auto begin = pr->profiled_graph_->block()->nodes().begin();
  auto end = pr->profiled_graph_->block()->nodes().end();
  auto linear = std::find_if(
      begin, end, [](Node* n) { return n->kind() == aten::linear; });
  ASSERT_NE(linear, end);
  std::vector<int64_t> bias_expected_shape = {1, 2};
  auto profiled_bias = linear->namedInput("bias")->node();
  checkShape(profiled_bias->ty(attr::profiled_type), bias_expected_shape);
  ASSERT_EQ(0, profiled_bias->i(attr::seen_none));

  auto none_bias = c10::IValue();

  stack.clear();
  stack.emplace_back(input);
  stack.emplace_back(weight);
  stack.emplace_back(none_bias);
  is = InterpreterState{cd};
  is.run(stack);

  // make sure we recorded that "None" was seen.
  begin = pr->profiled_graph_->block()->nodes().begin();
  end = pr->profiled_graph_->block()->nodes().end();
  linear = std::find_if(
      begin, end, [](Node* n) { return n->kind() == aten::linear; });
  ASSERT_NE(linear, end);
  profiled_bias = linear->namedInput("bias")->node();
  checkShape(profiled_bias->ty(attr::profiled_type), bias_expected_shape);
  ASSERT_EQ(1, profiled_bias->i(attr::seen_none));
}

TEST(CallStackTest, Basic) {
  const auto text = R"(
def ham(x):
    return x/7

def bar(x):
    return x*3

def baz(x):
    return ham(x)*x

def foo(x):
    return bar(x)*baz(x)*11
  )";
  auto cu = compile(text);
  const auto& foo = toGraphFunction(cu->get_function("foo"));
  for (Node* n : foo.optimized_graph()->nodes()) {
    if (n->kind() == prim::Constant) {
      if (!n->hasAttribute(attr::value) ||
          n->kindOf(attr::value) != AttributeKind::i) {
        continue;
      }
      int v = n->i(attr::value);
      switch (v) {
        case 3: {
          // Const 3 comes from function 'bar', which gets inlined to 'foo'.
          // The callstack for the corresponding node should contain only the
          // function 'bar'.
          ASSERT_TRUE(n->callstack());
          auto callstack_vector = (*n->callstack())->vec();
          ASSERT_EQ(callstack_vector.size(), 1);
          ASSERT_EQ(std::get<0>(callstack_vector[0]), &cu->get_function("bar"));
          break;
        }
        case 7: {
          // Const 7 comes from function 'ham', which gets inlined to 'baz',
          // which is then inlined to 'foo'. The callstack for the corresponding
          // node should contain these two functions.
          ASSERT_TRUE(n->callstack());
          auto callstack_vector = (*n->callstack())->vec();
          ASSERT_EQ(callstack_vector.size(), 2);
          ASSERT_EQ(std::get<0>(callstack_vector[0]), &cu->get_function("baz"));
          ASSERT_EQ(std::get<0>(callstack_vector[1]), &cu->get_function("ham"));
          break;
        }
        case 11: {
          // Const 11 comes from function 'foo', which is not inlined anywhere
          // and thus it should not have a callstack.
          ASSERT_FALSE(n->callstack());
          break;
        }
      }
    }
  }

  // Check that inlining doesn't corrupt callstack of the callee's nodes.
  const auto& baz = toGraphFunction(cu->get_function("baz"));
  for (Node* n : baz.optimized_graph()->nodes()) {
    if (n->kind() == prim::Constant) {
      if (!n->hasAttribute(attr::value) ||
          n->kindOf(attr::value) != AttributeKind::i) {
        continue;
      }
      int v = n->i(attr::value);
      ASSERT_TRUE(v == 7);
      // Const 7 comes from function 'ham', which gets inlined to 'baz'. 'baz'
      // was also inlined into 'foo', but when looking at the graph of 'baz' we
      // should only see a callstack of depth 1 (containing only 'ham').
      ASSERT_TRUE(n->callstack());
      auto callstack_vector = (*n->callstack())->vec();
      ASSERT_EQ(callstack_vector.size(), 1);
      ASSERT_EQ(std::get<0>(callstack_vector[0]), &cu->get_function("ham"));
    }
  }
}

TEST(CallStackTest, Caching) {
  const auto text = R"(

def a(x):
    print("a1")
    print("a2")
    return x

def b(x):
    print("b1")
    print("b2")
    a(x)
    return x

def c(x):
    print("c1")
    print("c2")
    b(x)
    return x
  )";
  auto cu = compile(text);
  const auto& baz = toGraphFunction(cu->get_function("c"));
  std::unordered_map<std::string, InlinedCallStack*> callstack_objects;
  for (Node* n : baz.optimized_graph()->nodes()) {
    if (n->kind() == prim::Constant) {
      if (!n->hasAttribute(attr::value) ||
          n->kindOf(attr::value) != AttributeKind::s) {
        continue;
      }
      // NOLINTNEXTLINE(performance-unnecessary-copy-initialization)
      std::string v = n->s(attr::value);
      if (n->callstack()) {
        callstack_objects[v] = n->callstack()->get();
      }
    }
  }
  // We expect to see nodes prim::Constant[value="a1"] and
  // prim::Constant[value="a2"] inlined to function 'c'. Their callstacks are
  // the same (a->b->c), so we want to make sure we're not creating different
  // callstack entries for them.
  ASSERT_TRUE(callstack_objects.count("a1") && callstack_objects.count("a2"));
  ASSERT_TRUE(callstack_objects.at("a1") == callstack_objects.at("a2"));
}

TEST(InlinedCallStackTest, BlockAnnotation) {
  Module a("A");
  a.define(R"(
    def forward(self, x, y, z: int):
      if (z == 1):
        return x + y
      else:
        return x * y
  )");
  Module b("B");
  b.define(R"(
    def forward(self, x):
      return x + 2
  )");
  Module c("C");
  c.register_module("A0", a);
  c.register_module("B0", b);
  c.define(R"(
    def forward(self, x, y, z: int):
      return self.A0.forward(x, y, z) + self.B0.forward(x)
  )");

  auto graph =
      toGraphFunction(c.get_method("forward").function()).optimized_graph();
  std::stringstream add_ss, mul_ss;
  for (Node* n : graph->nodes()) {
    if (n->kind() == prim::If) {
      for (Block* block : n->blocks()) {
        for (Node* if_node : block->nodes()) {
          if (if_node->kind() == aten::add) {
            for (const auto& e : if_node->callstack().value()->vec()) {
              add_ss << std::get<1>(e);
            }
            add_ss << if_node->sourceRange();
          }
          if (if_node->kind() == aten::mul) {
            for (const auto& e : if_node->callstack().value()->vec()) {
              mul_ss << std::get<1>(e);
            }
            mul_ss << if_node->sourceRange();
          }
        }
      }
    }
  }
  ASSERT_NE(add_ss.str().find("line 3"), std::string::npos);
  ASSERT_NE(add_ss.str().find("line 4"), std::string::npos);
  ASSERT_NE(
      add_ss.str().find("return self.A0.forward(x, y, z)"), std::string::npos);
  ASSERT_NE(add_ss.str().find("return x + y"), std::string::npos);
  ASSERT_NE(mul_ss.str().find("line 3"), std::string::npos);
  ASSERT_NE(mul_ss.str().find("line 6"), std::string::npos);
  ASSERT_NE(
      mul_ss.str().find("return self.A0.forward(x, y, z)"), std::string::npos);
  ASSERT_NE(mul_ss.str().find("return x * y"), std::string::npos);
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(InlinedCallStackTest, SelfCallMethods) {
  Module a("A");
  a.define(R"(
    def my_new_method(self, x):
      return x * 3
    def forward_impl_(self, x, y):
      return self.my_new_method(x) + y
    def forward(self, x, y):
      y = y + 2
      return self.forward_impl_(x, y)
  )");
  Module b("B");
  b.define(R"(
    def forward(self, x):
      return x + 2
  )");
  Module c("C");
  c.register_module("A0", a);
  c.register_module("B0", b);
  c.define(R"(
    def call_b(self, x):
      return self.B0.forward(x)
    def forward(self, x, y):
      return self.A0.forward(x, y) + self.call_b(x)
  )");

  auto graph =
      toGraphFunction(c.get_method("forward").function()).optimized_graph();
  std::unordered_map<std::string, size_t> module_hierarchies;
  for (Node* n : graph->nodes()) {
    auto hierarchy = torch::jit::utils::getNodesModuleHierarchy(*n);
    if (module_hierarchies.count(hierarchy) == 0) {
      module_hierarchies[hierarchy] = 0;
    }
    module_hierarchies[hierarchy] += 1;
  }
  ASSERT_EQ(module_hierarchies["A0(A)"], 2);
  ASSERT_EQ(module_hierarchies["A0(A).SELF(A).SELF(A)"], 2);
  ASSERT_EQ(module_hierarchies["A0(A).SELF(A)"], 1);
  ASSERT_EQ(module_hierarchies["SELF(C)"], 1);
  ASSERT_EQ(module_hierarchies["SELF(C).B0(B)"], 1);
}

TEST(AutogradSymbolsTest, Basic) {
  Symbol sym = Symbol::fromQualString("aten::test_symbol");
  Graph graph;
  auto node = graph.create(sym);
  TORCH_CHECK(canRunWithAutograd(node));

  sym = Symbol::fromQualString("prim::test_symbol");
  node = graph.create(sym);
  TORCH_CHECK(canRunWithAutograd(node));

  sym = Symbol::fromQualString("prim::FusionGroup");
  node = graph.create(sym);
  TORCH_CHECK(!canRunWithAutograd(node));

  sym = Symbol::fromQualString("custom::test_symbol");
  node = graph.create(sym);
  TORCH_CHECK(!canRunWithAutograd(node));
}

TEST(DefaultArgTypeHintingTest, Basic) {
  const auto text_non_hinted = R"(

def a(x, y=1):
    print("a1")
    print("a2")
    return x
  )";

  const auto text_hinted = R"(

def a(x, y:int=1):
    print("a1")
    print("a2")
    return x
  )";

  try {
    compile(text_non_hinted);
    ASSERT_TRUE(0);
  } catch (const std::exception& c) {
  }

  auto cu = compile(text_hinted);
}

// Basic set case.
TEST(FuturesTest, Basic) {
  auto f1 = c10::make_intrusive<Future>(IntType::get());
  ASSERT_FALSE(f1->completed());
  ASSERT_FALSE(f1->hasValue());
  int32_t sat1 = 0;
  int32_t sat2 = 0;
  f1->addCallback([&](Future& /* unused */) { ++sat1; });
  f1->markCompleted(43);
  ASSERT_TRUE(f1->completed());
  ASSERT_TRUE(f1->hasValue());
  ASSERT_FALSE(f1->hasError());
  ASSERT_EQ(sat1, 1);
  ASSERT_EQ(f1->constValue().toInt(), 43);
  ASSERT_EQ(f1->value().toInt(), 43);
  f1->addCallback([&](Future& /* unused */) { ++sat2; });
  ASSERT_EQ(sat1, 1);
  ASSERT_EQ(sat2, 1);
}

// Sparse CUDA tensor test
TEST(FutureTest, SparseTensor) {
  // Skip test if CUDA is not available.
  bool has_cuda = at::globalContext().hasCUDA();
  if (!has_cuda) {
    LOG(INFO) << "CUDA not available, skipping test";
  }
  for (int i = 0; i < 2; ++i) {
    auto f = c10::make_intrusive<Future>(TensorType::get());
    at::TensorOptions opts = at::TensorOptions().device(at::DeviceType::CUDA);
    auto sparse_tensor = i == 0 ? at::ones(10).to_sparse()
                                : at::sparse_coo_tensor(
                                      at::arange(10).unsqueeze(0).to(at::kLong),
                                      at::ones({10, 10}),
                                      opts);
    // Runs storage extraction for sparse CUDA tensors
    f->markCompleted(sparse_tensor);
    ASSERT_TRUE(f->completed());
    ASSERT_FALSE(f->hasError());
  }
}

// Basic error cases.
TEST(FuturesTest, Error) {
  auto f1 = c10::make_intrusive<Future>(IntType::get());
  int sat1 = 0;
  int sat2 = 0;
  f1->addCallback([&](Future& /* unused */) { ++sat1; });
  f1->setError(
      std::make_exception_ptr(c10::ivalue::Future::FutureError("Failed")));
  ASSERT_EQ(sat1, 1);
  ASSERT_TRUE(f1->completed());
  ASSERT_TRUE(f1->hasError());
  ASSERT_FALSE(f1->hasValue());
  try {
    (void)f1->value();
    ASSERT_TRUE(false); // Supposed to throw.
  } catch (const std::exception& e) {
    ASSERT_TRUE(strcmp(e.what(), "Failed") == 0);
  }
  f1->addCallback([&](Future& /* unused */) { ++sat2; });
  ASSERT_EQ(sat1, 1);
  ASSERT_EQ(sat2, 1);
  f1->setErrorIfNeeded(
      std::make_exception_ptr(c10::ivalue::Future::FutureError("Dup")));
  ASSERT_TRUE(strcmp(f1->tryRetrieveErrorMessage().c_str(), "Failed") == 0);
  ASSERT_EQ(sat1, 1);
  ASSERT_EQ(sat2, 1);
  try {
    (void)f1->constValue();
    ASSERT_TRUE(false); // Supposed to throw.
  } catch (const std::exception& e) {
    // Original error should be logged.
    ASSERT_TRUE(std::string(e.what()).find("Failed") != std::string::npos);
  }
}

// then
TEST(FuturesTest, Then) {
  auto f1 = c10::make_intrusive<Future>(IntType::get());
  auto f2 = f1->then(
      [](Future& f1) -> IValue { return f1.constValue().toInt() + 1; },
      IntType::get());
  auto f3 = f2->then(
      [](Future& f2) -> IValue { return f2.constValue().toInt() * 3; },
      IntType::get());
  bool done = false;
  f3->addCallback([&done](Future& f3) {
    ASSERT_EQ(f3.constValue().toInt(), (42 + 1) * 3);
    done = true;
  });
  ASSERT_FALSE(done);
  f1->markCompleted(42);
  ASSERT_TRUE(done);
}

// collectAll()
TEST(FuturesTest, CollectAll) {
  auto s1 = c10::make_intrusive<Future>(IntType::get());
  auto s2 = c10::make_intrusive<Future>(IntType::get());
  auto s3 = c10::make_intrusive<Future>(IntType::get());

  // Empty case
  c10::List<intrusive_ptr<ivalue::Future>> futures(
      FutureType::create(IntType::get()));
  auto c1 = collectAll(futures);
  ASSERT_TRUE(c1->completed());
  ASSERT_EQ(c1->value().toList().size(), 0);
  ASSERT_TRUE(
      *(c1->value().toList().elementType()) ==
      *FutureType::create(IntType::get()));

  // 1-element, initially not completed.
  futures.push_back(s1);
  auto c2 = collectAll(futures);
  ASSERT_FALSE(c2->completed());
  s1->markCompleted(5);
  ASSERT_TRUE(c2->completed());
  ASSERT_EQ(c2->value().toList().size(), 1);
  ASSERT_TRUE(
      *(c2->value().toList().elementType()) ==
      *FutureType::create(IntType::get()));
  ASSERT_EQ(c2->value().toList().get(0).toFuture()->value().toInt(), 5);

  // 1-element, already completed
  auto c3 = collectAll(futures);
  ASSERT_TRUE(c3->completed());
  ASSERT_EQ(c3->value().toList().size(), 1);
  ASSERT_EQ(c3->value().toList().get(0).toFuture()->value().toInt(), 5);

  // 3 elements.
  futures.push_back(s2);
  futures.push_back(s3);
  auto c4 = collectAll(futures);
  ASSERT_FALSE(c4->completed());
  s3->markCompleted(7);
  ASSERT_FALSE(c4->completed());
  s2->markCompleted(6);
  ASSERT_TRUE(c4->completed());
  ASSERT_EQ(c4->value().toList().size(), 3);
  ASSERT_EQ(c4->value().toList().get(0).toFuture()->value().toInt(), 5);
  ASSERT_EQ(c4->value().toList().get(1).toFuture()->value().toInt(), 6);
  ASSERT_EQ(c4->value().toList().get(2).toFuture()->value().toInt(), 7);
  ASSERT_TRUE(
      *(c4->value().toList().elementType()) ==
      *FutureType::create(IntType::get()));

  // Handle exception in the list.
  auto s4 = c10::make_intrusive<Future>(IntType::get());
  futures.push_back(s4);
  auto c5 = collectAll(futures);
  ASSERT_FALSE(c5->completed());
  s4->setError(
      std::make_exception_ptr(c10::ivalue::Future::FutureError("Failed")));
  ASSERT_TRUE(c5->completed());
  try {
    c5->value();
    ASSERT_TRUE(false); // supposed to throw
  } catch (const std::exception& e) {
    ASSERT_EQ(std::string(e.what()), "Failed");
  }
}

// collectAny()
TEST(FuturesTest, CollectAny) {
  auto s1 = c10::make_intrusive<Future>(IntType::get());

  // Empty case
  c10::List<intrusive_ptr<ivalue::Future>> futures(
      FutureType::create(IntType::get()));
  auto c1 = collectAny(futures);
  ASSERT_TRUE(c1->completed());

  // 1 element, not yet satisfied
  futures.push_back(s1);
  auto c2 = collectAny(futures);
  ASSERT_FALSE(c2->completed());
  s1->markCompleted(5);
  ASSERT_TRUE(c2->completed());
  ASSERT_TRUE(c2->value().isInt());
  ASSERT_EQ(c2->value().toInt(), 5);

  // 1 element already satisfied.
  auto c3 = collectAny(futures);
  ASSERT_TRUE(c3->completed());
  ASSERT_TRUE(c3->value().isInt());
  ASSERT_EQ(c3->value().toInt(), 5);

  // 2 elements
  futures.clear();
  auto s2 = c10::make_intrusive<Future>(IntType::get());
  auto s3 = c10::make_intrusive<Future>(IntType::get());
  futures.push_back(s2);
  futures.push_back(s3);
  auto c4 = collectAny(futures);
  ASSERT_FALSE(c4->completed());
  s3->markCompleted(7);
  ASSERT_TRUE(c4->completed());
  ASSERT_EQ(c4->value().toInt(), 7);
  s2->markCompleted(1);
  ASSERT_EQ(c4->value().toInt(), 7);
}

TEST(TLSFutureCallbacksTest, Basic) {
  // cb that verifies the profiler is enabled
  auto profilerEnabledCb = [](Future& /* unused */) {
    ASSERT_TRUE(torch::autograd::profiler::profilerEnabled());
  };
  // test running callbacks with propagation of TLS state.
  {
    // Enable the profiler in this thread
    torch::autograd::profiler::enableProfilerLegacy(
        torch::autograd::profiler::ProfilerConfig(
            torch::autograd::profiler::ProfilerState::CPU, false, false));
    auto s1 = c10::make_intrusive<Future>(IntType::get());
    s1->addCallback(wrapPropagateTLSState(profilerEnabledCb));
    std::thread t([s1 = std::move(s1)]() { s1->markCompleted(); });
    // Since we join here, we can ensure that all callbacks corresponding to
    // markCompleted() have finished.
    t.join();
    torch::autograd::profiler::disableProfilerLegacy();
  }
  // then() with TLS State
  {
    // Enable the profiler in this thread
    torch::autograd::profiler::enableProfilerLegacy(
        torch::autograd::profiler::ProfilerConfig(
            torch::autograd::profiler::ProfilerState::CPU, false, false));
    auto s1 = c10::make_intrusive<Future>(IntType::get());
    auto s2 = s1->then(
        wrapPropagateTLSState([&profilerEnabledCb](Future& s1) {
          profilerEnabledCb(s1);
          return at::IValue(1);
        }),
        IntType::get());
    std::thread t([s1 = std::move(s1)]() { s1->markCompleted(); });
    t.join();
    s2->wait();
    torch::autograd::profiler::disableProfilerLegacy();
  }
}

TEST(ProfilerDisableInCallbackTest, Basic) {
  // cb that verifies the profiler is enabled
  auto profilerEnabledCb = []() {
    ASSERT_TRUE(torch::autograd::profiler::profilerEnabled());
  };
  torch::autograd::profiler::enableProfilerLegacy(
      torch::autograd::profiler::ProfilerConfig(
          torch::autograd::profiler::ProfilerState::CPU, false, false));
  auto s1 = c10::make_intrusive<Future>(IntType::get());
  auto verifyProfilerCb =
      wrapPropagateTLSState([&profilerEnabledCb](Future& /* unused */) {
        // Ensure the profiler is still enabled in this thread.
        profilerEnabledCb();
        auto t1 = torch::ones({2, 2});
        auto t2 = torch::ones({2, 2});
        torch::add(t1, t2);
        // Don't cleanup TLSState, and just consolidate.
        auto opts =
            torch::autograd::profiler::ProfilerDisableOptions(false, true);
        auto thread_event_lists =
            // NOLINTNEXTLINE(performance-move-const-arg)
            torch::autograd::profiler::disableProfilerLegacy(std::move(opts));
        // Ensure that the events from this thread are still profiled and we
        // obtain the expected in events in our consolidated list when calling
        // disableProfilerLegacy().
        bool found_ones = false;
        bool found_add = false;
        for (const auto& li : thread_event_lists) {
          for (const auto& evt : li) {
            if (strcmp(evt.name(), "aten::add") == 0) {
              found_add = true;
            } else if (strcmp(evt.name(), "aten::ones") == 0) {
              found_ones = true;
            }
          }
          if (found_add && found_ones) {
            break;
          }
        }
        ASSERT_TRUE(found_ones);
        ASSERT_TRUE(found_add);
      });

  s1->addCallback(verifyProfilerCb);
  // Disable the profiler, but do not consolidate results in the main thread.
  auto opts = torch::autograd::profiler::ProfilerDisableOptions(true, false);
  // NOLINTNEXTLINE(performance-move-const-arg)
  torch::autograd::profiler::disableProfilerLegacy(std::move(opts));
  std::thread t([s1 = std::move(s1)]() { s1->markCompleted(at::IValue(1)); });
  t.join();

  // Similar to above test, but verifies correctness in the case where
  // continuation runs on the main thread.
  torch::autograd::profiler::enableProfilerLegacy(
      torch::autograd::profiler::ProfilerConfig(
          torch::autograd::profiler::ProfilerState::CPU, false, false));
  s1 = c10::make_intrusive<Future>(IntType::get());
  s1->addCallback(verifyProfilerCb);
  // Runs callback inline
  s1->markCompleted(at::IValue(1));
  opts = torch::autograd::profiler::ProfilerDisableOptions(true, false);
  // NOLINTNEXTLINE(performance-move-const-arg)
  torch::autograd::profiler::disableProfilerLegacy(std::move(opts));
}

TEST(RecordDebugHandles, Basic) {
  // Enable the profiler in this thread
  const std::set<torch::autograd::profiler::ActivityType> activities(
      {torch::autograd::profiler::ActivityType::CPU});
  torch::autograd::profiler::prepareProfiler(
      torch::autograd::profiler::ProfilerConfig(
          torch::autograd::profiler::ProfilerState::KINETO, false, false),
      activities);
  torch::autograd::profiler::enableProfiler(
      torch::autograd::profiler::ProfilerConfig(
          torch::autograd::profiler::ProfilerState::KINETO, false, false),
      activities);
  {
    RECORD_EDGE_SCOPE_WITH_DEBUG_HANDLE_AND_INPUTS("my_function", 42, {});
    float x{5.9999}, y{2.1212};
    float z = x / y;
    (void)z;
  }
  {
    RECORD_USER_SCOPE_WITH_INPUTS("not_my_function", {});
    float x{5.9999}, y{2.1212};
    float z = x / y;
    (void)z;
  }
  auto profiler_results_ptr = torch::autograd::profiler::disableProfiler();
  const auto& kineto_events = profiler_results_ptr->events();
  size_t my_events{0};
  for (const auto& e : kineto_events) {
    if (e.name() == "my_function") {
      ASSERT_EQ(e.debugHandle(), 42);
      my_events++;
    } else if (e.name() == "not_my_function") {
      ASSERT_EQ(e.debugHandle(), -1);
      my_events++;
    }
  }
  ASSERT_EQ(my_events, 2);
}

TEST(RecordDebugHandles, ScopedCallbacks) {
  // Enable the profiler in this thread
  torch::autograd::profiler::prepareProfiler(
      torch::autograd::profiler::ProfilerConfig(
          torch::autograd::profiler::ProfilerState::KINETO, false, false),
      {torch::autograd::profiler::ActivityType::CPU});
  torch::autograd::profiler::enableProfiler(
      torch::autograd::profiler::ProfilerConfig(
          torch::autograd::profiler::ProfilerState::KINETO, false, false),
      {torch::autograd::profiler::ActivityType::CPU});

  {
    auto a = torch::rand({128, 128});
    auto b = torch::rand({128, 128});
    auto c = a + b;
  }
  auto profiler_results_ptr = torch::autograd::profiler::disableProfiler();
  ASSERT_TRUE(profiler_results_ptr->events().size() > 0);

  // Enable the profiler in this thread
  torch::autograd::profiler::prepareProfiler(
      torch::autograd::profiler::ProfilerConfig(
          torch::autograd::profiler::ProfilerState::KINETO, false, false),
      {torch::autograd::profiler::ActivityType::CPU});
  torch::autograd::profiler::enableProfiler(
      torch::autograd::profiler::ProfilerConfig(
          torch::autograd::profiler::ProfilerState::KINETO, false, false),
      {torch::autograd::profiler::ActivityType::CPU},
      {at::RecordScope::LITE_INTERPRETER});
  {
    auto a = torch::rand({128, 128});
    auto b = torch::rand({128, 128});
    auto c = a + b;
  }
  profiler_results_ptr = torch::autograd::profiler::disableProfiler();
  ASSERT_TRUE(profiler_results_ptr->events().size() == 0);

  torch::autograd::profiler::prepareProfiler(
      torch::autograd::profiler::ProfilerConfig(
          torch::autograd::profiler::ProfilerState::KINETO, false, false),
      {torch::autograd::profiler::ActivityType::CPU});
  torch::autograd::profiler::enableProfiler(
      torch::autograd::profiler::ProfilerConfig(
          torch::autograd::profiler::ProfilerState::KINETO, false, false),
      {torch::autograd::profiler::ActivityType::CPU},
      {at::RecordScope::LITE_INTERPRETER});
  {
    RECORD_EDGE_SCOPE_WITH_DEBUG_HANDLE_AND_INPUTS("my_function", 42, {});
    auto a = torch::rand({128, 128});
    auto b = torch::rand({128, 128});
    auto c = a + b;
  }
  {
    RECORD_USER_SCOPE_WITH_INPUTS("not_my_function", {});
    auto a = torch::rand({128, 128});
    auto b = torch::rand({128, 128});
    auto c = a + b;
  }
  profiler_results_ptr = torch::autograd::profiler::disableProfiler();
  const auto& kineto_events = profiler_results_ptr->events();
  for (const auto& e : kineto_events) {
    if (e.name() == "my_function") {
      ASSERT_EQ(e.debugHandle(), 42);
    }
  }
  ASSERT_TRUE(profiler_results_ptr->events().size() == 1);
}

TEST(IValueKWargsTest, Basic) {
  const auto text = R"(
    def foo(a : int, b : int, c : int = 4):
      return a + 2*b + 3*c
  )";
  auto cu = compile(text);
  auto result = cu->get_function("foo")({1}, {{"b", 3}});
  ASSERT_EQ(result.toInt(), 19);
}

TEST(ComputeFlopsTest, Basic) {
  uint64_t flops = 0;

  // Test unknown operator
  std::unordered_map<std::string, c10::IValue> extra_args;
  flops = torch::profiler::impl::computeFlops(
      std::string("aten::unknown"), extra_args);
  ASSERT_EQ(flops, 0);

  // Test aten::conv2d
  extra_args.clear();
  std::vector<int64_t> input_size = {4, 5, 6, 7};
  std::vector<int64_t> weight_size = {3, 5, 2, 1};
  std::vector<int64_t> padding = {1, 0};
  std::vector<int64_t> stride = {1, 1};
  std::vector<int64_t> dilation = {0, 0};
  extra_args["input_size"] = at::IValue(at::IntArrayRef(input_size));
  extra_args["weight_size"] = at::IValue(at::IntArrayRef(weight_size));
  extra_args["groups"] = 1;
  extra_args["padding"] = at::IValue(at::IntArrayRef(padding));
  extra_args["stride"] = at::IValue(at::IntArrayRef(stride));
  extra_args["dilation"] = at::IValue(at::IntArrayRef(dilation));
  flops = torch::profiler::impl::computeFlops(
      std::string("aten::conv2d"), extra_args);
  ASSERT_EQ(flops, 13440);

  // Test aten::conv2d fail
  input_size = {4, 5, 6, 7};
  weight_size = {4, 5, 6};
  extra_args["input_size"] = at::IValue(at::IntArrayRef(input_size));
  extra_args["weight_size"] = at::IValue(at::IntArrayRef(weight_size));
  flops = torch::profiler::impl::computeFlops(
      std::string("aten::conv2d"), extra_args);
  ASSERT_EQ(flops, 0);

  // Test aten::conv2d fail 2
  weight_size = {3, 5, 2, 1};
  stride = {0, 0};
  extra_args["weight_size"] = at::IValue(at::IntArrayRef(input_size));
  extra_args["stride"] = at::IValue(at::IntArrayRef(stride));
  flops = torch::profiler::impl::computeFlops(
      std::string("aten::conv2d"), extra_args);
  ASSERT_EQ(flops, 0);

  // Test aten::conv2d fail 3
  extra_args.clear();
  input_size = {4, 5, 6, 7};
  extra_args["input_size"] = at::IValue(at::IntArrayRef(input_size));
  flops = torch::profiler::impl::computeFlops(
      std::string("aten::conv2d"), extra_args);
  ASSERT_EQ(flops, 0);

  // Test aten::mm
  extra_args.clear();
  std::vector<int64_t> mat1_sizes = {3, 4, 5, 6};
  std::vector<int64_t> mat2_sizes = {6, 5, 4, 3};
  extra_args["mat1_size"] = at::IValue(at::IntArrayRef(mat1_sizes));
  extra_args["mat2_size"] = at::IValue(at::IntArrayRef(mat2_sizes));
  flops =
      torch::profiler::impl::computeFlops(std::string("aten::mm"), extra_args);
  ASSERT_EQ(flops, 43200);

  // Test aten::addmm
  flops = torch::profiler::impl::computeFlops(
      std::string("aten::addmm"), extra_args);
  ASSERT_EQ(flops, 43200);

  // Test aten::bmm
  extra_args.clear();
  mat1_sizes = {7, 5, 6};
  mat2_sizes = {7, 6, 3};
  extra_args["mat1_size"] = at::IValue(at::IntArrayRef(mat1_sizes));
  extra_args["mat2_size"] = at::IValue(at::IntArrayRef(mat2_sizes));
  flops =
      torch::profiler::impl::computeFlops(std::string("aten::bmm"), extra_args);
  ASSERT_EQ(flops, 1260);

  // Test aten::baddbmm
  flops = torch::profiler::impl::computeFlops(
      std::string("aten::baddbmm"), extra_args);
  ASSERT_EQ(flops, 1260);

  // Test mm out of range
  extra_args.clear();
  flops =
      torch::profiler::impl::computeFlops(std::string("aten::mm"), extra_args);
  ASSERT_EQ(flops, 0);

  // Test aten::add.Tensor
  extra_args.clear();
  std::vector<int64_t> mat_sizes = {3, 4, 5, 6};
  extra_args["mat_size"] = at::IValue(at::IntArrayRef(mat_sizes));
  flops =
      torch::profiler::impl::computeFlops(std::string("aten::add"), extra_args);
  ASSERT_EQ(flops, 360);

  // Test aten::mul.Tensor
  extra_args.clear();
  mat_sizes = {3, 4, 5, 6};
  extra_args["mat_size"] = at::IValue(at::IntArrayRef(mat_sizes));
  flops =
      torch::profiler::impl::computeFlops(std::string("aten::mul"), extra_args);
  ASSERT_EQ(flops, 360);
}

TEST(TestConstant, TensorGrad) {
  auto graph = std::make_shared<Graph>();
  IValue ten = torch::randn({3, 5}).requires_grad_(true);
  auto con = tryInsertConstant(*graph, ten);
  ASSERT_TRUE(con == c10::nullopt);
}

TEST(TestMutation, Basic) {
  auto graph = std::make_shared<Graph>();
  std::unordered_map<std::string, Value*> vmap;
  parseIR(
      R"IR(
graph(%x.1 : Tensor):
  %2 : int = prim::Constant[value=1]()
  %9 : int = prim::Constant[value=4]()
  %x.3 : Tensor = aten::add(%x.1, %2, %2)
  %7 : Tensor = aten::add_(%x.3, %2, %2)
  %y.1 : Tensor = aten::add(%x.3, %9, %2)
  return (%y.1))IR",
      &*graph,
      vmap);
  RemoveTensorMutation(graph, [](Node*) { return false; });
  testing::FileCheck().check("aten::add_")->run(*graph);
  RemoveTensorMutation(graph, [](Node*) { return true; });
  testing::FileCheck().check_not("aten::add_")->run(*graph);
}

TEST(TestInplaceToFunctionalActivation, Basic) {
  auto graph = std::make_shared<Graph>();
  std::unordered_map<std::string, Value*> vmap;
  parseIR(
      R"IR(
graph(%x.1 : Tensor):
  %2 : int = prim::Constant[value=1]()
  %x.3 : Tensor = aten::add(%x.1, %2, %2)
  %y : Tensor = aten::relu_(%x.3)
  return (%y))IR",
      &*graph,
      vmap);
  InplaceToFunctionalActivation(graph);
  testing::FileCheck().check("aten::relu")->run(*graph);
  testing::FileCheck().check_not("aten::relu_")->run(*graph);
}

TEST(TestRegisterShapeOp, Basic) {
  auto graph = std::make_shared<Graph>();
  std::unordered_map<std::string, Value*> vmap;
  parseIR(
      R"IR(
graph():
  %2 : int = prim::Constant[value=5]()
  %3: int[] = prim::ListConstruct(%2, %2)
  return (%3))IR",
      &*graph,
      vmap);

  auto g2 = std::make_shared<Graph>();
  parseIR(
      R"IR(
graph():
  %2 : Tensor = prim::MakeTestTensor()
  return (%2))IR",
      &*g2,
      vmap);

  const FunctionSchema& schema = g2->nodes().begin()->schema();
  torch::jit::RegisterShapeComputeGraphForSchema(schema, graph);
  PropagateShapesOnGraph(g2);
  testing::FileCheck().check("5, 5")->run(*g2);
}

TEST(TestFunctionalToInplaceActivation, Basic) {
  auto graph = std::make_shared<Graph>();
  std::unordered_map<std::string, Value*> vmap;
  parseIR(
      R"IR(
graph(%x.1 : Tensor):
  %2 : int = prim::Constant[value=1]()
  %x.3 : Tensor = aten::add(%x.1, %2, %2)
  %y : Tensor = aten::relu(%x.3)
  return (%y))IR",
      &*graph,
      vmap);
  FunctionalToInplaceActivation(graph);
  testing::FileCheck().check("aten::relu_")->run(*graph);
  testing::FileCheck().check_not("aten::relu(")->run(*graph);
}

TEST(TestFunctionExecutor, SimpleExecutorTest) {
  auto graph = std::make_shared<Graph>();
  parseIR(
      R"IR(
graph(%x.1 : Tensor):
  %2 : int = prim::Constant[value=1]()
  %x.3 : Tensor = aten::add(%x.1, %2, %2)
  %y : Tensor = aten::relu(%x.3)
  return (%y))IR",
      &*graph);
  {
    auto func = std::make_unique<GraphFunction>(
        "name", graph, [](GraphFunction&) {}, ExecutorExecutionMode::PROFILING);
    auto a = at::rand({2, 2, 2}, TensorOptions(kCPU).dtype(at::kFloat));
    Stack stack = {a};
    func->run(stack);
    auto g = lastExecutedOptimizedGraph();
    testing::FileCheck()
        .check("prim::profile")
        ->check("aten::add")
        ->check("aten::relu")
        ->run(*g);
  }
  {
    auto func = std::make_unique<GraphFunction>(
        "name", graph, [](GraphFunction&) {}, ExecutorExecutionMode::SIMPLE);
    auto a = at::rand({2, 2, 2}, TensorOptions(kCPU).dtype(at::kFloat));
    Stack stack = {a};
    func->run(stack);
    auto g = func->getDebugState().graph;
    testing::FileCheck()
        .check_not("prim::profile")
        ->check("aten::add")
        ->check("aten::relu")
        ->run(*g);
  }
}

TEST(TestFunctionExecutor, RunDecompositionTest) {
  static auto* func = torch::jit::GetDecompositionExecutor(
      "aten::var(Tensor self, bool unbiased=True) -> Tensor");
  for (bool unbiased : {true, false}) {
    auto input = at::rand({4, 4});
    Stack stack = {input, unbiased};
    func->run(stack);
    at::Tensor out = pop(stack).toTensor();
    ASSERT_TRUE(at::allclose(out, input.var(unbiased)));
  }
}

TEST(TestShapeGraphLinting, Basic) {
  auto schemas = RegisteredShapeComputeSchemas();
  for (const auto& schema : schemas) {
    // arange does not actually support complex, leave as
    // union[int, float] for now
    if (schema->name() == "aten::arange") {
      continue;
    }
    auto g = shapeComputeGraphForSchema(*schema);
    TORCH_INTERNAL_ASSERT(g);
    LintShapeComputeGraph(schema, *g);
  }
}

// TODO: move to test_kernel when global settings are explicit
// fusion parameters
class Composed : public ::testing::Test {
 public:
  void SetUp() override {
    torch::jit::tensorexpr::getTEMustUseLLVMOnCPU() = false;
  }
};

TEST_F(Composed, ComposedOp) {
  struct WithCPUFuser {
    WithCPUFuser(bool val = true) : cpuFuserEnabled(canFuseOnCPU()) {
      overrideCanFuseOnCPU(val);
    }

    ~WithCPUFuser() {
      overrideCanFuseOnCPU(cpuFuserEnabled);
    }

    bool cpuFuserEnabled;
  };

#ifdef TORCH_ENABLE_LLVM
  const auto graph_string = R"IR(
      graph(%0 : Float(5, 3, strides=[3, 1], device=cpu),
            %1 : Float(5, 3, strides=[1, 5], device=cpu)):
        %2 : Float(5, 3, strides=[3, 1], device=cpu) = aten::mul(%0, %1)
        %3 : Float(5, 3, strides=[3, 1], device=cpu) = aten::mul(%0, %2)
        %4 : Float(5, 3, strides=[3, 1], device=cpu) = aten::mul(%0, %3)
        return (%3, %4))IR";
  auto graph = std::make_shared<Graph>();
  parseIR(graph_string, &*graph);

  // wrong input sizes so we hit the fallback path
  auto a = at::rand({2, 2, 2}, TensorOptions(kCPU).dtype(at::kFloat));
  auto b = at::rand({2, 2, 2}, TensorOptions(kCPU).dtype(at::kFloat))
               .transpose(0, 1);
  auto ref1 = a * (a * b);
  auto ref2 = a * ref1;
  WithCPUFuser g(true);
  bool fusable_on_device = torch::jit::tensorexpr::getTEMustUseLLVMOnCPU();
  torch::jit::tensorexpr::getTEMustUseLLVMOnCPU() = false;
  FuseTensorExprs(
      graph,
      /*min_group_size*/ 2,
      /*add_composed_op*/ true,
      /*fuse_to_dynamic_shapes*/ true);
  Code code(graph, "");
  InterpreterState interpreter{code};
  std::vector<IValue> stack = {a, b};
  interpreter.run(stack);
  at::Tensor out2 = pop(stack).toTensor();
  at::Tensor out1 = pop(stack).toTensor();
  ASSERT_TRUE(at::allclose(ref1, out1));
  ASSERT_TRUE(at::allclose(ref2, out2));

  auto inp_1 = at::ones({4, 4}, TensorOptions(kCPU).dtype(at::kFloat));
  auto inp_2 = at::ones({4, 4}, TensorOptions(kCPU).dtype(at::kFloat));
  stack = {inp_1, inp_2, a, b};
  InterpreterState interpreter2{code};
  interpreter2.run(stack);
  out2 = pop(stack).toTensor();
  out1 = pop(stack).toTensor();
  ASSERT_TRUE(at::allclose(ref1, out1));
  ASSERT_TRUE(at::allclose(ref2, out2));
  // inp_1 is on the bottom of the stack, and corresponds
  // to the second output. inp_2 is on the top corresponds to first output
  ASSERT_TRUE(at::allclose(inp_1, ref2));
  ASSERT_TRUE(at::allclose(inp_2, ref1));
  torch::jit::tensorexpr::getTEMustUseLLVMOnCPU() = fusable_on_device;
#endif
}

TEST(ConstantPropagation, CustomClassesCanBePropagated) {
#ifdef USE_PYTORCH_QNNPACK
  const auto src = R"IR(
    graph():
        %none: NoneType = prim::Constant()
        %dim: int = prim::Constant[value=3]()
        %shape: int[] = prim::ListConstruct(%dim, %dim)
        %weight: Tensor = aten::ones(%shape, %none, %none, %none, %none)
        %scale: float = prim::Constant[value=1.]()
        %zero_point: int = prim::Constant[value=0]()
        %dtype: int = prim::Constant[value=12]()
        %weight_q: Tensor = aten::quantize_per_tensor(%weight, %scale, %zero_point, %dtype)
        %params: __torch__.torch.classes.quantized.LinearPackedParamsBase = quantized::linear_prepack(%weight_q, %none)
        return (%params)
  )IR";
  auto graph = std::make_shared<Graph>();
  std::unordered_map<std::string, Value*> vmap;
  parseIR(src, graph.get(), vmap);

  ConstantPropagation(graph);

  testing::FileCheck().check_not("quantized::linear_prepack")->run(*graph);
#endif
}

} // namespace jit
} // namespace torch
