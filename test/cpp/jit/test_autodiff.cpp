#include <gtest/gtest.h>

#include "test/cpp/jit/test_utils.h"
#include "torch/csrc/jit/frontend/tracer.h"
#include "torch/csrc/jit/passes/common_subexpression_elimination.h"
#include "torch/csrc/jit/passes/constant_propagation.h"
#include "torch/csrc/jit/passes/create_autodiff_subgraphs.h"
#include "torch/csrc/jit/passes/dead_code_elimination.h"
#include "torch/csrc/jit/passes/graph_fuser.h"
#include "torch/csrc/jit/passes/lower_grad_of.h"
#include "torch/csrc/jit/passes/requires_grad_analysis.h"
#include "torch/csrc/jit/passes/shape_analysis.h"
#include "torch/csrc/jit/passes/utils/subgraph_utils.h"
#include "torch/csrc/jit/runtime/argument_spec.h"
#include "torch/csrc/jit/runtime/autodiff.h"
#include "torch/csrc/jit/runtime/graph_iterator.h"
#include "torch/csrc/jit/runtime/profiling_graph_executor_impl.h"
#include "torch/torch.h"

#include <ATen/ATen.h>
#include "torch/csrc/autograd/engine.h"
#include "torch/csrc/autograd/generated/variable_factories.h"
#include "torch/csrc/autograd/variable.h"

namespace torch {
namespace jit {

using namespace torch::autograd;

using var_meta_type = std::vector<int64_t>;
using var_meta_list = std::vector<var_meta_type>;
using test_fn_type = std::function<variable_list(const variable_list&)>;

struct ADTestSpec {
  ADTestSpec(
      const char* name,
      // NOLINTNEXTLINE(modernize-pass-by-value)
      var_meta_list input_meta,
      // NOLINTNEXTLINE(modernize-pass-by-value)
      test_fn_type test_fn,
      float clampMax = -1.0f)
      : name(name),
        input_meta(input_meta),
        test_fn(test_fn),
        clampMax(clampMax) {}

  variable_list operator()(const variable_list& inputs) const {
    return test_fn(inputs);
  };

  std::vector<Variable> make_vars() const {
    std::vector<Variable> out;
    for (const auto& m : input_meta) {
      if (clampMax > 0.0f) {
        out.push_back(torch::randn(m, at::requires_grad(true))
                          .clamp(-clampMax, clampMax));
        continue;
      }
      out.push_back(torch::randn(m, at::requires_grad(true)));
    }
    return out;
  }

  const char* name;
  var_meta_list input_meta;
  test_fn_type test_fn;
  float clampMax;
};

variable_list get_grad_outputs(const variable_list& vars) {
  return fmap(vars, [](const Variable& v) -> Variable {
    return at::randn(v.sizes(), v.options());
  });
}

variable_list grad(
    const variable_list& outputs,
    const variable_list& inputs,
    const variable_list& grad_outputs) {
  const auto get_edge = [](const Variable& v) {
    return torch::autograd::impl::gradient_edge(v);
  };
  auto& engine = torch::autograd::Engine::get_default_engine();
  return engine.execute(
      fmap(outputs, get_edge),
      grad_outputs,
      true,
      false,
      false,
      fmap(inputs, get_edge));
}

TEST(AutodiffTest, ADFormulas) {
  const auto cast = [](const Variable& v) {
    return static_cast<at::Tensor>(v);
  };

  using VL = variable_list;
  const var_meta_list binary_pointwise = {{2, 3, 4, 5}, {2, 3, 4, 5}};
  const var_meta_list unary_pointwise = {{2, 3, 4, 5}};
  const var_meta_list unary_pointwise_2d = {{2, 3}};
  const std::vector<ADTestSpec> ad_tests = {
      {"add",
       binary_pointwise,
       [](const VL& v) -> VL { return {v[0] + v[1]}; }},
      {"sub",
       binary_pointwise,
       [](const VL& v) -> VL { return {v[0] - v[1]}; }},
      {"mul",
       binary_pointwise,
       [](const VL& v) -> VL { return {v[0] * v[1]}; }},
      {"sigmoid",
       unary_pointwise,
       [](const VL& v) -> VL { return {v[0].sigmoid()}; }},
      // Clamp tanh input tensor values to [-3, 3]
      // to set a minimum on gradient absolute values
      {"tanh",
       unary_pointwise,
       [](const VL& v) -> VL { return {v[0].tanh()}; },
       3.0f},
      {"t", unary_pointwise_2d, [](const VL& v) -> VL { return {v[0].t()}; }},
      {"view",
       unary_pointwise_2d,
       [](const VL& v) -> VL {
         return {v[0].view({3, 2})};
       }},
      {"expand",
       {{2, 1}},
       [](const VL& v) -> VL {
         return {v[0].expand({2, 3})};
       }},
      {"mm",
       {{10, 12}, {12, 15}},
       [](const VL& v) -> VL { return {v[0].mm(v[1])}; }},
      // TODO: enable once we'll be able to capture lists across
      // forward-backward
      //{"chunk",   {{10, 12, 15}}, [](const VL& v) -> VL { return
      // fmap<Variable>(v[0].chunk(4, 1)); }},
      //{"chunk",   {{10, 12, 15}}, [](const VL& v) -> VL { return
      // fmap<Variable>(v[0].chunk(3, 2)); }},
      //{"split",   {{10, 12, 15}}, [](const VL& v) -> VL { return
      // fmap<Variable>(v[0].split(4, 1)); }},
      //{"split",   {{10, 12, 15}}, [](const VL& v) -> VL { return
      // fmap<Variable>(v[0].split(3, 2)); }},
  };

  for (const auto& test : ad_tests) {
    // Get reference values form autograd
    auto vars_in = test.make_vars();
    auto vars_out = test(vars_in);
    auto var_grads_in = get_grad_outputs(vars_out);
    auto var_grads_out = grad(vars_out, vars_in, var_grads_in);

    // Trace and differentiate the op
    auto graph = tracer::trace(
                     fmap<IValue>(vars_in),
                     [&test](Stack in) -> Stack {
                       auto ivalue_inps = fmap(in, [](const IValue& v) {
                         return Variable(v.toTensor());
                       });
                       return fmap<IValue>(test(ivalue_inps));
                     },
                     [](const Variable& var) { return ""; })
                     .first->graph;
    EliminateDeadCode(graph); // Tracing of some ops depends on the DCE trick
    ConstantPropagation(graph);
    auto grad_spec = differentiate(graph);
    LowerGradOf(*grad_spec.df);
    // Get outputs from the interpreter
    auto tensors_in = fmap(vars_in, cast);
    auto tensor_grads_in = fmap(var_grads_in, cast);
    tensor_list tensors_out, tensor_grads_out;
    std::tie(tensors_out, tensor_grads_out) =
        runGradient(grad_spec, tensors_in, tensor_grads_in);

    // Compare results
    auto expected_tensors_out = fmap(vars_out, cast);
    auto expected_tensor_grads_out = fmap(var_grads_out, cast);
    assertAllClose(tensors_out, expected_tensors_out);
    assertAllClose(tensor_grads_out, expected_tensor_grads_out);
  }
}

TEST(AutodiffTest, Differentiate) {
  // Note: can't use IRParser for this test due to issue #23989
  auto graph = std::make_shared<Graph>();
  std::vector<int64_t> sizes{2, 3, 4};
  std::vector<int64_t> strides{12, 4, 1};
  const auto type = TensorType::create(
      at::ScalarType::Float,
      at::kCPU,
      c10::VaryingShape<int64_t>{sizes},
      c10::VaryingShape<int64_t>{strides},
      true);

  // Builds graph a * b * a + b
  auto* a = graph->addInput()->setType(type);
  auto* b = graph->addInput()->setType(type);
  auto* cOne = graph->insertConstant(1);

  auto* ab = graph->insertNode(graph->create(aten::mul, /*num_outputs =*/1));
  ab->addInput(a);
  ab->addInput(b);

  auto* aba = graph->insertNode(graph->create(aten::mul, /*num_outputs =*/1));
  aba->addInput(ab->output());
  aba->addInput(a);

  auto* abaplusb =
      graph->insertNode(graph->create(aten::add, /*num_outputs =*/1));
  abaplusb->addInput(aba->output());
  abaplusb->addInput(b);
  abaplusb->addInput(cOne);

  graph->registerOutput(abaplusb->output());

  auto grad_spec = differentiate(graph);
  std::vector<size_t> expected_captured_inputs = {0, 1};
  std::vector<size_t> expected_captured_outputs = {1, 2, 3, 4, 5, 6, 7};
  std::vector<size_t> expected_input_vjps = {0, 1};
  std::vector<size_t> expected_output_vjps = {0, 1};
  ASSERT_EQ(grad_spec.f_real_outputs, 1);
  ASSERT_EQ(grad_spec.df_input_captured_inputs, expected_captured_inputs);
  ASSERT_EQ(grad_spec.df_input_captured_outputs, expected_captured_outputs);
  ASSERT_EQ(grad_spec.df_input_vjps, expected_input_vjps);
  ASSERT_EQ(grad_spec.df_output_vjps, expected_output_vjps);
  testing::FileCheck()
      .check_count("aten::mul", 2)
      ->check("aten::size")
      ->check("aten::add")
      ->run(*grad_spec.f);
  testing::FileCheck()
      .check("prim::GradOf[name=\"aten::add\"]")
      ->check_count("prim::GradOf[name=\"aten::mul\"]", 2)
      ->check_count("AutogradAdd", 2)
      ->run(*grad_spec.df);
}

TEST(AutodiffTest, DifferentiateWithRequiresGrad) {
  const auto graph_string = R"IR(
    graph(%0 : Tensor,
          %1 : Tensor):
      %2 : int = prim::Constant[value=1]()
      %3 : Tensor = aten::mul(%1, %1)
      %4 : Tensor = aten::add(%3, %1, %2)
      %5 : Tensor = aten::add(%4, %0, %2)
      %6 : Tensor = aten::mul(%5, %0)
      %7 : Tensor = aten::add(%6, %1, %2)
      return (%4, %7))IR";
  auto g = std::make_shared<Graph>();
  torch::jit::parseIR(graph_string, g.get());

  auto a_var = autograd::make_variable(
      at::empty_strided(2, 2, at::CPU(at::kFloat).options()), true);
  auto b_var = autograd::make_variable(
      at::empty_strided(2, 2, at::CPU(at::kFloat).options()), false);

  ArgumentSpecCreator asc(*g);
  asc.specializeTypes(*g, asc.create(true, {a_var, b_var}));

  PropagateInputShapes(g);
  PropagateRequiresGrad(g);

  auto grad_spec = differentiate(g);
  std::vector<size_t> expected_input_vjps = {1, 2}; // for e and %4 = (d + a)
  std::vector<size_t> expected_output_vjps = {0}; // only a requires grad
  ASSERT_EQ(grad_spec.f_real_outputs, 2);
  ASSERT_EQ(grad_spec.df_input_captured_inputs, std::vector<size_t>({0}));
  ASSERT_EQ(
      grad_spec.df_input_captured_outputs,
      std::vector<size_t>({2, 3, 4, 5, 6}));
  ASSERT_EQ(grad_spec.df_input_vjps, expected_input_vjps);
  ASSERT_EQ(grad_spec.df_output_vjps, expected_output_vjps);
  testing::FileCheck()
      .check("aten::mul")
      ->check_count("aten::add", 2)
      ->check("aten::mul")
      ->check("aten::size")
      ->check("aten::add")
      ->run(*grad_spec.f);

  testing::FileCheck()
      .check_count("prim::GradOf[name=\"aten::mul\"]", 1, /*exactly*/ true)
      ->run(*grad_spec.df);
}

class AutodiffRemoveUnusedGradientsTest : public ::testing::Test {
 protected:
  void SetUp() override {
    prev_exec = getExecutorMode();
    getExecutorMode() = true;
    prev_inline_autodiff = getAutodiffSubgraphInlining();
    debugSetAutodiffSubgraphInlining(false);
  }
  void TearDown() override {
    getExecutorMode() = prev_exec;
    debugSetAutodiffSubgraphInlining(prev_inline_autodiff);
  }

  bool prev_exec;
  bool prev_profiling;
  bool prev_inline_autodiff;
};

TEST_F(AutodiffRemoveUnusedGradientsTest, Linear) {
  auto graph = std::make_shared<Graph>();
  const std::string input =
      R"IR(
graph(%inp.1 : Tensor,
      %weight.1 : Tensor,
      %bias.1 : Tensor):
  %6 : Tensor = aten::linear(%inp.1, %weight.1, %bias.1)
  return (%6))IR";
  parseIR(input, graph.get());

  auto inp = torch::randn({10, 10}).requires_grad_(false);
  auto weight = torch::randn({10, 10}).requires_grad_(true);
  auto bias = torch::randn({1, 10}).requires_grad_(true);
  auto stack = createStack({inp, weight, bias});

  ProfilingGraphExecutorImpl executor(graph, "linear");

  // initial run to profile requires_grad information
  auto plan = executor.getPlanFor(stack, 20);
  InterpreterState is{plan.code};
  is.run(stack);

  auto optimized_plan = executor.getPlanFor(stack, 20);
  DepthFirstGraphNodeIterator it(optimized_plan.graph);
  Node* diff_graph_node = nullptr;

  while ((diff_graph_node = it.next()) != nullptr) {
    if (diff_graph_node->kind() == prim::DifferentiableGraph) {
      break;
    }
  }
  ASSERT_NE(nullptr, diff_graph_node);

  auto backward_graph = diff_graph_node->g(attr::ReverseSubgraph);

  // we expect to compute grad_weight (which requires a matmul) but we don't
  // expect to compute grad_input. So, we expect exactly 1 matmul.
  // Note: this could change, e.g. if mm is used instead
  testing::FileCheck().check_count("matmul", 1, true)->run(*backward_graph);
}

} // namespace jit
} // namespace torch
