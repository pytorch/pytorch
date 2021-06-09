#include <gtest/gtest.h>

#include <test/cpp/jit/test_utils.h>
#include <torch/csrc/jit/jit_log.h>
#include <torch/csrc/jit/passes/clear_undefinedness.h>
#include <torch/csrc/jit/runtime/custom_operator.h>

namespace torch {
namespace jit {

Stack createStack(std::vector<at::Tensor>&& list) {
  return Stack(
      std::make_move_iterator(list.begin()),
      std::make_move_iterator(list.end()));
}

void assertAllClose(const tensor_list& a, const tensor_list& b) {
  ASSERT_EQ(a.size(), b.size());
  for (size_t i = 0; i < a.size(); ++i) {
    ASSERT_TRUE(a[i].is_same_size(b[i]));
    ASSERT_TRUE(a[i].allclose(b[i]));
  }
}

std::vector<at::Tensor> run(
    InterpreterState& interp,
    const std::vector<at::Tensor>& inputs) {
  std::vector<IValue> stack(inputs.begin(), inputs.end());
  interp.run(stack);
  return fmap(stack, [](const IValue& i) { return i.toTensor(); });
}

static void unpackReturnTuple(Stack& stack) {
  auto tuple = pop(stack).toTuple();
  stack.insert(stack.end(), tuple->elements().begin(), tuple->elements().end());
}

std::pair<tensor_list, tensor_list> runGradient(
    Gradient& grad_spec,
    tensor_list& tensors_in,
    tensor_list& tensor_grads_in) {
  static const auto as_tensorlist = [](const Stack& stack) {
    return fmap(stack, [](const IValue& i) { return i.toTensor(); });
  };
  ClearUndefinedness(grad_spec.df);
  Code f_code{grad_spec.f, ""}, df_code{grad_spec.df, ""};
  InterpreterState f_interpreter{f_code}, df_interpreter{df_code};

  auto f_stack = fmap<IValue>(tensors_in);
  f_interpreter.run(f_stack);

  Stack df_stack;
  df_stack.insert(
      df_stack.end(), tensor_grads_in.begin(), tensor_grads_in.end());
  for (auto offset : grad_spec.df_input_captured_inputs)
    df_stack.push_back(tensors_in[offset]);
  for (auto offset : grad_spec.df_input_captured_outputs)
    df_stack.push_back(f_stack[offset]);
  df_interpreter.run(df_stack);
  unpackReturnTuple(df_stack);
  // Outputs of f needs to be sliced
  f_stack.erase(f_stack.begin() + grad_spec.f_real_outputs, f_stack.end());
  return std::make_pair(as_tensorlist(f_stack), as_tensorlist(df_stack));
}

std::shared_ptr<Graph> build_lstm() {
  const auto graph_string = R"IR(
    graph(%0 : Tensor,
          %1 : Tensor,
          %2 : Tensor,
          %3 : Tensor,
          %4 : Tensor):
      %5 : Tensor = aten::mm(%0, %3)
      %6 : Tensor = aten::mm(%1, %4)
      %7 : int = prim::Constant[value=1]()
      %8 : Tensor = aten::add(%5, %6, %7)
      %9 : Tensor, %10 : Tensor, %11 : Tensor, %12 : Tensor = prim::ConstantChunk[chunks=4, dim=1](%8)
      %13 : Tensor = aten::sigmoid(%9)
      %14 : Tensor = aten::sigmoid(%12)
      %15 : Tensor = aten::tanh(%11)
      %16 : Tensor = aten::sigmoid(%10)
      %17 : Tensor = aten::mul(%16, %2)
      %18 : Tensor = aten::mul(%13, %15)
      %19 : int = prim::Constant[value=1]()
      %20 : Tensor = aten::add(%17, %18, %19)
      %21 : Tensor = aten::tanh(%20)
      %22 : Tensor = aten::mul(%14, %21)
      return (%22, %20))IR";
  auto g = std::make_shared<Graph>();
  torch::jit::parseIR(graph_string, g.get());
  g->lint();

  return g;
}

std::shared_ptr<Graph> build_mobile_export_analysis_graph() {
  // We use following two schemas for this graph:
  //   1. slice.Tensor(Tensor(a) self, int dim=0, int? start=0,
  //                   int? end=9223372036854775807, int step=1) -> Tensor(a)
  //   2. slice.str(str string, int? start=0, int? end=9223372036854775807,
  //                  int step=1) -> str
  // %3 and %4 use slice.Tensor while %5 use slice.str.
  // Since we can see %3 and %4 have the same last argument that is never used
  // (same as default value of schema), we know we can ignore that last arg. For
  // %5, we see that last three args are same as schema default, hence
  // unnecessary.

  const auto graph_string = R"IR(
    graph(%0 : Tensor):
      %1 : int = prim::Constant[value=1]()
      %2 : int = prim::Constant[value=2]()
      %20 : int = prim::Constant[value=0]()
      %21 : int = prim::Constant[value=9223372036854775807]()
      %22 : str = prim::Constant[value="value"]()
      %3 : Tensor  = aten::slice(%0, %1, %20, %2, %1)
      %4 : Tensor = aten::slice(%0, %2, %20, %21, %1)
      %5 : str = aten::slice(%22, %20, %21, %1)
      return (%3, %4, %5))IR";

  auto g = std::make_shared<Graph>();
  torch::jit::parseIR(graph_string, g.get());
  g->lint();
  return g;
}

std::shared_ptr<Graph> build_mobile_export_analysis_graph_nested() {
  // this is pretty much same test as build_mobile_export_analysis_graph(),
  // but some aten::slice operators are hidden under block statement to check
  // if we are correctly recursing all the nodes in graph.
  const auto graph_string = R"IR(
    graph(%0 : Tensor):
      %1 : int = prim::Constant[value=1]()
      %2 : int = prim::Constant[value=2]()
      %20 : int = prim::Constant[value=0]()
      %21 : int = prim::Constant[value=9223372036854775807]()
      %22 : str = prim::Constant[value="value"]()
      %3 : Tensor  = aten::slice(%0, %1, %20, %2, %1)
      %23 : bool = aten::Bool(%3)
      %c : Tensor = prim::If(%23)
        block0():
          %4 : Tensor = aten::slice(%0, %2, %20, %21, %1)
          %5 : str = aten::slice(%22, %20, %21, %1)
          %c.1 : Tensor = aten::slice(%0, %1, %20, %2, %1)
          -> (%c.1)
        block1():
          -> (%3)
      return (%3, %3))IR";
  auto g = std::make_shared<Graph>();
  torch::jit::parseIR(graph_string, g.get());
  g->lint();
  return g;
}

std::shared_ptr<Graph> build_mobile_export_analysis_graph_with_vararg() {
  const auto graph_string = R"IR(
    graph(%0 : Tensor):
      %1 : int = prim::Constant[value=1]()
      %2 : int = prim::Constant[value=2]()
      %3 : int = prim::Constant[value=3]()
      %4 : int[]  = prim::tolist(%1, %2)
      %5 : int[] = prim::tolist(%1, %2, %3)
      return (%4, %5))IR";

  auto g = std::make_shared<Graph>();
  torch::jit::parseIR(graph_string, g.get());
  g->lint();
  return g;
}

std::shared_ptr<Graph> build_mobile_export_analysis_graph_non_const() {
  const auto graph_string = R"IR(
      graph(%input.1 : Tensor):
        %7 : int = prim::Constant[value=1]() # <string>:3:58
        %9 : int = prim::Constant[value=0]() # <string>:3:66
        %8 : int[] = prim::ListConstruct(%7, %7)
        %10 : int[] = prim::ListConstruct(%9, %9)
        %11 : int[] = prim::ListConstruct(%7, %7)
        %12 : Tensor = aten::conv2d(%input.1, %input.1, %input.1, %8, %10, %11, %7)
        return (%12))IR";

  auto g = std::make_shared<Graph>();
  torch::jit::parseIR(graph_string, g.get());
  g->lint();
  return g;
}

at::Tensor t_use(at::Tensor x) {
  return x;
}
at::Tensor t_def(at::Tensor x) {
  return x.t();
}

bool checkRtol(const at::Tensor& diff, const std::vector<at::Tensor> inputs) {
  double maxValue = 0.0;
  for (auto& tensor : inputs) {
    maxValue = fmax(tensor.abs().max().item<float>(), maxValue);
  }
  return diff.abs().max().item<float>() < 2e-6 * maxValue;
}
bool almostEqual(const at::Tensor& a, const at::Tensor& b) {
  return checkRtol(a - b, {a, b});
}

bool exactlyEqual(const at::Tensor& a, const at::Tensor& b) {
  return (a - b).abs().max().item<float>() == 0.f;
}

std::pair<at::Tensor, at::Tensor> lstm(
    at::Tensor input,
    at::Tensor hx,
    at::Tensor cx,
    at::Tensor w_ih,
    at::Tensor w_hh) {
  auto gates = input.mm(t_use(w_ih)) + hx.mm(t_use(w_hh));

  auto chunked_gates = gates.chunk(4, 1);
  auto ingate = chunked_gates[0];
  auto forgetgate = chunked_gates[1];
  auto cellgate = chunked_gates[2];
  auto outgate = chunked_gates[3];

  ingate = ingate.sigmoid();
  outgate = outgate.sigmoid();
  cellgate = cellgate.tanh();
  forgetgate = forgetgate.sigmoid();

  auto cy = (forgetgate * cx) + (ingate * cellgate);
  auto hy = outgate * cy.tanh();

  return {hy, cy};
}

inline c10::AliasAnalysisKind aliasAnalysisFromSchema() {
  return c10::AliasAnalysisKind::FROM_SCHEMA;
}

namespace {
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
RegisterOperators reg({
    // This operator is intended to be used in JIT analysis and transformation
    // pass unit tests in which Values with type Tensor are often required. It
    // should not be used in situations in which the graph is actually executed
    // because it always produces empty Tensors.
    Operator(
        "prim::MakeTestTensor() -> Tensor",
        [](Stack* stack) { push(stack, at::Tensor()); },
        aliasAnalysisFromSchema()),
});
} // namespace

} // namespace jit
} // namespace torch
