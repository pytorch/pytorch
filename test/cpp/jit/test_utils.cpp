#include <test/cpp/jit/test_utils.h>
#include <torch/csrc/jit/jit_log.h>
#include <torch/csrc/jit/passes/clear_undefinedness.h>

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

static void unpackReturnTuple(Stack &stack) {
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

} // namespace jit
} // namespace torch
