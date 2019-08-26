#include <test/cpp/jit/test_utils.h>

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

std::pair<tensor_list, tensor_list> runGradient(
    Gradient& grad_spec,
    tensor_list& tensors_in,
    tensor_list& tensor_grads_in) {
  static const auto as_tensorlist = [](const Stack& stack) {
    return fmap(stack, [](const IValue& i) { return i.toTensor(); });
  };
  Code f_code{grad_spec.f}, df_code{grad_spec.df};
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

  // Outputs of f needs to be sliced
  f_stack.erase(f_stack.begin() + grad_spec.f_real_outputs, f_stack.end());
  return std::make_pair(as_tensorlist(f_stack), as_tensorlist(df_stack));
}

std::tuple<Var, Var> build_lstm_body(
    Graph& g,
    Var input,
    Var hx,
    Var cx,
    Var w_ih,
    Var w_hh) {
  auto gates = input.mm(w_ih);
  gates = gates + hx.mm(w_hh);
  auto outputs = gates.chunk(4, 1);
  auto ingate = outputs[0];
  auto forgetgate = outputs[1];
  auto cellgate = outputs[2];
  auto outgate = outputs[3];
  ingate = ingate.sigmoid();
  outgate = outgate.sigmoid();
  cellgate = cellgate.tanh();
  forgetgate = forgetgate.sigmoid();

  auto cy = forgetgate * cx;
  cy = cy + ingate * cellgate;
  auto hy = outgate * cy.tanh();

  return std::make_tuple(hy, cy);
}

std::shared_ptr<Graph> build_lstm() {
  auto r = std::make_shared<Graph>();
  auto& g = *r;
  Value* input = g.addInput();
  Value* hx = g.addInput();
  Value* cx = g.addInput();
  Value* w_ih = g.addInput();
  Value* w_hh = g.addInput();

  Var hy;
  Var cy;
  std::tie(hy, cy) = build_lstm_body(g, input, hx, cx, w_ih, w_hh);

  hy.addAsOutput();
  cy.addAsOutput();
  g.lint();

  return r;
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
