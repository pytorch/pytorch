#include "test/cpp/jit/test_base.h"
#include "test/cpp/jit/test_utils.h"

#include <stdexcept>
namespace torch {
namespace jit {

void testTypeCheck() {
  {
    auto graph = std::make_shared<Graph>();
    std::unordered_map<std::string, Value*> vmap;
    parseIR(
        R"IR(
graph(%a.1 : Tensor,
      %b.1 : Tensor):
  %t0 : Float(2:2, 2:1, device=cpu, requires_grad=1), %t1 : Float(3:3, 3:1), %type_matched : bool = prim::TypeCheck(%a.1, %b.1)
  return (%t0, %t1, %type_matched)
  )IR",
        &*graph,
        vmap);

    Code function(graph, "");
    InterpreterState interp(function);
    {
      // TypeCheck yields to true! Shape, grad and device matches.
      auto a = at::zeros({2, 2}, at::kFloat);
      auto b = at::ones({3, 3}, at::kFloat);
      a.set_requires_grad(true);
      a = a.to(at::kCPU);
      std::vector<IValue> stack({a, b});
      interp.run(stack);
      ASSERT_TRUE(exactlyEqual(stack[0].toTensor(), a));
      ASSERT_TRUE(exactlyEqual(stack[1].toTensor(), b));
      ASSERT_TRUE(stack[2].toBool());
    }
    {
      auto a = at::zeros({2, 2}, at::kFloat);
      auto b = at::ones({2, 2}, at::kFloat); // Size mismatch
      a.set_requires_grad(true);
      a = a.to(at::kCPU);
      std::vector<IValue> stack({a, b});
      interp.run(stack);
      ASSERT_FALSE(stack[2].toBool());
    }
    {
      auto a = at::zeros({2, 2}, at::kFloat);
      auto b = at::ones({3, 3}, at::kFloat);
      a = a.to(at::kCPU);
      a.set_requires_grad(false); // Gradient mismatch
      std::vector<IValue> stack({a, b});
      interp.run(stack);
      ASSERT_FALSE(stack[2].toBool());
    }
    {
      auto a = at::zeros({2, 2}, at::kFloat);
      auto b = at::ones({3, 3}, at::kFloat);
      a = a.to(at::kCPU);
      a.set_requires_grad(true);
      a = a.to(at::kInt); // Scalar type mismatch
      std::vector<IValue> stack({a, b});
      interp.run(stack);
      ASSERT_FALSE(stack[2].toBool());
    }
    {
      auto a = at::zeros({2, 2}, at::kFloat);
      auto b = at::ones({3, 3}, at::kFloat);
      a.set_requires_grad(true);
      a = a.to(at::kCUDA); // Device mismatch
      std::vector<IValue> stack({a, b});
      interp.run(stack);
      ASSERT_FALSE(stack[2].toBool());
    }
  }

  try { // Test empty Typecheck raises an internal assertion
    auto graph = std::make_shared<Graph>();
    std::unordered_map<std::string, Value*> vmap;
    parseIR(
        R"IR(
graph(%a.1 : Tensor,
      %b.1 : Tensor):
  %type_matched : bool = prim::TypeCheck()
  return (%type_matched)
  )IR",
        &*graph,
        vmap);
    ASSERT_TRUE(false);
  } catch (const std::exception& e) {
  }
  try { // Test for assertion if num_inputs + 1 != num_outputs
    auto graph = std::make_shared<Graph>();
    std::unordered_map<std::string, Value*> vmap;
    parseIR(
        R"IR(
graph(%a.1 : Tensor,
      %b.1 : Tensor):
  %type_matched : bool = prim::TypeCheck(%a.1)
  return (%type_matched)
  )IR",
        &*graph,
        vmap);
    ASSERT_TRUE(false);
  } catch (const std::exception& e) {
  }
}
void testInterp() {
  constexpr int batch_size = 4;
  constexpr int input_size = 256;
  constexpr int seq_len = 32;

  int hidden_size = 2 * input_size;

  auto input = at::randn({seq_len, batch_size, input_size}, at::kCUDA);
  auto hx = at::randn({batch_size, hidden_size}, at::kCUDA);
  auto cx = at::randn({batch_size, hidden_size}, at::kCUDA);
  auto w_ih = t_def(at::randn({4 * hidden_size, input_size}, at::kCUDA));
  auto w_hh = t_def(at::randn({4 * hidden_size, hidden_size}, at::kCUDA));

  auto lstm_g = build_lstm();
  Code lstm_function(lstm_g, "");
  InterpreterState lstm_interp(lstm_function);
  auto outputs = run(lstm_interp, {input[0], hx, cx, w_ih, w_hh});
  std::tie(hx, cx) = lstm(input[0], hx, cx, w_ih, w_hh);

  ASSERT_TRUE(exactlyEqual(outputs[0], hx));
  ASSERT_TRUE(exactlyEqual(outputs[1], cx));
}
} // namespace jit
} // namespace torch
