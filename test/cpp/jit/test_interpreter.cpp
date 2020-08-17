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
  %t : Float(2:2, 2:1, device=cuda, requires_grad=1), %t0 : Float(3:3, 3:1), %type_matched : bool = prim::TypeCheck(%a.1, %b.1)
  %14 : Tensor = prim::If(%type_matched)
    block0():
      -> (%a.1)
    block1():
      -> (%b.1)
  return (%14)
  )IR",
        &*graph,
        vmap);

    Code print_function(graph, "");

    auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 1);
    auto a = at::zeros({2, 2}, options);
    auto b = at::ones({3, 3});
    auto a2 = at::ones({2, 2});
    auto a3 = at::ones({2, 2}); //.idtype(at::kInt);
    a3 = a3.to(at::kInt);

    a.set_requires_grad(true);
    a2.set_requires_grad(false);

    {
      // TypeCheck yields to true! Tensor a is returned
      InterpreterState print_interp(print_function);
      std::vector<IValue> stack({a, b});
      print_interp.run(stack);
      ASSERT_TRUE(exactlyEqual(stack[0].toTensor(), a));
    }
    {
      // TypeCheck yields to false because of size mismatch expected tensor
      // size (3,3) got (2,2). Tensor a2 is returned
      InterpreterState print_interp(print_function);
      std::vector<IValue> stack({a, a2});
      print_interp.run(stack);
      ASSERT_TRUE(exactlyEqual(stack[0].toTensor(), a2));
    }
    {
      // TypeCheck yields to false because of requires_grad mismatch.
      //  Tensor b is returned
      // FIXME: gradient is not checked! a2 is returned instead of b
      InterpreterState print_interp(print_function);
      std::vector<IValue> stack({a2, b});
      print_interp.run(stack);
      ASSERT_TRUE(exactlyEqual(stack[0].toTensor(), a2));
    }
    {
      // TypeCheck yields to false because of type mismatch: got
      // Int expected Float. Tensor b is returned
      // FIXME: type is not checked! a3 is returned instead of b
      InterpreterState print_interp(print_function);
      std::vector<IValue> stack({a3, b});
      print_interp.run(stack);
      ASSERT_TRUE(exactlyEqual(stack[0].toTensor(), a3));
    }
    {
      // TypeCheck yields to false because of device mismatch.
      // Tensor b is returned
      // FIXME: device is not checked! a is returned instead of b
      InterpreterState print_interp(print_function);
      a = a.to(at::kCPU);
      std::vector<IValue> stack({a, b});
      print_interp.run(stack);
      ASSERT_TRUE(exactlyEqual(stack[0].toTensor(), a));
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
  %14 : Tensor = prim::If(%type_matched)
    block0():
      -> (%a.1)
    block1():
      -> (%b.1)
  return (%14)
  )IR",
        &*graph,
        vmap);
    Code print_function(graph, "");
    auto a = at::zeros({2, 2});
    auto b = at::ones({3, 3});

    InterpreterState print_interp(print_function);
    std::vector<IValue> stack({a, b});
    print_interp.run(stack);
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
