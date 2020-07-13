#include "test/cpp/jit/test_base.h"
#include "test/cpp/jit/test_utils.h"

namespace torch {
namespace jit {

void testInterp() {
  {
    auto graph = std::make_shared<Graph>();
    std::unordered_map<std::string, Value*> vmap;
    parseIR(
        R"IR(
graph(%a.1 : Tensor,
      %b.1 : Tensor):
  %t : Float(2:2, 2:1), %type_matched : bool = prim::TypeCheck(%a.1)
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
    bool cond = false;
    auto a = at::zeros({2, 2});
    auto b = at::ones({2, 2});
    InterpreterState print_interp(print_function);
    std::vector<IValue> stack({a, b});
    print_interp.run(stack);
    // TODO: Check something, not crashing is not good enough
  }
  {
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

    // std::cout << almostEqual(outputs[0],hx) << "\n";
    ASSERT_TRUE(exactlyEqual(outputs[0], hx));
    ASSERT_TRUE(exactlyEqual(outputs[1], cx));
  }
}
} // namespace jit
} // namespace torch
