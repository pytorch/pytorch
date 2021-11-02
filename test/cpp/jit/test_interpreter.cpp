#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <ATen/Parallel.h>
#include <c10/core/DeviceType.h>
#include <test/cpp/jit/test_utils.h>
#include <torch/csrc/jit/runtime/instruction.h>
#include <torch/jit.h>
#include <torch/script.h>
#include <torch/torch.h>

namespace torch {
namespace jit {

class TypeCheckTest : public ::testing::Test {
 protected:
  TypeCheckTest() : interp(makeInterp()) {}

  // NOLINTNEXTLINE(cppcoreguidelines-non-private-member-variables-in-classes)
  InterpreterState interp;

 private:
  static InterpreterState makeInterp() {
    auto graph = std::make_shared<Graph>();
    std::unordered_map<std::string, Value*> vmap;
    parseIR(
        R"IR(
graph(%a.1 : Tensor,
      %b.1 : Tensor):
  %t0 : Float(2, 2, strides=[2, 1], device=cpu, requires_grad=1), %t1 : Float(3, 3, strides=[3, 1]), %type_matched : bool = prim::TypeCheck[types=[Float(2, 2, strides=[2, 1], device=cpu, requires_grad=1), Float(3, 3, strides=[3, 1])]](%a.1, %b.1)
  return (%t0, %t1, %type_matched)
  )IR",
        &*graph,
        vmap);

    Code function(graph, "");
    return InterpreterState(function);
  }
};

TEST_F(TypeCheckTest, MatchingType) {
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

TEST_F(TypeCheckTest, SizeMismatch) {
  auto a = at::zeros({2, 2}, at::kFloat);
  auto b = at::ones({2, 2}, at::kFloat); // Size mismatch
  a.set_requires_grad(true);
  a = a.to(at::kCPU);
  std::vector<IValue> stack({a, b});
  interp.run(stack);
  ASSERT_FALSE(stack[2].toBool());
}

TEST_F(TypeCheckTest, GradientMismatch) {
  auto a = at::zeros({2, 2}, at::kFloat);
  auto b = at::ones({3, 3}, at::kFloat);
  a = a.to(at::kCPU);
  a.set_requires_grad(false); // Gradient mismatch
  std::vector<IValue> stack({a, b});
  interp.run(stack);
  ASSERT_FALSE(stack[2].toBool());
}

TEST_F(TypeCheckTest, ScalarTypeMismatch) {
  auto a = at::zeros({2, 2}, at::kFloat);
  auto b = at::ones({3, 3}, at::kFloat);
  a = a.to(at::kCPU);
  a.set_requires_grad(true);
  a = a.to(at::kInt); // Scalar type mismatch
  std::vector<IValue> stack({a, b});
  interp.run(stack);
  ASSERT_FALSE(stack[2].toBool());
}

TEST_F(TypeCheckTest, DeviceMismatch_CUDA) {
  auto a = at::zeros({2, 2}, at::kFloat);
  auto b = at::ones({3, 3}, at::kFloat);
  a.set_requires_grad(true);
  a = a.to(at::kCUDA); // Device mismatch
  std::vector<IValue> stack({a, b});
  interp.run(stack);
  ASSERT_FALSE(stack[2].toBool());
}

// TODO: These tests weren't doing anything.
// TEST(TypeCheckErrorTest, EmptyCheckRaises) {
//   // Test empty Typecheck raises an internal assertion
//   auto graph = std::make_shared<Graph>();
//   std::unordered_map<std::string, Value*> vmap;
//   EXPECT_ANY_THROW(parseIR(
//       R"IR(
// graph(%a.1 : Tensor,
//       %b.1 : Tensor):
//   %type_matched : bool = prim::TypeCheck()
//   return (%type_matched)
//   )IR",
//       &*graph,
//       vmap));
// }

// TODO: These tests weren't doing anything.
// TEST(TypeCheckErrorTest, WrongInputOutputCountRaises) {
//   // Test for assertion if num_inputs + 1 != num_outputs
//   auto graph = std::make_shared<Graph>();
//   std::unordered_map<std::string, Value*> vmap;
//   EXPECT_ANY_THROW(parseIR(
//       R"IR(
// graph(%a.1 : Tensor,
//       %b.1 : Tensor):
//   %type_matched : bool = prim::TypeCheck(%a.1)
//   return (%type_matched)
//   )IR",
//       &*graph,
//       vmap));
// }

TEST(InterpreterTest, Basic_CUDA) {
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

TEST(InterpreterTest, IgnorableArgsInSchema) {
  auto graph = build_mobile_export_analysis_graph();
  MobileCode function(graph, "");
  auto op_to_specified_args = function.op_to_num_specified_args();
  ASSERT_TRUE(op_to_specified_args.size() == 2);
  ASSERT_TRUE(op_to_specified_args["aten::slice.Tensor"] == 4);
  ASSERT_TRUE(op_to_specified_args["aten::slice.str"] == 4);
  auto graph_vararg = build_mobile_export_analysis_graph_with_vararg();
  MobileCode function_vararg(graph_vararg, "");
  auto op_to_specified_args_vararg = function_vararg.op_to_num_specified_args();
  // should never register it
  ASSERT_TRUE(
      op_to_specified_args_vararg.find("prim::tolist") ==
      op_to_specified_args_vararg.end());

  auto graph_nested = build_mobile_export_analysis_graph_nested();
  MobileCode function_nested(graph_nested, "");
  auto op_to_specified_args_nested = function_nested.op_to_num_specified_args();
  ASSERT_TRUE(op_to_specified_args_nested["aten::slice.Tensor"] == 4);
  ASSERT_TRUE(op_to_specified_args_nested["aten::slice.str"] == 4);

  auto graph_non_const = build_mobile_export_analysis_graph_non_const();
  MobileCode function_non_const(graph_non_const, "");
  auto op_to_specified_args_non_const =
      function_non_const.op_to_num_specified_args();
  ASSERT_TRUE(op_to_specified_args_non_const["aten::conv2d"] == 6);
}

TEST(InterpreterTest, IgnorableArgsInSchemaWithOut) {
  auto graph = build_mobile_export_with_out();
  MobileCode function(graph, "");
  auto op_to_specified_args = function.op_to_num_specified_args();
  ASSERT_TRUE(op_to_specified_args.size() == 1);
  // this should be 3 when the add_out flag is set to True
  ASSERT_TRUE(op_to_specified_args["aten::add.out"] == 3);
}

TEST(InterpreterTest, runAsyncBasicTest) {
  /*
  TODO: there are some problem with C++ parsing script program involving
  fork. Use the test module below for now.
  issue about this: github.com/pytorch/pytorch/issues/46368
  The test module file is generated by following:
    class DemoModule(torch.nn.Module):
      def forward(self):
        r1 = torch.jit.fork(torch.mm, torch.rand(100,100),torch.rand(100,100))
        r2 = torch.jit.fork(torch.mm, torch.rand(100,100),torch.rand(100,100))
        return r1.wait() + r2.wait()
  demo = DemoModule()
  torch.jit.save(torch.jit.script(demo), 'test_interpreter_async.pt')
  */
  std::string filePath(__FILE__);
  auto testModelFile = filePath.substr(0, filePath.find_last_of("/\\") + 1);
  testModelFile.append("test_interpreter_async.pt");
  auto model = load(testModelFile);
  auto graph = model.get_method("forward").graph();
  Code function(graph, "");
  auto asyncCounter = 0;
  std::mutex mtx;
  // a dummy executor which actually use at::launch, but add up a counter
  auto launcher = [&](std::function<void()> f) {
    mtx.lock();
    ++asyncCounter;
    mtx.unlock();
    at::launch(f);
  };
  std::vector<IValue> stack;
  // NOLINTNEXTLINE(modernize-use-emplace)
  stack.push_back(model._ivalue());
  InterpreterState interp(function, launcher);
  interp.runAsync(stack)->wait();
  ASSERT_TRUE(asyncCounter > 0);
}

TEST(
    EnableRethrowCaughtExceptionTest,
    EnableRethrowCaughtExceptionTestRethrowsCaughtException) {
  auto graph = std::make_shared<Graph>();
  std::unordered_map<std::string, Value*> vmap;
  parseIR(
      R"IR(
graph(%0 : Tensor,
      %1 : Tensor):
  %2 : int = prim::Constant[value=2]()
  %3 : Tensor = aten::add(%0, %1, %2)
  return (%3)
  )IR",
      &*graph,
      vmap);
  Code function(graph, "");
  InterpreterState interp = InterpreterState(function);
  auto a = at::zeros({2, 2}, at::kFloat);
  auto b = at::ones({2, 3}, at::kFloat);
  a.set_requires_grad(true);
  a = a.to(at::kCPU);
  std::vector<IValue> stack({a, b});

  bool original_flag_value = FLAGS_torch_jit_enable_rethrow_caught_exception;
  bool exception_handled = false;
  try {
    FLAGS_torch_jit_enable_rethrow_caught_exception = false;
    interp.run(stack);
  } catch (std::runtime_error& e) {
    exception_handled = true;
    std::string exception_msg = e.what();
    EXPECT_THAT(
        exception_msg,
        ::testing::HasSubstr("%3 : Tensor = aten::add(%0, %1, %2)"));
    EXPECT_THAT(
        exception_msg,
        ::testing::HasSubstr(
            "The size of tensor a (2) must match the size of tensor b (3) at non-singleton dimension 1"));
  }
  EXPECT_TRUE(exception_handled);

  exception_handled = false;
  try {
    FLAGS_torch_jit_enable_rethrow_caught_exception = true;
    interp.run(stack);
  } catch (c10::Error& e) {
    exception_handled = true;
    std::string exception_msg = e.what_without_backtrace();
    EXPECT_STREQ(
        exception_msg.c_str(),
        "The size of tensor a (2) must match the size of tensor b (3) at non-singleton dimension 1");
  }
  EXPECT_TRUE(exception_handled);

  FLAGS_torch_jit_enable_rethrow_caught_exception = true;
  c10::intrusive_ptr<Future> future = interp.runAsync(stack);
  future->wait();
  ASSERT_TRUE(future->completed());
  ASSERT_TRUE(future->hasError());
  try {
    std::rethrow_exception(future->exception_ptr());
  } catch (c10::Error& e) {
    std::string exception_msg = e.what_without_backtrace();
    EXPECT_STREQ(
        exception_msg.c_str(),
        "The size of tensor a (2) must match the size of tensor b (3) at non-singleton dimension 1");
  }

  FLAGS_torch_jit_enable_rethrow_caught_exception = original_flag_value;
}

} // namespace jit
} // namespace torch
