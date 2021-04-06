#include <gtest/gtest.h>

#include <test/cpp/tensorexpr/test_base.h>

#include <test/cpp/tensorexpr/test_utils.h>
#include <torch/csrc/jit/tensorexpr/eval.h>
#include <torch/csrc/jit/tensorexpr/ir.h>
#include <torch/csrc/jit/tensorexpr/ir_printer.h>
#include <torch/csrc/jit/tensorexpr/ir_simplifier.h>
#include <torch/csrc/jit/tensorexpr/llvm_codegen.h>
#include <torch/csrc/jit/tensorexpr/loopnest.h>
#include <torch/csrc/jit/tensorexpr/tensor.h>

#include <ATen/NativeFunctions.h>

namespace torch {
namespace jit {
using namespace torch::jit::tensorexpr;

TEST(ExternalCall, Conv2d_float) {
  KernelScope kernel_scope;

  Placeholder Input("Input", kFloat, {1, 3, 224, 224});
  Placeholder Weight("Weight", kFloat, {16, 3, 3, 3});
  Placeholder Bias("Bias", kFloat, {16});
  BufHandle ResultBuf("Result", {1, 16, 112, 112}, kFloat);
  int64_t stride = 2;
  int64_t pad = 1;
  int64_t dilation = 1;
  int64_t groups = 1;

  Tensor* Result = new Tensor(
      ResultBuf.node(),
      ExternalCall::make(
          ResultBuf,
          "nnc_aten_conv2d",
          {BufHandle(Input.data()),
           BufHandle(Weight.data()),
           BufHandle(Bias.data())},
          {stride, stride, pad, pad, dilation, dilation, groups}));
  LoopNest l({Result});
  l.prepareForCodegen();
  l.simplify();

  auto options = at::TensorOptions()
                     .dtype(at::kFloat)
                     .layout(at::kStrided)
                     .device(at::kCPU)
                     .requires_grad(false);
  at::Tensor input = at::ones({1, 3, 224, 224}, options) * 5.f;
  at::Tensor weight = at::ones({16, 3, 3, 3}, options) * 6.f;
  at::Tensor bias = at::ones({16}, options) * 11.f;
  at::Tensor ref = at::conv2d(
      input,
      weight,
      bias,
      {stride, stride},
      {pad, pad},
      {dilation, dilation},
      groups);

  at::Tensor nnc_result;
  std::vector<float> input_buf(1 * 3 * 224 * 224, 5.f);
  std::vector<float> weight_buf(16 * 3 * 3 * 3, 6.f);
  std::vector<float> bias_buf(16, 11.f);
  std::vector<float> result_buf(1 * 16 * 112 * 112, -1.f);

#ifdef TORCH_ENABLE_LLVM
  LLVMCodeGen llvm_codegen(l.root_stmt(), {Input, Weight, Bias, Result});

  llvm_codegen.call({input_buf, weight_buf, bias_buf, result_buf});
  nnc_result = at::from_blob(result_buf.data(), {1, 16, 112, 112}, options);
  ASSERT_TRUE(at::allclose(nnc_result, ref));
#endif

  SimpleIREvaluator ir_eval(l.root_stmt(), {Input, Weight, Bias, Result});

  ir_eval.call({input_buf, weight_buf, bias_buf, result_buf});
  nnc_result = at::from_blob(result_buf.data(), {1, 16, 112, 112}, options);
  ASSERT_TRUE(at::allclose(nnc_result, ref));
}

TEST(ExternalCall, Conv2d_int) {
  // A similar test, but now using kInt tensors
  KernelScope kernel_scope;

  Placeholder Input("Input", kInt, {1, 3, 224, 224});
  Placeholder Weight("Weight", kInt, {16, 3, 3, 3});
  Placeholder Bias("Bias", kInt, {16});
  BufHandle ResultBuf("Result", {1, 16, 112, 112}, kInt);
  int64_t stride = 2;
  int64_t pad = 1;
  int64_t dilation = 1;
  int64_t groups = 1;

  Tensor* Result = new Tensor(
      ResultBuf.node(),
      ExternalCall::make(
          ResultBuf,
          "nnc_aten_conv2d",
          {BufHandle(Input.data()),
           BufHandle(Weight.data()),
           BufHandle(Bias.data())},
          {stride, stride, pad, pad, dilation, dilation, groups}));
  LoopNest l({Result});
  l.prepareForCodegen();
  l.simplify();

  auto options = at::TensorOptions()
                     .dtype(at::kInt)
                     .layout(at::kStrided)
                     .device(at::kCPU)
                     .requires_grad(false);
  at::Tensor input = at::ones({1, 3, 224, 224}, options) * 5;
  at::Tensor weight = at::ones({16, 3, 3, 3}, options) * 6;
  at::Tensor bias = at::ones({16}, options) * 11;
  at::Tensor ref = at::conv2d(
      input,
      weight,
      bias,
      {stride, stride},
      {pad, pad},
      {dilation, dilation},
      groups);

  at::Tensor nnc_result;
  std::vector<int32_t> input_buf(1 * 3 * 224 * 224, 5);
  std::vector<int32_t> weight_buf(16 * 3 * 3 * 3, 6);
  std::vector<int32_t> bias_buf(16, 11);
  std::vector<int32_t> result_buf(1 * 16 * 112 * 112, -1);

#ifdef TORCH_ENABLE_LLVM
  LLVMCodeGen llvm_codegen(l.root_stmt(), {Input, Weight, Bias, Result});

  llvm_codegen.call({input_buf, weight_buf, bias_buf, result_buf});
  nnc_result = at::from_blob(result_buf.data(), {1, 16, 112, 112}, options);
  ASSERT_TRUE(at::allclose(nnc_result, ref));
#endif

  SimpleIREvaluator ir_eval(l.root_stmt(), {Input, Weight, Bias, Result});

  ir_eval.call({input_buf, weight_buf, bias_buf, result_buf});
  nnc_result = at::from_blob(result_buf.data(), {1, 16, 112, 112}, options);
  ASSERT_TRUE(at::allclose(nnc_result, ref));
}

TEST(ExternalCall, Conv2d_nobias_noargs) {
  KernelScope kernel_scope;

  Placeholder Input("Input", kFloat, {1, 16, 112, 112});
  Placeholder Weight("Weight", kFloat, {16, 16, 1, 1});
  BufHandle ResultBuf("Result", {1, 16, 112, 112}, kFloat);

  Tensor* Result = new Tensor(
      ResultBuf.node(),
      ExternalCall::make(
          ResultBuf,
          "nnc_aten_conv2d",
          {BufHandle(Input.data()), BufHandle(Weight.data())},
          {}));
  LoopNest l({Result});
  l.prepareForCodegen();
  l.simplify();

  auto options = at::TensorOptions()
                     .dtype(at::kFloat)
                     .layout(at::kStrided)
                     .device(at::kCPU)
                     .requires_grad(false);
  at::Tensor input = at::ones({1, 16, 112, 112}, options) * 5.f;
  at::Tensor weight = at::ones({16, 16, 1, 1}, options) * 6.f;
  at::Tensor ref = at::conv2d(input, weight);

  at::Tensor nnc_result;
  std::vector<float> input_buf(1 * 16 * 112 * 112, 5.f);
  std::vector<float> weight_buf(16 * 16 * 1 * 1, 6.f);
  std::vector<float> result_buf(1 * 16 * 112 * 112, -1.f);

#ifdef TORCH_ENABLE_LLVM
  LLVMCodeGen llvm_codegen(l.root_stmt(), {Input, Weight, Result});

  llvm_codegen.call({input_buf, weight_buf, result_buf});
  nnc_result = at::from_blob(result_buf.data(), {1, 16, 112, 112}, options);
  ASSERT_TRUE(at::allclose(nnc_result, ref));
#endif

  SimpleIREvaluator ir_eval(l.root_stmt(), {Input, Weight, Result});

  ir_eval.call({input_buf, weight_buf, result_buf});
  nnc_result = at::from_blob(result_buf.data(), {1, 16, 112, 112}, options);
  ASSERT_TRUE(at::allclose(nnc_result, ref));
}

TEST(ExternalCall, BinaryFloat) {
  KernelScope kernel_scope;
  using TensorFunc = std::function<at::Tensor(at::Tensor, at::Tensor)>;
  using Test = std::tuple<
      std::vector<int64_t>,
      std::vector<int64_t>,
      std::vector<int64_t>,
      TensorFunc,
      std::string>;
  std::vector<Test> tests = {};
  tests.push_back(
      Test{{100, 200}, {200, 300}, {100, 300}, at::matmul, "nnc_aten_matmul"});
  tests.push_back(Test{{100, 300}, {300}, {100}, at::mv, "nnc_aten_mv"});
  tests.push_back(
      Test{{100, 200}, {200, 300}, {100, 300}, at::mm, "nnc_aten_mm"});
  for (auto curTest : tests) {
    std::vector<int64_t> aShape, bShape, resShape;
    TensorFunc torchFunc;
    std::string externCallName;
    std::tie(aShape, bShape, resShape, torchFunc, externCallName) = curTest;
    auto toExprHandleVec = [](std::vector<int64_t> v) {
      auto intV = std::vector<int>(v.begin(), v.end());
      return std::vector<ExprHandle>(intV.begin(), intV.end());
    };
    Placeholder A("A", kFloat, toExprHandleVec(aShape));
    Placeholder B("", kFloat, toExprHandleVec(bShape));
    BufHandle ResultBuf("Result", toExprHandleVec(resShape), kFloat);

    Tensor* Result = new Tensor(
        ResultBuf.node(),
        ExternalCall::make(
            ResultBuf,
            externCallName,
            {BufHandle(A.data()), BufHandle(B.data())},
            {}));
    LoopNest l({Result});
    l.prepareForCodegen();
    l.simplify();

    auto options = at::TensorOptions()
                       .dtype(at::kFloat)
                       .layout(at::kStrided)
                       .device(at::kCPU)
                       .requires_grad(false);
    at::Tensor a = at::ones(c10::IntArrayRef(aShape), options) * 5.f;
    at::Tensor b = at::ones(c10::IntArrayRef(bShape), options) * 6.f;
    at::Tensor ref = torchFunc(a, b);

    auto prod = [](std::vector<int64_t> v) {
      return std::accumulate(v.begin(), v.end(), 1, std::multiplies<int64_t>());
    };

    at::Tensor nnc_result;
    std::vector<float> a_buf(prod(aShape), 5.f);
    std::vector<float> b_buf(prod(bShape), 6.f);
    std::vector<float> result_buf(prod(resShape), -1.f);

#ifdef TORCH_ENABLE_LLVM
    LLVMCodeGen llvm_codegen(l.root_stmt(), {A, B, Result});

    llvm_codegen.call({a_buf, b_buf, result_buf});
    nnc_result =
        at::from_blob(result_buf.data(), c10::IntArrayRef(resShape), options);
    ASSERT_TRUE(at::allclose(nnc_result, ref));
#endif

    SimpleIREvaluator ir_eval(l.root_stmt(), {A, B, Result});
    ir_eval.call({a_buf, b_buf, result_buf});
    nnc_result =
        at::from_blob(result_buf.data(), c10::IntArrayRef(resShape), options);
    ASSERT_TRUE(at::allclose(nnc_result, ref));
  }
}

TEST(ExternalCall, UnaryFloat) {
  KernelScope kernel_scope;
  using TensorFunc = std::function<at::Tensor(at::Tensor)>;
  auto toExprHandleVec = [](std::vector<int64_t> v) {
    auto intV = std::vector<int>(v.begin(), v.end());
    return std::vector<ExprHandle>(intV.begin(), intV.end());
  };
  using Test = std::tuple<
      std::vector<int64_t>,
      std::vector<int64_t>,
      TensorFunc,
      std::string,
      std::vector<ExprHandle>>;
  std::vector<Test> tests = {};
  tests.push_back(Test{
      {1, 64, 8, 9},
      {1, 64, 5, 7},
      [](at::Tensor x) {
        return at::adaptive_avg_pool2d(x, {5, 7});
      },
      "nnc_aten_adaptive_avg_pool2d",
      toExprHandleVec({5, 7})});
  tests.push_back(Test{
      {100, 200},
      {100},
      [](at::Tensor x) { return at::mean(x, {1}); },
      "nnc_aten_mean",
      toExprHandleVec({1})});
  for (auto curTest : tests) {
    std::vector<int64_t> aShape, resShape;
    TensorFunc torchFunc;
    std::string externCallName;
    std::vector<ExprHandle> externCallArgs;
    std::tie(aShape, resShape, torchFunc, externCallName, externCallArgs) =
        curTest;
    Placeholder A("A", kFloat, toExprHandleVec(aShape));
    BufHandle ResultBuf("Result", toExprHandleVec(resShape), kFloat);

    Tensor* Result = new Tensor(
        ResultBuf.node(),
        ExternalCall::make(
            ResultBuf, externCallName, {BufHandle(A.data())}, externCallArgs));
    LoopNest l({Result});
    l.prepareForCodegen();
    l.simplify();

    auto options = at::TensorOptions()
                       .dtype(at::kFloat)
                       .layout(at::kStrided)
                       .device(at::kCPU)
                       .requires_grad(false);
    at::Tensor a = at::ones(c10::IntArrayRef(aShape), options) * 5.f;
    at::Tensor ref = torchFunc(a);

    auto prod = [](std::vector<int64_t> v) {
      return std::accumulate(v.begin(), v.end(), 1, std::multiplies<int64_t>());
    };

    at::Tensor nnc_result;
    std::vector<float> a_buf(prod(aShape), 5.f);
    std::vector<float> result_buf(prod(resShape), -1.f);

#ifdef TORCH_ENABLE_LLVM
    LLVMCodeGen llvm_codegen(l.root_stmt(), {A, Result});

    llvm_codegen.call({a_buf, result_buf});
    nnc_result =
        at::from_blob(result_buf.data(), c10::IntArrayRef(resShape), options);
    ASSERT_TRUE(at::allclose(nnc_result, ref));
#endif

    SimpleIREvaluator ir_eval(l.root_stmt(), {A, Result});
    ir_eval.call({a_buf, result_buf});
    nnc_result =
        at::from_blob(result_buf.data(), c10::IntArrayRef(resShape), options);
    ASSERT_TRUE(at::allclose(nnc_result, ref));
  }
}

TEST(ExternalCall, ComputeInterop) {
  // This test verifies that Tensors using external calls can be used by and can
  // use Tensors built with Compute API.
  KernelScope kernel_scope;

  BufHandle ConvResultBuf("ConvResult", {1, 16, 112, 112}, kFloat);
  BufHandle MatmulResultBuf("MatmulResult", {1, 16, 112, 112}, kFloat);

  Tensor* Input = Compute(
      "Input",
      {{1, "n"}, {16, "c"}, {112, "h"}, {112, "w"}},
      [&](const VarHandle& n,
          const VarHandle& c,
          const VarHandle& h,
          const VarHandle& w) { return FloatImm::make(5.0f); });
  Tensor* Weight = Compute(
      "Weight",
      {{16, "n"}, {16, "c"}, {1, "kh"}, {1, "kw"}},
      [&](const VarHandle& n,
          const VarHandle& c,
          const VarHandle& h,
          const VarHandle& w) { return FloatImm::make(6.0f); });

  Tensor* ConvResult = new Tensor(
      ConvResultBuf.node(),
      ExternalCall::make(
          ConvResultBuf,
          "nnc_aten_conv2d",
          {BufHandle(Input->buf()), BufHandle(Weight->buf())},
          {}));
  Tensor* MatmulResult = new Tensor(
      MatmulResultBuf.node(),
      ExternalCall::make(
          MatmulResultBuf,
          "nnc_aten_matmul",
          {BufHandle(ConvResult->buf()), BufHandle(ConvResult->buf())},
          {}));
  Tensor* Result = Compute(
      "Result",
      {{1, "n"}, {16, "c"}, {112, "h"}, {112, "w"}},
      [&](const VarHandle& n,
          const VarHandle& c,
          const VarHandle& h,
          const VarHandle& w) {
        return ConvResult->call(n, c, h, w) + MatmulResult->call(n, c, h, w);
      });

  LoopNest l({Input, Weight, ConvResult, MatmulResult, Result});

  // Inlining should not inline anything here since all Bufs are either defined
  // or used in ExternalCalls - we run it just for testing
  l.inlineIntermediateBufs(true);

  l.prepareForCodegen();
  l.simplify();

  auto options = at::TensorOptions()
                     .dtype(at::kFloat)
                     .layout(at::kStrided)
                     .device(at::kCPU)
                     .requires_grad(false);
  at::Tensor input = at::ones({1, 16, 112, 112}, options) * 5.f;
  at::Tensor weight = at::ones({16, 16, 1, 1}, options) * 6.f;
  at::Tensor t = at::conv2d(input, weight);
  at::Tensor t2 = at::matmul(t, t);
  at::Tensor ref = t + t2;

  at::Tensor nnc_result;
  std::vector<float> input_buf(1 * 16 * 112 * 112, 5.f);
  std::vector<float> weight_buf(16 * 16 * 1 * 1, 6.f);
  std::vector<float> conv_result_buf(1 * 16 * 112 * 112, -1.f);
  std::vector<float> matmul_result_buf(1 * 16 * 112 * 112, -1.f);
  std::vector<float> result_buf(1 * 16 * 112 * 112, -1.f);

#ifdef TORCH_ENABLE_LLVM
  LLVMCodeGen llvm_codegen(
      l.root_stmt(), {Input, Weight, ConvResult, MatmulResult, Result});

  llvm_codegen.call(
      {input_buf, weight_buf, conv_result_buf, matmul_result_buf, result_buf});
  nnc_result = at::from_blob(result_buf.data(), {1, 16, 112, 112}, options);
  ASSERT_TRUE(at::allclose(nnc_result, ref));
#endif

  SimpleIREvaluator ir_eval(
      l.root_stmt(), {Input, Weight, ConvResult, MatmulResult, Result});

  ir_eval.call(
      {input_buf, weight_buf, conv_result_buf, matmul_result_buf, result_buf});
  nnc_result = at::from_blob(result_buf.data(), {1, 16, 112, 112}, options);
  ASSERT_TRUE(at::allclose(nnc_result, ref));
}

} // namespace jit
} // namespace torch
