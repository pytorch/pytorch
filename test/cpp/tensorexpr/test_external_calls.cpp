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

TEST(ExternalCall, Matmul) {
  KernelScope kernel_scope;
  Placeholder A("A", kFloat, {10, 3, 100, 200});
  Placeholder B("", kFloat, {10, 3, 200, 300});
  BufHandle ResultBuf("Result", {10, 3, 100, 300}, kFloat);

  Tensor* Result = new Tensor(
      ResultBuf.node(),
      ExternalCall::make(
          ResultBuf,
          "nnc_aten_matmul",
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
  at::Tensor a = at::ones({10, 3, 100, 200}, options) * 5.f;
  at::Tensor b = at::ones({10, 3, 200, 300}, options) * 6.f;
  at::Tensor ref = at::matmul(a, b);

  at::Tensor nnc_result;
  std::vector<float> a_buf(10 * 3 * 100 * 200, 5.f);
  std::vector<float> b_buf(10 * 3 * 200 * 300, 6.f);
  std::vector<float> result_buf(10 * 3 * 100 * 300, -1.f);

#ifdef TORCH_ENABLE_LLVM
  LLVMCodeGen llvm_codegen(l.root_stmt(), {A, B, Result});

  llvm_codegen.call({a_buf, b_buf, result_buf});
  nnc_result = at::from_blob(result_buf.data(), {10, 3, 100, 300}, options);
  ASSERT_TRUE(at::allclose(nnc_result, ref));
#endif

  SimpleIREvaluator ir_eval(l.root_stmt(), {A, B, Result});

  ir_eval.call({a_buf, b_buf, result_buf});
  nnc_result = at::from_blob(result_buf.data(), {10, 3, 100, 300}, options);
  ASSERT_TRUE(at::allclose(nnc_result, ref));
}

} // namespace jit
} // namespace torch
