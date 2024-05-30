#include <gtest/gtest.h>

#include <test/cpp/tensorexpr/test_base.h>

#include <torch/csrc/jit/ir/irparser.h>
#include <torch/csrc/jit/passes/subgraph_rewrite.h>
#include <torch/csrc/jit/passes/tensorexpr_fuser.h>
#include <torch/csrc/jit/runtime/custom_operator.h>
#include <torch/csrc/jit/tensorexpr/kernel.h>

#include <test/cpp/tensorexpr/test_utils.h>
#include <torch/csrc/jit/runtime/operator.h>
#include <torch/csrc/jit/runtime/symbolic_shape_registry.h>
#include <torch/csrc/jit/tensorexpr/eval.h>
#include <torch/csrc/jit/tensorexpr/external_functions_registry.h>
#include <torch/csrc/jit/tensorexpr/ir.h>
#include <torch/csrc/jit/tensorexpr/ir_printer.h>
#include <torch/csrc/jit/tensorexpr/ir_simplifier.h>
#include <torch/csrc/jit/tensorexpr/llvm_codegen.h>
#include <torch/csrc/jit/tensorexpr/loopnest.h>
#include <torch/csrc/jit/tensorexpr/tensor.h>

#include <torch/csrc/jit/testing/file_check.h>
#include <torch/jit.h>

#include <ATen/NativeFunctions.h>
#include <ATen/core/dispatch/Dispatcher.h>
#include <ATen/native/xnnpack/OpContext.h>

namespace torch {
namespace jit {
using namespace torch::jit::tensorexpr;

TEST(ExternalCall, Conv1d_float) {
  BufHandle Input("Input", {1, 100, 115}, kFloat);
  BufHandle Weight("Weight", {100, 1, 7}, kFloat);
  BufHandle Bias("Bias", {100}, kFloat);
  BufHandle ResultBuf("Result", {1, 100, 115}, kFloat);
  int64_t stride = 1;
  int64_t pad = 3;
  int64_t dilation = 1;
  int64_t groups = 100;

  Tensor Result = Tensor(
      ResultBuf.node(),
      ExternalCall::make(
          ResultBuf,
          "nnc_aten_conv1d",
          {Input, Weight, Bias},
          {stride, pad, dilation, groups}));
  LoopNest l({Result});
  l.prepareForCodegen();
  l.simplify();

  auto options = at::TensorOptions()
                     .dtype(at::kFloat)
                     .layout(at::kStrided)
                     .device(at::kCPU)
                     .requires_grad(false);
  at::Tensor input = at::ones({1, 100, 115}, options) * 5.f;
  at::Tensor weight = at::ones({100, 1, 7}, options) * 6.f;
  at::Tensor bias = at::ones({100}, options) * 11.f;
  at::Tensor ref =
      at::conv1d(input, weight, bias, {stride}, {pad}, {dilation}, groups);

  at::Tensor nnc_result;
  std::vector<float> input_buf(1 * 100 * 115, 5.f);
  std::vector<float> weight_buf(100 * 1 * 7, 6.f);
  std::vector<float> bias_buf(100, 11.f);
  std::vector<float> result_buf(1 * 100 * 115, -1.f);

#ifdef TORCH_ENABLE_LLVM
  LLVMCodeGen llvm_codegen(l.root_stmt(), {Input, Weight, Bias, Result});

  llvm_codegen.call({input_buf, weight_buf, bias_buf, result_buf});
  nnc_result = at::from_blob(result_buf.data(), {1, 100, 115}, options);
  ASSERT_TRUE(at::allclose(nnc_result, ref));
#endif

  SimpleIREvaluator ir_eval(l.root_stmt(), {Input, Weight, Bias, Result});

  ir_eval.call({input_buf, weight_buf, bias_buf, result_buf});
  nnc_result = at::from_blob(result_buf.data(), {1, 100, 115}, options);
  ASSERT_TRUE(at::allclose(nnc_result, ref));
}

TEST(ExternalCall, Conv1d_int) {
  // A similar test, but now using kInt tensors
  BufHandle Input("Input", {1, 100, 115}, kInt);
  BufHandle Weight("Weight", {100, 1, 7}, kInt);
  BufHandle Bias("Bias", {100}, kInt);
  BufHandle ResultBuf("Result", {1, 100, 115}, kInt);
  int64_t stride = 1;
  int64_t pad = 3;
  int64_t dilation = 1;
  int64_t groups = 100;

  Tensor Result = Tensor(
      ResultBuf.node(),
      ExternalCall::make(
          ResultBuf,
          "nnc_aten_conv1d",
          {Input, Weight, Bias},
          {stride, pad, dilation, groups}));
  LoopNest l({Result});
  l.prepareForCodegen();
  l.simplify();

  auto options = at::TensorOptions()
                     .dtype(at::kInt)
                     .layout(at::kStrided)
                     .device(at::kCPU)
                     .requires_grad(false);
  at::Tensor input = at::ones({1, 100, 115}, options) * 5;
  at::Tensor weight = at::ones({100, 1, 7}, options) * 6;
  at::Tensor bias = at::ones({100}, options) * 11;
  at::Tensor ref =
      at::conv1d(input, weight, bias, {stride}, {pad}, {dilation}, groups);

  at::Tensor nnc_result;
  std::vector<int32_t> input_buf(1 * 100 * 115, 5);
  std::vector<int32_t> weight_buf(100 * 1 * 7, 6);
  std::vector<int32_t> bias_buf(100, 11);
  std::vector<int32_t> result_buf(1 * 100 * 115, -1);

#ifdef TORCH_ENABLE_LLVM
  LLVMCodeGen llvm_codegen(l.root_stmt(), {Input, Weight, Bias, Result});

  llvm_codegen.call({input_buf, weight_buf, bias_buf, result_buf});
  nnc_result = at::from_blob(result_buf.data(), {1, 100, 115}, options);
  ASSERT_TRUE(at::allclose(nnc_result, ref));
#endif

  SimpleIREvaluator ir_eval(l.root_stmt(), {Input, Weight, Bias, Result});

  ir_eval.call({input_buf, weight_buf, bias_buf, result_buf});
  nnc_result = at::from_blob(result_buf.data(), {1, 100, 115}, options);
  ASSERT_TRUE(at::allclose(nnc_result, ref));
}

TEST(ExternalCall, Conv1d_nobias_noargs) {
  BufHandle Input("Input", {1, 1, 115}, kFloat);
  BufHandle Weight("Weight", {10, 1, 7}, kFloat);
  BufHandle ResultBuf("Result", {1, 10, 109}, kFloat);

  Tensor Result = Tensor(
      ResultBuf.node(),
      ExternalCall::make(ResultBuf, "nnc_aten_conv1d", {Input, Weight}, {}));
  LoopNest l({Result});
  l.prepareForCodegen();
  l.simplify();

  auto options = at::TensorOptions()
                     .dtype(at::kFloat)
                     .layout(at::kStrided)
                     .device(at::kCPU)
                     .requires_grad(false);
  at::Tensor input = at::ones({1, 1, 115}, options) * 5.f;
  at::Tensor weight = at::ones({10, 1, 7}, options) * 6.f;
  at::Tensor ref = at::conv1d(input, weight);

  at::Tensor nnc_result;
  std::vector<float> input_buf(1 * 1 * 115, 5.f);
  std::vector<float> weight_buf(10 * 1 * 7, 6.f);
  std::vector<float> result_buf(1 * 10 * 109, -1.f);

#ifdef TORCH_ENABLE_LLVM
  LLVMCodeGen llvm_codegen(l.root_stmt(), {Input, Weight, Result});

  llvm_codegen.call({input_buf, weight_buf, result_buf});
  nnc_result = at::from_blob(result_buf.data(), {1, 10, 109}, options);
  ASSERT_TRUE(at::allclose(nnc_result, ref));
#endif

  SimpleIREvaluator ir_eval(l.root_stmt(), {Input, Weight, Result});

  ir_eval.call({input_buf, weight_buf, result_buf});
  nnc_result = at::from_blob(result_buf.data(), {1, 10, 109}, options);
  ASSERT_TRUE(at::allclose(nnc_result, ref));
}

TEST(ExternalCall, Conv2d_float) {
  BufHandle Input("Input", {1, 3, 224, 224}, kFloat);
  BufHandle Weight("Weight", {16, 3, 3, 3}, kFloat);
  BufHandle Bias("Bias", {16}, kFloat);
  BufHandle ResultBuf("Result", {1, 16, 112, 112}, kFloat);
  int64_t stride = 2;
  int64_t pad = 1;
  int64_t dilation = 1;
  int64_t groups = 1;

  Tensor Result = Tensor(
      ResultBuf.node(),
      ExternalCall::make(
          ResultBuf,
          "nnc_aten_conv2d",
          {Input, Weight, Bias},
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

  BufHandle Input("Input", {1, 3, 224, 224}, kInt);
  BufHandle Weight("Weight", {16, 3, 3, 3}, kInt);
  BufHandle Bias("Bias", {16}, kInt);
  BufHandle ResultBuf("Result", {1, 16, 112, 112}, kInt);
  int64_t stride = 2;
  int64_t pad = 1;
  int64_t dilation = 1;
  int64_t groups = 1;

  Tensor Result = Tensor(
      ResultBuf.node(),
      ExternalCall::make(
          ResultBuf,
          "nnc_aten_conv2d",
          {Input, Weight, Bias},
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
  BufHandle Input("Input", {1, 16, 112, 112}, kFloat);
  BufHandle Weight("Weight", {16, 16, 1, 1}, kFloat);
  BufHandle ResultBuf("Result", {1, 16, 112, 112}, kFloat);

  Tensor Result = Tensor(
      ResultBuf.node(),
      ExternalCall::make(ResultBuf, "nnc_aten_conv2d", {Input, Weight}, {}));
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

TEST(ExternalCall, Addmm_float) {
  BufHandle Input("Input", {100, 300}, kFloat);
  BufHandle Mat1("Mat1", {100, 200}, kFloat);
  BufHandle Mat2("Mat2", {200, 300}, kFloat);
  BufHandle ResultBuf("Result", {100, 300}, kFloat);
  int64_t beta = 2;
  int64_t alpha = 2;

  Tensor Result = Tensor(
      ResultBuf.node(),
      ExternalCall::make(
          ResultBuf, "nnc_aten_addmm", {Input, Mat1, Mat2}, {beta, alpha}));
  LoopNest l({Result});
  l.prepareForCodegen();
  l.simplify();

  auto options = at::TensorOptions()
                     .dtype(at::kFloat)
                     .layout(at::kStrided)
                     .device(at::kCPU)
                     .requires_grad(false);
  at::Tensor input = at::ones({100, 300}, options) * 5.f;
  at::Tensor mat1 = at::ones({100, 200}, options) * 6.f;
  at::Tensor mat2 = at::ones({200, 300}, options) * 11.f;
  at::Tensor ref = at::addmm(input, mat1, mat2, beta, alpha);

  at::Tensor nnc_result;
  std::vector<float> input_buf(100 * 300, 5.f);
  std::vector<float> mat1_buf(100 * 200, 6.f);
  std::vector<float> mat2_buf(200 * 300, 11.f);
  std::vector<float> result_buf(100 * 300, -1.f);

#ifdef TORCH_ENABLE_LLVM
  LLVMCodeGen llvm_codegen(l.root_stmt(), {Input, Mat1, Mat2, Result});

  llvm_codegen.call({input_buf, mat1_buf, mat2_buf, result_buf});
  nnc_result = at::from_blob(result_buf.data(), {100, 300}, options);
  ASSERT_TRUE(at::allclose(nnc_result, ref));
#endif

  SimpleIREvaluator ir_eval(l.root_stmt(), {Input, Mat1, Mat2, Result});

  ir_eval.call({input_buf, mat1_buf, mat2_buf, result_buf});
  nnc_result = at::from_blob(result_buf.data(), {100, 300}, options);
  ASSERT_TRUE(at::allclose(nnc_result, ref));
}

TEST(ExternalCall, Embedding) {
  BufHandle Weight("Weight", {256, 100}, kFloat);
  BufHandle Indices("Indices", {1, 115}, kLong);
  BufHandle ResultBuf("Result", {1, 115, 100}, kFloat);
  int64_t padding_idx = -1;
  bool scale_grad_by_freq = false;
  bool sparse = false;

  Tensor Result = Tensor(
      ResultBuf.node(),
      ExternalCall::make(
          ResultBuf,
          "nnc_aten_embedding",
          {Weight, Indices},
          {padding_idx, (int64_t)scale_grad_by_freq, (int64_t)sparse}));
  LoopNest l({Result});
  l.prepareForCodegen();
  l.simplify();

  auto options = at::TensorOptions()
                     .layout(at::kStrided)
                     .device(at::kCPU)
                     .requires_grad(false);

  at::Tensor weight = at::ones({256, 100}, options.dtype(at::kFloat)) * 5.f;
  at::Tensor indices = at::ones({1, 115}, options.dtype(at::kLong)) * 6;
  at::Tensor ref =
      at::embedding(weight, indices, padding_idx, scale_grad_by_freq, sparse);

  at::Tensor nnc_result;
  std::vector<float> weight_buf(256 * 100, 5.f);
  std::vector<int64_t> indices_buf(1 * 115, 6);
  std::vector<float> result_buf(1 * 115 * 100, -1.f);

#ifdef TORCH_ENABLE_LLVM
  LLVMCodeGen llvm_codegen(l.root_stmt(), {Weight, Indices, Result});

  llvm_codegen.call({weight_buf, indices_buf, result_buf});
  nnc_result = at::from_blob(
      result_buf.data(), {1, 115, 100}, options.dtype(at::kFloat));
  ASSERT_TRUE(at::allclose(nnc_result, ref));
#endif

  SimpleIREvaluator ir_eval(l.root_stmt(), {Weight, Indices, Result});

  ir_eval.call({weight_buf, indices_buf, result_buf});
  nnc_result = at::from_blob(
      result_buf.data(), {1, 115, 100}, options.dtype(at::kFloat));
  ASSERT_TRUE(at::allclose(nnc_result, ref));
}

TEST(ExternalCall, MaxReduction) {
  BufHandle Input("Input", {1, 115, 152}, kFloat);
  BufHandle ResultBuf("Result", {1, 152}, kFloat);
  int64_t dim = 1;
  bool keep_dim = false;

  Tensor Result = Tensor(
      ResultBuf.node(),
      ExternalCall::make(
          ResultBuf, "nnc_aten_max_red", {Input}, {dim, (int64_t)keep_dim}));
  LoopNest l({Result});
  l.prepareForCodegen();
  l.simplify();

  auto options = at::TensorOptions()
                     .dtype(at::kFloat)
                     .layout(at::kStrided)
                     .device(at::kCPU)
                     .requires_grad(false);

  at::Tensor input = at::ones({1, 115, 152}, options) * 5.f;
  at::Tensor ref = std::get<0>(at::max(input, dim, keep_dim));

  at::Tensor nnc_result;
  std::vector<float> input_buf(1 * 115 * 152, 5.f);
  std::vector<float> result_buf(1 * 152, -1.f);

#ifdef TORCH_ENABLE_LLVM
  LLVMCodeGen llvm_codegen(l.root_stmt(), {Input, Result});

  llvm_codegen.call({input_buf, result_buf});
  nnc_result = at::from_blob(result_buf.data(), {1, 152}, options);
  ASSERT_TRUE(at::allclose(nnc_result, ref));
#endif

  SimpleIREvaluator ir_eval(l.root_stmt(), {Input, Result});

  ir_eval.call({input_buf, result_buf});
  nnc_result = at::from_blob(result_buf.data(), {1, 152}, options);
  ASSERT_TRUE(at::allclose(nnc_result, ref));
}

#ifdef USE_XNNPACK

TEST(ExternalCall, Prepacked_Linear_float) {
  using namespace at::native::xnnpack;

  BufHandle Input("Input", {100, 200}, kFloat);
  BufHandle ResultBuf("Result", {100, 300}, kFloat);

  // Calculate reference result using at::linear.
  auto options = at::TensorOptions()
                     .dtype(at::kFloat)
                     .layout(at::kStrided)
                     .device(at::kCPU)
                     .requires_grad(false);
  at::Tensor input =
      at::linspace(-10.0, 10.0, 100 * 200, options).resize_({100, 200});
  at::Tensor weight =
      at::linspace(-10.0, 10.0, 300 * 200, options).resize_({300, 200});
  at::Tensor bias = at::linspace(-10.0, 10.0, 300, options);
  at::Tensor ref = at::linear(input, weight, bias);

  // Create prepacked xnnpack context object.
  auto linear_clamp_prepack_op =
      c10::Dispatcher::singleton()
          .findSchemaOrThrow("prepacked::linear_clamp_prepack", "")
          .typed<c10::intrusive_ptr<LinearOpContext>(
              at::Tensor,
              std::optional<at::Tensor>,
              const std::optional<at::Scalar>&,
              const std::optional<at::Scalar>&)>();
  auto prepacked = linear_clamp_prepack_op.call(
      weight, bias, std::optional<at::Scalar>(), std::optional<at::Scalar>());

  BufHandle DummyPrepacked("DummyPrepacked", {1}, kFloat);
  Tensor Result = Tensor(
      ResultBuf.node(),
      ExternalCall::make(
          ResultBuf,
          "nnc_prepacked_linear_clamp_run",
          {Input, DummyPrepacked},
          {}));
  LoopNest l({Result});
  l.prepareForCodegen();
  l.simplify();

  at::Tensor nnc_result;
  std::vector<float> input_buf(
      input.data_ptr<float>(), input.data_ptr<float>() + 100 * 200);
  std::vector<float> result_buf(100 * 300, -1.f);

#ifdef TORCH_ENABLE_LLVM
  LLVMCodeGen llvm_codegen(l.root_stmt(), {Input, DummyPrepacked, Result});

  llvm_codegen.call({input_buf, prepacked.get(), result_buf});
  nnc_result = at::from_blob(result_buf.data(), {100, 300}, options);
  ASSERT_TRUE(at::allclose(nnc_result, ref));
#endif

  SimpleIREvaluator ir_eval(l.root_stmt(), {Input, DummyPrepacked, Result});

  ir_eval.call({input_buf, prepacked.get(), result_buf});
  nnc_result = at::from_blob(result_buf.data(), {100, 300}, options);
  ASSERT_TRUE(at::allclose(nnc_result, ref));
}

TEST(ExternalCall, Prepacked_Conv2d_float) {
  using namespace at::native::xnnpack;

  BufHandle Input("Input", {1, 3, 224, 224}, kFloat);
  BufHandle ResultBuf("Result", {1, 16, 112, 112}, kFloat);
  int64_t stride = 2;
  int64_t pad = 1;
  int64_t dilation = 1;
  int64_t groups = 1;

  // Calculate reference result using at::conv2d.
  auto options = at::TensorOptions()
                     .dtype(at::kFloat)
                     .layout(at::kStrided)
                     .device(at::kCPU)
                     .requires_grad(false);
  at::Tensor input = at::linspace(-10.0, 10.0, 1 * 3 * 224 * 224, options)
                         .resize_({1, 3, 224, 224});
  at::Tensor weight =
      at::linspace(-10.0, 10.0, 16 * 3 * 3 * 3, options).resize_({16, 3, 3, 3});
  at::Tensor bias = at::linspace(-10.0, 10.0, 16, options);
  at::Tensor ref = at::conv2d(
      input,
      weight,
      bias,
      {stride, stride},
      {pad, pad},
      {dilation, dilation},
      groups);

  // Create prepacked xnnpack context object.
  auto conv2d_clamp_prepack_op =
      c10::Dispatcher::singleton()
          .findSchemaOrThrow("prepacked::conv2d_clamp_prepack", "")
          .typed<c10::intrusive_ptr<Conv2dOpContext>(
              at::Tensor,
              std::optional<at::Tensor>,
              std::vector<int64_t>,
              std::vector<int64_t>,
              std::vector<int64_t>,
              int64_t,
              const std::optional<at::Scalar>&,
              const std::optional<at::Scalar>&)>();
  auto prepacked = conv2d_clamp_prepack_op.call(
      weight,
      bias,
      {stride, stride},
      {pad, pad},
      {dilation, dilation},
      groups,
      std::optional<at::Scalar>(),
      std::optional<at::Scalar>());

  BufHandle DummyPrepacked("DummyPrepacked", {1}, kFloat);
  Tensor Result = Tensor(
      ResultBuf.node(),
      ExternalCall::make(
          ResultBuf,
          "nnc_prepacked_conv2d_clamp_run",
          {Input, DummyPrepacked},
          {}));
  LoopNest l({Result});
  l.prepareForCodegen();
  l.simplify();

  at::Tensor nnc_result;
  std::vector<float> input_buf(
      input.data_ptr<float>(), input.data_ptr<float>() + 1 * 3 * 224 * 224);
  std::vector<float> result_buf(1 * 16 * 112 * 112, -1.f);

#ifdef TORCH_ENABLE_LLVM
  LLVMCodeGen llvm_codegen(l.root_stmt(), {Input, DummyPrepacked, Result});

  llvm_codegen.call({input_buf, prepacked.get(), result_buf});
  nnc_result = at::from_blob(result_buf.data(), {1, 16, 112, 112}, options);
  ASSERT_TRUE(at::allclose(nnc_result, ref, 1e-03, 1e-03));
#endif

  SimpleIREvaluator ir_eval(l.root_stmt(), {Input, DummyPrepacked, Result});

  ir_eval.call({input_buf, prepacked.get(), result_buf});
  nnc_result = at::from_blob(result_buf.data(), {1, 16, 112, 112}, options);
  ASSERT_TRUE(at::allclose(nnc_result, ref, 1e-03, 1e-03));
}

#endif // USE_XNNPACK

TEST(ExternalCall, BinaryFloat) {
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
    BufHandle A("A", toExprHandleVec(aShape), kFloat);
    BufHandle B("B", toExprHandleVec(bShape), kFloat);
    BufHandle ResultBuf("Result", toExprHandleVec(resShape), kFloat);

    Tensor Result = Tensor(
        ResultBuf.node(),
        ExternalCall::make(ResultBuf, externCallName, {A, B}, {}));
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
      // NOLINTNEXTLINE(modernize-use-transparent-functors)
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
  tests.push_back(Test{// NOLINTNEXTLINE(cppcoreguidelines-avoid-magic-numbers)
                       {1, 64, 8, 9},
                       {1, 64, 5, 7},
                       [](at::Tensor x) {
                         return at::adaptive_avg_pool2d(x, {5, 7});
                       },
                       "nnc_aten_adaptive_avg_pool2d",
                       toExprHandleVec({5, 7})});
  tests.push_back(Test{// NOLINTNEXTLINE(cppcoreguidelines-avoid-magic-numbers)
                       {100, 200},
                       {100},
                       [](at::Tensor x) { return at::mean(x, {1}); },
                       "nnc_aten_mean",
                       toExprHandleVec({1, /*keepdim=*/0})});
  for (auto curTest : tests) {
    std::vector<int64_t> aShape, resShape;
    TensorFunc torchFunc;
    std::string externCallName;
    std::vector<ExprHandle> externCallArgs;
    std::tie(aShape, resShape, torchFunc, externCallName, externCallArgs) =
        curTest;
    BufHandle A("A", toExprHandleVec(aShape), kFloat);
    BufHandle ResultBuf("Result", toExprHandleVec(resShape), kFloat);

    Tensor Result = Tensor(
        ResultBuf.node(),
        ExternalCall::make(ResultBuf, externCallName, {A}, externCallArgs));
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
      // NOLINTNEXTLINE(modernize-use-transparent-functors)
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

  BufHandle ConvResultBuf("ConvResult", {1, 16, 32, 32}, kFloat);
  BufHandle MatmulResultBuf("MatmulResult", {1, 16, 32, 32}, kFloat);

  Tensor Input = Compute(
      "Input",
      {1, 16, 32, 32},
      [&](const VarHandle& n,
          const VarHandle& c,
          const VarHandle& h,
          const VarHandle& w) { return FloatImm::make(5.0f); });
  Tensor Weight = Compute(
      "Weight",
      {16, 16, 1, 1},
      [&](const VarHandle& n,
          const VarHandle& c,
          const VarHandle& h,
          const VarHandle& w) { return FloatImm::make(6.0f); });

  Tensor ConvResult = Tensor(
      ConvResultBuf.node(),
      ExternalCall::make(
          ConvResultBuf,
          "nnc_aten_conv2d",
          {BufHandle(Input.buf()), BufHandle(Weight.buf())},
          {}));
  Tensor MatmulResult = Tensor(
      MatmulResultBuf.node(),
      ExternalCall::make(
          MatmulResultBuf,
          "nnc_aten_matmul",
          {BufHandle(ConvResult.buf()), BufHandle(ConvResult.buf())},
          {}));
  Tensor Result = Compute(
      "Result",
      {1, 16, 32, 32},
      [&](const VarHandle& n,
          const VarHandle& c,
          const VarHandle& h,
          const VarHandle& w) {
        return ConvResult.load(n, c, h, w) + MatmulResult.load(n, c, h, w);
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
  at::Tensor input = at::ones({1, 16, 32, 32}, options) * 5.f;
  at::Tensor weight = at::ones({16, 16, 1, 1}, options) * 6.f;
  at::Tensor t = at::conv2d(input, weight);
  at::Tensor t2 = at::matmul(t, t);
  at::Tensor ref = t + t2;

  at::Tensor nnc_result;
  std::vector<float> input_buf(1 * 16 * 32 * 32, 5.f);
  std::vector<float> weight_buf(16 * 16 * 1 * 1, 6.f);
  std::vector<float> conv_result_buf(1 * 16 * 32 * 32, -1.f);
  std::vector<float> matmul_result_buf(1 * 16 * 32 * 32, -1.f);
  std::vector<float> result_buf(1 * 16 * 32 * 32, -1.f);

#ifdef TORCH_ENABLE_LLVM
  LLVMCodeGen llvm_codegen(
      l.root_stmt(), {Input, Weight, ConvResult, MatmulResult, Result});

  llvm_codegen.call(
      {input_buf, weight_buf, conv_result_buf, matmul_result_buf, result_buf});
  nnc_result = at::from_blob(result_buf.data(), {1, 16, 32, 32}, options);
  ASSERT_TRUE(at::allclose(nnc_result, ref));
#endif

  SimpleIREvaluator ir_eval(
      l.root_stmt(), {Input, Weight, ConvResult, MatmulResult, Result});

  ir_eval.call(
      {input_buf, weight_buf, conv_result_buf, matmul_result_buf, result_buf});
  nnc_result = at::from_blob(result_buf.data(), {1, 16, 32, 32}, options);
  ASSERT_TRUE(at::allclose(nnc_result, ref));
}

TEST(ExternalCall, Inlining) {
  // This test verifies that Tensors using external calls can be used by and
  // can use Tensors built with Compute API.

  BufHandle MatmulResultBuf("MatmulResult", {8, 8}, kFloat);

  Tensor A = Compute("A", {8, 8}, [&](const VarHandle& i, const VarHandle& j) {
    return FloatImm::make(5.0f);
  });
  Tensor B = Compute("B", {8, 8}, [&](const VarHandle& i, const VarHandle& j) {
    return FloatImm::make(4.0f);
  });
  Tensor MatmulResult = Tensor(
      MatmulResultBuf.node(),
      ExternalCall::make(
          MatmulResultBuf,
          "nnc_aten_matmul",
          {BufHandle(A.buf()), BufHandle(B.buf())},
          {}));
  Tensor Result =
      Compute("Result", {8, 8}, [&](const VarHandle& i, const VarHandle& j) {
        return MatmulResult.load(i, j) + FloatImm::make(3.0f);
      });

  StmtPtr root_stmt = alloc<torch::jit::tensorexpr::Block>(std::vector<StmtPtr>(
      {A.stmt(), B.stmt(), MatmulResult.stmt(), Result.stmt()}));
  LoopNest l(root_stmt, {Result.buf()});

  // Inlining should not inline anything here since all Bufs are either
  // defined or used in ExternalCalls
  l.inlineIntermediateBufs(false);

  l.prepareForCodegen();
  l.simplify();

  auto options = at::TensorOptions()
                     .dtype(at::kFloat)
                     .layout(at::kStrided)
                     .device(at::kCPU)
                     .requires_grad(false);
  at::Tensor a = at::ones({8, 8}, options) * 5.f;
  at::Tensor b = at::ones({8, 8}, options) * 4.f;
  at::Tensor t = at::matmul(a, b);
  at::Tensor ref = t + 3.f;

  at::Tensor nnc_result;
  std::vector<float> result_buf(8 * 8);

#ifdef TORCH_ENABLE_LLVM
  LLVMCodeGen llvm_codegen(l.root_stmt(), {Result});

  llvm_codegen.call({result_buf});
  nnc_result = at::from_blob(result_buf.data(), {8, 8}, options);
  ASSERT_TRUE(at::allclose(nnc_result, ref));
#endif

  SimpleIREvaluator ir_eval(l.root_stmt(), {Result});

  ir_eval.call({result_buf});
  nnc_result = at::from_blob(result_buf.data(), {8, 8}, options);
  ASSERT_TRUE(at::allclose(nnc_result, ref));
}

TEST(ExternalCall, JitCustomFusionOp) {
  const char* custom_op_schema_literal =
      "nnc_custom::add_mul(Tensor a, Tensor b, Tensor c) -> Tensor";
  const char* external_func_name = "nnc_add_mul";

  auto add_mul_lowering_func =
      [external_func_name](
          const std::vector<torch::jit::tensorexpr::ArgValue>& inputs,
          const std::vector<torch::jit::tensorexpr::ExprHandle>& output_shape,
          const std::vector<torch::jit::tensorexpr::ExprHandle>& output_strides,
          const std::optional<torch::jit::tensorexpr::ScalarType>& output_type,
          at::Device device) {
        auto output_dtype = Dtype(*output_type);
        torch::jit::tensorexpr::BufHandle result_buf(
            "nnc_add_mul_res_buf", output_shape, output_dtype);
        const torch::jit::tensorexpr::BufHandle& a =
            std::get<torch::jit::tensorexpr::BufHandle>(inputs[0]);
        const torch::jit::tensorexpr::BufHandle& b =
            std::get<torch::jit::tensorexpr::BufHandle>(inputs[1]);
        const torch::jit::tensorexpr::BufHandle& c =
            std::get<torch::jit::tensorexpr::BufHandle>(inputs[1]);
        torch::jit::tensorexpr::StmtPtr s =
            torch::jit::tensorexpr::ExternalCall::make(
                result_buf, external_func_name, {a, b, c}, {});
        return Tensor(result_buf.node(), s);
      };

  auto add_mul_external_func = [](int64_t bufs_num,
                                  void** buf_data,
                                  int64_t* buf_ranks,
                                  int64_t* buf_dims,
                                  int64_t* buf_strides,
                                  int8_t* buf_dtypes,
                                  int64_t args_num,
                                  int64_t* extra_args) {};

  torch::jit::RegisterOperators reg({Operator(
      custom_op_schema_literal,
      [](const Node* node) -> Operation {
        return [](Stack& _stack) {
          auto a = std::move(peek(_stack, 0, 3)).toTensor();
          auto b = std::move(peek(_stack, 1, 3)).toTensor();
          auto c = std::move(peek(_stack, 2, 3)).toTensor();
          drop(_stack, 3);
          auto result = (a + b) * c;
          pack(_stack, std::move(result));
          return 0;
        };
      },
      c10::AliasAnalysisKind::FROM_SCHEMA)});

  auto& custom_operator_set = torch::jit::tensorexpr::getCustomOperatorSet();
  custom_operator_set.insert({custom_op_schema_literal});

  auto& te_lowering_registry = torch::jit::tensorexpr::getNNCLoweringRegistry();
  te_lowering_registry.insert(
      parseSchema(custom_op_schema_literal), add_mul_lowering_func);

  auto& te_nnc_func_registry = torch::jit::tensorexpr::getNNCFunctionRegistry();
  te_nnc_func_registry[external_func_name] = add_mul_external_func;

  std::string graph_string = R"IR(
    graph(%a : Float(10, 20, strides=[20, 1], device=cpu),
          %b : Float(10, 20, strides=[20, 1], device=cpu),
          %c : Float(10, 20, strides=[20, 1], device=cpu)):
      %res : Float(10, 20, strides=[20, 1], device=cpu) = nnc_custom::add_mul(%a, %b, %c)
      return (%res))IR";

  auto graph = std::make_shared<Graph>();
  torch::jit::parseIR(graph_string, graph.get());

  std::string shape_compute_python_string = R"PY(
  def computOutput(a: List[int], b: List[int], c: List[int]):
    expandedSizes: List[int] = []
    dimsA = len(a)
    dimsB = len(b)
    dimsC = len(c)
    ndim = max(dimsA, dimsB, dimsC)
    for i in range(ndim):
        offset = ndim - 1 - i
        dimA = dimsA - 1 - offset
        dimB = dimsB - 1 - offset
        dimC = dimsC - 1 - offset
        sizeA = a[dimA] if (dimA >= 0) else 1
        sizeB = b[dimB] if (dimB >= 0) else 1
        sizeC = a[dimC] if (dimC >= 0) else 1

        if sizeA != sizeB and sizeB != sizeC and sizeA != 1 and sizeB != 1 and sizeC != 1:
            # TODO: only assertion error is bound in C++ compilation right now
            raise AssertionError(
                "The size of tensor a {} must match the size of tensor b ("
                "{} and c {}) at non-singleton dimension {}".format(sizeA, sizeB, sizeC, i)
            )

        expandedSizes.append(max(sizeA, sizeB, sizeC))

    return expandedSizes
  )PY";
  auto cu_ptr = torch::jit::compile(shape_compute_python_string);
  torch::jit::GraphFunction* gf =
      (torch::jit::GraphFunction*)&cu_ptr->get_function("computOutput");
  ASSERT_TRUE(gf);

#ifdef TORCH_ENABLE_LLVM
  auto static_graph_case = graph->copy();
  FuseTensorExprs(static_graph_case, 1);
  torch::jit::testing::FileCheck()
      .check("prim::TensorExprGroup_")
      ->check("nnc_custom::add_mul")
      ->run(*static_graph_case);

  auto dynamic_graph_case = graph->copy();
  auto custom_op = torch::jit::getOperatorForLiteral(custom_op_schema_literal);
  ASSERT_TRUE(custom_op);
  torch::jit::RegisterShapeComputeGraphForSchema(
      custom_op->schema(), gf->graph());
  FuseTensorExprs(dynamic_graph_case, 1, false, true);
  torch::jit::testing::FileCheck()
      .check("prim::TensorExprGroup_")
      ->check("nnc_custom::add_mul")
      ->run(*dynamic_graph_case);
#else
  torch::jit::testing::FileCheck().check("nnc_custom::add_mul")->run(*graph);
#endif
}

} // namespace jit
} // namespace torch
