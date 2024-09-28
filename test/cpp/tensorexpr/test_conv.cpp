#include <gtest/gtest.h>
#include <torch/csrc/jit/tensorexpr/ir_simplifier.h>
#include <torch/csrc/jit/tensorexpr/llvm_codegen.h>
#include <torch/csrc/jit/tensorexpr/loopnest.h>
#include <torch/csrc/jit/tensorexpr/operators/conv2d.h>
#include <torch/csrc/jit/tensorexpr/tensor.h>
#include <torch/torch.h>

namespace torch {
namespace jit {

namespace te = torch::jit::tensorexpr;
namespace F = torch::nn::functional;

#ifdef TORCH_ENABLE_LLVM

// Generate test data with few bits of precision, to minimize error
// accumulation from floating-point reordering.
static at::Tensor genTestData(c10::IntArrayRef args) {
  return at::trunc(at::randn(args) * 256.0f) / 256.0f;
}

TEST(Conv, DepthwiseConv2D) {
  constexpr int N = 1, C = 72, H = 56, W = 56;
  constexpr int K = 72, R = 3, S = 3;
  constexpr int kPad = 1, kStride = 2, kGroups = C;
  constexpr int CperG = C / kGroups;

  te::BufHandle input("input", {N, C, H, W}, te::kFloat);
  te::BufHandle weight("weight", {K, CperG, R, S}, te::kFloat);
  te::BufHandle bias("bias", {K}, te::kFloat);
  te::Tensor output =
      te::conv2d_depthwise(input, weight, bias, kStride, kPad, kGroups);

  te::LoopNest loop({output});
  loop.simplify();
  loop.prepareForCodegen();
  te::LLVMCodeGen cg(loop.root_stmt(), {input, weight, bias, output});

  auto it = genTestData({N, C, H, W});
  auto wt = genTestData({K, CperG, R, S});
  auto bt = genTestData({K});
  auto ref = at::conv2d(it, wt, bt, kStride, kPad, /*dilation=*/1, kGroups);
  auto ot = at::zeros_like(ref);
  cg.call(
      {it.data_ptr<float>(),
       wt.data_ptr<float>(),
       bt.data_ptr<float>(),
       ot.data_ptr<float>()});

  ASSERT_TRUE(at::allclose(ref, ot));
}

TEST(Conv, DepthwiseConv2DNoBias) {
  constexpr int N = 1, C = 72, H = 56, W = 56;
  constexpr int K = 72, R = 3, S = 3;
  constexpr int kPad = 1, kStride = 2, kGroups = C;
  constexpr int CperG = C / kGroups;

  te::BufHandle input("input", {N, C, H, W}, te::kFloat);
  te::BufHandle weight("weight", {K, CperG, R, S}, te::kFloat);
  te::Tensor output =
      te::conv2d_depthwise(input, weight, kStride, kPad, kGroups);

  te::LoopNest loop({output});
  loop.simplify();
  loop.prepareForCodegen();
  te::LLVMCodeGen cg(loop.root_stmt(), {input, weight, output});

  auto it = genTestData({N, C, H, W});
  auto wt = genTestData({K, CperG, R, S});
  auto ref =
      at::conv2d(it, wt, at::Tensor(), kStride, kPad, /*dilation=*/1, kGroups);
  auto ot = at::zeros_like(ref);
  cg.call({it.data_ptr<float>(), wt.data_ptr<float>(), ot.data_ptr<float>()});

  ASSERT_TRUE(at::allclose(ref, ot));
}

TEST(Conv, DepthwiseConv2DDynamicShapes) {
  te::VarHandle N_var("N", te::kInt);
  te::VarHandle C_var("C", te::kInt);
  te::VarHandle H_var("H", te::kInt);
  te::VarHandle W_var("W", te::kInt);
  te::VarHandle K_var("K", te::kInt);
  te::VarHandle CperG_var("CperG", te::kInt);
  te::VarHandle R_var("R", te::kInt);
  te::VarHandle S_var("S", te::kInt);
  te::VarHandle kPad_var("kPad", te::kInt);
  te::VarHandle kStride_var("kStride", te::kInt);
  te::VarHandle kGroups_var("kGroups", te::kInt);

  te::BufHandle input("input", {N_var, C_var, H_var, W_var}, te::kFloat);
  te::BufHandle weight("weight", {K_var, CperG_var, R_var, S_var}, te::kFloat);
  te::Tensor output = te::conv2d_depthwise(
      input,
      weight,
      N_var,
      C_var,
      H_var,
      W_var,
      K_var,
      CperG_var,
      R_var,
      S_var,
      kStride_var,
      kPad_var,
      kGroups_var);

  te::LoopNest loop({output});
  loop.simplify();
  loop.prepareForCodegen();
  std::vector<te::CodeGen::BufferArg> buffer_args = {
      input,
      weight,
      N_var,
      C_var,
      H_var,
      W_var,
      K_var,
      CperG_var,
      R_var,
      S_var,
      kPad_var,
      kStride_var,
      kGroups_var,
      output};
  te::LLVMCodeGen cg(loop.root_stmt(), buffer_args);

  constexpr int N = 1, C = 72, H = 56, W = 56;
  constexpr int K = 72, R = 3, S = 3;
  constexpr int kPad = 1, kStride = 2, kGroups = C;
  constexpr int CperG = C / kGroups;

  auto it = genTestData({N, C, H, W});
  auto wt = genTestData({K, CperG, R, S});
  auto ref =
      at::conv2d(it, wt, at::Tensor(), kStride, kPad, /*dilation=*/1, kGroups);
  auto ot = at::zeros_like(ref);
  std::vector<te::CodeGen::CallArg> call_args = {
      it.data_ptr<float>(),
      wt.data_ptr<float>(),
      N,
      C,
      H,
      W,
      K,
      CperG,
      R,
      S,
      kPad,
      kStride,
      kGroups,
      ot.data_ptr<float>()};
  cg.call(call_args);

  ASSERT_TRUE(at::allclose(ref, ot));
}

#endif

TEST(Conv, Conv2D) {
  // Input dimensions.
  constexpr int N = 1;
  constexpr int C = 3;
  constexpr int H = 11;
  constexpr int W = 11;

  // Filter dimensions.
  constexpr int K = 8;
  constexpr int R = 3;
  constexpr int S = 3;

  // Output dims.
  constexpr int OH = H - R + 1;
  constexpr int OW = W - S + 1;

  // Compute reference result.
  at::Tensor input = torch::randn({N, C, H, W});
  at::Tensor filter = torch::randn({K, C, R, S});
  at::Tensor ref = F::conv2d(input, filter);

  // Double check the output size is as expected.
  ASSERT_EQ(ref.size(0), N);
  ASSERT_EQ(ref.size(1), K);
  ASSERT_EQ(ref.size(2), OH);
  ASSERT_EQ(ref.size(3), OW);

  te::BufHandle inputB("input", {N, C, H, W}, te::kFloat);
  te::BufHandle filterB("filter", {K, C, R, S}, te::kFloat);

  te::Tensor conv = te::Reduce(
      "conv",
      {N, K, OH, OW},
      te::Sum(),
      // FIXME: We have to use a `std::vector` parameter here and then unpack
      // it, because we don't have an overload allowing for an arbitrary number
      // of ExprHandle/VarHandle parameters.
      [&](const std::vector<te::VarHandle>& v) {
        auto const& n = v[0];
        auto const& k = v[1];
        auto const& oh = v[2];
        auto const& ow = v[3];
        auto const& c = v[4];
        auto const& r = v[5];
        auto const& s = v[6];
        // FIXME: We have to use `call` and construct a `std::vector` here
        // because the `operator()` overload is only specialized for a small
        // number of arguments.
        return inputB.load(n, c, oh + r, ow + s) * filterB.load(k, c, r, s);
      },
      // FIXME: If you forget one of the reduction dims, you get a segfault.
      // Could that be caught by a verifier?
      {C, R, S});

  // FIXME: It'd be nice to have a single header that pulls in things like
  // LoopNest, IRSimplifier, etc.
  te::LoopNest loop({conv});
  loop.prepareForCodegen();
  te::StmtPtr s = loop.root_stmt();
  s = te::IRSimplifier::simplify(s);

  at::Tensor result = at::empty_like(ref);
  te::SimpleIREvaluator cg(s, {inputB, filterB, conv});
  cg.call(
      {input.data_ptr<float>(),
       filter.data_ptr<float>(),
       result.data_ptr<float>()});

  ASSERT_TRUE(at::allclose(ref, result, 1e-3, 1e-3));
}

} // namespace jit
} // namespace torch
