#include <gtest/gtest.h>
#include <torch/csrc/jit/tensorexpr/ir_simplifier.h>
#include <torch/csrc/jit/tensorexpr/loopnest.h>
#include <torch/csrc/jit/tensorexpr/tensor.h>
#include <torch/torch.h>

namespace torch {
namespace jit {

namespace te = torch::jit::tensorexpr;
namespace F = torch::nn::functional;

TEST(Conv, Conv2D) {
  te::KernelScope kernel_scope;

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

  te::Placeholder inputB(te::BufHandle("input", {N, C, H, W}, te::kFloat));
  te::Placeholder filterB(te::BufHandle("filter", {K, C, R, S}, te::kFloat));

  te::Tensor* conv = te::Reduce(
      "conv",
      {{N, "n"}, {K, "k"}, {OH, "oh"}, {OW, "ow"}},
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
      {{C, "c"}, {R, "r"}, {S, "s"}});

  // FIXME: It'd be nice to have a single header that pulls in things like
  // LoopNest, IRSimplifier, etc.
  te::LoopNest loop({conv});
  loop.prepareForCodegen();
  te::Stmt* s = loop.root_stmt();
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
