#include <gtest/gtest.h>

#include "test/cpp/tensorexpr/padded_buffer.h"
#include "test/cpp/tensorexpr/test_base.h"
#include "test/cpp/tensorexpr/test_train.h"
#include "test/cpp/tensorexpr/test_utils.h"
#include "torch/csrc/jit/tensorexpr/eval.h"
#include "torch/csrc/jit/tensorexpr/ir.h"
#include "torch/csrc/jit/tensorexpr/ir_printer.h"
#include "torch/csrc/jit/tensorexpr/loopnest.h"
#include "torch/csrc/jit/tensorexpr/tensor.h"

#include <algorithm>
#include <iterator>
#include <random>

namespace torch {
namespace jit {

struct T {
  T(VGraph& g, std::vector<std::string> shape) : vt_(g.create_tensor(shape)) {}
  T(VTensor* vt) : vt_(vt) {}
  T() = delete;
  VTensor* vt_ = nullptr;
  operator VTensor*() const {
    return vt_;
  }
  T operator+(const T& other) {
    return T(call("add", {vt_, other})[0]);
  }
  T operator*(const T& other) {
    return T(call("mul", {vt_, other})[0]);
  }
  T operator/(const T& other) {
    return T(call("div", {vt_, other})[0]);
  }
  T operator-(const T& other) {
    return T(call("sub", {vt_, other})[0]);
  }
  T sum() {
    return T(call("sum", {vt_})[0]);
  }
  T broadcast_like(const T& other) {
    return T(call("broadcast", {vt_, other})[0]);
  }
  T grad(const T& param, const T& jacob) {
    return T(::grad(vt_, param, jacob));
  }
};

TEST(Train, TrainBasic) {
  {
    VGraph graph;
    auto A = graph.create_tensor({"K"});
    auto B = graph.create_tensor({"K"});
    auto C = call("mul", {A, B})[0];

    Stmt* s;
    std::map<const VTensor*, Placeholder> inputs;
    std::map<const VTensor*, Tensor*> bindings;
    std::map<std::string, VarHandle> vbindings;

    KernelScope kernel_scope;
    std::tie(s, inputs, bindings, vbindings) = to_tensorexpr(graph);

    SimpleIREvaluator cg(
        s, {inputs.at(A), inputs.at(B), bindings.at(C), vbindings.at("K")});

    auto N = 1024;
    std::vector<float> a_vec(N, 21.0f);
    std::vector<float> b_vec(N, 2.0f);
    std::vector<float> c_vec(N, 0.0f);
    cg.call({a_vec.data(), b_vec.data(), c_vec.data(), N});
    assertAllEqual(c_vec, 42.0f);
  }
  {
    VGraph graph;
    auto A = graph.create_tensor({"K"});
    auto B = graph.create_tensor({"K"});
    auto C = call("mul", {A, B})[0];
    auto ones = graph.create_tensor({"K"});
    auto D = call("mul", {C, C})[0];
    // D = (A * B)^2
    // dD/dA = 2*(A*B)*B = 2*A*B^2
    auto dA = grad(D, A, ones);

    Stmt* s;
    std::map<const VTensor*, Placeholder> inputs;
    std::map<const VTensor*, Tensor*> bindings;
    std::map<std::string, VarHandle> vbindings;

    KernelScope kernel_scope;
    std::tie(s, inputs, bindings, vbindings) = to_tensorexpr(graph, {dA});

    SimpleIREvaluator cg(
        s,
        {inputs.at(A),
         inputs.at(B),
         inputs.at(ones),
         bindings.at(dA),
         vbindings.at("K")});

    auto N = 1024;
    std::vector<float> a_vec(N, 21.0f);
    std::vector<float> b_vec(N, 2.0f);
    std::vector<float> ones_vec(N, 1.0f);
    std::vector<float> da_vec(N, 0.0f);
    cg.call({a_vec.data(), b_vec.data(), ones_vec.data(), da_vec.data(), N});
    // 2*A*B^2
    assertAllEqual(da_vec, 168.0f);
  }
  // T wrapper
  {
    VGraph g;
    auto A = T(g.create_tensor({"K"}));
    auto B = T(g.create_tensor({"K"}));
    auto C = A + B;

    Stmt* s;
    std::map<const VTensor*, Placeholder> inputs;
    std::map<const VTensor*, Tensor*> bindings;
    std::map<std::string, VarHandle> vbindings;

    KernelScope kernel_scope;
    std::tie(s, inputs, bindings, vbindings) = to_tensorexpr(g);

    SimpleIREvaluator cg(
        s, {inputs.at(A), inputs.at(B), bindings.at(C), vbindings.at("K")});

    auto N = 1024;
    std::vector<float> a_vec(N, 21.0f);
    std::vector<float> b_vec(N, 2.0f);
    std::vector<float> c_vec(N, 0.0f);
    cg.call({a_vec.data(), b_vec.data(), c_vec.data(), N});
    assertAllEqual(c_vec, 23.0f);
  }
  {
    VGraph g;
    auto A = T(g.create_tensor({"K"}));
    auto B = T(g.create_tensor({"K"}));
    auto C = A * B;
    auto ones = T(g.create_tensor({"K"}));
    auto D = C * C;
    // D = (A * B)^2
    // dD/dA = 2*(A*B)*B = 2*A*B^2
    auto dA = D.grad(A, ones);

    Stmt* s;
    std::map<const VTensor*, Placeholder> inputs;
    std::map<const VTensor*, Tensor*> bindings;
    std::map<std::string, VarHandle> vbindings;

    KernelScope kernel_scope;
    std::tie(s, inputs, bindings, vbindings) = to_tensorexpr(g, {dA});

    SimpleIREvaluator cg(
        s,
        {inputs.at(A),
         inputs.at(B),
         inputs.at(ones),
         bindings.at(dA),
         vbindings.at("K")});

    auto N = 1024;
    std::vector<float> a_vec(N, 21.0f);
    std::vector<float> b_vec(N, 2.0f);
    std::vector<float> ones_vec(N, 1.0f);
    std::vector<float> da_vec(N, 0.0f);
    cg.call({a_vec.data(), b_vec.data(), ones_vec.data(), da_vec.data(), N});
    // 2*A*B^2
    assertAllEqual(da_vec, 168.0f);
  }
  // division gradient
  {
    VGraph g;
    auto A = T(g, {"K"});
    auto B = T(g, {"K"});
    auto C = (A * A) / B;
    auto ones = T(g, {"K"});
    // d (A^2 / B)^2 / dB = -2 A^4 / B^3
    auto dC = (C * C).grad(B, ones);

    Stmt* s;
    std::map<const VTensor*, Placeholder> inputs;
    std::map<const VTensor*, Tensor*> bindings;
    std::map<std::string, VarHandle> vbindings;

    KernelScope kernel_scope;
    std::tie(s, inputs, bindings, vbindings) = to_tensorexpr(g, {dC});

    SimpleIREvaluator cg(
        s,
        {inputs.at(A),
         inputs.at(B),
         inputs.at(ones),
         bindings.at(dC),
         vbindings.at("K")});
    auto N = 1024;
    std::vector<float> a_vec(N, 2.0f);
    std::vector<float> b_vec(N, 3.0f);
    std::vector<float> ones_vec(N, 1.0f);
    std::vector<float> dc_vec(N, 0.0f);
    cg.call({a_vec.data(), b_vec.data(), ones_vec.data(), dc_vec.data(), N});
    // -2 A^4 / B^3
    assertAllEqual(dc_vec, -1.185185185185f);
  }
  {
    VGraph g;
    auto X = T(g, {"K"});
    auto Y = X.sum();
    Stmt* s;
    std::map<const VTensor*, Placeholder> inputs;
    std::map<const VTensor*, Tensor*> bindings;
    std::map<std::string, VarHandle> vbindings;

    KernelScope kernel_scope;
    std::tie(s, inputs, bindings, vbindings) = to_tensorexpr(g, {Y});
    SimpleIREvaluator cg(s, {inputs.at(X), bindings.at(Y), vbindings.at("K")});
    auto N = 1024;
    std::vector<float> X_vec(N, 2.0f);
    std::vector<float> Y_vec(1, 0.0f);
    cg.call({X_vec.data(), Y_vec.data(), N});
    assertAllEqual(Y_vec, 2048.f);
  }

  {
    VGraph g;
    auto X = T(g, {"K"});
    auto Y = X.sum();
    auto Z = Y.broadcast_like(X);
    Stmt* s;
    std::map<const VTensor*, Placeholder> inputs;
    std::map<const VTensor*, Tensor*> bindings;
    std::map<std::string, VarHandle> vbindings;

    KernelScope kernel_scope;
    std::tie(s, inputs, bindings, vbindings) = to_tensorexpr(g, {Z});
    SimpleIREvaluator cg(s, {inputs.at(X), bindings.at(Z), vbindings.at("K")});
    auto N = 1024;
    std::vector<float> X_vec(N, 2.0f);
    std::vector<float> Z_vec(N, 0.0f);
    cg.call({X_vec.data(), Z_vec.data(), N});
    assertAllEqual(Z_vec, 2048.f);
  }

  // Linear regression
  {
    VGraph g;
    auto X = T(g, {"K"});
    auto W_ref = T(g, {"K"});
    // We want to fit W
    auto W = T(g, {"K"});

    auto Y_ref = X * W_ref;
    auto Y = X * W;
    auto diff = Y_ref - Y;
    auto K = T(g, {});
    // L2-norm
    auto loss = (diff * diff).sum() / K;

    // backward
    auto one = T(g, {});
    auto W_grad = loss.grad(W, one);
    auto LR = T(g, {});
    W_grad = W_grad * LR.broadcast_like(W_grad);
    auto new_W = W - W_grad;

    Stmt* s;
    std::map<const VTensor*, Placeholder> inputs;
    std::map<const VTensor*, Tensor*> bindings;
    std::map<std::string, VarHandle> vbindings;

    KernelScope kernel_scope;
    std::tie(s, inputs, bindings, vbindings) = to_tensorexpr(g, {new_W});

    SimpleIREvaluator cg(
        s,
        {inputs.at(X),
         inputs.at(W_ref),
         inputs.at(W),
         inputs.at(one),
         inputs.at(K),
         inputs.at(LR),
         bindings.at(new_W),
         vbindings.at("K")});

    auto N = 4;

    std::random_device rnd_device;
    std::mt19937 mersenne_engine{rnd_device()};
    std::uniform_real_distribution<float> dist{-1, 1};
    auto gen = [&dist, &mersenne_engine]() { return dist(mersenne_engine); };

    std::vector<float> X_(N, 0.0f);

    // Generate a random target vector
    std::vector<float> W_ref_(N, 3.0f);
    std::generate(W_ref_.begin(), W_ref_.end(), gen);

    std::vector<float> W_(N, 0.0f);
    std::vector<float> one_(1, 1.0f);
    std::vector<float> K_(N, 1.0f);
    std::vector<float> LR_(1, 0.1f);

    for (auto i = 0; i < 100; ++i) {
      std::generate(X_.begin(), X_.end(), gen);
      cg.call(
          {X_.data(),
           W_ref_.data(),
           W_.data(),
           one_.data(),
           K_.data(),
           LR_.data(),
           W_.data(),
           N});
    }
    // Less than 1% difference after running regression
    for (auto i = 0; i < W_.size(); ++i) {
      assert(std::abs(W_[i] - W_ref_[i]) < 0.01);
    }
  }
}
} // namespace jit
} // namespace torch
