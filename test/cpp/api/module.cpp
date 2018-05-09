#include <catch.hpp>

#include <torch/torch.h>

using namespace torch;
using namespace torch::nn;

TEST_CASE("module/training-mode") {
  auto model = make(Linear(3, 4));
  REQUIRE(model->is_training());
  SECTION("Enable eval mode") {
    model->eval();
    REQUIRE(!model->is_training());
  }
  SECTION("Enable train mode") {
    model->train();
    REQUIRE(model->is_training());
  }
}

TEST_CASE("module/zero-grad") {
  auto model = make(Linear(3, 4));
  auto weights = Var(at::ones(at::CPU(at::kFloat), {8, 3}));
  auto loss = model->forward({weights}).front().sum();
  backward(loss);
  for (auto& parameter : model->parameters()) {
    Variable grad = parameter.second.grad();
    REQUIRE(grad.defined());
    REQUIRE(grad.sum().toCFloat() != 0);
  }
  model->zero_grad();
  for (auto& parameter : model->parameters()) {
    Variable grad = parameter.second.grad();
    REQUIRE(grad.defined());
    REQUIRE(grad.sum().toCFloat() == 0);
  }
}

TEST_CASE("module/conversions", "[cuda]") {
  auto model = make(LSTM(128, 64).nlayers(3).dropout(0.2));
  SECTION("starts as float on CPU") {
    for (auto& parameter : model->parameters()) {
      REQUIRE(parameter.second.type().backend() == at::kCPU);
      REQUIRE(parameter.second.type().scalarType() == at::kFloat);
    }
  }
  SECTION("to(CUDA)") {
    model->cuda();
    for (auto& parameter : model->parameters()) {
      REQUIRE(parameter.second.type().backend() == at::kCUDA);
    }
  }
  SECTION("to(CPU)") {
    model->to(at::kCPU);
    for (auto& parameter : model->parameters()) {
      REQUIRE(parameter.second.type().backend() == at::kCPU);
    }
  }
  SECTION("to(Int)") {
    model->to(at::kInt);
    for (auto& parameter : model->parameters()) {
      REQUIRE(parameter.second.type().scalarType() == at::kInt);
    }
  }
  SECTION("to(Double)") {
    model->to(at::kDouble);
    for (auto& parameter : model->parameters()) {
      REQUIRE(parameter.second.type().scalarType() == at::kDouble);
    }
  }
  SECTION("to(CUDA(Float))") {
    model->to(at::CUDA(at::kFloat));
    for (auto& parameter : model->parameters()) {
      REQUIRE(parameter.second.type().backend() == at::kCUDA);
      REQUIRE(parameter.second.type().scalarType() == at::kFloat);
    }
  }
}
