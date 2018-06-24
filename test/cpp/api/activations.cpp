#include <catch.hpp>

#include <torch/functions.h>
#include <torch/nn/modules/activations.h>
#include <torch/nn/modules/sequential.h>
#include <torch/tensor.h>

#include <vector>

using namespace torch::nn;

bool is_between(torch::Variable tensor, double lower, double upper) {
  return (tensor >= lower).all().toCInt() && (tensor <= upper).all().toCInt();
}

// These tests are not to test the actual functions, but to ensure we pass on
// parameters correctly.

TEST_CASE("Activations/Threshold") {
  Threshold threshold(1.0, -1.0);

  REQUIRE(threshold->forward({torch::ones({1}) * 3})[0].toCFloat() == 3);
  REQUIRE(threshold->forward({torch::ones({1}) * 2})[0].toCFloat() == 2);

  REQUIRE(threshold->forward({torch::ones({1}) * 0.5})[0].toCFloat() == -1);
  REQUIRE(threshold->forward({torch::ones({1})})[0].toCFloat() == -1);
  REQUIRE(threshold->forward({torch::ones({1}) * -3})[0].toCFloat() == -1);
}

TEST_CASE("Activations/RReLU") {
  RRelu rrelu(-1.0, 1.0);

  REQUIRE(rrelu->forward({torch::ones({1}) * 0})[0].toCFloat() == 0);
  REQUIRE(rrelu->forward({torch::ones({1}) * 1})[0].toCFloat() == 1);
  REQUIRE(rrelu->forward({torch::ones({1}) * 5})[0].toCFloat() == 5);

  REQUIRE(is_between(rrelu->forward({torch::ones(1) * -0.5})[0], -0.5, 0.5));
  REQUIRE(is_between(rrelu->forward({torch::ones(1) * -1})[0], -1.0, 1.0));
  REQUIRE(is_between(rrelu->forward({torch::ones(1) * -5})[0], -5.0, 5.0));
}

TEST_CASE("Activations/GLU") {
  GLU glu(-1);
  auto output = glu->forward({torch::ones({2, 3, 4})})[0];
  auto expected = torch::ones({2, 3, 2}).mul(torch::ones({2, 3, 2}).sigmoid());
  REQUIRE(output.allclose(expected));
}

TEST_CASE("Activations/Hardshrink") {
  Hardshrink hardshrink(1);
  REQUIRE(hardshrink->forward({torch::ones({1}) * 0})[0].toCFloat() == 0);
  REQUIRE(hardshrink->forward({torch::ones({1}) * 0.5})[0].toCFloat() == 0);
  REQUIRE(hardshrink->forward({torch::ones({1}) * -0.5})[0].toCFloat() == 0);
  REQUIRE(hardshrink->forward({torch::ones({1}) * 1})[0].toCFloat() == 0);
  REQUIRE(hardshrink->forward({torch::ones({1}) * -1})[0].toCFloat() == 0);
  REQUIRE(hardshrink->forward({torch::ones({1}) * 5})[0].toCFloat() == 5);
  REQUIRE(hardshrink->forward({torch::ones({1}) * -5})[0].toCFloat() == -5);
}

TEST_CASE("Activations/LeakyRelu") {
  LeakyReLU leaky_relu(0.01);
  REQUIRE(leaky_relu->forward({torch::ones({1}) * 0})[0].toCFloat() == 0);
  REQUIRE(leaky_relu->forward({torch::ones({1}) * 1})[0].toCFloat() == 1);
  REQUIRE(leaky_relu->forward({torch::ones({1}) * 2})[0].toCFloat() == 2);
  REQUIRE(leaky_relu->forward({torch::ones({1}) * -100})[0].toCFloat() == -1);
}

TEST_CASE("Activations/Sequential") {
  Sequential sequential(Sigmoid{}, ReLU{});
  auto result = sequential.forward<std::vector<torch::Variable>>(
      std::vector<torch::Variable>{torch::zeros({1})});
  REQUIRE(result[0].toCFloat() == 0.5);
}
