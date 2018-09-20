#include <test/cpp/api/catch_utils.hpp>

#include <torch/nn/modules/functional.h>
#include <torch/nn/modules/linear.h>
#include <torch/nn/modules/sequential.h>
#include <torch/optim/optimizer.h>
#include <torch/optim/sgd.h>
#include <torch/serialize.h>
#include <torch/tensor.h>
#include <torch/utils.h>

#include <test/cpp/api/util.h>

#include <cstdio>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

using namespace torch::nn;
using namespace torch::serialize;

namespace {
Sequential xor_model() {
  return Sequential(
      Linear(2, 8),
      Functional(at::sigmoid),
      Linear(8, 1),
      Functional(at::sigmoid));
}

torch::Tensor save_and_load(torch::Tensor input) {
  torch::test::TempFile tempfile;
  torch::save(input, tempfile.str());
  return torch::load(tempfile.str());
}
} // namespace

CATCH_TEST_CASE("Serialize/Default/Basic") {
  torch::manual_seed(0);

  auto x = torch::randn({5, 5});
  auto y = save_and_load(x);

  CATCH_REQUIRE(y.defined());
  CATCH_REQUIRE(x.sizes().vec() == y.sizes().vec());
  CATCH_REQUIRE(x.allclose(y));
}

CATCH_TEST_CASE("Serialize/Default/Resized") {
  torch::manual_seed(0);

  auto x = torch::randn({11, 5});
  x.resize_({5, 5});
  auto y = save_and_load(x);

  CATCH_REQUIRE(y.defined());
  CATCH_REQUIRE(x.sizes().vec() == y.sizes().vec());
  CATCH_REQUIRE(x.allclose(y));
}

CATCH_TEST_CASE("Serialize/Default/Sliced") {
  torch::manual_seed(0);

  auto x = torch::randn({11, 5});
  x = x.slice(0, 1, 5);
  auto y = save_and_load(x);

  CATCH_REQUIRE(y.defined());
  CATCH_REQUIRE(x.sizes().vec() == y.sizes().vec());
  CATCH_REQUIRE(x.allclose(y));
}

CATCH_TEST_CASE("Serialize/Default/NonContiguous") {
  torch::manual_seed(0);

  auto x = torch::randn({11, 5});
  x = x.slice(1, 1, 4);
  auto y = save_and_load(x);

  CATCH_REQUIRE(y.defined());
  CATCH_REQUIRE(x.sizes().vec() == y.sizes().vec());
  CATCH_REQUIRE(x.allclose(y));
}

CATCH_TEST_CASE("Serialize/Default/XOR") {
  // We better be able to save and load an XOR model!
  auto getLoss = [](Sequential model, uint32_t batch_size) {
    auto inputs = torch::empty({batch_size, 2});
    auto labels = torch::empty({batch_size});
    for (size_t i = 0; i < batch_size; i++) {
      inputs[i] = torch::randint(2, {2}, torch::kInt64);
      labels[i] = inputs[i][0].toCLong() ^ inputs[i][1].toCLong();
    }
    auto x = model->forward<torch::Tensor>(inputs);
    return torch::binary_cross_entropy(x, labels);
  };

  auto model = xor_model();
  auto model2 = xor_model();
  auto model3 = xor_model();
  auto optimizer = torch::optim::SGD(
      model->parameters(),
      torch::optim::SGDOptions(1e-1).momentum(0.9).nesterov(true).weight_decay(
          1e-6));

  float running_loss = 1;
  int epoch = 0;
  while (running_loss > 0.1) {
    torch::Tensor loss = getLoss(model, 4);
    optimizer.zero_grad();
    loss.backward();
    optimizer.step();

    running_loss = running_loss * 0.99 + loss.sum().toCFloat() * 0.01;
    CATCH_REQUIRE(epoch < 3000);
    epoch++;
  }

  torch::test::TempFile tempfile;
  torch::save(model, tempfile.str());
  torch::load(model2, tempfile.str());

  auto loss = getLoss(model2, 100);
  CATCH_REQUIRE(loss.toCFloat() < 0.1);
}

CATCH_TEST_CASE("Serialize/Default/Optim") {
  auto model1 = Linear(5, 2);
  auto model2 = Linear(5, 2);
  auto model3 = Linear(5, 2);

  // Models 1, 2, 3 will have the same parameters.
  torch::test::TempFile model_tempfile;
  torch::save(model1, model_tempfile.str());
  torch::load(model2, model_tempfile.str());
  torch::load(model3, model_tempfile.str());

  auto param1 = model1->parameters();
  auto param2 = model2->parameters();
  auto param3 = model3->parameters();
  for (const auto& p : param1) {
    CATCH_REQUIRE(param1[p.key].allclose(param2[p.key]));
    CATCH_REQUIRE(param2[p.key].allclose(param3[p.key]));
  }

  // Make some optimizers with momentum (and thus state)
  auto optim1 = torch::optim::SGD(
      model1->parameters(), torch::optim::SGDOptions(1e-1).momentum(0.9));
  auto optim2 = torch::optim::SGD(
      model2->parameters(), torch::optim::SGDOptions(1e-1).momentum(0.9));
  auto optim2_2 = torch::optim::SGD(
      model2->parameters(), torch::optim::SGDOptions(1e-1).momentum(0.9));
  auto optim3 = torch::optim::SGD(
      model3->parameters(), torch::optim::SGDOptions(1e-1).momentum(0.9));
  auto optim3_2 = torch::optim::SGD(
      model3->parameters(), torch::optim::SGDOptions(1e-1).momentum(0.9));

  auto x = torch::ones({10, 5});

  auto step = [&x](torch::optim::Optimizer& optimizer, Linear model) {
    optimizer.zero_grad();
    auto y = model->forward(x).sum();
    y.backward();
    optimizer.step();
  };

  // Do 2 steps of model1
  step(optim1, model1);
  step(optim1, model1);

  // Do 2 steps of model 2 without saving the optimizer
  step(optim2, model2);
  step(optim2_2, model2);

  // Do 2 steps of model 3 while saving the optimizer
  step(optim3, model3);

  torch::test::TempFile optim_tempfile;
  torch::save(optim3, optim_tempfile.str());
  torch::load(optim3_2, optim_tempfile.str());
  step(optim3_2, model3);

  param1 = model1->parameters();
  param2 = model2->parameters();
  param3 = model3->parameters();
  for (const auto& p : param1) {
    const auto& name = p.key;
    // Model 1 and 3 should be the same
    CATCH_REQUIRE(
        param1[name].norm().toCFloat() == param3[name].norm().toCFloat());
    CATCH_REQUIRE(
        param1[name].norm().toCFloat() != param2[name].norm().toCFloat());
  }
}

CATCH_TEST_CASE("Serialize/Default/CUDA", "[cuda]") {
  torch::manual_seed(0);
  // We better be able to save and load a XOR model!
  auto getLoss = [](Sequential model, uint32_t batch_size) {
    auto inputs = torch::empty({batch_size, 2});
    auto labels = torch::empty({batch_size});
    for (size_t i = 0; i < batch_size; i++) {
      inputs[i] = torch::randint(2, {2}, torch::kInt64);
      labels[i] = inputs[i][0].toCLong() ^ inputs[i][1].toCLong();
    }
    auto x = model->forward<torch::Tensor>(inputs);
    return torch::binary_cross_entropy(x, labels);
  };

  auto model = xor_model();
  auto model2 = xor_model();
  auto model3 = xor_model();
  auto optimizer = torch::optim::SGD(
      model->parameters(),
      torch::optim::SGDOptions(1e-1).momentum(0.9).nesterov(true).weight_decay(
          1e-6));

  float running_loss = 1;
  int epoch = 0;
  while (running_loss > 0.1) {
    torch::Tensor loss = getLoss(model, 4);
    optimizer.zero_grad();
    loss.backward();
    optimizer.step();

    running_loss = running_loss * 0.99 + loss.sum().toCFloat() * 0.01;
    CATCH_REQUIRE(epoch < 3000);
    epoch++;
  }

  torch::test::TempFile tempfile;
  torch::save(model, tempfile.str());
  torch::load(model2, tempfile.str());

  auto loss = getLoss(model2, 100);
  CATCH_REQUIRE(loss.toCFloat() < 0.1);

  model2->to(torch::kCUDA);
  torch::test::TempFile tempfile2;
  torch::save(model2, tempfile2.str());
  torch::load(model3, tempfile2.str());

  loss = getLoss(model3, 100);
  CATCH_REQUIRE(loss.toCFloat() < 0.1);
}
