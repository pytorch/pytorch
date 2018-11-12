#include <gtest/gtest.h>

#include <torch/nn/modules/functional.h>
#include <torch/nn/modules/linear.h>
#include <torch/nn/modules/sequential.h>
#include <torch/optim/optimizer.h>
#include <torch/optim/sgd.h>
#include <torch/serialize.h>
#include <torch/types.h>
#include <torch/utils.h>

#include <test/cpp/api/support.h>

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
  std::stringstream stream;
  torch::save(input, stream);
  torch::Tensor tensor;
  torch::load(tensor, stream);
  return tensor;
}
} // namespace

TEST(SerializeTest, Basic) {
  torch::manual_seed(0);

  auto x = torch::randn({5, 5});
  auto y = save_and_load(x);

  ASSERT_TRUE(y.defined());
  ASSERT_EQ(x.sizes().vec(), y.sizes().vec());
  ASSERT_TRUE(x.allclose(y));
}

TEST(SerializeTest, BasicToFile) {
  torch::manual_seed(0);

  auto x = torch::randn({5, 5});

  auto tempfile = torch::utils::make_tempfile();
  torch::save(x, tempfile.name);

  torch::Tensor y;
  torch::load(y, tempfile.name);

  ASSERT_TRUE(y.defined());
  ASSERT_EQ(x.sizes().vec(), y.sizes().vec());
  ASSERT_TRUE(x.allclose(y));
}

TEST(SerializeTest, Resized) {
  torch::manual_seed(0);

  auto x = torch::randn({11, 5});
  x.resize_({5, 5});
  auto y = save_and_load(x);

  ASSERT_TRUE(y.defined());
  ASSERT_EQ(x.sizes().vec(), y.sizes().vec());
  ASSERT_TRUE(x.allclose(y));
}

TEST(SerializeTest, Sliced) {
  torch::manual_seed(0);

  auto x = torch::randn({11, 5});
  x = x.slice(0, 1, 5);
  auto y = save_and_load(x);

  ASSERT_TRUE(y.defined());
  ASSERT_EQ(x.sizes().vec(), y.sizes().vec());
  ASSERT_TRUE(x.allclose(y));
}

TEST(SerializeTest, NonContiguous) {
  torch::manual_seed(0);

  auto x = torch::randn({11, 5});
  x = x.slice(1, 1, 4);
  auto y = save_and_load(x);

  ASSERT_TRUE(y.defined());
  ASSERT_EQ(x.sizes().vec(), y.sizes().vec());
  ASSERT_TRUE(x.allclose(y));
}

TEST(SerializeTest, XOR) {
  // We better be able to save and load an XOR model!
  auto getLoss = [](Sequential model, uint32_t batch_size) {
    auto inputs = torch::empty({batch_size, 2});
    auto labels = torch::empty({batch_size});
    for (size_t i = 0; i < batch_size; i++) {
      inputs[i] = torch::randint(2, {2}, torch::kInt64);
      labels[i] = inputs[i][0].item<int64_t>() ^ inputs[i][1].item<int64_t>();
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

    running_loss = running_loss * 0.99 + loss.sum().item<float>() * 0.01;
    ASSERT_LT(epoch, 3000);
    epoch++;
  }

  auto tempfile = torch::utils::make_tempfile();
  torch::save(model, tempfile.name);
  torch::load(model2, tempfile.name);

  auto loss = getLoss(model2, 100);
  ASSERT_LT(loss.item<float>(), 0.1);
}

TEST(SerializeTest, Optim) {
  auto model1 = Linear(5, 2);
  auto model2 = Linear(5, 2);
  auto model3 = Linear(5, 2);

  // Models 1, 2, 3 will have the same parameters.
  auto model_tempfile = torch::utils::make_tempfile();
  torch::save(model1, model_tempfile.name);
  torch::load(model2, model_tempfile.name);
  torch::load(model3, model_tempfile.name);

  auto param1 = model1->named_parameters();
  auto param2 = model2->named_parameters();
  auto param3 = model3->named_parameters();
  for (const auto& p : param1) {
    ASSERT_TRUE(p->allclose(param2[p.key()]));
    ASSERT_TRUE(param2[p.key()].allclose(param3[p.key()]));
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

  auto optim_tempfile = torch::utils::make_tempfile();
  torch::save(optim3, optim_tempfile.name);
  torch::load(optim3_2, optim_tempfile.name);
  step(optim3_2, model3);

  param1 = model1->named_parameters();
  param2 = model2->named_parameters();
  param3 = model3->named_parameters();
  for (const auto& p : param1) {
    const auto& name = p.key();
    // Model 1 and 3 should be the same
    ASSERT_TRUE(
        param1[name].norm().item<float>() == param3[name].norm().item<float>());
    ASSERT_TRUE(
        param1[name].norm().item<float>() != param2[name].norm().item<float>());
  }
}

TEST(SerializeTest, XOR_CUDA) {
  torch::manual_seed(0);
  // We better be able to save and load a XOR model!
  auto getLoss = [](Sequential model, uint32_t batch_size) {
    auto inputs = torch::empty({batch_size, 2});
    auto labels = torch::empty({batch_size});
    for (size_t i = 0; i < batch_size; i++) {
      inputs[i] = torch::randint(2, {2}, torch::kInt64);
      labels[i] = inputs[i][0].item<int64_t>() ^ inputs[i][1].item<int64_t>();
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

    running_loss = running_loss * 0.99 + loss.sum().item<float>() * 0.01;
    ASSERT_LT(epoch, 3000);
    epoch++;
  }

  auto tempfile = torch::utils::make_tempfile();
  torch::save(model, tempfile.name);
  torch::load(model2, tempfile.name);

  auto loss = getLoss(model2, 100);
  ASSERT_LT(loss.item<float>(), 0.1);

  model2->to(torch::kCUDA);
  auto tempfile2 = torch::utils::make_tempfile();
  torch::save(model2, tempfile2.name);
  torch::load(model3, tempfile2.name);

  loss = getLoss(model3, 100);
  ASSERT_LT(loss.item<float>(), 0.1);
}
