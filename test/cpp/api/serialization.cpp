#include <catch.hpp>

#include <torch/nn/modules/functional.h>
#include <torch/nn/modules/linear.h>
#include <torch/nn/modules/sequential.h>
#include <torch/optim/optimizer.h>
#include <torch/optim/sgd.h>
#include <torch/serialization.h>
#include <torch/tensor.h>
#include <torch/utils.h>

#include <test/cpp/api/util.h>

#include <cereal/archives/portable_binary.hpp>

#include <memory>
#include <sstream>
#include <string>
#include <vector>

using namespace torch::nn;

namespace {
Sequential xor_model() {
  return Sequential(
      Linear(2, 8),
      Functional(at::sigmoid),
      Linear(8, 1),
      Functional(at::sigmoid));
}
} // namespace

TEST_CASE("serialization") {
  torch::manual_seed(0);
  SECTION("undefined") {
    auto x = torch::Tensor();

    REQUIRE(!x.defined());

    auto y = torch::randn({5});

    std::stringstream ss;
    torch::save(ss, &x);
    torch::load(ss, &y);

    REQUIRE(!y.defined());
  }

  SECTION("cputypes") {
    for (int i = 0; i < static_cast<int>(torch::Dtype::NumOptions); i++) {
      if (i == static_cast<int>(torch::Dtype::Half)) {
        // XXX can't serialize half tensors at the moment since contiguous() is
        // not implemented for this type;
        continue;
      } else if (i == static_cast<int>(torch::Dtype::Undefined)) {
        // We can't construct a tensor for this type. This is tested in
        // serialization/undefined anyway.
        continue;
      }

      auto x = torch::ones(
          {5, 5}, torch::getType(torch::Backend::CPU, static_cast<torch::Dtype>(i)));
      auto y = torch::empty({});

      std::stringstream ss;
      torch::save(ss, &x);
      torch::load(ss, &y);

      REQUIRE(y.defined());
      REQUIRE(x.sizes().vec() == y.sizes().vec());
      if (torch::isIntegralType(static_cast<torch::Dtype>(i))) {
        REQUIRE(x.equal(y));
      } else {
        REQUIRE(x.allclose(y));
      }
    }
  }

  SECTION("binary") {
    auto x = torch::randn({5, 5});
    auto y = torch::Tensor();

    std::stringstream ss;
    {
      cereal::BinaryOutputArchive archive(ss);
      archive(x);
    }
    {
      cereal::BinaryInputArchive archive(ss);
      archive(y);
    }

    REQUIRE(y.defined());
    REQUIRE(x.sizes().vec() == y.sizes().vec());
    REQUIRE(x.allclose(y));
  }
  SECTION("portable_binary") {
    auto x = torch::randn({5, 5});
    auto y = torch::Tensor();

    std::stringstream ss;
    {
      cereal::PortableBinaryOutputArchive archive(ss);
      archive(x);
    }
    {
      cereal::PortableBinaryInputArchive archive(ss);
      archive(y);
    }

    REQUIRE(y.defined());
    REQUIRE(x.sizes().vec() == y.sizes().vec());
    REQUIRE(x.allclose(y));
  }

  SECTION("resized") {
    auto x = torch::randn({11, 5});
    x.resize_({5, 5});
    auto y = torch::Tensor();

    std::stringstream ss;
    {
      cereal::BinaryOutputArchive archive(ss);
      archive(x);
    }
    {
      cereal::BinaryInputArchive archive(ss);
      archive(y);
    }

    REQUIRE(y.defined());
    REQUIRE(x.sizes().vec() == y.sizes().vec());
    REQUIRE(x.allclose(y));
  }
  SECTION("sliced") {
    auto x = torch::randn({11, 5});
    x = x.slice(0, 1, 3);
    auto y = torch::Tensor();

    std::stringstream ss;
    {
      cereal::BinaryOutputArchive archive(ss);
      archive(x);
    }
    {
      cereal::BinaryInputArchive archive(ss);
      archive(y);
    }

    REQUIRE(y.defined());
    REQUIRE(x.sizes().vec() == y.sizes().vec());
    REQUIRE(x.allclose(y));
  }

  SECTION("noncontig") {
    auto x = torch::randn({11, 5});
    x = x.slice(1, 1, 4);
    auto y = torch::Tensor();

    std::stringstream ss;
    {
      cereal::BinaryOutputArchive archive(ss);
      archive(x);
    }
    {
      cereal::BinaryInputArchive archive(ss);
      archive(y);
    }

    REQUIRE(y.defined());
    REQUIRE(x.sizes().vec() == y.sizes().vec());
    REQUIRE(x.allclose(y));
  }

  SECTION("xor") {
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
        torch::optim::SGDOptions(1e-1)
            .momentum(0.9)
            .nesterov(true)
            .weight_decay(1e-6));

    float running_loss = 1;
    int epoch = 0;
    while (running_loss > 0.1) {
      torch::Tensor loss = getLoss(model, 4);
      optimizer.zero_grad();
      loss.backward();
      optimizer.step();

      running_loss = running_loss * 0.99 + loss.sum().toCFloat() * 0.01;
      REQUIRE(epoch < 3000);
      epoch++;
    }

    std::stringstream ss;
    torch::save(ss, model);
    torch::load(ss, model2);

    auto loss = getLoss(model2, 100);
    REQUIRE(loss.toCFloat() < 0.1);
  }

  SECTION("optim") {
    auto model1 = Linear(5, 2);
    auto model2 = Linear(5, 2);
    auto model3 = Linear(5, 2);

    // Models 1, 2, 3 will have the same params
    std::stringstream ss;
    torch::save(ss, model1.get());
    torch::load(ss, model2.get());
    ss.seekg(0, std::ios::beg);
    torch::load(ss, model3.get());

    auto param1 = model1->parameters();
    auto param2 = model2->parameters();
    auto param3 = model3->parameters();
    for (const auto& p : param1) {
      REQUIRE(param1[p.key].allclose(param2[p.key]));
      REQUIRE(param2[p.key].allclose(param3[p.key]));
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
    ss.clear();
    torch::save(ss, &optim3);
    torch::load(ss, &optim3_2);
    step(optim3_2, model3);

    param1 = model1->parameters();
    param2 = model2->parameters();
    param3 = model3->parameters();
    for (const auto& p : param1) {
      const auto& name = p.key;
      // Model 1 and 3 should be the same
      REQUIRE(param1[name].norm().toCFloat() == param3[name].norm().toCFloat());
      REQUIRE(param1[name].norm().toCFloat() != param2[name].norm().toCFloat());
    }
  }
}

TEST_CASE("serialization_cuda", "[cuda]") {
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
    REQUIRE(epoch < 3000);
    epoch++;
  }

  std::stringstream ss;
  torch::save(ss, model);
  torch::load(ss, model2);

  auto loss = getLoss(model2, 100);
  REQUIRE(loss.toCFloat() < 0.1);

  model2->to(torch::kCUDA);
  ss.clear();
  torch::save(ss, model2);
  torch::load(ss, model3);

  loss = getLoss(model3, 100);
  REQUIRE(loss.toCFloat() < 0.1);
}
