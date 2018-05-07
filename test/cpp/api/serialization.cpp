#include <catch.hpp>

#include <torch/torch.h>

#include "cereal/archives/portable_binary.hpp"

using namespace torch;
using namespace torch::nn;

TEST_CASE("serialization") {
  SECTION("undefined") {
    auto x = at::Tensor();

    REQUIRE(!x.defined());

    auto y = at::CPU(at::kFloat).randn({5});

    std::stringstream ss;
    save(ss, &x);
    load(ss, &y);

    REQUIRE(!y.defined());
  }

  SECTION("cputypes") {
    for (int i = 0; i < static_cast<int>(at::ScalarType::NumOptions); i++) {
      if (i == static_cast<int>(at::ScalarType::Half)) {
        // XXX can't serialize half tensors at the moment since contiguous() is
        // not implemented for this type;
        continue;
      } else if (i == static_cast<int>(at::ScalarType::Undefined)) {
        // We can't construct a tensor for this type. This is tested in
        // serialization/undefined anyway.
        continue;
      }

      auto x =
          at::getType(at::kCPU, static_cast<at::ScalarType>(i)).ones({5, 5});
      auto y = at::Tensor();

      std::stringstream ss;
      save(ss, &x);
      load(ss, &y);

      REQUIRE(y.defined());
      REQUIRE(x.sizes().vec() == y.sizes().vec());
      if (at::isIntegralType(static_cast<at::ScalarType>(i))) {
        REQUIRE(x.equal(y));
      } else {
        REQUIRE(x.allclose(y));
      }
    }
  }

  SECTION("binary") {
    auto x = at::CPU(at::kFloat).randn({5, 5});
    auto y = at::Tensor();

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
    auto x = at::CPU(at::kFloat).randn({5, 5});
    auto y = at::Tensor();

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
    auto x = at::CPU(at::kFloat).randn({11, 5});
    x.resize_({5, 5});
    auto y = at::Tensor();

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
    auto x = at::CPU(at::kFloat).randn({11, 5});
    x = x.slice(0, 1, 3);
    auto y = at::Tensor();

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
    auto x = at::CPU(at::kFloat).randn({11, 5});
    x = x.slice(1, 1, 4);
    auto y = at::Tensor();

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
    auto makeModel = []() {
      ContainerList list;
      list.append(make(Linear(2, 8)));
      list.append(make(Linear(8, 1)));
      return make(list);
    };
    auto getLoss = [](std::shared_ptr<ContainerList> model, uint32_t bs) {
      auto inp = at::CPU(at::kFloat).tensor({bs, 2});
      auto lab = at::CPU(at::kFloat).tensor({bs});
      for (auto i = 0U; i < bs; i++) {
        auto a = std::rand() % 2;
        auto b = std::rand() % 2;
        auto c = a ^ b;
        inp[i][0] = a;
        inp[i][1] = b;
        lab[i] = c;
      }

      // forward
      auto x = Var(inp);
      auto y = Var(lab, false);
      for (auto layer : *model)
        x = layer->forward({x})[0].sigmoid_();
      return at::binary_cross_entropy(x, y);
    };

    auto model = makeModel();
    auto model2 = makeModel();
    auto model3 = makeModel();
    auto optim =
        SGD(model, 1e-1).momentum(0.9).nesterov().weight_decay(1e-6).make();

    float running_loss = 1;
    int epoch = 0;
    while (running_loss > 0.1) {
      Variable loss = getLoss(model, 4);
      optim->zero_grad();
      backward(loss);
      optim->step();

      running_loss = running_loss * 0.99 + loss.data().sum().toCFloat() * 0.01;
      REQUIRE(epoch < 3000);
      epoch++;
    }

    std::stringstream ss;
    save(ss, model);
    load(ss, model2);

    auto loss = getLoss(model2, 100);
    REQUIRE(loss.toCFloat() < 0.1);
  }

  SECTION("optim") {
    auto model1 = make(Linear(5, 2));
    auto model2 = make(Linear(5, 2));
    auto model3 = make(Linear(5, 2));

    // Models 1, 2, 3 will have the same params
    std::stringstream ss;
    save(ss, model1);
    load(ss, model2);
    ss.seekg(0, std::ios::beg);
    load(ss, model3);

    // Make some optimizers with momentum (and thus state)
    auto optim1 = SGD(model1, 1e-1).momentum(0.9).make();
    auto optim2 = SGD(model2, 1e-1).momentum(0.9).make();
    auto optim2_2 = SGD(model2, 1e-1).momentum(0.9).make();
    auto optim3 = SGD(model3, 1e-1).momentum(0.9).make();
    auto optim3_2 = SGD(model3, 1e-1).momentum(0.9).make();

    auto x = Var(at::CPU(at::kFloat).ones({10, 5}), true);

    auto step = [&](Optimizer optim, std::shared_ptr<Module> model) {
      optim->zero_grad();
      auto y = model->forward({x})[0].sum();
      backward(y);
      optim->step();
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
    save(ss, optim3);
    load(ss, optim3_2);
    step(optim3_2, model3);

    auto param1 = model1->parameters();
    auto param2 = model2->parameters();
    auto param3 = model3->parameters();
    for (auto& p : param1) {
      auto name = p.first;
      // Model 1 and 3 should be the same
      REQUIRE(param1[name].norm().toCFloat() == param3[name].norm().toCFloat());
      REQUIRE(param1[name].norm().toCFloat() != param2[name].norm().toCFloat());
    }
  }
}

TEST_CASE("serialization_cuda", "[cuda]") {
  SECTION("xor") {
    // We better be able to save and load a XOR model!
    auto makeModel = []() {
      ContainerList list;
      list.append(make(Linear(2, 8)));
      list.append(make(Linear(8, 1)));
      return make(list);
    };
    auto getLoss = [](std::shared_ptr<ContainerList> model, uint32_t bs) {
      auto inp = at::CPU(at::kFloat).tensor({bs, 2});
      auto lab = at::CPU(at::kFloat).tensor({bs});
      for (auto i = 0U; i < bs; i++) {
        auto a = std::rand() % 2;
        auto b = std::rand() % 2;
        auto c = a ^ b;
        inp[i][0] = a;
        inp[i][1] = b;
        lab[i] = c;
      }

      // forward
      auto x = Var(inp);
      auto y = Var(lab, false);
      for (auto layer : *model)
        x = layer->forward({x})[0].sigmoid_();
      return at::binary_cross_entropy(x, y);
    };

    auto model = makeModel();
    auto model2 = makeModel();
    auto model3 = makeModel();
    auto optim =
        SGD(model, 1e-1).momentum(0.9).nesterov().weight_decay(1e-6).make();

    float running_loss = 1;
    int epoch = 0;
    while (running_loss > 0.1) {
      Variable loss = getLoss(model, 4);
      optim->zero_grad();
      backward(loss);
      optim->step();

      running_loss = running_loss * 0.99 + loss.data().sum().toCFloat() * 0.01;
      REQUIRE(epoch < 3000);
      epoch++;
    }

    std::stringstream ss;
    save(ss, model);
    load(ss, model2);

    auto loss = getLoss(model2, 100);
    REQUIRE(loss.toCFloat() < 0.1);

    model2->cuda();
    ss.clear();
    save(ss, model2);
    load(ss, model3);

    loss = getLoss(model3, 100);
    REQUIRE(loss.toCFloat() < 0.1);
  }
}
