#include <gtest/gtest.h>

#include <c10/util/tempfile.h>

#include <torch/torch.h>

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

  auto tempfile = c10::make_tempfile();
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

  auto tempfile = c10::make_tempfile();
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
  auto model_tempfile = c10::make_tempfile();
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

  auto optim_tempfile = c10::make_tempfile();
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
  auto getLoss = [](Sequential model,
                    uint32_t batch_size,
                    bool is_cuda = false) {
    auto inputs = torch::empty({batch_size, 2});
    auto labels = torch::empty({batch_size});
    if (is_cuda) {
      inputs = inputs.cuda();
      labels = labels.cuda();
    }
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

  auto tempfile = c10::make_tempfile();
  torch::save(model, tempfile.name);
  torch::load(model2, tempfile.name);

  auto loss = getLoss(model2, 100);
  ASSERT_LT(loss.item<float>(), 0.1);

  model2->to(torch::kCUDA);
  loss = getLoss(model2, 100, true);
  ASSERT_LT(loss.item<float>(), 0.1);

  auto tempfile2 = c10::make_tempfile();
  torch::save(model2, tempfile2.name);
  torch::load(model3, tempfile2.name);

  loss = getLoss(model3, 100, true);
  ASSERT_LT(loss.item<float>(), 0.1);
}

TEST(
    SerializeTest,
    CanSerializeModulesWithIntermediateModulesWithoutParametersOrBuffers) {
  struct C : torch::nn::Module {
    C() {
      register_buffer("foo", torch::ones(5, torch::kInt32));
    }
  };
  struct B : torch::nn::Module {};
  struct A : torch::nn::Module {
    A() {
      register_module("b", std::make_shared<B>());
      register_module("c", std::make_shared<C>());
    }
  };
  struct M : torch::nn::Module {
    M() {
      register_module("a", std::make_shared<A>());
    }
  };

  auto out = std::make_shared<M>();
  std::stringstream ss;
  torch::save(out, ss);
  auto in = std::make_shared<M>();
  torch::load(in, ss);

  const int output = in->named_buffers()["a.c.foo"].sum().item<int>();
  ASSERT_EQ(output, 5);
}

TEST(SerializeTest, VectorOfTensors) {
  torch::manual_seed(0);

  std::vector<torch::Tensor> x_vec = { torch::randn({1, 2}), torch::randn({3, 4}) };

  std::stringstream stream;
  torch::save(x_vec, stream);

  std::vector<torch::Tensor> y_vec;
  torch::load(y_vec, stream);

  for (int64_t i = 0; i < x_vec.size(); i++) {
    auto& x = x_vec[i];
    auto& y = y_vec[i];
    ASSERT_TRUE(y.defined());
    ASSERT_EQ(x.sizes().vec(), y.sizes().vec());
    ASSERT_TRUE(x.allclose(y));
  }
}

// NOTE: if a `Module` contains unserializable submodules (e.g. `nn::Functional`),
// we expect those submodules to be skipped when the `Module` is being serialized.
TEST(SerializeTest, UnserializableSubmoduleIsSkippedWhenSavingModule) {
  struct A : torch::nn::Module {
    A() {
      register_module("relu", torch::nn::Functional(torch::relu));
    }
  };

  auto out = std::make_shared<A>();
  std::stringstream ss;
  torch::save(out, ss);

  torch::serialize::InputArchive archive;
  archive.load_from(ss);
  torch::serialize::InputArchive relu_archive;

  // Submodule with name "relu" should not exist in the `InputArchive`,
  // because the "relu" submodule is an `nn::Functional` and is not serializable.
  ASSERT_FALSE(archive.try_read("relu", relu_archive));
}

// NOTE: If a `Module` contains unserializable submodules (e.g. `nn::Functional`),
// we don't check the existence of those submodules in the `InputArchive` when
// deserializing.
TEST(SerializeTest, UnserializableSubmoduleIsIgnoredWhenLoadingModule) {
  struct B : torch::nn::Module {
    B() {
      register_module("relu1", torch::nn::Functional(torch::relu));
      register_buffer("foo", torch::zeros(5, torch::kInt32));
    }
  };
  struct A : torch::nn::Module {
    A() {
      register_module("b", std::make_shared<B>());
      register_module("relu2", torch::nn::Functional(torch::relu));
    }
  };

  auto out = std::make_shared<A>();
  // Manually change the values of "b.foo", so that we can check whether the buffer
  // contains these values after deserialization.
  out->named_buffers()["b.foo"].fill_(1);
  auto tempfile = c10::make_tempfile();
  torch::save(out, tempfile.name);

  torch::serialize::InputArchive archive;
  archive.load_from(tempfile.name);
  torch::serialize::InputArchive archive_b;
  torch::serialize::InputArchive archive_relu;
  torch::Tensor tensor_foo;

  ASSERT_TRUE(archive.try_read("b", archive_b));
  ASSERT_TRUE(archive_b.try_read("foo", tensor_foo, /*is_buffer=*/true));

  // Submodule with name "relu1" should not exist in `archive_b`, because the "relu1"
  // submodule is an `nn::Functional` and is not serializable.
  ASSERT_FALSE(archive_b.try_read("relu1", archive_relu));

  // Submodule with name "relu2" should not exist in `archive`, because the "relu2"
  // submodule is an `nn::Functional` and is not serializable.
  ASSERT_FALSE(archive.try_read("relu2", archive_relu));

  auto in = std::make_shared<A>();
  // `torch::load(...)` works without error, even though `A` contains the `nn::Functional`
  // submodules while the serialized file doesn't, because the `nn::Functional` submodules
  // are not serializable and thus ignored when deserializing.
  torch::load(in, tempfile.name);

  // Check that the "b.foo" buffer is correctly deserialized from the file.
  const int output = in->named_buffers()["b.foo"].sum().item<int>();
  // `output` should equal to the sum of the values we manually assigned to "b.foo" before
  // serialization.
  ASSERT_EQ(output, 5);  
}
