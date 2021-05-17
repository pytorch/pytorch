#include <gtest/gtest.h>

#include <c10/core/TensorOptions.h>
#include <torch/csrc/autograd/generated/variable_factories.h>
#include <torch/csrc/jit/api/module.h>
#include <torch/csrc/jit/mobile/import.h>
#include <torch/csrc/jit/mobile/import_data.h>
#include <torch/csrc/jit/mobile/module.h>
#include <torch/csrc/jit/mobile/train/export_data.h>
#include <torch/csrc/jit/mobile/train/optim/sgd.h>
#include <torch/csrc/jit/mobile/train/random.h>
#include <torch/csrc/jit/mobile/train/sequential.h>
#include <torch/csrc/jit/serialization/import.h>
#include <torch/data/dataloader.h>
#include <torch/torch.h>

// Tests go in torch::jit
namespace torch {
namespace jit {

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(LiteTrainerTest, Params) {
  Module m("m");
  m.register_parameter("foo", torch::ones({1}, at::requires_grad()), false);
  m.define(R"(
    def forward(self, x):
      b = 1.0
      return self.foo * x + b
  )");
  double learning_rate = 0.1, momentum = 0.1;
  int n_epoc = 10;
  // init: y = x + 1;
  // target: y = 2 x + 1
  std::vector<std::pair<Tensor, Tensor>> trainData{
      {1 * torch::ones({1}), 3 * torch::ones({1})},
  };
  // Reference: Full jit
  std::stringstream ms;
  m.save(ms);
  auto mm = load(ms);
  //  mm.train();
  std::vector<::at::Tensor> parameters;
  for (auto parameter : mm.parameters()) {
    parameters.emplace_back(parameter);
  }
  ::torch::optim::SGD optimizer(
      parameters, ::torch::optim::SGDOptions(learning_rate).momentum(momentum));
  for (int epoc = 0; epoc < n_epoc; ++epoc) {
    for (auto& data : trainData) {
      auto source = data.first, targets = data.second;
      optimizer.zero_grad();
      std::vector<IValue> train_inputs{source};
      auto output = mm.forward(train_inputs).toTensor();
      auto loss = ::torch::l1_loss(output, targets);
      loss.backward();
      optimizer.step();
    }
  }
  std::stringstream ss;
  m._save_for_mobile(ss);
  mobile::Module bc = _load_for_mobile(ss);
  std::vector<::at::Tensor> bc_parameters = bc.parameters();
  ::torch::optim::SGD bc_optimizer(
      bc_parameters,
      ::torch::optim::SGDOptions(learning_rate).momentum(momentum));
  for (int epoc = 0; epoc < n_epoc; ++epoc) {
    for (auto& data : trainData) {
      auto source = data.first, targets = data.second;
      bc_optimizer.zero_grad();
      std::vector<IValue> train_inputs{source};
      auto output = bc.forward(train_inputs).toTensor();
      auto loss = ::torch::l1_loss(output, targets);
      loss.backward();
      bc_optimizer.step();
    }
  }
  AT_ASSERT(parameters[0].item<float>() == bc_parameters[0].item<float>());
}

// TODO Renable these tests after parameters are correctly loaded on mobile
/*
TEST(MobileTest, NamedParameters) {
  Module m("m");
  m.register_parameter("foo", torch::ones({}), false);
  m.define(R"(
    def add_it(self, x):
      b = 4
      return self.foo + x + b
  )");
  Module child("m2");
  child.register_parameter("foo", 4 * torch::ones({}), false);
  child.register_parameter("bar", 4 * torch::ones({}), false);
  m.register_module("child1", child);
  m.register_module("child2", child.clone());
  std::stringstream ss;
  m._save_for_mobile(ss);
  mobile::Module bc = _load_for_mobile(ss);

  auto full_params = m.named_parameters();
  auto mobile_params = bc.named_parameters();
  AT_ASSERT(full_params.size() == mobile_params.size());
  for (const auto& e : full_params) {
    AT_ASSERT(e.value.item().toInt() ==
    mobile_params[e.name].item().toInt());
  }
}

TEST(MobileTest, SaveLoadParameters) {
  Module m("m");
  m.register_parameter("foo", torch::ones({}), false);
  m.define(R"(
    def add_it(self, x):
      b = 4
      return self.foo + x + b
  )");
  Module child("m2");
  child.register_parameter("foo", 4 * torch::ones({}), false);
  child.register_parameter("bar", 3 * torch::ones({}), false);
  m.register_module("child1", child);
  m.register_module("child2", child.clone());
  auto full_params = m.named_parameters();
  std::stringstream ss;
  std::stringstream ss_data;
  m._save_for_mobile(ss);

  // load mobile module, save mobile named parameters
  mobile::Module bc = _load_for_mobile(ss);
  _save_parameters(bc.named_parameters(), ss_data);

  // load back the named parameters, compare to full-jit Module's
  auto mobile_params = _load_parameters(ss_data);
  AT_ASSERT(full_params.size() == mobile_params.size());
  for (const auto& e : full_params) {
    AT_ASSERT(e.value.item<int>() == mobile_params[e.name].item<int>());
  }
}
*/

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(MobileTest, SaveLoadParametersEmpty) {
  Module m("m");
  m.define(R"(
    def add_it(self, x):
      b = 4
      return x + b
  )");
  Module child("m2");
  m.register_module("child1", child);
  m.register_module("child2", child.clone());
  std::stringstream ss;
  std::stringstream ss_data;
  m._save_for_mobile(ss);

  // load mobile module, save mobile named parameters
  mobile::Module bc = _load_for_mobile(ss);
  _save_parameters(bc.named_parameters(), ss_data);

  // load back the named parameters, test is empty
  auto mobile_params = _load_parameters(ss_data);
  AT_ASSERT(mobile_params.size() == 0);
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(LiteTrainerTest, SGD) {
  Module m("m");
  m.register_parameter("foo", torch::ones({1}, at::requires_grad()), false);
  m.define(R"(
    def forward(self, x):
      b = 1.0
      return self.foo * x + b
  )");
  double learning_rate = 0.1, momentum = 0.1;
  int n_epoc = 10;
  // init: y = x + 1;
  // target: y = 2 x + 1
  std::vector<std::pair<Tensor, Tensor>> trainData{
      {1 * torch::ones({1}), 3 * torch::ones({1})},
  };
  // Reference: Full jit and torch::optim::SGD
  std::stringstream ms;
  m.save(ms);
  auto mm = load(ms);
  std::vector<::at::Tensor> parameters;
  for (auto parameter : mm.parameters()) {
    parameters.emplace_back(parameter);
  }
  ::torch::optim::SGD optimizer(
      parameters, ::torch::optim::SGDOptions(learning_rate).momentum(momentum));
  for (int epoc = 0; epoc < n_epoc; ++epoc) {
    for (auto& data : trainData) {
      auto source = data.first, targets = data.second;
      optimizer.zero_grad();
      std::vector<IValue> train_inputs{source};
      auto output = mm.forward(train_inputs).toTensor();
      auto loss = ::torch::l1_loss(output, targets);
      loss.backward();
      optimizer.step();
    }
  }
  // Test: lite interpreter and torch::jit::mobile::SGD
  std::stringstream ss;
  m._save_for_mobile(ss);
  mobile::Module bc = _load_for_mobile(ss);
  std::vector<::at::Tensor> bc_parameters = bc.parameters();
  ::torch::jit::mobile::SGD bc_optimizer(
      bc_parameters,
      ::torch::jit::mobile::SGDOptions(learning_rate).momentum(momentum));
  for (int epoc = 0; epoc < n_epoc; ++epoc) {
    for (auto& data : trainData) {
      auto source = data.first, targets = data.second;
      bc_optimizer.zero_grad();
      std::vector<IValue> train_inputs{source};
      auto output = bc.forward(train_inputs).toTensor();
      auto loss = ::torch::l1_loss(output, targets);
      loss.backward();
      bc_optimizer.step();
    }
  }
  AT_ASSERT(parameters[0].item<float>() == bc_parameters[0].item<float>());
}

namespace {
struct DummyDataset : torch::data::datasets::Dataset<DummyDataset, int> {
  explicit DummyDataset(size_t size = 100) : size_(size) {}

  int get(size_t index) override {
    // NOLINTNEXTLINE(bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions)
    return 1 + index;
  }
  torch::optional<size_t> size() const override {
    return size_;
  }

  size_t size_;
};
} // namespace

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(LiteTrainerTest, SequentialSampler) {
  // test that sampler can be used with dataloader
  const int kBatchSize = 10;
  auto data_loader = torch::data::make_data_loader<mobile::SequentialSampler>(
      DummyDataset(25), kBatchSize);
  int i = 1;
  for (const auto& batch : *data_loader) {
    for (const auto& example : batch) {
      AT_ASSERT(i == example);
      i++;
    }
  }
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(LiteTrainerTest, RandomSamplerReturnsIndicesInCorrectRange) {
  mobile::RandomSampler sampler(10);

  std::vector<size_t> indices = sampler.next(3).value();
  for (auto i : indices) {
    AT_ASSERT(i >= 0);
    AT_ASSERT(i < 10);
  }

  indices = sampler.next(5).value();
  for (auto i : indices) {
    AT_ASSERT(i >= 0);
    AT_ASSERT(i < 10);
  }

  indices = sampler.next(2).value();
  for (auto i : indices) {
    AT_ASSERT(i >= 0);
    AT_ASSERT(i < 10);
  }

  AT_ASSERT(sampler.next(10).has_value() == false);
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(LiteTrainerTest, RandomSamplerReturnsLessValuesForLastBatch) {
  mobile::RandomSampler sampler(5);
  AT_ASSERT(sampler.next(3).value().size() == 3);
  AT_ASSERT(sampler.next(100).value().size() == 2);
  AT_ASSERT(sampler.next(2).has_value() == false);
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(LiteTrainerTest, RandomSamplerResetsWell) {
  mobile::RandomSampler sampler(5);
  AT_ASSERT(sampler.next(5).value().size() == 5);
  AT_ASSERT(sampler.next(2).has_value() == false);
  sampler.reset();
  AT_ASSERT(sampler.next(5).value().size() == 5);
  AT_ASSERT(sampler.next(2).has_value() == false);
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(LiteTrainerTest, RandomSamplerResetsWithNewSizeWell) {
  mobile::RandomSampler sampler(5);
  AT_ASSERT(sampler.next(5).value().size() == 5);
  AT_ASSERT(sampler.next(2).has_value() == false);
  sampler.reset(7);
  AT_ASSERT(sampler.next(7).value().size() == 7);
  AT_ASSERT(sampler.next(2).has_value() == false);
  sampler.reset(3);
  AT_ASSERT(sampler.next(3).value().size() == 3);
  AT_ASSERT(sampler.next(2).has_value() == false);
}

} // namespace jit
} // namespace torch
