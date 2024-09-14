#include <test/cpp/jit/test_utils.h>

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

TEST(MobileTest, SaveParametersDefaultsToZip) {
  // Save some empty parameters.
  std::map<std::string, at::Tensor> empty_parameters;
  std::stringstream ss_data;
  _save_parameters(empty_parameters, ss_data);

  // Verify that parameters were serialized to a ZIP container.
  EXPECT_GE(ss_data.str().size(), 4);
  EXPECT_EQ(ss_data.str()[0], 'P');
  EXPECT_EQ(ss_data.str()[1], 'K');
  EXPECT_EQ(ss_data.str()[2], '\x03');
  EXPECT_EQ(ss_data.str()[3], '\x04');
}

TEST(MobileTest, SaveParametersCanUseFlatbuffer) {
  // Save some empty parameters using flatbuffer.
  std::map<std::string, at::Tensor> empty_parameters;
  std::stringstream ss_data;
  _save_parameters(empty_parameters, ss_data, /*use_flatbuffer=*/true);

  // Verify that parameters were serialized to a flatbuffer. The flatbuffer
  // magic bytes should be at offsets 4..7. The first four bytes contain an
  // offset to the actual flatbuffer data.
  EXPECT_GE(ss_data.str().size(), 8);
  EXPECT_EQ(ss_data.str()[4], 'P');
  EXPECT_EQ(ss_data.str()[5], 'T');
  EXPECT_EQ(ss_data.str()[6], 'M');
  EXPECT_EQ(ss_data.str()[7], 'F');
}

TEST(MobileTest, SaveLoadParametersUsingFlatbuffers) {
  // Create some simple parameters to save.
  std::map<std::string, at::Tensor> input_params;
  input_params["four_by_ones"] = 4 * torch::ones({});
  input_params["three_by_ones"] = 3 * torch::ones({});

  // Serialize them using flatbuffers.
  std::stringstream data;
  _save_parameters(input_params, data, /*use_flatbuffer=*/true);

  // The flatbuffer magic bytes should be at offsets 4..7.
  EXPECT_EQ(data.str()[4], 'P');
  EXPECT_EQ(data.str()[5], 'T');
  EXPECT_EQ(data.str()[6], 'M');
  EXPECT_EQ(data.str()[7], 'F');

  // Read them back and check that they survived the trip.
  auto output_params = _load_parameters(data);
  EXPECT_EQ(output_params.size(), 2);
  {
    auto four_by_ones = 4 * torch::ones({});
    EXPECT_EQ(
        output_params["four_by_ones"].item<int>(), four_by_ones.item<int>());
  }
  {
    auto three_by_ones = 3 * torch::ones({});
    EXPECT_EQ(
        output_params["three_by_ones"].item<int>(), three_by_ones.item<int>());
  }
}

TEST(MobileTest, LoadParametersUnexpectedFormatShouldThrow) {
  // Manually create some data that doesn't look like a ZIP or Flatbuffer file.
  // Make sure it's longer than 8 bytes, since getFileFormat() needs that much
  // data to detect the type.
  std::stringstream bad_data;
  bad_data << "abcd"
           << "efgh"
           << "ijkl";

  // Loading parameters from it should throw an exception.
  EXPECT_ANY_THROW(_load_parameters(bad_data));
}

TEST(MobileTest, LoadParametersEmptyDataShouldThrow) {
  // Loading parameters from an empty data stream should throw an exception.
  std::stringstream empty;
  EXPECT_ANY_THROW(_load_parameters(empty));
}

TEST(MobileTest, LoadParametersMalformedFlatbuffer) {
  // Manually create some data with Flatbuffer header.
  std::stringstream bad_data;
  bad_data << "PK\x03\x04PTMF\x00\x00"
           << "*}NV\xb3\xfa\xdf\x00pa";

  // Loading parameters from it should throw an exception.
  ASSERT_THROWS_WITH_MESSAGE(
      _load_parameters(bad_data), "Malformed Flatbuffer module");
}

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

TEST(LiteTrainerTest, RandomSamplerReturnsIndicesInCorrectRange) {
  mobile::RandomSampler sampler(10);

  std::vector<size_t> indices = sampler.next(3).value();
  for (auto i : indices) {
    AT_ASSERT(i < 10);
  }

  indices = sampler.next(5).value();
  for (auto i : indices) {
    AT_ASSERT(i < 10);
  }

  indices = sampler.next(2).value();
  for (auto i : indices) {
    AT_ASSERT(i < 10);
  }

  AT_ASSERT(sampler.next(10).has_value() == false);
}

TEST(LiteTrainerTest, RandomSamplerReturnsLessValuesForLastBatch) {
  mobile::RandomSampler sampler(5);
  AT_ASSERT(sampler.next(3).value().size() == 3);
  AT_ASSERT(sampler.next(100).value().size() == 2);
  AT_ASSERT(sampler.next(2).has_value() == false);
}

TEST(LiteTrainerTest, RandomSamplerResetsWell) {
  mobile::RandomSampler sampler(5);
  AT_ASSERT(sampler.next(5).value().size() == 5);
  AT_ASSERT(sampler.next(2).has_value() == false);
  sampler.reset();
  AT_ASSERT(sampler.next(5).value().size() == 5);
  AT_ASSERT(sampler.next(2).has_value() == false);
}

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
