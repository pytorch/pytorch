#include <c10/core/TensorOptions.h>
#include <test/cpp/jit/test_base.h>
#include <torch/csrc/autograd/generated/variable_factories.h>
#include <torch/csrc/jit/api/module.h>
#include <torch/csrc/jit/mobile/export.h>
#include <torch/csrc/jit/mobile/import.h>
#include <torch/csrc/jit/mobile/import_data.h>
#include <torch/csrc/jit/mobile/module.h>
#include <torch/csrc/jit/mobile/optim/sgd.h>
#include <torch/csrc/jit/serialization/import.h>
#include <torch/torch.h>

// Tests go in torch::jit
namespace torch {
namespace jit {

void testLiteInterpreterParams() {
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

void testMobileNamedParameters() {
  Module m("m");
  m.register_parameter("foo", torch::ones({}), false);
  m.define(R"(
    def add_it(self, x):
      b = 4
      return self.foo + x + b
  )");
  Module child("m2");
  child.register_parameter("foo", 4 * torch::ones({}), false);
  m.register_module("child1", child);
  m.register_module("child2", child);
  std::stringstream ss;
  m._save_for_mobile(ss);
  mobile::Module bc = _load_for_mobile(ss);


  auto full_params = m.named_parameters();
  std::cout << "full: before modification " << std::endl;
  for (const auto& e : full_params) {
    // note that child1.foo and child2.foo are supposed to start with the same value of 4
    // however, by the time child2.foo is printed here, it will already be -4
    // this indicates that modifying child1.foo is also modifying child2.foo
    std::cout << e.name << ", " << e.value << std::endl;
    auto tensor = e.value;
    tensor *= -1;
  }
  std::cout << "full: after modification " << std::endl;
  full_params = m.named_parameters();
  for (const auto& e : full_params) {
    std::cout << e.name << ", " << e.value << std::endl;
  }

  std::cout << std::endl;
  std::cout << "mobile: before modification " << std::endl;
  auto mobile_params = bc.named_parameters();
  for (const auto& e : mobile_params) {
    std::cout << e.first << ", " << e.second << std::endl;
    auto tensor = e.second;
    tensor *= -1;
  }
  std::cout << "mobile: after modification " << std::endl;
  auto delta = bc.named_parameters();
  for (const auto& e : delta) {
    std::cout << e.first << ", " << e.second << std::endl;
  }

//  AT_ASSERT(full_params.size() == mobile_params.size());
//  for (const auto& e : full_params) {
//    std::cout << e.name << std::endl;
//    std::cout << mobile_params[e.name].item().toInt() << std::endl;
//    AT_ASSERT(e.value.item().toInt() == (mobile_params[e.name].item().toInt() * -1));
//  }
}

void testMobileSaveLoadData() {
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
  m.register_module("child2", child);
  auto full_params = m.named_parameters();

  std::stringstream ss;
  std::stringstream ss_data;
  m._save_for_mobile(ss);
  mobile::Module bc = _load_for_mobile(ss);

  _save_parameters(bc, ss_data);
  auto mobile_params = _load_parameters(ss_data);
  AT_ASSERT(full_params.size() == mobile_params.size());
  for (const auto& e : full_params) {
    AT_ASSERT(e.value.item<int>() == mobile_params[e.name].item<int>());
  }
}

void testLiteSGD() {
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

} // namespace jit
} // namespace torch
