#include <gtest/gtest.h>

#include <torch/torch.h>

#include <test/cpp/api/optim_baseline.h>
#include <test/cpp/api/support.h>

#include <cmath>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <memory>
#include <random>
#include <vector>

using namespace torch::nn;
using namespace torch::optim;

template <typename OptimizerClass, typename Options>
bool test_optimizer_xor(Options options) {
  torch::manual_seed(0);

  Sequential model(
      Linear(2, 8),
      Functional(torch::sigmoid),
      Linear(8, 1),
      Functional(torch::sigmoid));

  const int64_t kBatchSize = 200;
  const int64_t kMaximumNumberOfEpochs = 3000;

  OptimizerClass optimizer(model->parameters(), options);

  float running_loss = 1;
  int epoch = 0;
  while (running_loss > 0.1) {
    auto inputs = torch::empty({kBatchSize, 2});
    auto labels = torch::empty({kBatchSize});
    for (size_t i = 0; i < kBatchSize; i++) {
      inputs[i] = torch::randint(2, {2}, torch::kInt64);
      labels[i] = inputs[i][0].item<int64_t>() ^ inputs[i][1].item<int64_t>();
    }

    inputs.set_requires_grad(true);

    auto step = [&](OptimizerClass& optimizer, Sequential model, torch::Tensor inputs, torch::Tensor labels) {
      auto closure = [&]() {
        optimizer.zero_grad();
        auto x = model->forward(inputs);
        auto loss = torch::binary_cross_entropy(x, labels);
        loss.backward();
        return loss;
      };
      return optimizer.step(closure);
    };

    torch::Tensor loss = step(optimizer, model, inputs, labels);

    running_loss = running_loss * 0.99 + loss.item<float>() * 0.01;
    if (epoch > kMaximumNumberOfEpochs) {
      std::cout << "Loss is too high after epoch " << epoch << ": "
                << running_loss << std::endl;
      return false;
    }
    epoch++;
  }
  return true;
}

template <typename Parameters>
void assign_parameter(
    const Parameters& parameters,
    const char* name,
    torch::Tensor new_tensor) {
  auto parameter = parameters[name];
  parameter.set_requires_grad(false);
  parameter.flatten().copy_(new_tensor);
  parameter.set_requires_grad(true);
}

template <typename OptimizerClass, typename Options>
void check_exact_values(
    Options options,
    std::vector<std::vector<torch::Tensor>> expected_parameters) {
  const size_t kIterations = 1001;
  const size_t kSampleEvery = 100;

  torch::manual_seed(0);

  Sequential model(
      Linear(2, 3),
      Functional(torch::sigmoid),
      Linear(3, 1),
      Functional(torch::sigmoid));

  model->to(torch::kFloat64);

  // Use exact input values because matching random values is hard.
  auto parameters = model->named_parameters();
  assign_parameter(
      parameters,
      "0.weight",
      torch::tensor({-0.2109, -0.4976, -0.1413, -0.3420, -0.2524, 0.6976}, torch::kFloat64));
  assign_parameter(
      parameters, "0.bias", torch::tensor({-0.1085, -0.2979, 0.6892}, torch::kFloat64));
  assign_parameter(
      parameters, "2.weight", torch::tensor({-0.0508, -0.3941, -0.2843}, torch::kFloat64));
  assign_parameter(parameters, "2.bias", torch::tensor({-0.0711}, torch::kFloat64));

  auto optimizer = OptimizerClass(parameters.values(), options);
  torch::Tensor input =
      torch::tensor({0.1, 0.2, 0.3, 0.4, 0.5, 0.6}, torch::kFloat64).reshape({3, 2});

  for (size_t i = 0; i < kIterations; ++i) {
    optimizer.zero_grad();
    auto output = model->forward(input);
    auto loss = output.sum();
    loss.backward();

    auto closure = []() { return torch::tensor({10}); };
    optimizer.step(closure);

    if (i % kSampleEvery == 0) {
      ASSERT_TRUE(
          expected_parameters.at(i / kSampleEvery).size() == parameters.size());
      for (size_t p = 0; p < parameters.size(); ++p) {
        ASSERT_TRUE(parameters[p]->defined());
        // Always compare using double dtype, regardless of the original dtype of the tensors
        auto computed = parameters[p]->flatten().to(torch::kFloat64);
        auto expected = expected_parameters.at(i / kSampleEvery).at(p).to(torch::kFloat64);
        if (!computed.allclose(expected, /*rtol=*/1e-3, /*atol=*/5e-4)) {
          std::cout << "Iteration " << i << ": " << computed
                    << " != " << expected << " (parameter " << p << ")"
                    << std::endl;
          ASSERT_TRUE(false);
        }
      }
    }
  }
}

TEST(OptimTest, OptimizerAccessors) {
  auto options = AdagradOptions(1.0);
  std::vector<torch::Tensor> params;
  for (size_t i = 0; i < 3; i++) {
    params.push_back(torch::randn(10));
  }
  auto optimizer = Adagrad(params, options);
  // test for defaults() method with non-const reference
  auto& options_ = static_cast<AdagradOptions&>(optimizer.defaults());
  ASSERT_TRUE(options == options_);
  // test for param_groups() with non-const reference return
  auto& params_groups = optimizer.param_groups();
  params_groups.push_back(OptimizerParamGroup(params));
  auto& params_1 = params_groups[1].params();
  for (size_t i = 0; i < params_1.size(); i++) {
    torch::equal(params[i], params_1[i]);
  }

  // test for add_param_group() when one or more params existing in another param_group
  // are passed in the new param group to be added
  ASSERT_THROWS_WITH(
    optimizer.add_param_group(OptimizerParamGroup(params)), "some parameters appear in more than one parameter group");

  // test for state() with non-const reference return
  auto& state_ = static_cast<AdagradParamState&>(*(optimizer.state()[c10::guts::to_string(params_1[0].unsafeGetTensorImpl())]));
  state_.step(state_.step()+1);

  const auto& optimizer_ = Adagrad(params, options);
  optimizer_.defaults();
  // test for param_groups() with const reference return
  const auto& params_2 = optimizer_.param_groups();
  // test for state() with const reference return
  optimizer_.state();
}

#define OLD_INTERFACE_WARNING_CHECK(func)       \
  {                                             \
    torch::test::WarningCapture warnings;       \
    func;                                       \
    ASSERT_EQ(                                  \
        torch::test::count_substr_occurrences(  \
            warnings.str(), "will be removed"), \
        1);                                     \
  }

struct MyOptimizerOptions : public OptimizerCloneableOptions<MyOptimizerOptions> {
  MyOptimizerOptions(double lr = 1.0) : lr_(lr) {};
  TORCH_ARG(double, lr) = 1.0;
};

TEST(OptimTest, OldInterface) {
  struct MyOptimizer : Optimizer {
    using Optimizer::Optimizer;
    torch::Tensor step(LossClosure closure = nullptr) override { return {};}
    explicit MyOptimizer(
        std::vector<at::Tensor> params, MyOptimizerOptions defaults = {}) :
          Optimizer({std::move(OptimizerParamGroup(params))}, std::make_unique<MyOptimizerOptions>(defaults)) {}
  };
  std::vector<torch::Tensor> parameters = {
      torch::ones({2, 3}), torch::zeros({2, 3}), torch::rand({2, 3})};
  {
    MyOptimizer optimizer(parameters);
    size_t size;
    OLD_INTERFACE_WARNING_CHECK(size = optimizer.size());
    ASSERT_EQ(size, parameters.size());
  }
  {
    std::vector<at::Tensor> params;
    MyOptimizer optimizer(params);

    size_t size;
    OLD_INTERFACE_WARNING_CHECK(size = optimizer.size());
    ASSERT_EQ(size, 0);

    OLD_INTERFACE_WARNING_CHECK(optimizer.add_parameters(parameters));

    OLD_INTERFACE_WARNING_CHECK(size = optimizer.size());
    ASSERT_EQ(size, parameters.size());

    std::vector<torch::Tensor> params_;
    OLD_INTERFACE_WARNING_CHECK(params_ = optimizer.parameters());
    for (size_t p = 0; p < size; ++p) {
      ASSERT_TRUE(params_[p].allclose(parameters[p]));
    }
  }
  {
    Linear linear(3, 4);
    MyOptimizer optimizer(linear->parameters());

    size_t size;
    OLD_INTERFACE_WARNING_CHECK(size = optimizer.size());
    ASSERT_EQ(size, linear->parameters().size());
  }
}

TEST(OptimTest, XORConvergence_SGD) {
  ASSERT_TRUE(test_optimizer_xor<SGD>(
      SGDOptions(0.1).momentum(0.9).nesterov(true).weight_decay(1e-6)));
}

TEST(OptimTest, XORConvergence_LBFGS) {
  ASSERT_TRUE(test_optimizer_xor<LBFGS>(LBFGSOptions(1.0)));
  ASSERT_TRUE(test_optimizer_xor<LBFGS>(LBFGSOptions(1.0).line_search_fn("strong_wolfe")));
}

TEST(OptimTest, XORConvergence_Adagrad) {
  ASSERT_TRUE(test_optimizer_xor<Adagrad>(
      AdagradOptions(1.0).weight_decay(1e-6).lr_decay(1e-3)));
}

TEST(OptimTest, XORConvergence_RMSprop) {
  ASSERT_TRUE(test_optimizer_xor<RMSprop>(RMSpropOptions(0.1).centered(true)));
}

TEST(OptimTest, XORConvergence_RMSpropWithMomentum) {
  ASSERT_TRUE(test_optimizer_xor<RMSprop>(
      RMSpropOptions(0.1).momentum(0.9).weight_decay(1e-6)));
}

TEST(OptimTest, XORConvergence_Adam) {
  ASSERT_TRUE(test_optimizer_xor<Adam>(AdamOptions(0.1).weight_decay(1e-6)));
}

TEST(OptimTest, XORConvergence_AdamWithAmsgrad) {
  ASSERT_TRUE(test_optimizer_xor<Adam>(
      AdamOptions(0.1).weight_decay(1e-6).amsgrad(true)));
}

TEST(OptimTest, ProducesPyTorchValues_Adam) {
  check_exact_values<Adam>(AdamOptions(1.0), expected_parameters::Adam());
}

TEST(OptimTest, ProducesPyTorchValues_AdamWithWeightDecay) {
  check_exact_values<Adam>(
      AdamOptions(1.0).weight_decay(1e-2),
      expected_parameters::Adam_with_weight_decay());
}

TEST(OptimTest, ProducesPyTorchValues_AdamWithWeightDecayAndAMSGrad) {
  check_exact_values<Adam>(
      AdamOptions(1.0).weight_decay(1e-6).amsgrad(true),
      expected_parameters::Adam_with_weight_decay_and_amsgrad());
}

TEST(OptimTest, XORConvergence_AdamW) {
  ASSERT_TRUE(test_optimizer_xor<AdamW>(AdamWOptions(0.1)));
}

TEST(OptimTest, XORConvergence_AdamWWithAmsgrad) {
  ASSERT_TRUE(test_optimizer_xor<AdamW>(
      AdamWOptions(0.1).amsgrad(true)));
}

TEST(OptimTest, ProducesPyTorchValues_AdamW) {
  check_exact_values<AdamW>(AdamWOptions(1.0), expected_parameters::AdamW());
}

TEST(OptimTest, ProducesPyTorchValues_AdamWWithoutWeightDecay) {
  check_exact_values<AdamW>(
      AdamWOptions(1.0).weight_decay(0),
      expected_parameters::AdamW_without_weight_decay());
}

TEST(OptimTest, ProducesPyTorchValues_AdamWWithAMSGrad) {
  check_exact_values<AdamW>(
      AdamWOptions(1.0).amsgrad(true),
      expected_parameters::AdamW_with_amsgrad());
}

TEST(OptimTest, ProducesPyTorchValues_Adagrad) {
  check_exact_values<Adagrad>(
      AdagradOptions(1.0), expected_parameters::Adagrad());
}

TEST(OptimTest, ProducesPyTorchValues_AdagradWithWeightDecay) {
  check_exact_values<Adagrad>(
      AdagradOptions(1.0).weight_decay(1e-2),
      expected_parameters::Adagrad_with_weight_decay());
}

TEST(OptimTest, ProducesPyTorchValues_AdagradWithWeightDecayAndLRDecay) {
  check_exact_values<Adagrad>(
      AdagradOptions(1.0).weight_decay(1e-6).lr_decay(1e-3),
      expected_parameters::Adagrad_with_weight_decay_and_lr_decay());
}

TEST(OptimTest, ProducesPyTorchValues_RMSprop) {
  check_exact_values<RMSprop>(
      RMSpropOptions(0.1), expected_parameters::RMSprop());
}

TEST(OptimTest, ProducesPyTorchValues_RMSpropWithWeightDecay) {
  check_exact_values<RMSprop>(
      RMSpropOptions(0.1).weight_decay(1e-2),
      expected_parameters::RMSprop_with_weight_decay());
}

TEST(OptimTest, ProducesPyTorchValues_RMSpropWithWeightDecayAndCentered) {
  check_exact_values<RMSprop>(
      RMSpropOptions(0.1).weight_decay(1e-6).centered(true),
      expected_parameters::RMSprop_with_weight_decay_and_centered());
}

TEST(
    OptimTest,
    ProducesPyTorchValues_RMSpropWithWeightDecayAndCenteredAndMomentum) {
  check_exact_values<RMSprop>(
      RMSpropOptions(0.1).weight_decay(1e-6).centered(true).momentum(0.9),
      expected_parameters::
          RMSprop_with_weight_decay_and_centered_and_momentum());
}

TEST(OptimTest, ProducesPyTorchValues_SGD) {
  check_exact_values<SGD>(SGDOptions(0.1), expected_parameters::SGD());
}

TEST(OptimTest, ProducesPyTorchValues_SGDWithWeightDecay) {
  check_exact_values<SGD>(
      SGDOptions(0.1).weight_decay(1e-2),
      expected_parameters::SGD_with_weight_decay());
}

TEST(OptimTest, ProducesPyTorchValues_SGDWithWeightDecayAndMomentum) {
  check_exact_values<SGD>(
      SGDOptions(0.1).weight_decay(1e-2).momentum(0.9),
      expected_parameters::SGD_with_weight_decay_and_momentum());
}

TEST(OptimTest, ProducesPyTorchValues_SGDWithWeightDecayAndNesterovMomentum) {
  check_exact_values<SGD>(
      SGDOptions(0.1).weight_decay(1e-6).momentum(0.9).nesterov(true),
      expected_parameters::SGD_with_weight_decay_and_nesterov_momentum());
}

TEST(OptimTest, ProducesPyTorchValues_LBFGS) {
  check_exact_values<LBFGS>(
      LBFGSOptions(1.0),
      expected_parameters::LBFGS());
}

TEST(OptimTest, ProducesPyTorchValues_LBFGS_with_line_search) {
  check_exact_values<LBFGS>(
      LBFGSOptions(1.0).line_search_fn("strong_wolfe"),
      expected_parameters::LBFGS_with_line_search());
}

TEST(OptimTest, ZeroGrad) {
  torch::manual_seed(0);

  Linear model(2, 8);
  SGD optimizer(model->parameters(), 0.1);

  for (const auto& parameter : model->parameters()) {
    ASSERT_FALSE(parameter.grad().defined());
  }

  auto output = model->forward(torch::ones({5, 2}));
  auto loss = output.sum();
  loss.backward();

  for (const auto& parameter : model->parameters()) {
    ASSERT_TRUE(parameter.grad().defined());
    ASSERT_GT(parameter.grad().sum().item<float>(), 0);
  }

  optimizer.zero_grad();

  for (const auto& parameter : model->parameters()) {
    ASSERT_TRUE(parameter.grad().defined());
    ASSERT_EQ(parameter.grad().sum().item<float>(), 0);
  }
}

TEST(OptimTest, ExternalVectorOfParameters) {
  torch::manual_seed(0);

  std::vector<torch::Tensor> parameters = {
      torch::randn({2, 2}), torch::randn({3, 3}), torch::randn({4, 4})};
  std::vector<torch::Tensor> original_parameters = {
      parameters[0].clone(), parameters[1].clone(), parameters[2].clone()};

  // Set all gradients to one
  for (auto& parameter : parameters) {
    parameter.mutable_grad() = torch::ones_like(parameter);
  }

  SGD optimizer(parameters, 1.0);

  optimizer.step();

  ASSERT_TRUE(parameters[0].allclose(original_parameters[0] - 1.0));
  ASSERT_TRUE(parameters[1].allclose(original_parameters[1] - 1.0));
  ASSERT_TRUE(parameters[2].allclose(original_parameters[2] - 1.0));
}

TEST(OptimTest, AddParameter_LBFGS) {
  torch::manual_seed(0);

  std::vector<torch::Tensor> parameters = {torch::randn({5, 5})};
  std::vector<torch::Tensor> original_parameters = {parameters[0].clone()};

  // Set all gradients to one
  for (auto& parameter : parameters) {
    parameter.mutable_grad() = torch::ones_like(parameter);
  }

  LBFGS optimizer(std::vector<torch::Tensor>{}, 1.0);
  OLD_INTERFACE_WARNING_CHECK(optimizer.add_parameters(parameters));

  optimizer.step([]() { return torch::tensor(1); });

  // REQUIRE this doesn't throw
}

//Check whether the learning rate of the parameter groups in the optimizer are the
//same as the expected learning rates given in the epoch:learning rate map
void check_lr_change(
    Optimizer& optimizer,
    LRScheduler& lr_scheduler,
    std::map<unsigned, double> expected_epoch_lrs) {

  //Find maximum epoch in map
  unsigned kIterations =
    std::max_element(expected_epoch_lrs.begin(),
                     expected_epoch_lrs.end(),
                     [] (const std::pair<unsigned, double>& a,
                         const std::pair<unsigned, double>& b) -> bool {
                       return a.second > b.second;
                     })->first;

  for(unsigned i = 0; i <= kIterations; i++) {
    const auto epoch_iter = expected_epoch_lrs.find(i);
    if(epoch_iter != expected_epoch_lrs.end())
    {
      //Compare the similarity of the two floating point learning rates
      ASSERT_TRUE(fabs(epoch_iter->second - optimizer.param_groups()[0].options().get_lr()) <
                  std::numeric_limits<double>::epsilon());
    }
    optimizer.step();
    lr_scheduler.step();
  }

}

TEST(OptimTest, CheckLRChange_StepLR_Adam) {

  torch::Tensor parameters = torch::zeros({1});
  auto optimizer = Adam({parameters}, AdamOptions().lr(1e-3));

  const unsigned step_size = 20;
  const double gamma = 0.5;
  StepLR step_lr_scheduler(optimizer, step_size, gamma);

  //The learning rate should have halved at epoch 20
  const std::map<unsigned, double> expected_epoch_lrs = {
    {1, 1e-3},
    {25, 5e-4}
  };

  check_lr_change(optimizer, step_lr_scheduler, expected_epoch_lrs);
}
