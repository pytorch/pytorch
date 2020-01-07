#include <gtest/gtest.h>

#include <torch/csrc/autograd/functions/comm.h>
#include <torch/nn/module.h>
#include <torch/nn/modules/conv.h>
#include <torch/nn/modules/linear.h>
#include <torch/nn/parallel/data_parallel.h>
#include <torch/nn/pimpl.h>
#include <torch/optim/sgd.h>
#include <torch/types.h>
#include <torch/utils.h>

#include <test/cpp/api/support.h>

#include <iostream>
#include <memory>
#include <utility>
#include <vector>

using namespace torch::autograd;
using namespace torch::nn;

struct ParallelTest : torch::test::SeedingFixture {};

TEST_F(ParallelTest, DifferentiableScatter_MultiCUDA) {
  Scatter scatter(
      {torch::Device(torch::kCUDA, 0), torch::Device(torch::kCUDA, 1)});

  auto input = torch::ones(10, torch::requires_grad(true));
  auto output = scatter.apply({input});

  ASSERT_EQ(output.size(), 2);
  ASSERT_EQ(output[0].size(0), 5);
  ASSERT_EQ(output[1].size(0), 5);

  ASSERT_TRUE(torch::cat({output[0].to(torch::kCPU), output[1].to(torch::kCPU)})
                  .allclose(input));

  torch::Tensor sum = output[0].to({torch::kCUDA, 1}) + output[1];
  sum.backward(torch::ones_like(sum));

  ASSERT_TRUE(input.grad().defined());
  ASSERT_TRUE(input.grad().device().is_cpu());
  ASSERT_EQ(input.grad().sum().item<int32_t>(), 10);
}

TEST_F(ParallelTest, DifferentiableGather_MultiCUDA) {
  Gather gather(torch::Device(torch::kCUDA, 1));

  auto a = torch::ones(5, torch::requires_grad(true).device(torch::kCUDA, 0));
  auto b = torch::ones(5, torch::requires_grad(true).device(torch::kCUDA, 1));

  auto outputs = gather.apply({a, b});
  ASSERT_EQ(outputs.size(), 1);
  torch::Tensor output = outputs.front();

  ASSERT_EQ(output.size(0), 10);
  ASSERT_EQ(output.device(), torch::Device(torch::kCUDA, 1));

  auto chunks = output.chunk(2);
  ASSERT_TRUE(chunks[0].to({torch::kCUDA, 0}).allclose(a));
  ASSERT_TRUE(chunks[1].allclose(b));

  output.backward(torch::ones_like(output));

  ASSERT_TRUE(a.grad().defined());
  ASSERT_EQ(a.grad().device(), torch::Device(torch::kCUDA, 0));
  ASSERT_EQ(a.grad().sum().item<int32_t>(), 5);

  ASSERT_TRUE(b.grad().defined());
  ASSERT_EQ(b.grad().device(), torch::Device(torch::kCUDA, 1));
  ASSERT_EQ(b.grad().sum().item<int32_t>(), 5);
}

TEST_F(ParallelTest, Replicate_MultiCUDA) {
  Linear linear(3, 4);
  auto replicas = parallel::replicate(
      linear, {torch::Device(torch::kCUDA, 0), torch::Device(torch::kCUDA, 1)});
  ASSERT_EQ(replicas.size(), 2);

  auto original_parameters = linear->parameters();

  auto replica1_parameters = replicas[0]->parameters();
  for (auto& parameter : replica1_parameters) {
    ASSERT_EQ(parameter.device(), torch::Device(torch::kCUDA, 0));
  }
  replicas[0]->to(torch::kCPU);
  ASSERT_EQ(replica1_parameters.size(), original_parameters.size());
  for (size_t i = 0; i < original_parameters.size(); ++i) {
    ASSERT_TRUE(replica1_parameters[i].allclose(original_parameters[i]));
    ASSERT_TRUE(
        replica1_parameters[i].data_ptr<float>() !=
        original_parameters[i].data_ptr<float>());
  }

  auto replica2_parameters = replicas[1]->parameters();
  for (auto& parameter : replica2_parameters) {
    ASSERT_EQ(parameter.device(), torch::Device(torch::kCUDA, 1));
  }
  replicas[1]->to(torch::kCPU);
  ASSERT_EQ(replica2_parameters.size(), original_parameters.size());
  for (size_t i = 0; i < original_parameters.size(); ++i) {
    ASSERT_TRUE(replica2_parameters[i].allclose(original_parameters[i]));
    ASSERT_TRUE(
        replica2_parameters[i].data_ptr<float>() !=
        original_parameters[i].data_ptr<float>());
  }
}

TEST_F(ParallelTest, ParallelApply_MultiCUDA) {
  Linear a(3, 4);

  Linear b(std::dynamic_pointer_cast<LinearImpl>(a->clone()));
  b->to({torch::kCUDA, 0});

  Linear c(std::dynamic_pointer_cast<LinearImpl>(a->clone()));
  c->to({torch::kCUDA, 1});

  std::vector<Linear> modules = {a, b, c};
  std::vector<torch::Tensor> inputs = {
      torch::ones({2, 3}),
      torch::ones({2, 3}, torch::device({torch::kCUDA, 0})),
      torch::ones({2, 3}, torch::device({torch::kCUDA, 1}))};

  auto outputs = parallel::parallel_apply(modules, inputs);

  ASSERT_EQ(outputs.size(), 3);
  ASSERT_TRUE(outputs[0].device().is_cpu());

  ASSERT_EQ(outputs[1].device(), torch::Device(torch::kCUDA, 0));
  ASSERT_TRUE(outputs[1].to(torch::kCPU).allclose(outputs[0]));

  ASSERT_EQ(outputs[2].device(), torch::Device(torch::kCUDA, 1));
  ASSERT_TRUE(outputs[2].to(torch::kCPU).allclose(outputs[0]));
}

TEST_F(ParallelTest, ParallelApplyWithDifferentOutputDevice_MultiCUDA) {
  struct M : torch::nn::Module {
    torch::Tensor forward(torch::Tensor input) {
      return torch::ones(5, torch::kInt32);
    }
  };

  std::vector<std::shared_ptr<M>> modules = {
      std::make_shared<M>(), std::make_shared<M>(), std::make_shared<M>()};
  std::vector<torch::Tensor> inputs = {
      torch::empty({}), torch::empty({}), torch::empty({})};
  std::vector<torch::Device> devices = {
      {torch::kCUDA, 1}, {torch::kCUDA, 0}, {torch::kCPU}};

  auto outputs = parallel::parallel_apply(modules, inputs, devices);

  ASSERT_EQ(outputs.size(), 3);
  ASSERT_TRUE(outputs[0].device().is_cuda());
  ASSERT_EQ(outputs[0].device(), torch::Device(torch::kCUDA, 1));

  ASSERT_TRUE(outputs[1].device().is_cuda());
  ASSERT_EQ(outputs[1].device(), torch::Device(torch::kCUDA, 0));

  ASSERT_TRUE(outputs[2].device().is_cpu());
}

TEST_F(ParallelTest, ParallelApplyRethrowsException_MultiCUDA) {
  struct M : torch::nn::Cloneable<M> {
    void reset() override {}
    torch::Tensor forward(torch::Tensor input) {
      throw std::runtime_error("Badness!");
    }
  };

  auto m = std::make_shared<M>();
  auto input = torch::ones({10, 3});
  ASSERT_THROWS_WITH(parallel::data_parallel(m, input), "Badness!");
}

TEST_F(
    ParallelTest,
    DataParallelPlacesTheOutputOnTheRequestedDevice_MultiCUDA) {
  struct M : torch::nn::Cloneable<M> {
    void reset() override {}
    torch::Tensor forward(torch::Tensor input) {
      // The returned tensor should be on the output device.
      return torch::ones(3);
    }
  };
  auto m = std::make_shared<M>();
  auto input = torch::ones({10, 3});
  {
    auto output = parallel::data_parallel(
        m,
        input,
        /*devices=*/torch::nullopt,
        /*output_device=*/torch::Device(torch::kCUDA, 1));
    ASSERT_TRUE(output.defined());
    ASSERT_TRUE(output.device().is_cuda());
    ASSERT_EQ(output.device().index(), 1);
  }
  {
    // Verify for the single-device case (where we don't scatter/gather).
    auto output = parallel::data_parallel(
        m,
        input,
        /*devices=*/std::vector<torch::Device>{torch::Device(torch::kCUDA, 0)},
        /*output_device=*/torch::Device(torch::kCUDA, 1));
    ASSERT_TRUE(output.defined());
    ASSERT_TRUE(output.device().is_cuda());
    ASSERT_EQ(output.device().index(), 1);
  }
}

TEST_F(ParallelTest, DataParallelUsesAllAvailableCUDADevices_CUDA) {
  struct M : torch::nn::Cloneable<M> {
    void reset() override {}
    torch::Tensor forward(torch::Tensor input) {
      return torch::tensor({input.device().index()});
    }
  };

  auto m = std::make_shared<M>();
  auto input = torch::ones({10, 3});
  auto output = parallel::data_parallel(m, input);

  const auto device_count = torch::cuda::device_count();
  ASSERT_EQ(output.numel(), device_count);
  for (size_t i = 0; i < device_count; ++i) {
    ASSERT_EQ(output[i].item<int32_t>(), i);
  }
}

TEST_F(ParallelTest, DataParallelNumericalEquivalence_MultiCUDA) {
  struct M : torch::nn::Cloneable<M> {
      M() {
        reset();
      }

      void reset() override {
        conv = register_module("conv",
            torch::nn::Conv2d(torch::nn::Conv2dOptions(2, 2, /*kernel_size=*/2)));
        fc = register_module("fc", torch::nn::Linear(8, 2));
      }

      torch::Tensor forward(torch::Tensor x) {
        x = conv->forward(x);
        x = torch::relu(x);
        x = x.view({-1, 8});
        x = fc->forward(x);
        return torch::log_softmax(x, /*dim=*/1);
      }

      torch::nn::Conv2d conv{nullptr};
      torch::nn::Linear fc{nullptr};
    };

    // prepare modules and inputs
    auto input = torch::ones({16, 2, 3, 3});
    auto input_dp = torch::ones({16, 2, 3, 3});
    auto model = std::make_shared<M>();
    auto model_dp = std::dynamic_pointer_cast<M>(model->clone());

    // run 3 training iterations
    for (int i = 0; i < 3; ++i) {
      input += i;
      input_dp += i;

      // non-prallel training
      torch::optim::SGD optim(
          model->parameters(), torch::optim::SGDOptions(0.1));
      auto output = model->forward(input);
      auto loss = torch::mse_loss(output, torch::zeros_like(output));
      loss.backward();
      optim.step();

      // data-parallel training
      torch::optim::SGD optim_dp(
          model_dp->parameters(), torch::optim::SGDOptions(0.1));
      auto output_dp = parallel::data_parallel(model_dp, input_dp);
      auto loss_dp = torch::mse_loss(output_dp, torch::zeros_like(output_dp));
      loss_dp.backward();
      optim_dp.step();

      // make sure that weights are the same
      model->to(torch::kCPU);
      model_dp->to(torch::kCPU);
      auto params = model->parameters();
      auto params_dp = model_dp->parameters();
      ASSERT_EQ(params.size(), params_dp.size());
      for (auto it = params.begin(), it_dp = params_dp.begin();
          it != params.end() && it_dp != params.end();
          ++it, ++it_dp) {
        ASSERT_TRUE(torch::allclose(*it, *it_dp));
      }
    }
}
