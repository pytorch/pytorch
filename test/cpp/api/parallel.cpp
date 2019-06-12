#include <gtest/gtest.h>

#include <torch/csrc/autograd/functions/comm.h>
#include <torch/nn/module.h>
#include <torch/nn/modules/linear.h>
#include <torch/nn/parallel/data_parallel.h>
#include <torch/nn/pimpl.h>
#include <torch/types.h>

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
  sum.backward();

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

  output.backward();

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
        replica1_parameters[i].data<float>() !=
        original_parameters[i].data<float>());
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
        replica2_parameters[i].data<float>() !=
        original_parameters[i].data<float>());
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
      return torch::ones({5}, torch::dtype(torch::kInt32));
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
      // Intermediate tensors should be on the replica's current device.
      intermediate_tensor = torch::rand(5);
      // The returned tensor should be on the output device.
      return torch::ones(3);
    }
    torch::Tensor intermediate_tensor;
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
    ASSERT_TRUE(m->intermediate_tensor.defined());
    ASSERT_TRUE(m->intermediate_tensor.device().is_cuda());
    ASSERT_EQ(m->intermediate_tensor.device().index(), 0);
    ASSERT_TRUE(output.defined());
    ASSERT_TRUE(output.device().is_cuda());
    ASSERT_EQ(output.device().index(), 1);
  }
}

TEST_F(ParallelTest, DataParallelUsesAllAvailableCUDADevices_CUDA) {
  struct M : torch::nn::Cloneable<M> {
    void reset() override {}
    torch::Tensor forward(torch::Tensor input) {
      return torch::tensor(torch::getDefaultTensorOptions().device().index());
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
