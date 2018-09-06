#include <catch.hpp>

#include <torch/csrc/autograd/functions/comm.h>
#include <torch/nn/module.h>
#include <torch/nn/modules/linear.h>
#include <torch/nn/parallel/data_parallel.h>
#include <torch/nn/pimpl.h>
#include <torch/tensor.h>

#include <iostream>
#include <memory>
#include <utility>
#include <vector>

using Catch::StartsWith;

using namespace torch::autograd;
using namespace torch::nn;

#ifdef USE_CUDA

TEST_CASE("Parallel/DifferentiableScatter", "[multi-cuda]") {
  Scatter scatter(
      {torch::Device(torch::kCUDA, 0), torch::Device(torch::kCUDA, 1)});

  auto input = torch::ones(10, torch::requires_grad(true));
  auto output = scatter.apply({input});

  REQUIRE(output.size() == 2);
  REQUIRE(output[0].size(0) == 5);
  REQUIRE(output[1].size(0) == 5);

  REQUIRE(torch::cat({output[0].to(torch::kCPU), output[1].to(torch::kCPU)})
              .allclose(input));

  auto sum = output[0].to({torch::kCUDA, 1}) + output[1];
  sum.backward();

  REQUIRE(input.grad().defined());
  REQUIRE(input.grad().device().is_cpu());
  REQUIRE(input.grad().sum().toCInt() == 10);
}

TEST_CASE("Parallel/DifferentiableGather", "[multi-cuda]") {
  Gather gather(torch::Device(torch::kCUDA, 1));

  auto a = torch::ones(5, torch::requires_grad(true).device({torch::kCUDA, 0}));
  auto b = torch::ones(5, torch::requires_grad(true).device({torch::kCUDA, 1}));

  auto outputs = gather.apply({a, b});
  REQUIRE(outputs.size() == 1);
  auto& output = outputs.front();

  REQUIRE(output.size(0) == 10);
  REQUIRE(output.device() == torch::Device(torch::kCUDA, 1));

  auto chunks = output.chunk(2);
  REQUIRE(chunks[0].to({torch::kCUDA, 0}).allclose(a));
  REQUIRE(chunks[1].allclose(b));

  output.backward();

  REQUIRE(a.grad().defined());
  REQUIRE(a.grad().device() == torch::Device(torch::kCUDA, 0));
  REQUIRE(a.grad().sum().toCInt() == 5);

  REQUIRE(b.grad().defined());
  REQUIRE(b.grad().device() == torch::Device(torch::kCUDA, 1));
  REQUIRE(b.grad().sum().toCInt() == 5);
}

TEST_CASE("Parallel/Replicate", "[multi-cuda]") {
  Linear linear(3, 4);
  auto replicas = parallel::replicate(
      linear, {torch::Device(torch::kCUDA, 0), torch::Device(torch::kCUDA, 1)});
  REQUIRE(replicas.size() == 2);

  auto original_parameters = linear->parameters();

  auto replica1_parameters = replicas[0]->parameters();
  for (auto& parameter : replica1_parameters) {
    REQUIRE(parameter->device() == torch::Device(torch::kCUDA, 0));
  }
  replicas[0]->to(torch::kCPU);
  REQUIRE(replica1_parameters.size() == original_parameters.size());
  for (size_t i = 0; i < original_parameters.size(); ++i) {
    REQUIRE(replica1_parameters[i]->allclose(*original_parameters[i]));
    REQUIRE(
        replica1_parameters[i].data<float>() !=
        original_parameters[i].data<float>());
  }

  auto replica2_parameters = replicas[1]->parameters();
  for (auto& parameter : replica2_parameters) {
    REQUIRE(parameter->device() == torch::Device(torch::kCUDA, 1));
  }
  replicas[1]->to(torch::kCPU);
  REQUIRE(replica2_parameters.size() == original_parameters.size());
  for (size_t i = 0; i < original_parameters.size(); ++i) {
    REQUIRE(replica2_parameters[i]->allclose(*original_parameters[i]));
    REQUIRE(
        replica2_parameters[i].data<float>() !=
        original_parameters[i].data<float>());
  }
}

TEST_CASE("Parallel/ParallelApply", "[multi-cuda]") {
  Linear a(3, 4);

  Linear b(std::static_pointer_cast<LinearImpl>(a->clone()));
  b->to({torch::kCUDA, 0});

  Linear c(std::static_pointer_cast<LinearImpl>(a->clone()));
  c->to({torch::kCUDA, 1});

  std::vector<Linear> modules = {a, b, c};
  std::vector<torch::Tensor> inputs = {
      torch::ones({2, 3}),
      torch::ones({2, 3}, torch::device({torch::kCUDA, 0})),
      torch::ones({2, 3}, torch::device({torch::kCUDA, 1}))};

  auto outputs = parallel::parallel_apply(modules, inputs);

  REQUIRE(outputs.size() == 3);
  REQUIRE(outputs[0].device().is_cpu());

  REQUIRE(outputs[1].device() == torch::Device(torch::kCUDA, 0));
  REQUIRE(outputs[1].to(torch::kCPU).allclose(outputs[0]));

  REQUIRE(outputs[2].device() == torch::Device(torch::kCUDA, 1));
  REQUIRE(outputs[2].to(torch::kCPU).allclose(outputs[0]));
}

TEST_CASE("Parallel/ParallelApplyWithDifferentOutputDevice", "[multi-cuda]") {
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

  REQUIRE(outputs.size() == 3);
  REQUIRE(outputs[0].device().is_cuda());
  REQUIRE(outputs[0].device() == torch::Device(torch::kCUDA, 1));

  REQUIRE(outputs[1].device().is_cuda());
  REQUIRE(outputs[1].device() == torch::Device(torch::kCUDA, 0));

  REQUIRE(outputs[2].device().is_cpu());
}

TEST_CASE("Parallel/ParallelApplyRethrowsException", "[multi-cuda]") {
  struct M : torch::nn::Cloneable<M> {
    void reset() override {}
    torch::Tensor forward(torch::Tensor input) {
      throw std::runtime_error("Badness!");
    }
  };

  auto m = std::make_shared<M>();
  auto input = torch::ones({10, 3});
  REQUIRE_THROWS_WITH(
      parallel::data_parallel(m, input), StartsWith("Badness!"));
}

TEST_CASE(
    "Parallel/DataParallelPlacesTheOutputOnTheRequestedDevice",
    "[multi-cuda]") {
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
        /*devices=*/at::nullopt,
        /*output_device=*/torch::Device(torch::kCUDA, 1));
    REQUIRE(output.defined());
    REQUIRE(output.device().is_cuda());
    REQUIRE(output.device().index() == 1);
  }
  {
    // Verify for the single-device case (where we don't scatter/gather).
    auto output = parallel::data_parallel(
        m,
        input,
        /*devices=*/std::vector<torch::Device>{torch::Device(torch::kCUDA, 0)},
        /*output_device=*/torch::Device(torch::kCUDA, 1));
    REQUIRE(m->intermediate_tensor.defined());
    REQUIRE(m->intermediate_tensor.device().is_cuda());
    REQUIRE(m->intermediate_tensor.device().index() == 0);
    REQUIRE(output.defined());
    REQUIRE(output.device().is_cuda());
    REQUIRE(output.device().index() == 1);
  }
}

TEST_CASE("Parallel/DataParallelUsesAllAvailableCUDADevices", "[cuda]") {
  struct M : torch::nn::Cloneable<M> {
    void reset() override {}
    torch::Tensor forward(torch::Tensor input) {
      return torch::tensor(torch::DefaultTensorOptions::get().device().index());
    }
  };

  auto m = std::make_shared<M>();
  auto input = torch::ones({10, 3});
  auto output = parallel::data_parallel(m, input);

  const auto device_count = torch::cuda::device_count();
  REQUIRE(output.numel() == device_count);
  for (size_t i = 0; i < device_count; ++i) {
    REQUIRE(output[i].toCInt() == i);
  }
}

#endif
