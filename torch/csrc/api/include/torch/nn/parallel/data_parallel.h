#pragma once

#include <torch/cuda.h>
#include <torch/nn/module.h>
#include <torch/nn/pimpl.h>
#include <torch/types.h>

#include <torch/csrc/autograd/functions/comm.h>
#include <torch/csrc/autograd/functions/utils.h>
#include <ATen/core/functional.h>

#include <ATen/Device.h>
#include <ATen/Parallel.h>
#include <c10/core/TensorOptions.h>
#include <c10/util/Exception.h>

#include <cstddef>
#include <exception>
#include <memory>
#include <mutex>
#include <vector>

namespace torch {
namespace nn {

namespace {

// Note [Replicating Modules]
// ~~~~~~~~~~~~~~~~~~~~~~~~~~
//
// Module replication is implemented in the following two steps:
// 1) create a module replica on each destination device using Module.clone().
// 2) manually add a gradient edge pointing from every parameter X in every
//    module replica to the same parameter X in the original module, using
//    ReduceAdd as the grad_fn.
//
// ReduceAdd can ONLY be used during the backward pass of data parallel. Forward
// pass cannot use this function as it does not setup gradient function and
// history at all. Do NOT try to use ReduceAdd for any other purposes.
//
// NB: An alternative is to add Broadcast and ReduceAddCoalesce to
// torch/csrc/autograd/functions/comm.cpp as normal autograd functions,
// implement a Replicatable (like cloneable) class and add it as a friend class
// in Module.h. In the forward pass, the Replicatable could use the Broadcast
// function to replicate every module parameter and set gradient functions using
// ReduceAddCoalesce (like how it is implemented in Python). However, unlike in
// Python, where changes to Linear._parameters["weight"] would also apply to
// Linear.weight (using Linear as an example), Linear.weight and
// Linear.parameters_["weight"] are two tensor objects pointing to the same
// TensorImpl. Assigning a new tensor to Linear.parameters_["weight"] will not
// change Linear.weight. To make this work, we will have to:
// 1) force every module to also inherit from Replicatable
// 2) force every module to implement an additional function, e.g.,
//    Replicatable::load_params(), to pick up changes from parameters_ to their
//    own member fields.
// This will be an overkill as Replicatable will only be used in data_parallel,
// not even ddp.

// Autograd function for the replicate step in data parallel. This is only used
// in data parallel, and should not be exposed as a user API.
struct ReduceAdd : public autograd::Node {
  explicit ReduceAdd(const at::Device& destination_device)
      : destination_device_(destination_device) {};
  ~ReduceAdd() override {}

  autograd::variable_list apply(autograd::variable_list&& inputs) override {
    TORCH_CHECK(!torch::autograd::compute_requires_grad(inputs),
        "ReduceAdd can only be used during the backward pass of data parallel.");

    Tensor output = torch::zeros_like(inputs[0], {destination_device_});

    for (auto& input: inputs) {
      TORCH_CHECK(input.sizes() == inputs[0].sizes(),
          "All inputs of ReduceAdd must have the same size, but got ",
          input.sizes(), " and ", inputs[0].sizes());

      TORCH_CHECK(input.dtype() == inputs[0].dtype(),
          "All inputs of ReduceAdd must have the same dtype, but got ",
          input.dtype(), " and ", inputs[0].dtype());

      // TODO: use nccl reduce
      output.add_(input.to(destination_device_));
    }

    return {output};
  }

 private:
  at::Device destination_device_;
};

} // namespace

// A friend function to Module, it recursively sets gradient edges pointing from
// every parameter X in every module replica to the same parameter X in the
// original module. See [Replicating Modules]
template <typename ModuleType>
void replicate_grad_edges(
    const std::shared_ptr<Module>& module,
    const std::vector<std::shared_ptr<ModuleType>>& replicas,
    const std::vector<Device>& devices) {

  for (auto& parameter : module->named_parameters(/*recurse=*/false)) {
    auto grad_fn = std::make_shared<ReduceAdd>((*parameter).device());
    grad_fn->set_next_edges(autograd::collect_next_edges(*parameter));

    for (size_t i = 0; i < devices.size(); ++i) {
      autograd::set_history(replicas[i]->parameters_[parameter.key()], grad_fn);
    }
  }

  for (auto& buffer : module->named_buffers(/*recurse=*/false)) {
    if (buffer.value().requires_grad()){
      auto grad_fn = std::make_shared<ReduceAdd>((*buffer).device());
      grad_fn->set_next_edges(autograd::collect_next_edges(*buffer));

      for (size_t i = 0; i < devices.size(); ++i) {
        autograd::set_history(replicas[i]->buffers_[buffer.key()], grad_fn);
      }
    }
  }

  for (auto& child : module->children_) {
    std::vector<std::shared_ptr<Module>> child_replicas;
    child_replicas.reserve(devices.size());
    for (auto& replica : replicas) {
      child_replicas.push_back(replica->children_[child.key()]);
    }

    // recursively set gradient edges for all children
    replicate_grad_edges(*child, child_replicas, devices);
  }
}

namespace parallel {

/// Replicates a module on the given list of devices.
/// A replica is created by calling `clone()` on the module. For this, the
/// module must inherit from `nn::Cloneable`, or define its own `clone()`
/// method, which is expected to perform a deep copy of the module.
template <typename ModuleType>
std::vector<std::shared_ptr<ModuleType>> replicate(
    const std::shared_ptr<ModuleType>& module,
    const std::vector<Device>& devices) {
  std::vector<std::shared_ptr<ModuleType>> replicas;
  replicas.reserve(devices.size());
  for (const auto& device : devices) {
    replicas.push_back(
        std::dynamic_pointer_cast<ModuleType>(module->clone(device)));
  }
  // Configure gradient edges to point from replcia parameters to original
  // module parameters. See [Replicating Modules]
  replicate_grad_edges(module, replicas, devices);
  return replicas;
}

/// Replicates a module holder on the given list of devices.
/// This method allows calling `replicate()` with a module holder, such as
/// `Linear`.
template <typename ModuleType>
std::vector<ModuleHolder<ModuleType>> replicate(
    const ModuleHolder<ModuleType>& module,
    const std::vector<Device>& devices) {
  auto ptrs = replicate(module.ptr(), devices);
  return std::vector<ModuleHolder<ModuleType>>(ptrs.begin(), ptrs.end());
}

/// Applies the given inputs to the given modules in a parallel fashion.
/// Conceptually, a thread is spawned for each `(module, input)` pair, in which
/// `forward()` is called on the module with its corresponding input. The
/// outputs of the individual calls are stored in a vector and returned.
///
/// The first exception caught by any thread is stashed and rethrown after all
/// threads have completed their operation.
///
/// Further remarks:
/// 1. The length of the module container must match the length of the inputs.
/// 2. If a list of devices is supplied, it must match the list of modules in
/// length. Each device will be set to the current default device during the
/// invocation of the respective module. This means any tensors allocated on the
/// default device inside the module will be constructed on this device.
template <typename ModuleType>
std::vector<Tensor> parallel_apply(
    std::vector<ModuleType>& modules,
    const std::vector<Tensor>& inputs,
    const optional<std::vector<Device>>& devices = nullopt) {
  TORCH_CHECK(
      modules.size() == inputs.size(), "Must have as many inputs as modules");
  if (devices) {
    TORCH_CHECK(
        modules.size() == devices->size(),
        "Must have as many devices as modules");
  }

  std::vector<Tensor> outputs(modules.size());
  std::mutex mutex;

  // std::exception_ptr can be passed between threads:
  // > An instance of std::exception_ptr may be passed to another function,
  // > possibly on another thread, where the exception may be rethrown [...].
  // https://en.cppreference.com/w/cpp/error/exception_ptr
  std::exception_ptr exception;

  at::parallel_for(
      /*begin=*/0,
      /*end=*/modules.size(),
      /*grain_size=*/1,
      [&modules, &inputs, &devices, &outputs, &mutex, &exception](
          int64_t index, int64_t stop) {
        for (; index < stop; ++index) {
          try {
            auto output = modules[index]->forward(inputs[index]);
            output =
                output.to(devices ? (*devices)[index] : inputs[index].device());
            std::lock_guard<std::mutex> lock(mutex);
            outputs[index] = output;
          } catch (...) {
            std::lock_guard<std::mutex> lock(mutex);
            if (!exception) {
              exception = std::current_exception();
            }
          }
        }
      });

  if (exception) {
    std::rethrow_exception(exception);
  }

  return outputs;
}

/// Evaluates `module(input)` in parallel across the given `devices`. If
/// `devices` is not supplied, the invocation is parallelized across all
/// available CUDA devices. If `output_device` is supplied, the final, combined
/// tensor will be placed on this device. If not, it defaults to the first
/// device in `devices`.
///
/// In detail, this method performs the following four distinct steps:
/// 1. *Scatter* the input to the given devices,
/// 2. *Replicate* (deep clone) the model on each device,
/// 3. *Evaluate* each module with its input on its device,
/// 4. *Gather* the outputs of each replica into a single output tensor, located
/// on the `output_device`.
template <typename ModuleType>
Tensor data_parallel(
    ModuleType module,
    Tensor input,
    optional<std::vector<Device>> devices = nullopt,
    optional<Device> output_device = nullopt,
    int64_t dim = 0) {
  if (!devices) {
    const auto device_count = torch::cuda::device_count();
    TORCH_CHECK(
        device_count > 0, "Expected at least one CUDA device to be available");
    devices = std::vector<Device>();
    devices->reserve(device_count);
    for (size_t index = 0; index < device_count; ++index) {
      devices->emplace_back(kCUDA, index);
    }
  }
  if (!output_device) {
    output_device = devices->front();
  }

  if (devices->size() == 1) {
    module->to(devices->front());
    input = input.to(devices->front());
    return module->forward(std::move(input)).to(*output_device);
  }

  autograd::Scatter scatter(*devices, /*chunk_sizes=*/nullopt, dim);
  auto scattered_inputs = fmap<Tensor>(scatter.apply({std::move(input)}));

  auto replicas = replicate(module, *devices);
  auto outputs = parallel_apply(replicas, scattered_inputs, *devices);
  return autograd::Gather(*output_device, dim)
      .apply(fmap<autograd::Variable>(std::move(outputs)))
      .front();
}

} // namespace parallel
} // namespace nn
} // namespace torch
