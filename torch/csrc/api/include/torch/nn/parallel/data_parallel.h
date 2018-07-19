#pragma once

#include <torch/cuda.h>
#include <torch/nn/module.h>
#include <torch/nn/pimpl.h>
#include <torch/tensor.h>

#include <torch/csrc/autograd/functions/comm.h>
#include <torch/csrc/cuda/comm.h>

#include <ATen/Device.h>
#include <ATen/Error.h>
#include <ATen/OptionsGuard.h>
#include <ATen/Parallel.h>
#include <ATen/TensorOptions.h>
#include <ATen/optional.h>

#include <cstddef>
#include <exception>
#include <memory>
#include <mutex>
#include <vector>

namespace torch {
namespace nn {
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
    // Here we rely on the property tensors are never (or should never be)
    // allocated on any particular device, but always the default device, e.g.
    // in `torch::ones({3, 4})`, the device is unspecified and pulled from the
    // current thread local default options. As such, we can here modify these
    // thread local default options and thereby cause all tensors in the cloned
    // module to be constructed directly on the device we want.
    OptionsGuard guard(device);
    replicas.push_back(std::static_pointer_cast<ModuleType>(module->clone()));
  }
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
    const at::optional<std::vector<Device>>& devices = at::nullopt) {
  AT_CHECK(
      modules.size() == inputs.size(), "Must have as many inputs as modules");
  if (devices) {
    AT_CHECK(
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
            torch::OptionsGuard options_guard(
                devices ? (*devices)[index] : inputs[index].device());
            auto output = modules[index]->forward(inputs[index]);
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
    at::optional<std::vector<Device>> devices = at::nullopt,
    at::optional<Device> output_device = at::nullopt,
    int64_t dim = 0) {
  if (!devices) {
    const auto device_count = torch::cuda::device_count();
    AT_CHECK(device_count > 0, "Expected at least one CUDA device");
    devices.emplace();
    devices->reserve(device_count);
    for (size_t index = 0; index < device_count; ++index) {
      devices->emplace_back(kCUDA, index);
    }
  }
  if (!output_device) {
    output_device = devices->front();
  }

  if (devices->size() == 1) {
    OptionsGuard guard(devices->front());
    return module->forward(std::move(input)).to(*output_device);
  }

  autograd::Scatter scatter(*devices, /*chunk_sizes=*/at::nullopt, dim);
  auto scattered_inputs = scatter.apply({std::move(input)});

  auto replicas = replicate(module, *devices);
  auto outputs = parallel_apply(replicas, scattered_inputs, *devices);
  return autograd::Gather(*output_device, dim)
      .apply(std::move(outputs))
      .front();
}

} // namespace parallel
} // namespace nn
} // namespace torch
