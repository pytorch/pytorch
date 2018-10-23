#include "torch/csrc/jit/fuser/executor.h"

#include "ATen/ATen.h"
#include "ATen/ExpandUtils.h"
#include "c10/util/Optional.h"
#include "torch/csrc/utils/functional.h"
#include "torch/csrc/jit/stack.h"
#include "torch/csrc/jit/fuser/interface.h"
#include "torch/csrc/jit/fuser/kernel_cache.h"
#include "torch/csrc/jit/fuser/kernel_spec.h"
#include "torch/csrc/jit/fuser/compiler.h"

#include <vector>
#include <tuple>
#include <stdexcept>
#include <algorithm>
#include <map>
#include <iostream> // TODO: remove, debugging only

namespace torch { namespace jit { namespace fuser {

static c10::optional<std::vector<int64_t>> getMapSize(
  const KernelSpec& spec
, at::TensorList args
, at::IntList arg_subset) {
  
  int64_t dim_after_broadcast = 0;
  for (const auto arg_idx : arg_subset) {
    dim_after_broadcast = std::max(dim_after_broadcast, args[arg_idx].dim());
  }
  // TODO: this keeps reallocating map_size at every iteration, but we know
  // exactly how much storage do we need, so this could be fixed in-place at
  // every step. We're just missing a few functions for ATen, but the fix
  // should be straightforward.
  // Note: left unitialized since empty shape is broadcastable to any shape
  std::vector<int64_t> map_size;
  for (size_t i = 0; i < arg_subset.size(); ++i) {
    auto& arg = args.at(arg_subset[i]);
    auto& chunk_desc = spec.inputChunks().at(arg_subset[i]);
    if (chunk_desc.nSubTensors() == 1) {
      try {
        map_size = at::infer_size(map_size, arg.sizes());
      } catch (...) {
        return c10::nullopt;
      }
    } else {
      auto tensor_sizes = arg.sizes().vec();
      const auto num_chunks = chunk_desc.nSubTensors();
      const auto dim = at::maybe_wrap_dim(chunk_desc.dim(), tensor_sizes.size());
      if (tensor_sizes[dim] % num_chunks != 0) {
        return c10::nullopt;
      }
      tensor_sizes[dim] /= num_chunks;
      try {
        map_size = at::infer_size(map_size, tensor_sizes);
      } catch (...) {
        return c10::nullopt;
      }
    }
  }

  return {map_size};
}

static c10::optional<std::vector<int64_t>> canRunKernel(
  const KernelSpec& spec
, at::TensorList args) {
  // Short-circuits on size mismath
  AT_CHECK(args.size() == spec.inputChunks().size(),
           "Expected ", spec.inputChunks().size(), " arguments, but got ", args.size());

  c10::optional<std::vector<int64_t>> map_size;
  for (const auto& broadcast_group : spec.inputBroadcastGroups()) {
    if (!map_size) {
      map_size = getMapSize(spec, args, broadcast_group);
      if (!map_size) {
        return c10::nullopt;
      }
    } else {
      auto group_map_size = getMapSize(spec, args, broadcast_group);
      // NB: this checks that group_map_size is defined AND equal to map_size
      if (map_size != group_map_size) {
        return c10::nullopt;
      }
    }
  }

  return map_size;
}

// Note: Arguments are mutated by this call, although map_size is restored
// to its original value.
static void expandArgs(
  const KernelSpec& spec
, std::vector<at::Tensor>& args
, std::vector<int64_t>& map_size) {
  for (size_t i = 0; i < args.size(); ++i) {
    auto& arg = args[i];
    const auto& pdesc = spec.inputChunks()[i];
    if (pdesc.nSubTensors() == 1) {
      if (arg.sizes().equals(map_size)) continue;
      arg = arg.expand(map_size);
    } else {
      map_size.at(pdesc.dim()) *= pdesc.nSubTensors();
      if (!arg.sizes().equals(map_size)) {
        arg = arg.expand(map_size);
      }
      map_size.at(pdesc.dim()) /= pdesc.nSubTensors();
    }
  }
}

// void launchFusion(
//   const FusedKernel& fusion
// , const int device
// , const at::ArrayRef<at::Tensor>& inputs
// , std::vector<at::Tensor>& outputs) {
//   // Switches to device to run the fusion on
//   at::DeviceGuard guard{device};
     
     // Allocates tensors for outputs
//   auto& ref_type = inputs[0].type();
//   outputs.reserve(output_desc_.size());
//   for (const auto& od : output_desc_) {
//     outputs.push_back(at::empty({0}, ref_type.options().dtype(od.scalar_type)));
//   }

      //TODO: launch_with_tensors here
// }


void runFusion(
  const int64_t key
, Stack& stack) {
  // Short-circuits if fusion isn't enabled
  if (!canFuseOnCPU() && !canFuseOnGPU())
    throw std::runtime_error("Fusion not enabled.");

  // Acquires the FusionSpec
  auto maybe_spec = retrieve(key);
  if (!maybe_spec) 
    throw std::runtime_error("Failed to find fusion specification to run.");
  auto& spec = *maybe_spec;
  
  // Short-circuits if the spec isn't fusable
  if (!spec.isFusable()) 
    throw std::runtime_error("Non-fusable specification.");

  // Determines device to dispatch to
  // Acquires inputs from stack
  auto inputs = fmap(last(stack, spec.nInputs()), [](const IValue& i) {
    return i.toTensor();
  });
  int32_t device = kCPUDevice;
  for (const auto& t : inputs) {
    const auto cur_device = t.device().index();
    if (cur_device < 0) continue;
    if (device < 0) device = cur_device;
    else if (device != cur_device) 
      throw std::runtime_error("Cannot fuse CUDA tensors on different devices.");
  }

  // Validates sizes and expands inputs as needed
  auto maybe_map_size = canRunKernel(spec, inputs);
  if (!maybe_map_size)
    throw std::runtime_error("Incompatible map size preventing fusion.");
  expandArgs(spec, inputs, *maybe_map_size);

  // Retrieves the kernel, compiling if necessary
  ArgSpec arg_spec{inputs};
  auto maybe_kernel = spec.findKernel(arg_spec);
  if (!maybe_kernel) {
    const auto kernel = compileKernel(spec, arg_spec, *maybe_map_size, device);
    spec.cacheKernel(arg_spec, kernel);
  }
  maybe_kernel = spec.findKernel(arg_spec);
  if (!maybe_kernel)
    throw std::runtime_error("Failed to find cached fused kernel.");

  // Launches the kernel
  std::vector<at::Tensor> outputs;
  (*maybe_kernel)->launch(inputs, outputs);
  drop(stack, spec.nInputs());
  stack.insert(
    stack.end()
  , std::make_move_iterator(outputs.begin())
  , std::make_move_iterator(outputs.end()));
}

} // namespace fuser
} // namespace jit
} // namespace torch
