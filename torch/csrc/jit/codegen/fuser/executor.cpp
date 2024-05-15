#include <torch/csrc/jit/codegen/fuser/executor.h>

#include <ATen/ATen.h>
#include <ATen/ExpandUtils.h>
#include <ATen/core/functional.h>
#include <ATen/core/stack.h>
#include <c10/util/Optional.h>
#include <c10/util/irange.h>
#include <torch/csrc/jit/codegen/fuser/compiler.h>
#include <torch/csrc/jit/codegen/fuser/interface.h>
#include <torch/csrc/jit/codegen/fuser/kernel_cache.h>
#include <torch/csrc/jit/codegen/fuser/kernel_spec.h>
#include <torch/csrc/jit/codegen/fuser/tensor_info.h>
#include <torch/csrc/jit/passes/graph_fuser.h>

#include <algorithm>
#include <iostream> // TODO: remove, debugging only
#include <map>
#include <stdexcept>
#include <tuple>
#include <vector>

namespace torch {
namespace jit {
namespace fuser {

// Returns the "map size" for this run, which is the common size for all
// intermediate tensors.
static std::optional<std::vector<int64_t>> getMapSize(
    const KernelSpec& spec,
    at::TensorList args,
    at::IntArrayRef arg_subset) {
  // TODO: this keeps reallocating map_size at every iteration, but we know
  // exactly how much storage do we need, so this could be fixed in-place at
  // every step. We're just missing a few functions for ATen, but the fix
  // should be straightforward.
  // Note: left unitialized since empty shape is broadcastable to any shape
  std::vector<int64_t> map_size;
  map_size.reserve(8);
  for (const auto arg_idx : arg_subset) {
    auto& arg = args.at(arg_idx);
    auto& chunk_desc = spec.inputChunks().at(arg_idx);
    if (chunk_desc.nSubTensors() == 1) {
      try {
        map_size = at::infer_size(map_size, arg.sizes());
      } catch (...) {
        return c10::nullopt;
      }
    } else {
      auto tensor_sizes = arg.sizes().vec();
      const auto num_chunks = chunk_desc.nSubTensors();
      const auto dim =
          at::maybe_wrap_dim(chunk_desc.dim(), tensor_sizes.size());
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

// Tries to determine a map size for the instantiated kernel (see above)
static std::optional<std::vector<int64_t>> canRunKernel(
    const KernelSpec& spec,
    at::TensorList args) {
  // Short-circuits on size mismatch
  TORCH_CHECK(
      args.size() == spec.inputChunks().size(),
      "Expected ",
      spec.inputChunks().size(),
      " arguments, but got ",
      args.size());

  std::optional<std::vector<int64_t>> map_size;
  for (const auto& broadcast_group : spec.inputBroadcastGroups()) {
    if (!map_size) {
      map_size = getMapSize(spec, args, broadcast_group);
      if (!map_size)
        return c10::nullopt;
    } else {
      const auto group_map_size = getMapSize(spec, args, broadcast_group);
      // Note: this checks that group_map_size is defined AND equal to map_size
      if (map_size != group_map_size)
        return c10::nullopt;
    }
  }

  return map_size;
}

// Arguments are expanded to a common shape, referred to as the "map size,"
// (see above).
// Note: Arguments are mutated by this call, although map_size is restored
// to its original value.
static bool expandArgs(
    const KernelSpec& spec,
    std::vector<at::Tensor>& args,
    std::vector<int64_t>& map_size,
    bool dry_run) {
  bool has_broadcast = false;
  for (size_t i = 0; i < args.size(); ++i) {
    auto& arg = args[i];
    const auto& pdesc = spec.inputChunks()[i];
    if (pdesc.nSubTensors() == 1) {
      if (arg.sizes().equals(map_size))
        continue;
      if (!dry_run) {
        arg = arg.expand(map_size);
        has_broadcast = true;
      } else {
        return true;
      }
    } else {
      map_size.at(pdesc.dim()) *= pdesc.nSubTensors();
      if (!arg.sizes().equals(map_size)) {
        if (!dry_run) {
          arg = arg.expand(map_size);
          has_broadcast = true;
        } else {
          return true;
        }
      }
      map_size.at(pdesc.dim()) /= pdesc.nSubTensors();
    }
  }
  return has_broadcast;
}

static bool shouldExpandArgs(
    const KernelSpec& spec,
    std::vector<at::Tensor>& args,
    std::vector<int64_t>& map_size) {
  return expandArgs(spec, args, map_size, /*dry_run=*/true);
}

// Note: assumes that inputs are 32-bit addressable
static uint32_t computeNumel(const at::ArrayRef<int64_t> sizes) {
  uint32_t result = 1;

  for (const auto& size : sizes)
    result *= size;

  return result;
}

// Note: Assumes that after at::chunk, all inputs are the same size
static std::vector<int64_t> computeMapSize(
    const at::Tensor& tensor,
    const PartitionDesc& chunkDesc) {
  std::vector<int64_t> sizes(tensor.sizes().begin(), tensor.sizes().end());
  AT_ASSERT(sizes[chunkDesc.dim()] % chunkDesc.nSubTensors() == 0);
  sizes[chunkDesc.dim()] /= chunkDesc.nSubTensors();
  return sizes;
}

// Tries to compress sizes and strides according to cont. Emits the result t
// c_sizes, c_strides and throws an error on failure (if can't compress)
static void compressContiguous(
    const at::IntArrayRef& sizes,
    const at::IntArrayRef& strides,
    const std::vector<bool>& cont,
    uint32_t* c_sizes,
    uint32_t* c_strides) {
  size_t compressed_dims = 0;
  size_t cur = 0;
  size_t ndim = sizes.size();
  while (cur < ndim) {
    size_t total_size = sizes[cur];
    cur++;
    while (cont[cur - 1] && cur < ndim) {
      AT_ASSERT(strides[cur - 1] == sizes[cur] * strides[cur]);
      total_size *= sizes[cur];
      cur++;
    }
    c_sizes[compressed_dims] = total_size;
    c_strides[compressed_dims] = strides[cur - 1];
    compressed_dims++;
  }

  if (ndim > 0)
    AT_ASSERT(!cont.back() || strides.back() == 1);
}

// Launches the requested fusion on the given device with the given inputs.
// Output pointers are stored in outputs (to be put on the stack later).
static void launchFusion(
    const FusedKernel& fusion,
    const at::Device device,
    const at::ArrayRef<at::Tensor>& inputs,
    const at::ArrayRef<IValue>& all_inputs,
    std::vector<at::Tensor>& outputs) {
  // Fails if fusion and given inputs disagree
  AT_ASSERT(inputs.size() == fusion.inputDesc().size());

  // Computes number of flattened inputs and outputs
  size_t flat_inputs_size = 0;
  size_t flat_outputs_size = 0;
  for (const auto& c : fusion.chunkDesc())
    flat_inputs_size += c.nSubTensors();
  for (const auto& c : fusion.concatDesc())
    flat_outputs_size += c.nSubTensors();

  // Fails if the elements of the first (any) tensor are not expressable as
  // a 32-bit integer.
  // Note: this code assumes that inputs are 32-bit addressable
  // Note: this code assumes that all inputs are of the same size
  AT_ASSERT(inputs[0].numel() <= std::numeric_limits<uint32_t>::max());

  // Computes map_size, numel from the first input
  at::IntArrayRef map_size;
  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  uint32_t numel;
  std::vector<int64_t> keep_alive_size;
  if (fusion.chunkDesc()[0].isNoop()) {
    map_size = inputs[0].sizes();
    numel = inputs[0].numel();
  } else {
    keep_alive_size = computeMapSize(inputs[0], fusion.chunkDesc()[0]);
    map_size = keep_alive_size;
    numel = computeNumel(map_size);
  }

  // compute number of scalar inputs and convert them to float
  std::vector<double> scalar_inputs;
  scalar_inputs.reserve(all_inputs.size());
  for (auto const& input : all_inputs) {
    if (input.isDouble())
      scalar_inputs.push_back(input.to<float>());
  }

  // Computes the storage needed to store TensorInfo structs for inputs and
  // outputs.
  size_t uncompressedDim = fusion.inputDesc().at(0).contiguity.size();
  size_t maxPossibleTensorInfoSize =
      sizeof(TensorInfo) + 2 * sizeof(uint32_t) * uncompressedDim;
  size_t maxPossibleBufferSize =
      maxPossibleTensorInfoSize * (flat_inputs_size + flat_outputs_size);
  std::vector<char> buffer(maxPossibleBufferSize);
  char* buffer_next = buffer.data();

  // A vector of arguments to the kernel (numel, *input_desc_s, *output_desc_s)
  std::vector<void*> arguments;
  arguments.reserve(
      3 + scalar_inputs.size() + flat_inputs_size + flat_outputs_size);
  arguments.push_back(&numel);

  auto addTensorInfoRaw = [&](const TensorDesc& desc,
                              void* data_ptr,
                              at::IntArrayRef sizes,
                              at::IntArrayRef strides) {
    const auto nDim = desc.nDim(); // NOTE: this is the compressed dim
    AT_ASSERT(nDim <= uncompressedDim); // We'd overflow the space otherwise
    auto ti = reinterpret_cast<TensorInfo*>(buffer_next);
    ti->data = data_ptr;
    compressContiguous(
        sizes, strides, desc.contiguity, ti->sizes(nDim), ti->strides(nDim));
    buffer_next += maxPossibleTensorInfoSize;
    arguments.push_back(ti);
  };

  // Asserts that t's dims can be compressed in the same way as in desc
  // (that's what the kernel assumes), and appends it to the arguments vector.
  auto addTensorInfo = [&](const TensorDesc& desc, const at::Tensor& t) {
    addTensorInfoRaw(desc, t.data_ptr(), t.sizes(), t.strides());
  };

  // Adds (flattened) input arguments
  for (size_t i = 0; i < fusion.inputDesc().size(); ++i) {
    const auto& chunk = fusion.chunkDesc()[i];
    const at::Tensor& tensor = inputs[i];
    if (chunk.isNoop()) {
      addTensorInfo(fusion.inputDesc()[i], tensor);
    } else {
      size_t chunk_offset = map_size[chunk.dim()] * tensor.stride(chunk.dim()) *
          elementSize(tensor.scalar_type());
      char* data_ptr = reinterpret_cast<char*>(tensor.data_ptr());
      for (size_t chunks = 0; chunks < chunk.nSubTensors(); ++chunks) {
        addTensorInfoRaw(
            *chunk.subTensorDesc(), data_ptr, map_size, tensor.strides());
        data_ptr += chunk_offset;
      }
    }
  }
  // Adds scalar arguments
  for (double& s : scalar_inputs) {
    arguments.push_back(&s);
  }

  // Adds (flattened) output arguments
  outputs.reserve(fusion.outputDesc().size());
  const auto& ref_options = inputs[0].options();
  for (size_t i = 0; i < fusion.outputDesc().size(); ++i) {
    const auto& c = fusion.concatDesc()[i];
    if (c.isNoop()) {
      outputs.push_back(at::empty(
          map_size, ref_options.dtype(fusion.outputDesc()[i].scalar_type)));
      addTensorInfo(fusion.outputDesc()[i], outputs[i]);
    } else {
      size_t small_size = map_size[c.dim()];
      std::vector<int64_t> concat_size(map_size.begin(), map_size.end());
      concat_size[c.dim()] = small_size * c.nSubTensors();
      outputs.push_back(at::empty(concat_size, ref_options));
      const auto& o = outputs[i];
      size_t offset = 0;
      for (size_t j = 0; j < c.nSubTensors(); ++j) {
        // because the concatenated_output stays live, the underlying data
        // in this view remains live through the end of this function
        // so there is not need to hold onto this tensor
        const auto view = o.narrow(c.dim(), offset, small_size);
        addTensorInfo(*c.subTensorDesc(), view);
        offset += small_size;
      }
    }
  }
  // Skip launching the kernel for zero-element tensor inputs
  // launches are skipped, empty zero-sized output is returned
  if (numel > 0) {
    fusion.launch_raw(numel, arguments);
  }
}

bool runFusion(const int64_t key, Stack& stack, std::string* code_out) {
  // Short-circuits if fusion isn't enabled
  if (!canFuseOnCPULegacy() && !canFuseOnGPU())
    return false;

  // Acquires the FusionSpec
  auto maybe_spec = retrieve(key);
  AT_ASSERT(maybe_spec);
  auto& spec = *(*maybe_spec);
  // Acquires inputs from stack
  auto all_inputs = last(stack, spec.nInputs());
  std::vector<at::Tensor> inputs;
  inputs.reserve(spec.nTensorInputs());
  // we know that tensor inputs are first
  for (const auto i : c10::irange(spec.nTensorInputs())) {
    inputs.emplace_back(all_inputs[i].toTensor());
  }

  if (!inputs.at(0).defined()) {
    return false;
  }

  // Determines device to dispatch to.
  at::Device device = inputs.at(0).device();
  // If there's a device mismatch in the inputs or if one of the input is a
  // sparse tensor, we use the fallback (which should give a nice error
  // message).
  for (const auto& t : at::TensorList(inputs).slice(1)) {
    // Sparse tensor could not by supported by CUDA fusion, so we bail out.
    if (t.device() != device || t.is_sparse()) {
      return false;
    }
  }

  // Attempts to run fallback if device fusion is disabled
  if (device.is_cuda() && !canFuseOnGPU())
    return false;
  if (device.is_cpu() && !canFuseOnCPULegacy())
    return false;
  if (device.is_xpu())
    return false;

  // Validates sizes and expands inputs as needed
  auto maybe_map_size = canRunKernel(spec, inputs);

  // Tries to run fallback if map size can't be computed
  if (!maybe_map_size)
    return false;
  if (spec.hasRandom()) {
    bool hasBroadcast = shouldExpandArgs(spec, inputs, *maybe_map_size);
    if (hasBroadcast)
      return false;
  }
  expandArgs(spec, inputs, *maybe_map_size, /*dry_run=*/false);

  // Retrieves the kernel, compiling (and caching) if necessary
  ArgSpec arg_spec{inputs, device.index()};
  auto maybe_kernel = spec.findKernel(arg_spec);
  if (!maybe_kernel) {
    const auto kernel = compileKernel(spec, arg_spec, *maybe_map_size, device);
    spec.cacheKernel(arg_spec, kernel);
  }
  maybe_kernel = spec.findKernel(arg_spec);
  AT_ASSERT(maybe_kernel);

  if (code_out) {
    *code_out = maybe_kernel.value()->code();
  }

  // Launches fusion
  std::vector<at::Tensor> outputs;
  launchFusion(*(*maybe_kernel), device, inputs, all_inputs, outputs);

  // Updates stack
  drop(stack, spec.nInputs());
  stack.insert(
      stack.end(),
      std::make_move_iterator(outputs.begin()),
      std::make_move_iterator(outputs.end()));

  return true;
}

} // namespace fuser
} // namespace jit
} // namespace torch
