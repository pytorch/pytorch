#include "torch/csrc/jit/fuser/executor.h"

#include "ATen/ATen.h"
#include "ATen/ExpandUtils.h"
#include "c10/util/Optional.h"
#include "torch/csrc/utils/functional.h"
#include "torch/csrc/jit/stack.h"
#include "torch/csrc/jit/fuser/config.h"
#include "torch/csrc/jit/fuser/interface.h"
#include "torch/csrc/jit/fuser/kernel_cache.h"
#include "torch/csrc/jit/fuser/kernel_spec.h"
#include "torch/csrc/jit/fuser/compiler.h"
#include "torch/csrc/jit/fuser/tensor_info.h"

#include <vector>
#include <tuple>
#include <stdexcept>
#include <algorithm>
#include <map>
#include <iostream> // TODO: remove, debugging only

namespace torch { namespace jit { namespace fuser {

// Returns the "map size" for this run, which is the common size for all
// intermediate tensors.
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

// Tries to determine a map size for the instantiated kernel (see above)
static c10::optional<std::vector<int64_t>> canRunKernel(
  const KernelSpec& spec
, at::TensorList args) {
  // Short-circuits on size mismatch
  AT_CHECK(
    args.size() == spec.inputChunks().size()
  , "Expected ", spec.inputChunks().size(), " arguments, but got ", args.size());

  c10::optional<std::vector<int64_t>> map_size;
  for (const auto& broadcast_group : spec.inputBroadcastGroups()) {
    if (!map_size) {
      map_size = getMapSize(spec, args, broadcast_group);
      if (!map_size) return c10::nullopt;
    } else {
      const auto group_map_size = getMapSize(spec, args, broadcast_group);
      // Note: this checks that group_map_size is defined AND equal to map_size
      if (map_size != group_map_size) return c10::nullopt;
    }
  }

  return map_size;
}

// Arguments are expanded to a common shape, referred to as the "map size,"
// (see above).
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

// Note: assumes that inputs are 32-bit addressable
static uint32_t computeNumel(const at::ArrayRef<int64_t>& sizes) {
  uint32_t result = 1;

  for (const auto& size : sizes)
    result *= size;

  return result;
}

// Note: Assumes that after at::chunk, all inputs are the same size
static std::vector<int64_t> computeMapSize(
  const at::Tensor& tensor
, const PartitionDesc& chunkDesc) {
  std::vector<int64_t> sizes(tensor.sizes().begin(), tensor.sizes().end());
  JIT_ASSERT(sizes[chunkDesc.dim()] % chunkDesc.nSubTensors() == 0);
  sizes[chunkDesc.dim()] /= chunkDesc.nSubTensors();
  return sizes;
}

// Tries to compress sizes and strides according to cont. Emits the result t
// c_sizes, c_strides and throws an error on failure (if can't compress)
static void compressContiguous(
  const at::IntList& sizes
, const at::IntList& strides
, const std::vector<bool>& cont
, uint32_t* c_sizes
, uint32_t* c_strides) {
  size_t compressed_dims = 0;
  size_t cur = 0;
  size_t ndim = sizes.size();
  while (cur < ndim) {
    size_t total_size = sizes[cur];
    cur++;
    while (cont[cur-1] && cur < ndim) {
      JIT_ASSERT(strides[cur-1] == sizes[cur]*strides[cur]);
      total_size *= sizes[cur];
      cur++;
    }
    c_sizes[compressed_dims] = total_size;
    c_strides[compressed_dims] = strides[cur-1];
    compressed_dims++;
  }

  if (ndim > 0) JIT_ASSERT(!cont.back() || strides.back() == 1);
}

// Launches the requested fusion on the given device with the given inputs.
// Output pointers are stored in outputs (to be put on the stack later).
void launchFusion(
  const FusedKernel& fusion
, const int device
, const at::ArrayRef<at::Tensor>& inputs
, std::vector<at::Tensor>& outputs) {
  // Allocates tensors for outputs
  auto& ref_type = inputs[0].type();
  outputs.reserve(fusion.outputDesc().size());
  for (const auto& od : fusion.outputDesc()) {
    if (device >= 0) // GPU
      outputs.push_back(at::empty({0}, ref_type.options().dtype(od.scalar_type).device_index(device)));
    else // CPU
      outputs.push_back(at::empty({0}, ref_type.options().dtype(od.scalar_type).device(at::Device{at::DeviceType::CPU})));
  }

  // Fails if fusion and given inputs disagree
  JIT_ASSERT(inputs.size() == fusion.inputDesc().size());

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
  JIT_ASSERT(inputs[0].numel() <= std::numeric_limits<uint32_t>::max());

  // Computes map_size, numel from the first input
  at::IntList map_size;
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

  // Computes the storage needed to store TensorInfo structs for inputs and outputs.
  size_t uncompressedDim = fusion.inputDesc().at(0).contiguity.size();
  size_t maxPossibleTensorInfoSize = sizeof(TensorInfo) + 2 * sizeof(uint32_t) * uncompressedDim;
  size_t maxPossibleBufferSize = maxPossibleTensorInfoSize * (flat_inputs_size + flat_outputs_size);
  std::vector<char> buffer(maxPossibleBufferSize);
  char* buffer_next = buffer.data();

  // A vector of arguments to the kernel (numel, *input_desc_s, *output_desc_s)
  std::vector<void*> arguments;
  arguments.reserve(3 + flat_inputs_size + flat_outputs_size);
  arguments.push_back(&numel);

  auto addTensorInfoRaw = [&](
    const TensorDesc& desc
  , void* data_ptr
  , at::IntList sizes
  , at::IntList strides) {
    const auto nDim = desc.nDim(); // NOTE: this is the compressed dim
    JIT_ASSERT(nDim <= uncompressedDim); // We'd overflow the space otherwise
    auto ti = reinterpret_cast<TensorInfo*>(buffer_next);
    ti->data = data_ptr;
    compressContiguous(
      sizes
    , strides
    , desc.contiguity
    , ti->sizes(nDim)
    , ti->strides(nDim));
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
      size_t chunk_offset = map_size[chunk.dim()] * tensor.stride(chunk.dim()) * elementSize(tensor.type().scalarType());
      char* data_ptr = reinterpret_cast<char*>(tensor.data_ptr());
      for (size_t chunks = 0; chunks < chunk.nSubTensors(); ++chunks) {
        addTensorInfoRaw(*chunk.subTensorDesc(), data_ptr, map_size, tensor.strides());
        data_ptr += chunk_offset;
      }
    }
  }

  // Adds (flattened) output arguments
  for (size_t i = 0; i < fusion.outputDesc().size(); ++i) {
    const auto& c = fusion.concatDesc()[i];
    auto& o = outputs[i];
    if (c.isNoop()) {
      o.resize_(map_size);
      addTensorInfo(fusion.outputDesc()[i], outputs[i]);
    } else {
      size_t small_size = map_size[c.dim()];
      std::vector<int64_t> concat_size(map_size.begin(), map_size.end());
      concat_size[c.dim()] = small_size * c.nSubTensors();
      o.resize_(concat_size);
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

  fusion.launch_raw(numel, arguments);
}


bool runFusion(
  const int64_t key
, Stack& stack) {
  // Short-circuits if fusion isn't enabled
  if (!canFuseOnCPU() && !canFuseOnGPU()) return false;

  // Acquires the FusionSpec
  auto maybe_spec = retrieve(key);
  JIT_ASSERT(maybe_spec);
  auto& spec = *(*maybe_spec);

  // Acquires inputs from stack
  auto inputs = fmap(last(stack, spec.nInputs()), [](const IValue& i) {
    return i.toTensor();
  });

  // Determines device to dispatch to. If there's a device mismatch in the inputs,
  // we use the fallback (which should give a nice error message).
  int32_t device = inputs.at(0).device().index();
  at::ScalarType dtype = inputs[0].type().scalarType();
  for (const auto& t : at::TensorList(inputs).slice(1)) {
    if (t.device().index() != device) {
      return false;
    }
    if (t.type().scalarType() != dtype) {
      return false;
    }
  }

  // The codegen only supports float and half inputs at the moment, so bail out
  // if we see anything else.
  if (dtype != at::kFloat && dtype != at::kHalf) return false;

  // Attempts to run fallback if device fusion is disabled
  if (device != kCPUDevice && !canFuseOnGPU()) return false;
  if (device == kCPUDevice && !canFuseOnCPU()) return false;

  // Validates sizes and expands inputs as needed
  auto maybe_map_size = canRunKernel(spec, inputs);

  // Tries to run fallback if map size can't be computed
  if (!maybe_map_size) return false;
  expandArgs(spec, inputs, *maybe_map_size);

  // Retrieves the kernel, compiling (and caching) if necessary
  ArgSpec arg_spec{inputs, device};
  auto maybe_kernel = spec.findKernel(arg_spec);
  if (!maybe_kernel) {
    const auto kernel = compileKernel(spec, arg_spec, *maybe_map_size, device);
    spec.cacheKernel(arg_spec, kernel);
  }
  maybe_kernel = spec.findKernel(arg_spec);
  JIT_ASSERT(maybe_kernel);

  // Launches fusion
  std::vector<at::Tensor> outputs;
  launchFusion(*(*maybe_kernel), device, inputs, outputs);

  // Updates stack
  drop(stack, spec.nInputs());
  stack.insert(
    stack.end()
  , std::make_move_iterator(outputs.begin())
  , std::make_move_iterator(outputs.end()));

  return true;
}

} // namespace fuser
} // namespace jit
} // namespace torch
