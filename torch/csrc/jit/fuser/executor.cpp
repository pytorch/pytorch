#include <torch/csrc/jit/fuser/executor.h>

#include <ATen/ATen.h>
#include <ATen/ExpandUtils.h>
#include <ATen/core/functional.h>
#include <ATen/core/stack.h>
#include <c10/util/Optional.h>
#include <torch/csrc/jit/fuser/compiler.h>
#include <torch/csrc/jit/fuser/interface.h>
#include <torch/csrc/jit/fuser/kernel_cache.h>
#include <torch/csrc/jit/fuser/kernel_spec.h>
#include <torch/csrc/jit/fuser/tensor_info.h>

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
static c10::optional<std::vector<int64_t>> getMapSize(
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
static c10::optional<std::vector<int64_t>> canRunKernel(
    const KernelSpec& spec,
    at::TensorList args) {
  // Short-circuits on size mismatch
  AT_CHECK(
      args.size() == spec.inputChunks().size(),
      "Expected ",
      spec.inputChunks().size(),
      " arguments, but got ",
      args.size());

  c10::optional<std::vector<int64_t>> map_size;
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
static uint32_t computeNumel(const at::ArrayRef<int64_t>& sizes) {
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
void launchFusion(
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
  std::vector<float> scalar_inputs;
  scalar_inputs.reserve(all_inputs.size());
  for (auto const &input: all_inputs){
    if (input.isDouble()) scalar_inputs.push_back(input.to<float>());
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
  arguments.reserve(3 + scalar_inputs.size() + flat_inputs_size + flat_outputs_size);
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
  for (float &s: scalar_inputs){
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

  fusion.launch_raw(numel, arguments);
}

bool runFusion(const int64_t key, Stack& stack) {
  // Short-circuits if fusion isn't enabled
  if (!canFuseOnCPU() && !canFuseOnGPU())
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
  for (int64_t i = 0; i < spec.nTensorInputs(); i++) {
    inputs.emplace_back(all_inputs[i].toTensor());
  }

  // If the length of the SumToSize of FusedConcat outputs change, the
  // concat dimension changes, too. Thus the specifications won't be
  // compatible.
  std::vector<int64_t> list_input_lengths;
  list_input_lengths.reserve(all_inputs.size() - spec.nTensorInputs());
  for (int64_t i = spec.nTensorInputs(); i < all_inputs.size(); i++) {
    if (all_inputs[i].isIntList()) {
      list_input_lengths.push_back(all_inputs[i].toIntListRef().size());
    }
  }

  // Determines device to dispatch to. If there's a device mismatch in the
  // inputs, we use the fallback (which should give a nice error message).
  at::Device device = inputs.at(0).device();
  for (const auto& t : at::TensorList(inputs).slice(1)) {
    if (t.device() != device) {
      return false;
    }
  }

  // Attempts to run fallback if device fusion is disabled
  if (device.is_cuda() && !canFuseOnGPU())
    return false;
  if (device.is_cpu() && !canFuseOnCPU())
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
  ArgSpec arg_spec{inputs, device.index(), list_input_lengths};
  auto maybe_kernel = spec.findKernel(arg_spec);
  if (!maybe_kernel) {
    const auto kernel =
        compileKernel(spec, arg_spec, *maybe_map_size, device, all_inputs);
    spec.cacheKernel(arg_spec, kernel);
  }
  maybe_kernel = spec.findKernel(arg_spec);
  AT_ASSERT(maybe_kernel);

  std::vector<at::Tensor> raw_outputs;
  // Launches fusion
  auto& fusion = *(*maybe_kernel);
  launchFusion(fusion, device, inputs, raw_outputs);

  // now we need to do all the sum to size.
  auto outputs =
      fmap(spec.outputMapAndSizes(), [&](const OutputMapAndSize& omap) {
        auto ooffset = omap.offset();
        const auto& c = fusion.concatDesc()[ooffset];
        if (c.isNoop() && omap.needsSumToSize()) {
          // the easy case: a single output
          return at::sum_to(
              raw_outputs[ooffset],
              all_inputs[omap.sizeInput()].toIntList()->elements());
        } else if (omap.needsSumToSize()) {
          // FusedConcat output that potentially needs sum_to, but possibly also
          // on the concatenation axis and differently for the members.
          // If it were not for chunk returning differently-sized tensors
          // when the number of chunks does not divide the size, we would
          // just restrict this to the "regular case". However, we can construct
          // example where that is not the case, see the tests.
          // Keep in mind that we undo broadcasting in all its beauty (singleton
          // dimensions but also leading dimensions) but we require that the
          // concat members are compatible.

          auto size_inputs = omap.sizeInputs(); // get the size inputs
          AT_ASSERT(c.nSubTensors() > 0);

          // size_to_sum_to will be the size we sum to. Initially we assume that
          // it is that we don't sum.
          auto size_to_sum_to = raw_outputs[ooffset].sizes().vec();

          // Note: this will potentially be adjusted. After the comparison for
          // the first element, this is the dimension of the concat after the
          // sumtosize (c.dim() is before)
          int64_t concat_dim = c.dim();
          // In the concat dimension, the tensors may have different sizes.
          // Again our base hypothesis is that we don't sum.
          std::vector<int64_t> sizes_concat_dim(
              size_inputs.size(), size_to_sum_to[c.dim()] / c.nSubTensors());

          // we need to treat the concat dimension separately from the others
          // so this records our workload
          bool needs_sumtosize_other_dim = false;
          bool needs_sumtosize_concat_dim = false;
          // how many leading dimensions do we need to remove
          int64_t dims_to_go = 0;
          // we get our expectation for the non-concatenated dimensions from
          if (size_inputs[0] != -1) {
            auto target_size = all_inputs[size_inputs[0]].toIntListRef();
            if (target_size.size() != size_to_sum_to.size()) {
              // we have dimensions to delete...
              needs_sumtosize_other_dim = true;
              dims_to_go = size_to_sum_to.size() - target_size.size();
              AT_ASSERTM(dims_to_go > 0, "inconsistent concat sizes in fusion");
              concat_dim = c.dim() - dims_to_go;
              for (size_t d = 0; d < dims_to_go; d++) {
                size_to_sum_to[d] = 1;
              }
            }
            for (size_t d = 0; d < target_size.size(); d++) {
              if (d != concat_dim) {
                needs_sumtosize_other_dim |=
                    (size_to_sum_to[d + dims_to_go] != target_size[d]);
                size_to_sum_to[d + dims_to_go] = target_size[d];
              } else {
                needs_sumtosize_concat_dim |=
                    (sizes_concat_dim[0] != target_size[d]);
                sizes_concat_dim[0] = target_size[d];
              }
            }
          }

          // check the others for compatibility in other dimensions
          // and scan the target sizes of the concat dimension
          for (size_t i = 1; i < size_inputs.size(); i++) {
            if (size_inputs[i] == -1) {
              AT_ASSERTM(
                  !needs_sumtosize_other_dim,
                  "inconsistent concat sizes in fusion");
            } else {
              auto target_size = all_inputs[size_inputs[i]].toIntListRef();
              AT_ASSERTM(
                  target_size.size() + dims_to_go == size_to_sum_to.size(),
                  "inconsistent concat sizes in fusion");
              for (size_t d = 0; d < target_size.size(); d++) {
                if (d == concat_dim) {
                  needs_sumtosize_concat_dim |=
                      (sizes_concat_dim[i] != target_size[d]);
                  sizes_concat_dim[i] = target_size[d];
                } else {
                  // other dimensions must match first tensors example
                  AT_ASSERTM(
                      size_to_sum_to[d + dims_to_go] == target_size[d],
                      "inconsistent concat sizes in fusion");
                }
              }
            }
          }
          // now we do the actual summation
          if (!needs_sumtosize_concat_dim && !needs_sumtosize_other_dim) {
            // nothing to do
            return raw_outputs[ooffset];
          } else if (!needs_sumtosize_concat_dim) {
            // If we don't need to change the concat dim, we can
            // insert a dimensions so that each tensor has
            // one index in the new dimension.
            // Then we can use one sum_to call to do the sum_to_size
            const auto& ro = raw_outputs[ooffset];
            size_to_sum_to.erase(
                size_to_sum_to.begin(), size_to_sum_to.begin() + dims_to_go);
            return at::sum_to(ro, size_to_sum_to);
          } else {
            // If the tensors differ in the concat dim, we need to
            // work tensor by tensor. This can happen because e.g. chunk(4)
            // will work on a size 7 tensor and return outputs
            // of sizes 2 2 2 1.
            // As we don't have sum_to_size_out, we spell out
            // the summation dimensions and use sum_out
            const auto& ro = raw_outputs[ooffset];
            auto ro_size_d = ro.size(c.dim()) / c.nSubTensors();
            // calculate output size and reserve output vecotr
            size_to_sum_to[c.dim()] = 0;
            for (int64_t s : sizes_concat_dim)
              size_to_sum_to[c.dim()] += s;
            auto output = at::empty(size_to_sum_to, ro.options());

            // now find the summation dimensions, except the concat dimension
            std::vector<int64_t> summation_dims;
            for (int64_t d = 0; d < size_to_sum_to.size(); d++) {
              if (size_to_sum_to[d] == 1 && ro.size(d) != 1 && d != c.dim()) {
                summation_dims.push_back(d);
              }
            }
            // now along the concat dimension, it can be that we have
            // to sum some tensors but not others. that's why we keep
            // adding it back as needed
            int64_t output_offset = 0;
            for (int64_t i = 0; i < sizes_concat_dim.size(); i++) {
              auto out_chunk =
                  output.narrow(c.dim(), output_offset, sizes_concat_dim[i]);
              auto raw_chunk = ro.narrow(c.dim(), i * ro_size_d, ro_size_d);
              if (sizes_concat_dim[i] == 1) {
                summation_dims.push_back(c.dim());
                at::sum_out(
                    out_chunk, raw_chunk, summation_dims, /*keepdim*/ true);
                summation_dims.pop_back();
              } else if (!summation_dims.empty()) {
                at::sum_out(
                    out_chunk, raw_chunk, summation_dims, /*keepdim*/ true);
              } else {
                // note: passing the empty list of dims to sum_out does not work
                // as expected here (sums over all dims?)
                out_chunk.copy_(raw_chunk);
              }
              output_offset += sizes_concat_dim[i];
            }
            // remove the leading dimensions we don't want
            size_to_sum_to.erase(
                size_to_sum_to.begin(), size_to_sum_to.begin() + dims_to_go);
            return output.view(size_to_sum_to);
          }
        } else {
          return raw_outputs[ooffset];
        }
      });

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
