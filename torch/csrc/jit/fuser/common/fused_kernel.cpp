#include "torch/csrc/jit/fuser/common/fused_kernel.h"

#include "ATen/ATen.h"
#include "torch/csrc/jit/code_template.h"
#include "torch/csrc/jit/ir.h"
#include "torch/csrc/jit/assertions.h"
#include "torch/csrc/jit/fuser/interface.h"
#include "torch/csrc/jit/fuser/cpu/resource_strings.h"
#include "torch/csrc/jit/fuser/cuda/resource_strings.h"
#include "torch/csrc/jit/fuser/common/partition_desc.h"
#include "torch/csrc/jit/fuser/common/tensor_desc.h"
#include "torch/csrc/jit/fuser/common/tensor_info.h"

#if USE_CUDA_FUSER
  #include "THC/THCTensorRandom.h"
  #include "THC/THCGenerator.hpp"
  THCGenerator* THCRandom_getGenerator(THCState* state);
#endif // USE_CUDA_FUSER

#include <tuple>
#include <iostream>
#include <sstream>
#include <cstdint>
#include <vector>
#include <cmath>

namespace torch { namespace jit { namespace fuser {

// XXX: this code assumes that inputs are 32-bit addressable
static uint32_t computeNumel(at::ArrayRef<int64_t> sizes) {
  uint32_t result = 1;
  if (sizes.size() == 0) {
    return 1; // scalar tensor
  }
  for (int64_t size : sizes) {
    result *= size;
  }
  return result;
}

// XXX: Assumes that after at::chunk, all inputs are the same size
static std::vector<int64_t> computeMapSize(
    const at::Tensor& tensor,
    const PartitionDesc& chunkDesc) {
  std::vector<int64_t> sizes(tensor.sizes().begin(), tensor.sizes().end());
  // Should have been checked in graph fuser
  JIT_ASSERT(sizes[chunkDesc.dim] % chunkDesc.nSubtensors == 0);
  sizes[chunkDesc.dim] /= chunkDesc.nSubtensors;
  return sizes;
}

// Tries to compress sizes and strides according to cont. Emits the result t
// c_sizes, c_strides and throws an error on failure (if can't compress)
static void compressContiguous(
  at::IntList sizes
, at::IntList strides
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
   // cur starts pointing at the beginning of run to compress
   // cur ends one _after_ the terminating false or end of list.
   // total_size is the size of all dimensions [begin,end)
   // examples:
   // f = not cont.
   // t = cont.
   // x = don't care, including past end of list
   // s = start of cur
   // e = end of cur


   // f x x x
   // s e

   //  t f x x
   //  s   e

   //  t t f x
   //  s     e

    c_sizes[compressed_dims] = total_size;
    c_strides[compressed_dims] = strides[cur-1];
    compressed_dims++;
  }
  if (ndim > 0) {
    JIT_ASSERT(!cont.back() || strides.back() == 1);
  }
}

void FusedKernel::launch_with_tensors(
  at::ArrayRef<at::Tensor> inputs
, at::ArrayRef<at::Tensor> outputs) {
  at::DeviceGuard device_guard(inputs);
  JIT_ASSERT(inputs.size() == input_desc_.size());
  JIT_ASSERT(outputs.size() == output_desc_.size());
  size_t flat_inputs_size = 0;
  size_t flat_outputs_size = 0;
  for (const auto& c : chunk_desc_)
    flat_inputs_size += c.nSubtensors;
  for (const auto& c : concat_desc_)
    flat_outputs_size += c.nSubtensors;
  // XXX: this code assumes that inputs are 32-bit addressable
  // XXX: this code assumes that all inputs are of the same size
  JIT_ASSERT(inputs[0].numel() <= std::numeric_limits<uint32_t>::max());

  // Compute map_size, numel from the first input
  at::IntList map_size;
  uint32_t numel;
  std::vector<int64_t> keep_alive_size;
  if (chunk_desc_[0].isNoop()) {
    map_size = inputs[0].sizes();
    numel = inputs[0].numel();
  } else {
    keep_alive_size = computeMapSize(inputs[0], chunk_desc_[0]);
    map_size = keep_alive_size;
    numel = computeNumel(map_size);
  }

  // Compute the storage needed to store TensorInfo structs for inputs and outputs.
  size_t uncompressedDim = input_desc_.at(0).contiguity.size();
  size_t maxPossibleTensorInfoSize = sizeof(TensorInfo) + 2 * sizeof(uint32_t) * uncompressedDim;
  size_t maxPossibleBufferSize = maxPossibleTensorInfoSize * (flat_inputs_size + flat_outputs_size);
  std::vector<char> buffer(maxPossibleBufferSize);
  char* buffer_next = buffer.data();
  // A vector of arguments to the kernel. It's (numel, *input_desc_s, *output_desc_s)
  std::vector<void*> arguments;
  arguments.reserve(3 + flat_inputs_size + flat_outputs_size);
  auto addTensorInfoRaw = [&](const TensorDesc& desc, void* data_ptr, at::IntList sizes, at::IntList strides) {
    size_t nDim = desc.nDim(); // NOTE: this is the compressed dim
    JIT_ASSERT(nDim <= uncompressedDim); // We'd overflow the space otherwise
    auto ti = reinterpret_cast<TensorInfo*>(buffer_next);
    ti->data = data_ptr;
    compressContiguous(sizes, strides, desc.contiguity, ti->sizes(nDim), ti->strides(nDim));
    buffer_next += maxPossibleTensorInfoSize;
    arguments.push_back(ti);
  };
  // Asserts that t's dims can be compressed in the same way as in desc
  // (that's what the kernel assumes), and appends it to the arguments vector.
  auto addTensorInfo = [&](const TensorDesc& desc, const at::Tensor& t) {
    addTensorInfoRaw(desc, t.data_ptr(), t.sizes(), t.strides());
  };
  arguments.push_back(&numel);
  for (size_t i = 0; i < input_desc_.size(); ++i) {
    auto & chunk = chunk_desc_[i];
    const at::Tensor& tensor = inputs[i];
    if (chunk.isNoop()) {
      addTensorInfo(input_desc_[i], tensor);
    } else {
      size_t chunk_offset = map_size[chunk.dim] * tensor.stride(chunk.dim) * elementSize(tensor.type().scalarType());
      char * data_ptr = reinterpret_cast<char*>(tensor.data_ptr());
      for (size_t chunks = 0; chunks < chunk.nSubtensors; ++chunks) {
        addTensorInfoRaw(*chunk.subtensorDesc, data_ptr, map_size, tensor.strides());
        data_ptr += chunk_offset;
      }
    }
  }
  for (size_t i = 0; i < output_desc_.size(); ++i) {
    const auto& c = concat_desc_[i];
    at::Tensor o = outputs[i];
    if (c.isNoop()) {
      o.resize_(map_size);
      addTensorInfo(output_desc_[i], outputs[i]);
    } else {
      size_t small_size = map_size[c.dim];
      std::vector<int64_t> concat_size(map_size.begin(), map_size.end());
      concat_size[c.dim] = small_size * c.nSubtensors;
      o.resize_(concat_size);
      size_t offset = 0;
      for (size_t j = 0; j < c.nSubtensors; ++j) {
        // because the concatenated_output stays live, the underlying data
        // in this view remains live through the end of this function
        // so there is not need to hold onto this tensor
        auto view = o.narrow(c.dim, offset, small_size);
        addTensorInfo(*c.subtensorDesc, view);
        offset += small_size;
      }
    }
  }

  // If the kernel call contains a random op, we need to pass in random seeds as
  // well.
  #if USE_CUDA_FUSER
    if (has_random_ && this->backend() == at::Backend::CUDA) {
      auto gen_ = THCRandom_getGenerator(at::globalContext().getTHCState());
      uint64_t offset =
          gen_->state.philox_seed_offset.fetch_add(this->get_rand_offset(numel));
      arguments.push_back(&gen_->state.initial_seed);
      arguments.push_back(&offset);
    }
  #endif // USE_CUDA_FUSER
  
  launch_raw(numel, arguments.data());
}

void FusedKernel::launch(
  at::ArrayRef<at::Tensor> inputs
, std::vector<at::Tensor>& outputs) {
  at::DeviceGuard guard(inputs.back());
  JIT_ASSERT(inputs.size() > 0);
  auto& ref_type = inputs[0].type();
  outputs.clear();
  outputs.reserve(output_desc_.size());
  for (const auto& od : output_desc_) {
    outputs.push_back(at::empty({0}, ref_type.options().dtype(od.scalar_type)));
  }

  launch_with_tensors(inputs, outputs);
}

} // namespace fuser
} // namespace jit
} // namespace torch
