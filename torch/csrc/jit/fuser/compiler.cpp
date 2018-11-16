#include "torch/csrc/jit/fuser/compiler.h"

#include "ATen/ATen.h"
#include "torch/csrc/jit/ir.h"
#include "torch/csrc/jit/type.h"
#include "torch/csrc/jit/code_template.h"
#include "torch/csrc/jit/assertions.h"
#include "torch/csrc/jit/passes/shape_analysis.h"
#include "torch/csrc/jit/fuser/interface.h"
#include "torch/csrc/jit/fuser/kernel_cache.h"
#include "torch/csrc/jit/fuser/codegen.h"
#include "torch/csrc/jit/fuser/tensor_desc.h"

#if USE_CUDA_FUSER
  #include "torch/csrc/jit/fuser/cuda/fused_kernel.h"
#endif // USE_CUDA_FUSER

#if USE_CPU_FUSER
  #include "torch/csrc/jit/fuser/cpu/fused_kernel.h"
#endif // USE_CUDA_FUSER

#include <iostream>
#include <memory>
#include <unordered_set>
#include <utility>
#include <string>
#include <atomic>
#include <sstream>
#include <stdexcept>
#include <tuple>

namespace torch { namespace jit { namespace fuser {

// Counter for number of kernels compiled, used for debugging and
// creating arbitrary kernel names.
static std::atomic<size_t> next_kernel_id{0};

size_t nCompiledKernels() { return next_kernel_id.load(); }

// If the given node is used once by a chunk node, returns that node. 
// Returns nullptr otherwise.
static const Node* usedInFusedChunk(const Value* input) {
  const auto uses = input->uses();
  if (uses.size() == 1) {
    const Node *user = uses[0].user;
    if (user->kind() == prim::ConstantChunk) {
      return user;
    }
  }
  return nullptr;
}

static void setInputChunkDescriptors(KernelSpec& spec) {
  spec.inputChunks().reserve((spec.graph())->inputs().size());
  for (const Value* input : (spec.graph())->inputs()) {
    if (const Node* chunk = usedInFusedChunk(input)) {
      spec.inputChunks().emplace_back(chunk->i(attr::chunks), chunk->i(attr::dim));
    } else {
      spec.inputChunks().emplace_back(1, 0);
    }
  }
}

// Run a DFS traversal to find all inputs that affect a given output value
static std::vector<int64_t> getInputDependencies(const Value* output) {
  std::vector<const Value*> queue{output};
  std::unordered_set<const Value*> inputs;
  std::unordered_set<const Value*> seen;
  while (!queue.empty()) {
    const Value* val = queue.back(); queue.pop_back();
    const Node* producer = val->node();
    if (producer->kind() == prim::Param) {
      inputs.insert(val);
      continue;
    }
    for (const Value* input : producer->inputs()) {
      if (/*bool inserted = */seen.insert(input).second) {
        queue.push_back(input);
      }
    }
  }

  // Convert Value* into offsets into the graph's input list
  std::vector<int64_t> offsets;
  offsets.reserve(inputs.size());
  for (const Value* input : inputs) {
    offsets.push_back(input->offset());
  }

  std::sort(offsets.begin(), offsets.end());
  return offsets;
}

static void setInputBroadcastGroups(KernelSpec& spec) {
  std::unordered_set<std::vector<int64_t>, torch::hash<std::vector<int64_t>>> broadcast_groups;
  for (const Value* output : (spec.graph())->outputs()) {
    broadcast_groups.insert(getInputDependencies(output));
  }
  std::copy(
    broadcast_groups.begin()
  , broadcast_groups.end()
  , std::back_inserter(spec.inputBroadcastGroups()));
}

// Performs "upfront" compilation where storage is known but shapes are not.
// Currently identifies how to expand all tensors so that all intermediate
// tensors are the same shape, simplifying code generation.
// Broadcast groups and chunks are identified without shape information
// using logical properties of how each works. In particular, tensors
// are always expandable to the outputs of pointwise operations they
// or their descendants are involved in, which means that in a DAG of
// pointwise operations all tensors are expandable to the (single) output. 
// Note: The logic is slightly complicated by concatenation and chunking.
static void upfrontCompilation(KernelSpec& spec) {
  setInputBroadcastGroups(spec);
  setInputChunkDescriptors(spec);
}

int64_t registerFusion(const Node* fusion_group) {
  // Creates and stores the FusionSpec
  auto graph = fusion_group->g(attr::Subgraph)->copy();
  EraseShapeInformation(*graph);
  const auto key = store(graph);
  
  if (canFuseOnCPU() || canFuseOnGPU()) {
    const auto maybe_spec = retrieve(key);
    JIT_ASSERT(maybe_spec);
    upfrontCompilation(**maybe_spec);
  }  

  return key;
}

std::shared_ptr<FusedKernel> compileKernel(
  const KernelSpec& spec
, const ArgSpec& arg_spec
, const std::vector<int64_t>& map_size
, const int device) {
  const std::vector<TensorDesc>& input_desc = arg_spec.descs();

  // Note: this assumes fused kernels only operate on floating point values
  c10::optional<at::ScalarType> scalar_type;
  for (const auto& desc : input_desc) {
    if (isFloatingType(desc.scalar_type)) {
      scalar_type = desc.scalar_type;
      break;
    }
  }
  JIT_ASSERT(scalar_type);

  // Creates output descriptions
  std::vector<TensorDesc> output_desc;
  for (const Value* output : (spec.graph())->outputs()) {
    std::vector<int64_t> sizes = map_size;
    if (output->node()->kind() == prim::FusedConcat) {
      sizes.at(output->node()->i(attr::dim)) *= output->node()->inputs().size();
    }
    auto type = CompleteTensorType::create(*scalar_type, device, sizes);
    output_desc.emplace_back(std::move(type));
  }

  const std::string name = "kernel_" + std::to_string(next_kernel_id++);
  const bool use_cuda = (device >= 0);
  std::string code;
  std::vector<PartitionDesc> chunk_desc;
  std::vector<PartitionDesc> concat_desc;
  bool has_random;
  std::tie(code, chunk_desc, concat_desc, has_random) 
    = generateKernel(
        name
      , *(spec.graph())
      , input_desc
      , output_desc
      , use_cuda);

  std::shared_ptr<FusedKernel> fused_kernel;
  if (device != kCPUDevice) {
    #if USE_CUDA_FUSER
      fused_kernel = std::make_shared<cuda::FusedKernelCUDA>(
        device
      , name
      , code
      , input_desc
      , output_desc
      , chunk_desc
      , concat_desc
      , has_random);
    #else
      throw std::runtime_error("CUDA Fusion is not supported on this build.");
    #endif // USE_CUDA_FUSER
  } else {
    #if USE_CPU_FUSER
      fused_kernel = std::make_shared<cpu::FusedKernelCPU>(
        name
      , code
      , input_desc
      , output_desc
      , chunk_desc
      , concat_desc
      , has_random);
    #else
      throw std::runtime_error("CPU Fusion is not supported on this build.");
    #endif // USE_CPU_FUSER
  }

  return fused_kernel;
}

} // namespace fuser
} // namespace jit
} // namespace torch
