#pragma once

#include <torch/csrc/jit/ir.h>
#include "torch/csrc/utils/disallow_copy.h"
#include "torch/csrc/utils/hash.h"
#include <torch/csrc/jit/assertions.h>
#include <torch/csrc/jit/stack.h>
#include <torch/csrc/jit/argument_spec.h>
#include <torch/csrc/jit/interpreter.h>

#include "ATen/ATen.h"
#include <string>
#include <algorithm>
#include <unordered_map>
#include <vector>
#include <memory>

namespace torch { namespace jit {

struct FusedKernel;
struct FusionCompiler;

// type information needed by the compiler for input/outputs
// contiguity[i] is true if the dim i is contiguous with dim i + 1.
// contiguity.back() == true means strides.back() == 1.
struct TensorDesc {
  at::ScalarType scalar_type;
  std::vector<bool> contiguity;

  TensorDesc(const at::ScalarType& type, const std::vector<bool>& contiguity)
  : scalar_type(type), contiguity(contiguity) {
    nDim_ = std::count(contiguity.begin(), contiguity.end(), false) + (lastIsContiguous() ? 1 : 0);
  }

  TensorDesc(const at::ScalarType& type, const at::IntList& sizes, const at::IntList& strides)
  : TensorDesc(type, TensorDesc::findContiguous(sizes, strides)) {}
  TensorDesc(const at::Tensor& t)
    : TensorDesc(t.type().scalarType(), t.sizes(), t.strides()) {}
  TensorDesc(CompleteTensorTypePtr type)
    : TensorDesc(type->scalarType(), type->sizes(), type->strides()) {}

  // number of dimensions after contiguity compression
  size_t nDim() const {
    return nDim_;
  }

  // do we have inner stride == 1?
  bool lastIsContiguous() const {
    // NB: A scalar tensor does not have a "last dimension" because it has 0 dims.
    return contiguity.size() != 0 && contiguity.back();
  }

  static std::vector<bool> findContiguous(
    const at::IntList& sizes,
    const at::IntList& strides);

  bool operator==(const TensorDesc & desc) const {
    return scalar_type == desc.scalar_type && contiguity == desc.contiguity;
  }
  bool operator!=(const TensorDesc & desc) const {
    return !(*this == desc);
  }
  static size_t hash(const TensorDesc& spec) {
    return torch::get_hash(spec.scalar_type, spec.nDim_, std::hash<std::vector<bool>>{}(spec.contiguity));
  }

private:
  size_t nDim_;
};

inline std::ostream& operator<<(std::ostream & out, const TensorDesc & d) {
  out << d.scalar_type << "[";
  for(auto b : d.contiguity)
    out << b << ";";
  out << "]";
  return out;
}

struct FusedKernelArgSpec {
  FusedKernelArgSpec(at::TensorList inputs)
    : descs_(fmap<TensorDesc>(inputs))
    , hash_code_(torch::get_hash(inputs.size(), descs_)) {}

  bool operator==(const FusedKernelArgSpec & spec) const {
    return hash_code_ == spec.hash_code_ && descs_ == spec.descs_;
  }
  bool operator!=(const FusedKernelArgSpec & spec) const {
    return !(*this == spec);
  }
  static size_t hash(const FusedKernelArgSpec& spec) {
    return spec.hash_code_;
  }
  const std::vector<TensorDesc>& descs() const {
    return descs_;
  }

private:
  std::vector<TensorDesc> descs_;
  size_t hash_code_;
};

constexpr int kCPUDevice = -1;
struct AnnotatedGraph {
  // short-term storage only, so it borrows Graph.
  AnnotatedGraph(Graph & graph, int device)
  : graph(&graph), device(device) {}
  Graph* graph = nullptr; // TODO: this should really be const
  int device = kCPUDevice;
  std::vector<TensorDesc> input_desc;
  std::vector<TensorDesc> output_desc;
};

// FusionCompiler has very limited shape information available at the time getOrCompile
// is called, and this is why it can't really prepare the kernels at that time. Instead,
// it returns this object, which will take care of matching the run-time shapes to whatever
// kernels we have compiled already.
//
// Two configurations are considered eligible for the same fused kernel if:
//   - the shapes satisfy graph invariants for our fused code (e.g. that all intermediate shapes
//     are the same - see fusion_compiler.cpp for more details).
//   - their FusedKernelArgSpecs compare equal
struct FusedKernelCache {
  FusedKernelCache(FusionCompiler& compiler, std::shared_ptr<Graph> graph, int device);

  void run(Stack& inputs);
private:
  struct PartitionInfo {
    PartitionInfo(int64_t nsub, int64_t dim)
      : nSubtensors(nsub), dim(dim) {};
    int64_t nSubtensors;
    int64_t dim;
  };

  void runFallback(Stack& stack);
  void expandArgs(std::vector<at::Tensor>& args, std::vector<int64_t>& map_size);
  at::optional<std::vector<int64_t>> canRunKernel(at::TensorList args);
  at::optional<std::vector<int64_t>> getMapSize(at::TensorList args, at::IntList arg_subset);
  std::vector<std::vector<int64_t>> getInputBroadcastGroups();
  std::vector<PartitionInfo> getInputChunkDescriptors();
  std::unique_ptr<FusedKernel> compileSpec(
        const FusedKernelArgSpec& spec, const std::vector<int64_t>& map_size);

  static std::atomic<size_t> next_kernel_id;

  int device;
  Code fallback_code;
  FusionCompiler& compiler;
  std::shared_ptr<Graph> graph;
  std::vector<std::vector<int64_t>> input_broadcast_groups;
  std::vector<PartitionInfo> input_chunks;
  std::unordered_map<FusedKernelArgSpec, std::unique_ptr<FusedKernel>, torch::hash<FusedKernelArgSpec>> kernels;
};

struct FusionCompilerConfig {
  std::string cxx = "g++"; // compiler location
  bool debug = false; // emit debugging information about fusions
  bool openmp = true;
};

// caching compiler
struct FusionCompiler {
  friend struct FusedKernelCache;

  FusionCompiler();
  TH_DISALLOW_COPY_AND_ASSIGN(FusionCompiler);

  // uses type annotations in fusion_group to create Annotated graph
  std::shared_ptr<FusedKernelCache> getOrCompile(Node * fusion_group);

  // debugging function that lets you do everything from compilation to execution
  // in one step.
  // this should not be used in the hot path of execution because it has to serialize
  // the graph each time
  std::vector<at::Tensor> debugLaunchGraph(Graph & graph, int device, at::ArrayRef<at::Tensor> inputs);
  bool canCompileOnCPU() const {
    return config_.cxx.size() > 0;
  }
private:
  FusionCompilerConfig config_;
  std::unordered_map<std::string, std::shared_ptr<FusedKernelCache>> cache_map;
};

FusionCompiler & sharedFusionCompiler();

}}
