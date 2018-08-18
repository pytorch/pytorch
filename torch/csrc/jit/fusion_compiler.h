#pragma once

#include <torch/csrc/jit/ir.h>
#include "torch/csrc/utils/disallow_copy.h"
#include <torch/csrc/jit/assertions.h>

#include "ATen/ATen.h"
#include <string>
#include <algorithm>
#include <unordered_map>
#include <vector>

namespace torch { namespace jit {

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
  TensorDesc(TensorTypePtr type)
    : TensorDesc(type->scalarType(), type->sizes(), type->strides()) {}

  // number of dimensions after contiguity compression
  size_t nDim() const {
    return nDim_;
  }

  // do we have inner stride == 1?
  bool lastIsContiguous() const {
    return contiguity.size() == 0 || contiguity.back();
  }

  static std::vector<bool> findContiguous(
    const at::IntList& sizes,
    const at::IntList& strides);

private:
  size_t nDim_;
};

constexpr int kCPUDevice = -1;
struct AnnotatedGraph {
  // short-term storage only, so it borrows Graph.
  AnnotatedGraph(Graph & graph, int device)
  : graph(&graph), device(device) {}
  Graph* graph = nullptr;
  int device = kCPUDevice;
  std::vector<TensorDesc> input_desc;
  std::vector<TensorDesc> output_desc;
};

// Descriptor for chunk-ing an input tensor into subtensors
// OR concat-ing an output tensor from subtensors
struct PartitionDesc {
  size_t nSubtensors; // == 1 for tensors that should not be operated on via chunk/cat
  size_t dim; // dimension along which the chunk/concat occurs
  std::unique_ptr<TensorDesc> subtensorDesc; // descriptor for the subtensor, if it exists
  PartitionDesc()
  : nSubtensors(1), dim(0) {}

  // Constructor for cat descriptors
  // desc: TensorDesc for output tensor
  PartitionDesc(const TensorDesc & desc, size_t nSubtensors, size_t dim)
  : nSubtensors(nSubtensors), dim(dim) {
    JIT_ASSERT(nSubtensors > 1);
    std::vector<bool> cont = desc.contiguity;
    if(dim > 0) {
      // when we narrow the concatenated output
      // we make the size[dim] smaller while keeping the stride[dim] the same,
      // meaning: stride[dim - 1] != stride[dim]*size[dim]
      // so dim - 1 is no longer contiguous
      cont[dim - 1] = false;
    }
    subtensorDesc.reset(new TensorDesc(desc.scalar_type, cont));
  }

  // Constructor for chunk descriptors
  // tensor_type: the type of the input to the chunk node
  // ignored: XXX: Compiler gets confused without this arg
  PartitionDesc(TensorTypePtr tensor_type, size_t chunks, size_t dim, bool ignored)
  : nSubtensors(chunks), dim(dim) {
    (void)ignored;
    JIT_ASSERT(chunks > 1);

    std::vector<int64_t> sizes(tensor_type->sizes().begin(), tensor_type->sizes().end());
    JIT_ASSERT(sizes[dim] % chunks == 0); // Should have been checked in graph fuser
    sizes[dim] /= chunks;
    // Computes contiguity, which is what we really care about
    subtensorDesc.reset(new TensorDesc(tensor_type->scalarType(), sizes, tensor_type->strides()));
  }

  bool isNoop() const {
    return nSubtensors == 1;
  }
};

struct CompiledFusionFunction {
  TH_DISALLOW_COPY_AND_ASSIGN(CompiledFusionFunction);

  CompiledFusionFunction(const std::string & name, AnnotatedGraph & agraph);
  virtual ~CompiledFusionFunction() = default;

  // expects outputs to be pre-allocated
  void launch_with_tensors(at::ArrayRef<at::Tensor> inputs, at::ArrayRef<at::Tensor> outputs);

  // creates new tensors for outputs
  void launch(at::ArrayRef<at::Tensor> inputs, std::vector<at::Tensor> & outputs);
  const std::vector<TensorDesc> & outputDescriptors() const {
    return output_desc;
  }
protected:
  virtual at::Backend backend() const = 0;

  // arguments is a list of pointers to the arguments for the compiled CUDA/CPU
  // code.
  // The format of arguments is suitable for directly passing to a call to
  // cuLaunchKernel as the kernel arguments.
  // Currently the first argument is a pointer to numel (for passing to
  // CUDA code), and the remainder are pointers to the TensorInfo<T> structs
  // that compiled code uses to load Tensor data.
  // launch_with_tensors handles packing at::Tensors into this arguments array.
  // CPU code uses the same convension so that launch_with_tensors can be shared.
  virtual void launch_raw(uint32_t numel, void ** arguments) = 0;

  virtual uint64_t get_rand_offset(uint32_t numel) = 0;
  bool has_random;
  std::string name;
  // We keep these around for debugging
  std::string compilation_unit;
  std::vector<TensorDesc> input_desc;
  std::vector<TensorDesc> output_desc;

  // same size as output_desc, describes whether
  // an output is actually a concatenation of
  // many subtensors that the fusion group produces
  std::vector<PartitionDesc> concat_desc;

  // same size as input_desc, describes whether an
  // input should be broken into subtensors (chunks)
  // to be consumed by the fusion group
  std::vector<PartitionDesc> chunk_desc;
};

struct FusionCompilerConfig {
  std::string cxx = "g++"; // compiler location
  bool debug = false; // emit debugging information about fusions
  bool openmp = true;
};

// caching compiler
struct FusionCompiler {
  TH_DISALLOW_COPY_AND_ASSIGN(FusionCompiler);
  FusionCompiler();

  // ignores types in graph, and uses specific contiguity annotations
  std::shared_ptr<CompiledFusionFunction> getOrCompile(AnnotatedGraph & agraph);
  // uses type annotations in fusion_group to create Annotated graph
  std::shared_ptr<CompiledFusionFunction> getOrCompile(Node * fusion_group);

  // uses inputs/outputs as examples to infer continuity, does not run the graph
  std::shared_ptr<CompiledFusionFunction> getOrCompile(Graph & graph,
                                                       int device,
                                                       at::ArrayRef<at::Tensor> inputs,
                                                       at::ArrayRef<at::Tensor> outputs);
  // debugging function that lets you do everything from compilation to execution
  // in one step.
  // this should not be used in the hot path of execution because it has to serialize
  // the graph each time
  void debugLaunchGraph(Graph & graph, int device, at::ArrayRef<at::Tensor> inputs, at::ArrayRef<at::Tensor> outputs);
  bool canCompileOnCPU() const {
    return config_.cxx.size() > 0;
  }
private:
  FusionCompilerConfig config_;
  std::unordered_map<std::string, std::shared_ptr<CompiledFusionFunction>> cache;
};

FusionCompiler & sharedFusionCompiler();

}}
