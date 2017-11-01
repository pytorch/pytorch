#pragma once
#include <torch/csrc/jit/ir.h>
#include "torch/csrc/utils/disallow_copy.h"
#include "ATen/ATen.h"
#include <cuda.h>
#include <cuda_runtime.h>
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
  TensorDesc(TensorType *type)
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

// short-term storage only, so it borrows Graph.
// this type is probably temporary.
// it will be replaced when the needed TensorDesc information is encoded
// directly in the information in the IR (e.g. in the Type object)
struct AnnotatedGraph {
  Graph* graph;
  std::vector<TensorDesc> input_desc;
  std::vector<TensorDesc> output_desc;
};

struct ConcatDesc {
  size_t nSubtensors; // == 1 for outputs that are not concats, otherwise it is the number tensors concatenated
  size_t dim; // dimension along which the concat occurs
  std::unique_ptr<TensorDesc> subtensorDesc; // descriptor for the subtensor, if it exists
  ConcatDesc()
  : nSubtensors(1), dim(0) {}
  ConcatDesc(const TensorDesc & desc, size_t nSubtensors, size_t dim)
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
};

struct CompiledFusionFunction {
  TH_DISALLOW_COPY_AND_ASSIGN(CompiledFusionFunction);

  CompiledFusionFunction(const std::string & name, AnnotatedGraph & agraph);
  ~CompiledFusionFunction();

  void launch(at::ArrayRef<at::Tensor> inputs, at::ArrayRef<at::Tensor> outputs);
  const std::vector<TensorDesc> & outputDescriptors() const {
    return output_desc;
  }
private:
  void launch(uint32_t numel, void ** arguments);
  std::string name;
  // We keep these around for debugging
  std::string compilation_unit;
  std::vector<char> ptx;
  CUmodule module;
  CUfunction function;

  // we record prop/device so if they are availiable for launch heuristics
  // querying at launch is too slow for device properties.
  int device;
  cudaDeviceProp prop;
  int blockSize = 128;
  int maxBlocks;

  std::vector<TensorDesc> input_desc;
  std::vector<TensorDesc> output_desc;

  // same size as output_desc, describes whether
  // an output is actually a concatenation of
  // many subtensors that the fusion group produces
  std::vector<ConcatDesc> concat_desc;
};

// caching compiler
struct FusionCompiler {
  TH_DISALLOW_COPY_AND_ASSIGN(FusionCompiler);
  FusionCompiler() {}

  // ignores types in graph, and uses specific contiguity annotations
  std::shared_ptr<CompiledFusionFunction> getOrCompile(AnnotatedGraph & agraph);
  // uses type annotations in graph to create Annotated graph
  std::shared_ptr<CompiledFusionFunction> getOrCompile(Graph & graph);

  // debugging function that lets you do everything from compilation to execution
  // in one step.
  // this should not be used in the hot path of execution because it has to serialize
  // the graph each time
  void debugLaunchGraph(Graph & graph, at::ArrayRef<at::Tensor> inputs, at::ArrayRef<at::Tensor> outputs);
private:
  std::unordered_map<std::string, std::shared_ptr<CompiledFusionFunction>> cache;
};

FusionCompiler & sharedFusionCompiler();

}}
