#pragma once
#include <torch/csrc/jit/ir.h>
#include "torch/csrc/jit/DisallowCopy.h"
#include "ATen/ATen.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <string>
#include <algorithm>
#include <unordered_map>
#include <vector>

namespace torch { namespace jit {

std::vector<bool> findContiguous(
  at::IntList sizes,
  at::IntList strides);

// type information needed by the compiler for input/outputs
// contiguity[i] is true if the dim i is contiguous with dim i + 1.
// contiguity.back() == true means strides.back() == 1.
struct TensorDesc {
  at::ScalarType scalar_type;
  std::vector<bool> contiguity;
  TensorDesc(at::ScalarType scalar_type, const std::vector<bool> & contiguity)
  : scalar_type(scalar_type), contiguity(contiguity) {
    calcDim();
  }
  TensorDesc(const at::Tensor & t)
  : scalar_type(t.type().scalarType()),
    contiguity(findContiguous(t.sizes(), t.strides())) {
    calcDim();
  }
  TensorDesc(at::ScalarType st, at::IntList sizes, at::IntList strides)
  : scalar_type(st), contiguity(findContiguous(sizes, strides)) {
    calcDim();
  }
  // number of dimensions after contiguity compression
  size_t nDim() const {
    return nDim_;
  }
  // do we have inner stride == 1?
  bool lastIsContiguous() const {
    return contiguity.size() == 0 || contiguity.back();
  }
private:
  size_t nDim_;
  void calcDim() {
    nDim_ = std::count(contiguity.begin(), contiguity.end(), false)
    + (lastIsContiguous() ? 1 : 0);
  }
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
  //we keep these around for debugging
  std::string compliation_unit;
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
};

// caching compiler
struct FusionCompiler {
  TH_DISALLOW_COPY_AND_ASSIGN(FusionCompiler);
  FusionCompiler();
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
  std::unordered_map<std::string, std::shared_ptr<CompiledFusionFunction> > cache;
};

FusionCompiler & sharedFusionCompiler();

}}
