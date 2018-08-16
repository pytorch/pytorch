#if !(defined _WIN32)
#pragma once

#include "torch/csrc/jit/fusers/fuser_interface.h"

#include "torch/csrc/jit/fusers/cpu/tensor_desc.h"
#include "torch/csrc/jit/fusers/cpu/annotated_graph.h"
#include "torch/csrc/jit/fusers/cpu/concat_desc.h"

#include "torch/csrc/jit/ir.h"
#include "torch/csrc/utils/disallow_copy.h"
#include "torch/csrc/jit/assertions.h"

#include "ATen/ATen.h"

#include <string>
#include <algorithm>
#include <unordered_map>
#include <vector>

namespace torch { namespace jit { namespace cpufuser {

struct CompiledCPUFusionFunction : public CompiledFusionFunction {
  TH_DISALLOW_COPY_AND_ASSIGN(CompiledCPUFusionFunction);

  CompiledCPUFusionFunction(
    const std::string& name
  , AnnotatedGraph& agraph);
  
  virtual ~CompiledCPUFusionFunction() override = default;

  // Note: Creates new tensors for outputs
  virtual void launch(
    at::ArrayRef<at::Tensor> inputs
  , std::vector<at::Tensor>& outputs) override;

  // Note: expects outputs to be pre-allocated
  void launch_with_tensors(
    at::ArrayRef<at::Tensor> inputs
  , at::ArrayRef<at::Tensor> outputs);

  const std::vector<TensorDesc>& outputDescriptors() const {
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
  virtual void launch_raw(uint32_t numel, void** arguments) = 0;

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
  std::vector<ConcatDesc> concat_desc;
};

struct CPUFusionCompilerConfig {
  std::string cxx = "g++"; // compiler location
  bool debug = false; // emit debugging information about fusions
  bool openmp = true;
};

struct CPUFusionCompiler {
  TH_DISALLOW_COPY_AND_ASSIGN(CPUFusionCompiler);

  CPUFusionCompiler();

  // ignores types in graph, and uses specific contiguity annotations
  std::shared_ptr<CompiledCPUFusionFunction> getOrCompile(
    AnnotatedGraph& agraph);

  // uses inputs/outputs as examples to infer continuity, does not run the graph
  std::shared_ptr<CompiledCPUFusionFunction> getOrCompile(
    Graph& graph
  , int device
  , at::ArrayRef<at::Tensor> inputs
  , at::ArrayRef<at::Tensor> outputs);

// debugging function that lets you do everything from compilation to execution
  // in one step.
  // this should not be used in the hot path of execution because it has to serialize
  // the graph each time
  void debugLaunchGraph(
    Graph& graph
  , int device
  , at::ArrayRef<at::Tensor> inputs
  , at::ArrayRef<at::Tensor> outputs);

  bool canCompileOnCPU() const { return config_.cxx.size() > 0; }

private:
  CPUFusionCompilerConfig config_;
  std::unordered_map<
    std::string
  , std::shared_ptr<CompiledCPUFusionFunction>> cache;
};

CPUFusionCompiler& getCompiler();

} // namespace cpufuser
} // namespace jit
} // namespace torch

#endif // !(defined _WIN32)
