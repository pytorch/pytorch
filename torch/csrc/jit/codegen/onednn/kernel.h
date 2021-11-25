#pragma once

#include <ATen/native/mkldnn/LlgaTensorImpl.h>
#include <torch/csrc/jit/codegen/onednn/graph_helper.h>
#include <torch/csrc/jit/codegen/onednn/rw_mutex.hpp>
#include <unordered_map>

#include <oneapi/dnnl/dnnl_graph.hpp>
#include <torch/csrc/jit/ir/ir.h>
#include <torch/csrc/jit/runtime/interpreter.h>

namespace torch {
namespace jit {
namespace fuser {
namespace onednn {

using ArgSpec = at::LlgaTensorDesc;
using ArgSpecs = std::vector<ArgSpec>;
using RunArg = dnnl::graph::tensor;
using RunArgs = std::vector<RunArg>;
using TensorArgs = std::vector<at::Tensor>;

class LlgaKernel {
 public:
  explicit LlgaKernel(const Node* fusionNode);

  void run(Stack& stack);

  const std::string& debugName() const {
    return debugName_;
  }

 private:
  bool useOpaqueLayout(size_t offset) const;

  // PyTorch copy constants inside the subgraph instead of referencing them.
  // Constants inputs to the partition are no longer in the graph->inputs().
  // Need use the tid retrieved from the partition to find the missing
  // constant inputs.
  void initializeConstantInputs();

  ArgSpecs initializeInputSpecs(const TensorArgs& inputs);

  ArgSpecs initializeOutputSpecs(
      const dnnl::graph::partition& partition,
      const ArgSpecs& inputSpecs) const;

  dnnl::graph::compiled_partition compile(
      const dnnl::graph::partition& partition);

  std::tuple<RunArgs, RunArgs> prepareRunArgs(
      const TensorArgs& inputs,
      TensorArgs& outputs) const;

  static std::string genDebugName() {
    static size_t debugId = 0;
    return "LlgaPartition_" + std::to_string(debugId++);
  }

  static dnnl::graph::logical_tensor toLogicalTensor(const ArgSpec& s) {
    return s.logical_tensor();
  }

  void lock_read() {
    rw_mutex_.lock_read();
  }

  void lock_write() {
    rw_mutex_.lock_write();
  }

  void unlock_read() {
    rw_mutex_.unlock_read();
  }

  void unlock_write() {
    rw_mutex_.unlock_write();
  }

  at::Device device_ = at::kCPU;
  const Node* fusionNode_;
  std::shared_ptr<Graph> graph_;
  int64_t nGraphInputs_ = 0; // number of inputs to graph_ on the IR
  int64_t nOutputs_ = 0;
  std::map<size_t, Value*> tensorIdToValue_;
  dnnl::graph::partition partition_;
  // nPartitionInputs_ is the actual number of inputs to partition_ of graph_
  // needed by the backend.
  // nPartitionInputs_ = nGraphInputs_ + constantInputs_.size() since Constant
  // inputs are copied to the inside of the subgraph
  int64_t nPartitionInputs_;
  dnnl::graph::compiled_partition compilation_;
  std::set<size_t> initializedInputIds_;
  std::vector<Value*> constantValues_;
  TensorArgs constantInputs_;
  ArgSpecs inputSpecs_;
  ArgSpecs outputSpecs_;
  std::unordered_map<size_t, size_t> inplacePairs_; // output id -> input offset
  std::string debugName_;
  rw_mutex_t rw_mutex_;
  bool is_initialized_ = false;
};

struct Engine {
  // CPU engine singleton
  static dnnl::graph::engine& getEngine();
  Engine(const Engine&) = delete;
  void operator=(const Engine&) = delete;
};

struct Stream {
  // CPU stream singleton
  static dnnl::graph::stream& getStream();
  Stream(const Stream&) = delete;
  void operator=(const Stream&) = delete;
};

} // namespace onednn
} // namespace fuser
} // namespace jit
} // namespace torch
