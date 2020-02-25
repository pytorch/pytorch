#pragma once

#include <torch/csrc/WindowsTorchApiMacro.h>
#include <ATen/ATen.h>
#include <ATen/core/ivalue.h>
#include <ATen/core/jit_type.h>
#include <ATen/core/stack.h>
#include <torch/csrc/jit/ir.h>

#include <list>
#include <map>
#include <vector>

namespace torch {
namespace jit {

using ::c10::TensorTypePtr;

struct ProfilingRunRecord {
  std::map<int64_t, int64_t> dims2symbols_;
  std::map<int64_t, int64_t> symbols2dims_;
  std::map<Value*, c10::TensorTypePtr> symbolic_shapes_;
  std::map<int64_t, std::map<int64_t, int64_t>> split_symbols_;
};

struct ProfilingRecord {
  // N.B. ProfilingRecord's copy and move c-tor are disabled, so we won't
  // end up accidentally copying or moving ProfilingRecords whose addresses
  // are captured in callbacks_
  ProfilingRecord(const ProfilingRecord&) = delete;
  ProfilingRecord(ProfilingRecord&&) noexcept = delete;
  TORCH_API static std::unique_ptr<ProfilingRecord> instrumentGraph(
      const std::shared_ptr<Graph>& graph);

  std::shared_ptr<Graph> profiled_graph_;
  std::mutex mutex_;
  size_t profiling_count_;
  std::map<int64_t, ProfilingRunRecord> profiling_records_;
  size_t num_symbols = 0;

  int64_t getNewSymbol() {
    num_symbols++;
    return -num_symbols;
  }

  std::vector<int64_t> mergeSymbolicShapes(
      at::IntArrayRef sizes,
      c10::VaryingShape sym_shapes,
      std::map<int64_t, int64_t>& dims2symbols,
      std::map<int64_t, int64_t>& symbols2dims,
      std::map<int64_t, std::map<int64_t, int64_t>> split_symbols);
  int64_t toSymbol(int64_t val, std::map<int64_t, int64_t>& dims2symbols);
  bool ready() const {
    return profiling_count_ == 0;
  }
  std::shared_ptr<Graph> graph() const {
    return profiled_graph_;
  }
 private:
  ProfileOp* createProfileNode(
      const std::function<void(Stack&)>& fp,
      at::ArrayRef<Value*> inputs);
  void instrumentBlock(Block* block);
  void insertShapeProfile(Node *n, Value *i);
  ProfilingRecord(std::shared_ptr<Graph> g);
};

} // namespace jit
} // namespace torch
