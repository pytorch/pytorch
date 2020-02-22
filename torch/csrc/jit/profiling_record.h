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
  std::map<size_t, int64_t> dims2symbols_;
  // figure out concurrency and data races
  std::map<int64_t, size_t> symbols2dims_;
  std::map<int64_t, c10::optional<size_t>> static_sizes_;

  void convertToStaticShapes(Block* b);
  void updateStaticSizes(int64_t key, size_t dim);
  int64_t toSymbol(size_t val);
  // size_t toDimension(int64_t symbol, size_t);
  // std::vector<c10::optional<int64_t>> mergeSymbolicShapes(VaryingShape& vs,
  // at::IntArrayRef sizes)
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
