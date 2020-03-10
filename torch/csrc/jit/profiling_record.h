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


// using SymbolOrStaticShape = int64_t;
// using ShapeValue = int64_t;
// using c10::ShapeSymbol = int64_t;

struct ShapeSymbolTable {
  bool isBound(c10::ShapeSymbol s) { return data_.count(s) != 0; }
  void reset() { data_.clear(); };

  c10::ShapeSymbol getValue(c10::ShapeSymbol s) { return data_[s]; }
  void assign(c10::ShapeSymbol s, c10::ShapeSymbol v) { data_[s] = v; }
  std::map<c10::ShapeSymbol, c10::ShapeSymbol> data_;
};

struct ShapeSymbolSets {

  void reset() { sets_.clear(); };
  std::map<c10::ShapeSymbol, std::map<c10::ShapeSymbol, c10::ShapeSymbol>> sets_;
  
  std::map <c10::ShapeSymbol, c10::ShapeSymbol>& getGlobalSet() {
    return getSetForSymbol(c10::ShapeSymbol(-1));
  }

  std::map<c10::ShapeSymbol, c10::ShapeSymbol>& getSetForSymbol(c10::ShapeSymbol s) {
    return sets_[s];
  }

};


struct FrameSymbols {
  ShapeSymbolSets symbol_sets_;
  ShapeSymbolTable symbol_table_;
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
  std::map<int64_t, std::map<Value*, TensorTypePtr>> profiled_types_per_frame_;
  std::map<int64_t, FrameSymbols> symbols_per_frame_;
  size_t num_symbols = 1; // -1 is special to denote the global set of all symbols for a run

  c10::ShapeSymbol getNewSymbol() {
    num_symbols++;
    return c10::ShapeSymbol(-num_symbols, false);
  }

  std::vector<c10::ShapeSymbol> mergeSymbolicShapes(
    const std::vector<c10::ShapeSymbol>& new_sizes,
    c10::optional<std::vector<c10::ShapeSymbol>> sym_shapes,
    ShapeSymbolSets& symbol_sets,
    ShapeSymbolTable& symbol_table);

  c10::ShapeSymbol toSymbol(c10::ShapeSymbol, std::map<c10::ShapeSymbol, c10::ShapeSymbol>& dims2symbols);
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
