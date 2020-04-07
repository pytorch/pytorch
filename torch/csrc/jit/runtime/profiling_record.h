#pragma once

#include <ATen/ATen.h>
#include <ATen/core/ivalue.h>
#include <ATen/core/jit_type.h>
#include <ATen/core/stack.h>
#include <torch/csrc/WindowsTorchApiMacro.h>
#include <torch/csrc/jit/ir/ir.h>

#include <list>
#include <map>
#include <vector>

namespace torch {
namespace jit {

using ::c10::TensorTypePtr;
using Dimension = int64_t;

struct ProfilingRecord;
struct ShapeSymbolSets {
  void reset() {
    sets_.clear();
  };
  std::map<c10::ShapeSymbol, std::map<Dimension, c10::ShapeSymbol>> sets_;

  std::map<Dimension, c10::ShapeSymbol>& getGlobalSet() {
    return getSetForSymbol(c10::ShapeSymbol(-1));
  }

  std::map<Dimension, c10::ShapeSymbol>& getSetForSymbol(c10::ShapeSymbol s) {
    return sets_[s];
  }
};

struct ShapeSymbolTable {
  bool isBound(c10::ShapeSymbol s) {
    if (s.statik_) {
      return true;
    }
    return data_.count(s) != 0;
  }
  void reset() {
    data_.clear();
    sets_.reset();
  };

  Dimension getValue(c10::ShapeSymbol s) {
    if (s.statik_) {
      return s.value_;
    }
    return data_[s];
  }
  void assign(c10::ShapeSymbol s, Dimension v) {
    TORCH_INTERNAL_ASSERT(!s.statik_);
    data_[s] = v;
  }
  std::map<c10::ShapeSymbol, Dimension> data_;
  ShapeSymbolSets sets_;
  ProfilingRecord* pr_;

  c10::ShapeSymbol GetSymbolInSet(
      Dimension,
      c10::ShapeSymbol set,
      ProfilingRecord* pr);
  c10::ShapeSymbol toSymbol(
      Dimension,
      std::map<Dimension, c10::ShapeSymbol>& dims2symbols,
      ProfilingRecord* pr);
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
  std::map<int64_t, std::map<Value*, TensorTypePtr>> profiled_types_per_frame_;
  size_t num_symbols =
      1; // -1 is special to denote the global set of all symbols for a run

  c10::ShapeSymbol getNewSymbol() {
    num_symbols++;
    return c10::ShapeSymbol(-num_symbols, false);
  }

  // A very brief high-level description of an algorithm for
  // constructing sets of dynamic/symbolic shapes.
  // Each set is represented by a ShapeSymbol which can be
  // static or dynamic
  // We implicitly keep track of elements of sets (Value*, i)
  // by assigning a ShapeSymbol to a dimension i in sizes_ of the TensorType of
  // a Value if sizes_[1] of %1 and sizes_[2] of %2 have the same
  // ShapeSymbol(-3, false) they belong to the same set. The algorithm has two
  // main stages. The first stage is to construct symbolic sets for one
  // profiling execution of a graph
  //  * if `sizes_[i]` was never assigned to a set before, it is assigned a
  //  static ShapeSymbol(dim_value, true)
  //    this gives us the initial sets.
  //  * if `sizes_[i]` was assigned before (this could happen if a profiled use
  //  is in a loop) there are 4 cases to handle (`mergeSymbolicShapes`)
  //    * if the ShapeSymbol X at `sizes_[i]` is static and a new symbol is
  //    equal to it (belong to the same set), we keep the original symbol
  //    * if the ShapeSymbol X at sizes_[i] is static and a new symbol is not
  //    equal, we will assign sizes_[i] to a new symbol (this is equivalent to
  //    creating a new subset)
  //    * if the ShapeSymbol X at sizes_[i] is dynamic but we didn't assign it
  //    yet, we will sizes_[i] to a new symbol.
  //         Note, ShapeSymbol can be used both as a Symbol but also as a value
  //         assigned to a Symbol
  //    * if the ShapeSymbol X at sizes_[i] is dynamic and assigned and a new
  //    symbol is equal to the assigned sizes_[i], we keep the original symbol
  //    * if the ShapeSymbol X at sizes_[i] is dynamic and assigned isn't equal
  //    to the assigned sizes_[i], we create a new subset that is represented by
  //    a new symbol whenever we see the same new symbol is being matched
  //    against the ShapeSymbol X we will put it in the same subset.
  // The second stage is to merge symbolic sets from all profiling executions of
  // the graph Since merge/refinement operation is commutative, the order in
  // which we merge symbolic sets from different runs doesn't matter we use the
  // sets from the first run as our initial sets and we essentially reuse the
  // same merge algorithm `mergeSymbolicShapes` the stage 2 adds one extra case
  // we need to deal with is when the runs we processed so far don't have
  // profiling information for some uses that the current run we are merging the
  // information from has. This could happen if the first run executed the then
  // arm of an if and the second run executed the else arm. Since some
  // ShapeSymbols in the current run might already belong to a set we do a
  // reverse search to find which set the ShapeSymbol belongs. Ideally, we would
  // like to put it in the biggest set to be optimal but currently for
  // simplicity we put in a set whose ShapeSymbol was mapped to it the latest.
  std::vector<c10::optional<c10::ShapeSymbol>> mergeSymbolicShapes(
      c10::VaryingShape<c10::ShapeSymbol> new_sizes,
      c10::VaryingShape<c10::ShapeSymbol> sym_shapes,
      ShapeSymbolTable& symbol_table);

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
  void insertShapeProfile(Node* n, Value* i);
  ProfilingRecord(std::shared_ptr<Graph> g);
};

} // namespace jit
} // namespace torch
