#pragma once

#include <ATen/ATen.h>
#include <ATen/core/ivalue.h>
#include <ATen/core/jit_type.h>
#include <ATen/core/stack.h>
#include <torch/csrc/WindowsTorchApiMacro.h>
#include <torch/csrc/jit/ir/ir.h>

#include <list>
#include <map>
#include <unordered_map>
#include <vector>

// We would like to assign each position/axis of a tensor an abstract size
// * For each `tensor` we have a profiled `Value` of a `TensorType` describing
// the properties of the `tensor`.
// * `TensorType` has a property called `symbolic_sizes_` to describe observed
// `tensor.sizes()`
// * `symbolic_sizes_` is a vector of abstract sizes (or
// `std::vector<ShapeSymbol>`) where
//   * `ShapeSymbol`at `symbolic_sizes_[i]`  describes the size value
//   (`Dimension`) at `tensor.sizes()[i]`
// * We may see the same `Dimension` at different positions `i` in
// `tensor.sizes()` or even in different `tensor`
//   * First, we would like associate the same `ShapeSymbol` to the same
//   `Dimension` across **one** profiling execution or run of a TorchScript
//   function.
//     * The same `ShapeSymbol`s in different positions of `symbolic_shapes_` in
//     possibly different `TensorType`s (i.e. `TensorType`s for different
//     profiled values) form an implicit set. The elements of such a set are
//     called *dimension locations*.
//     * These sets allow us to track how the shapes of input arguments of some
//     operation relate to operation's output shapes as the input and output
//     shapes might share the same `ShapeSymbol`s
// * For **every** profiling run, we would like to maintain the invariant that
// *the same `ShapeSymbol` is always associated with the same `Dimension`*.
// * To maintain this invariant we merge the profiling information from all
// profiling runs,
//   * For every two runs, we iterate over all `symbic_shapes_`  and compare
//   their `ShapeSymbol`s in the same position.
//     * if we observe that for every dimension location that has
//     the`ShapeSymbol S1`  in run #1 there is **only one** `ShapeSymbol S2` in
//     the same dimension location in run #2, we conclude that the invariant
//     holds.
//     * However, if we observe some dimension locations in run #2 have
//     `ShapeSymbol S2` and the other ones have `ShapeSymbol S3` we would like
//     to partition the virtual set of dimension locations associated with
//     `ShapeSymbol S1` into two new subsets, so the invariant holds.
//     * The partitioning works by assigning a new symbol to the dimension
//     locations (associated with `ShapeSymbol S1`) that have `ShapeSymbol S2`
//     and another new symbol to the dimension locations that have `ShapeSymbol
//     S3`. In other words,
//       * Subset #1 will consist of the dimension locations that in run #2 have
//       `ShapeSymbol S2`  and will have `ShapeSymbol S4`  in those dimension
//       locations
//       * Subset #2 will consist of the dimension locations that in run #2 have
//       `ShapeSymbol S4`  and will have `ShapeSymbol S5`  in those dimension
//       locations
//     * The effective result of merging the profiling information from two runs
//     is new `TensorTypes` whose `symbolic_sizes_` /dimension locations have
//     either `ShapeSymbol S4` or `ShapeSymbol S5`.
//     * Partitioning can be done even before we have seen all the dimension
//     locations associated with `ShapeSymbol S1`
//       * We use `getSymbolInSet` of `ShapeSymbolTable` to remember all
//       `ShapeSymbols` from run #2 we observed in the dimension locations
//       associated with `ShapeSymbol S1` .
//       * For every `ShapeSymbol` from run #2 in the dimension location
//       associated with `ShapeSymbol S1`  `getSymbolInSet` returns a symbol
//       that we assign to the dimension location in a new TensorType.
//         * It's important to point out that the same `ShapeSymbol S2` from run
//         #2 in two dimension locations that have different `ShapeSymbol`s in
//         run #1 are different! These dimension locations will belong to
//         different subsets and have different `ShapeSymbol`s after merge.
//         * On the other hand, for the same `ShapeSymbol S2` in two dimension
//         locations that have `ShapeSymbol S1` in run #1`getSymbolInSet` will
//         return the same symbol.

namespace torch {
namespace jit {

using ::c10::TensorTypePtr;
using Dimension = int64_t;

struct ProfilingRecord;

// `SetPartitioningHelper` is used to maintain the following invariant:
// For **every** profiling run, *the same `ShapeSymbol` is always associated
// with the same `Dimension`*.
// while merging the profiling information from multiple runs.
struct SetPartitioningHelper {
  std::map<c10::ShapeSymbol, std::map<Dimension, c10::ShapeSymbol>>
      sets2subsets_;

  // `partitionSetByDimension` partitions a virtual set
  // of dimension locations associated with ShapeSymbol `symbol` into subsets.
  // Partitioning is equivalent to giving (or renaming) a particular
  // dimension location a new `ShapeSymbol`.
  // The same `Dimension` value in different dimension locations
  // that used to have `symbol` will receive the same
  // new `ShapeSymbol`, effectively forming a new set.
  c10::ShapeSymbol partitionSetByDimension(
      Dimension new_size,
      c10::ShapeSymbol symbol) {
    auto& dims2symbols = getSetForSymbol(symbol);

    if (dims2symbols.count(new_size) == 0) {
      auto new_sym = c10::ShapeSymbol::newSymbol();
      dims2symbols[new_size] = new_sym;
      return new_sym;
    }

    return dims2symbols[new_size];
  }

 private:
  std::map<Dimension, c10::ShapeSymbol>& getSetForSymbol(c10::ShapeSymbol s) {
    auto& set = sets2subsets_[s];
    // N.B. adding a mapping { s.static_size(), s }
    // makes sure we preserve the fact that
    // some dimension values remain the same
    // across all profiled runs
    if (s.is_static()) {
      set.insert({s.static_size(), s});
    }
    return set;
  }
};

// ShapeSymbolTable is used by Interpreter
// to assign dimension values to ShapeSymbols
// and fail a guard if the same symbol
// is assigned more than one dimension value.
struct ShapeSymbolTable {
  // N.B. we treat static symbols as always assigned
  // to themselves
  bool isBound(c10::ShapeSymbol s) {
    if (s.is_static()) {
      return true;
    }
    return data_.count(s) != 0;
  }

  // N.B. we treat static symbols as always assigned
  // to themselves
  Dimension getValue(c10::ShapeSymbol s) {
    if (s.is_static()) {
      return s.static_size();
    }
    return data_[s];
  }
  void assign(c10::ShapeSymbol s, Dimension v) {
    TORCH_INTERNAL_ASSERT(!s.is_static());
    data_[s] = v;
  }
  std::map<c10::ShapeSymbol, Dimension> data_;
  // Tries to assign dimension values from `new_sizes` to
  // `ShapeSymbol`s `sym_shapes`.
  // Returns `true` if every dimension value from `new_sizes`
  // can be assigned to the corresponding `ShapeSymbol` from
  // `sym_shapes`
  // A dimension value can be assigned to a `ShapeSymbol`
  // * if the symbol isn't assigned yet any dimension value
  // * if the symbol is assigned and its value is equal to
  // the dimension value from `new_sizes`
  bool bindSymbolicShapes(
      at::IntArrayRef new_sizes,
      const c10::SymbolicShape& sym_shapes);
};

struct ProfilingRecord {
  // N.B. ProfilingRecord's copy and move c-tor are disabled, so we won't
  // end up accidentally copying or moving ProfilingRecords whose addresses
  // are captured in callbacks_
  ProfilingRecord(const ProfilingRecord&) = delete;
  ProfilingRecord(ProfilingRecord&&) noexcept = delete;
  TORCH_API static std::unique_ptr<ProfilingRecord> instrumentGraph(
      const std::shared_ptr<Graph>& graph);
  TORCH_API static void removeProfilingNodes(Block* b);
  TORCH_API static void removeProfileCounter(Block* b);

  std::shared_ptr<Graph> profiled_graph_;
  std::mutex mutex_;
  size_t profiling_count_;
  // the key is a frame id
  // the value is a mapping from a Value in a graph
  // to a profiled TensorType
  std::map<int64_t, std::map<Value*, TensorTypePtr>> profiled_types_per_frame_;

  // A thin wrapper around `partitionSetByDimension` to ensure
  // `new_sizes` and `sym_shapes` are of the same rank

  c10::SymbolicShape mergeSymbolicShapes(
      const c10::SymbolicShape& new_sizes,
      const c10::SymbolicShape& sym_shapes,
      SetPartitioningHelper& partition_helper);

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
  ProfileOptionalOp* createProfileOptionalNode(
      const std::function<void(Stack&)>& fp,
      at::ArrayRef<Value*> inputs);
  void instrumentBlock(Block* block);
  void insertShapeProfile(Node* n, size_t offset);
  ProfilingRecord(std::shared_ptr<Graph> g);
};

} // namespace jit
} // namespace torch
