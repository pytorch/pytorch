#pragma once

#include <c10/util/FbcodeMaps.h>

#include <torch/nativert/executor/memory/FunctionSchema.h>
#include <torch/nativert/executor/memory/LayoutPlannerAlgorithm.h>
#include <torch/nativert/graph/Graph.h>

namespace torch::nativert {

class AliasAnalyzer {
 public:
  explicit AliasAnalyzer(
      const Graph& graph,
      const c10::FastMap<std::string /* target */, FunctionSchema>& schemas);

  const c10::FastSet<const Value*>* get_sources_of_alias(
      const Value* value) const {
    const auto it = aliases_.find(value);
    if (it == aliases_.end()) {
      return nullptr;
    }
    return &it->second;
  }

  const AllocationLifetime& lifetime(const Value* value) const {
    return lifetimes_.at(value);
  }

  bool is_alias(const Value* value) const {
    return aliases_.find(value) != aliases_.end();
  }

  bool is_storage_associated_with_output(const Value* value) const {
    return values_associated_with_outputs_.find(value) !=
        values_associated_with_outputs_.end();
  }

  const c10::FastSet<const Value*>& values_associated_with_output_storage()
      const {
    return values_associated_with_outputs_;
  }

  const std::vector<const Value*>& alive_values_at_time(size_t time) const {
    TORCH_CHECK(time < alive_values_at_time_.size());
    return alive_values_at_time_[time];
  }

 private:
  // listunpack operations who take a list that has
  // been created with a listpack operation should
  // be transparent with respect to aliasing
  //
  // e.g., given the op
  // %t[] = prim.ListPack(l0=%t0, l1=%t1)
  // %x1, %x2 = prim.ListUnpack(self=%t)
  // x1 should directly alias t0
  // and likewise x2 should directly alias t1
  //
  // this will make sure that the lifetimes of x1 and x2
  // are not just the max of the lifetimes of t0 and t1
  // which can make tensor-packing more efficient if list
  // element EOL's differ by large amounts
  bool /* applied */ update_aliases_if_packed_listunpack(
      const Node& node,
      size_t i);

  // use the schema aliasing spec, or if none is provided,
  // assume all outputs alias all inputs
  void maybe_update_aliases_from_schema(
      const Node& node,
      const c10::FastMap<std::string /* target */, FunctionSchema>& schemas);

  void create_or_update_lifetime(const Value* value, size_t i);

  // work our way from the DAG's output node to the input node
  // propagating the maximum EOL of all aliases back to their
  // source value(s).
  //
  // in addition, if a graph output is an alias, we need to ensure
  // that the source values are treated as graph outputs
  // so that we don't free them before the graph output is copied
  // back to the user (and we ignore them when creating a memory plan
  // even if they aren't explicitly considered outputs)
  void maybe_extend_lifetimes(const Graph& graph);

  // in the event that we have aliases-of-aliases
  // we want to make sure that the 'sources'
  // are propagated
  //
  // e.g.,
  // %x0 = ...
  // %x1 = some_aliasing_op(x0)
  // %x2 = some_aliasing_op(x1)
  //
  // we want aliases_[x2] = x0
  // instead of aliases[x2] = x1
  //
  // the result is aliases_ will contain a
  // mapping from each alias to its backed
  // source (i.e., the value that owns its
  // associated dataptr)
  void squash_deep_aliases(const Graph& graph);

  void log_state() const;

  // mapping from alias to its source
  c10::FastMap<const Value*, c10::FastSet<const Value*>> aliases_;
  c10::FastMap<const Value*, AllocationLifetime> lifetimes_;
  // non-aliasing outputs or non-aliasing intermediates that are aliased by
  // outputs
  c10::FastSet<const Value*> values_associated_with_outputs_;
  // alive_values_at_time_[i] = values that are "alive" during the
  // computation of node i
  std::vector<std::vector<const Value*>> alive_values_at_time_;
};

} // namespace torch::nativert
