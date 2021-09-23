#include <torch/csrc/jit/runtime/static/impl.h>

#include <ATen/MemoryOverlap.h>
#include <ATen/core/interned_strings.h>
#include <c10/core/CPUAllocator.h>
#include <c10/core/InferenceMode.h>
#include <c10/util/irange.h>
#include <caffe2/core/scope_guard.h>
#include <caffe2/core/timer.h>
#include <torch/csrc/jit/ir/alias_analysis.h>
#include <torch/csrc/jit/passes/canonicalize.h>
#include <torch/csrc/jit/passes/dead_code_elimination.h>
#include <torch/csrc/jit/passes/freeze_module.h>
#include <torch/csrc/jit/passes/remove_mutation.h>
#include <torch/csrc/jit/passes/subgraph_rewrite.h>
#include <torch/csrc/jit/passes/variadic_ops.h>
#include <torch/csrc/jit/runtime/static/memory_planner.h>
#include <torch/csrc/jit/runtime/static/ops.h>
#include <torch/csrc/jit/runtime/static/passes.h>
#include <torch/csrc/jit/runtime/vararg_functions.h>
#include <stdexcept>

#ifdef FBCODE_CAFFE2
#include <folly/dynamic.h>
#include <folly/json.h>
#endif

namespace torch {
namespace jit {

// graph must be frozen or canEnableStaticRuntime would return false if there's
// any prim::CallMethod op left in the graph
bool canEnableStaticRuntime(const std::shared_ptr<torch::jit::Graph>& graph) {
  // check for sub-blocks
  bool can_support = true;
  bool has_blocks = false;
  for (auto* node : graph->block()->nodes()) {
    if (node->blocks().size() > 0) {
      has_blocks = true;
      VLOG(1) << "Found nested sub-blocks in graph at node: "
              << PrintNode(node);
    }
    if (node->kind() == prim::Constant) {
      continue;
    }
    // check if can get op from Node
    const Operator* op = node->maybeOperator();
    if (!op && !nativeOpIsRegistered(node->kind())) {
      can_support = false;
      LOG(WARNING) << "Found unsupported op: " << node->kind().toQualString();
    }
  }
  if (has_blocks) {
    LOG(WARNING)
        << "Found nested sub-block in graph. Static Runtime doesn't support nested sub-blocks.";
    can_support = false;
  }
  return can_support;
}

namespace {

void OptimizeGraph(
    std::shared_ptr<torch::jit::Graph>& graph,
    const StaticModuleOptions& opts) {
  Inline(*graph);
  ConstantPropagation(graph);
  Canonicalize(graph);
  ConstantPropagation(graph);
  RemoveTensorMutation(graph);
  ConstantPropagation(graph);
  EliminateDeadCode(graph);
  FuseInferenceOpsForSparseNN(graph);
  UseVariadicCat(graph);
  UseVariadicStack(graph);
  if (opts.enable_out_variant) {
    FuseSignLog1P(graph);
  }

  // TODO: we can avoid this guard by moving operations
  // to exposed folders.
#ifdef FBCODE_CAFFE2
  if (opts.enable_out_variant) {
    FuseListUnpack(graph);
    ReplaceWithCopy(graph);
    EnableStaticRuntimeLayerNorm(graph);
  }
#endif
  ConstantPropagation(graph);
  RemoveImmutableInputDictLookups(graph);
  UseVariadicTupleUnpack(graph);
}

// remove unused input 0 from graph
bool RemoveSelfFromGraphInput(std::shared_ptr<torch::jit::Graph>& graph) {
  if (graph->inputs().at(0)->type()->is_module()) {
    if (graph->inputs().at(0)->hasUses()) {
      return false;
    }
    graph->eraseInput(0);
  }
  return true;
}

// remove "self" from function schema
c10::FunctionSchema RemoveSelfFromSchema(const c10::FunctionSchema& s) {
  TORCH_CHECK(s.arguments().size() >= 1 && s.arguments()[0].name() == "self");
  std::vector<Argument> args({s.arguments().begin() + 1, s.arguments().end()});
  return s.cloneWithArguments(args);
}

bool mayContainAlias(AliasDb& db, const Value* a, const Value* b) {
  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
  return db.mayContainAlias(const_cast<Value*>(a), const_cast<Value*>(b));
}

bool mayContainAlias(
    AliasDb& db,
    const FastSet<const Value*>& a,
    const FastSet<const Value*>& b) {
  std::vector<Value*> as;
  std::vector<Value*> bs;
  as.reserve(a.size());
  for (auto* v : a) {
    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
    as.emplace_back(const_cast<Value*>(v));
  }
  bs.reserve(b.size());
  for (auto* v : b) {
    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
    bs.emplace_back(const_cast<Value*>(v));
  }
  return db.mayContainAlias(as, bs);
}

// Get set of all inputs/outputs/constants (always alive) and their aliases
FastSet<const Value*> GetAlwaysAliveValues(
    const std::shared_ptr<torch::jit::Graph>& graph,
    AliasDb& db) {
  // a set of Values whose live-range exceed current inference
  FastSet<const Value*> always_alive;

  // mark inputs, constants, outputs as always_alive
  for (const auto* input : graph->inputs()) {
    always_alive.insert(input);
  }
  for (const auto* output : graph->outputs()) {
    always_alive.insert(output);
  }
  for (const auto* node : graph->nodes()) {
    if (node->kind() == prim::Constant) {
      for (const auto* output : node->outputs()) {
        always_alive.insert(output);
      }
    }
  }

  // insert aliases of always live Values
  for (const auto* node : graph->nodes()) {
    // constants are already in the always_alive set
    if (node->kind() != prim::Constant) {
      for (const auto* v : node->outputs()) {
        if (mayContainAlias(db, {v}, always_alive)) {
          always_alive.insert(v);
        }
      }
    }
  }
  return always_alive;
}

//  Map each value to all values that are alive at the same time.
using LivenessMap = FastMap<const Value*, FastSet<const Value*>>;

//  The algorithm does a traversal of the execution graph
//  while keeping track of the live values.
LivenessMap GetLivenessMap(
    const std::shared_ptr<torch::jit::Graph>& graph,
    const FastSet<const Value*>& always_alive,
    AliasDb& db) {
  // map a Value to a set of Values that overlap live-ranges with the Value's
  FastMap<const Value*, FastSet<const Value*>> liveness_map;

  // map Values to its creation order in graph (Note: only traverse top-level
  // nodes such that nodes under control-flows are represented by top-level
  // block nodes)
  std::vector<const Value*> values_in_creation_order;
  FastMap<const Value*, size_t> values_to_idx_in_creation_order;
  for (const auto* node : graph->nodes()) {
    for (const auto* v : node->outputs()) {
      values_to_idx_in_creation_order.emplace(
          v, values_in_creation_order.size());
      values_in_creation_order.emplace_back(v);
    }
  }

  // presence of a Value in live_values_use_chain means the Value alive
  // Value mapped to set of Nodes that may use the Value (i.e., use-chain of
  // Value)
  FastMap<const Value*, FastSet<const Node*>> live_values_use_chain;
  // Node mapped to set of Values that the Node may use (i.e., def-chain of node
  // inputs)
  FastMap<const Node*, FastSet<const Value*>> live_nodes_def_chain;

  // add v to the current liveness_map
  std::function<void(const Value* v)> add_live_value_fn = [&](const Value* v) {
    if (liveness_map.count(v)) {
      return;
    }

    auto& v_live_set = liveness_map[v] = {};

    for (const auto& live_v : live_values_use_chain) {
      v_live_set.insert(live_v.first);
      liveness_map.at(live_v.first).insert(v);
    }

    // only add values to the live set if they
    // have deps, otherwise they die immediately
    if (v->uses().size()) {
      live_values_use_chain[v] = {};
    }

    // record the relationship between v (Value) and its uses (Node)
    for (const auto& u : v->uses()) {
      const auto* node = u.user;
      live_values_use_chain.at(v).insert(node);
      live_nodes_def_chain[node].insert(v);
    }

    // FIXME(penguin): the following alias refinement seems to assume
    // that `v` refers to a new  tensor created by the node that defines
    // v, thus other Values "before" the node that defines `v` cannot
    // possibly be aliased to `v`.
    // TODO(penguin): Is it a limitation of TS alias analysis
    // so that we need to do such refinement? If so, better improve
    // alias analysis so that we dont need this special handling here
    //
    // Refine aliases of v by include only those created after v
    std::vector<const Value*> refined_aliases;
    auto idx = values_to_idx_in_creation_order[v];
    for (; idx < values_in_creation_order.size(); ++idx) {
      auto* alias_v = values_in_creation_order[idx];
      if (mayContainAlias(db, v, alias_v)) {
        refined_aliases.emplace_back(alias_v);
      }
    }
    // for all the values in the alias set,
    // we set them "alive"
    for (auto* aliased_v : refined_aliases) {
      add_live_value_fn(aliased_v);
      for (const auto& u : aliased_v->uses()) {
        const auto* node = u.user;
        // track deps of the aliased values is if they
        // are our own
        live_values_use_chain.at(v).insert(node);
        live_nodes_def_chain[node].insert(v);
      }
    }
  };

  auto traverse_node_fn = [&](const Node* node,
                              std::vector<const Value*>& dead) {
    if (live_nodes_def_chain.count(node)) {
      for (const auto* v : live_nodes_def_chain.at(node)) {
        live_values_use_chain.at(v).erase(node);
        if (!live_values_use_chain.at(v).size()) {
          dead.emplace_back(v);
        }
      }
    }
  };

  for (const auto* node : graph->nodes()) {
    for (const auto* v : node->outputs()) {
      if (always_alive.count(v) == 0) {
        add_live_value_fn(v);
      }
    }

    std::vector<const Value*> dead;
    traverse_node_fn(node, dead);
    for (const auto* dead_value : dead) {
      live_values_use_chain.erase(dead_value);
    }
  }

  for (const auto& v : live_values_use_chain) {
    TORCH_CHECK(always_alive.count(v.first));
  }

  for (const auto* node : graph->nodes()) {
    auto inputs = node->inputs();
    auto outputs = node->outputs();
    for (const auto* input : inputs) {
      for (const auto* output : outputs) {
        auto input_it = liveness_map.find(input);
        if (input_it == liveness_map.end()) {
          continue;
        }
        auto output_it = liveness_map.find(output);
        if (output_it == liveness_map.end()) {
          continue;
        }
        input_it->second.insert(output);
        output_it->second.insert(input);
      }
    }
    auto insert_all_pairs_in_liveness_map =
        [&](at::ArrayRef<const Value*> values) {
          for (size_t i = 0; !values.empty() && i < values.size() - 1; ++i) {
            auto value_it = liveness_map.find(values[i]);
            if (value_it == liveness_map.end()) {
              continue;
            }
            for (size_t j = i + 1; j < values.size(); ++j) {
              auto value2_it = liveness_map.find(values[j]);
              if (value2_it != liveness_map.end()) {
                value_it->second.insert(values[j]);
                value2_it->second.insert(values[i]);
              }
            }
          }
        };
    // All inputs should be alive at the same time.
    insert_all_pairs_in_liveness_map(inputs);

    // All outputs should be alive at the same time.
    insert_all_pairs_in_liveness_map(outputs);
  };

  return liveness_map;
};

// Collect the set of Values that are candidates for memory planning:
//   - Values that are used in in-place operators (i.e., _out variants), and
//   - excluding those that are either inputs or outputs of
//     non in-place operators
//
// Returns
//   first: Values that are candidates for memory planning
//   second: A deterministc order of all values
std::pair<std::vector<const Value*>, std::vector<const Value*>>
GetMemoryPlanningCandidates(
    const std::shared_ptr<torch::jit::Graph>& graph,
    const FastMap<Node*, bool>& node_has_out_variant) {
  // for determinism
  FastSet<const Value*> seen_values;
  std::vector<const Value*> all_values;
  FastSet<const Value*> can_reuse;
  // values used by unsupported ops (as either inputs or outputs)
  // these need to be removed from "can_reuse" after analyzing all nodes
  FastSet<const Value*> cannot_reuse;
  for (auto* n : graph->nodes()) {
    bool can_reuse_inputs_outputs =
        canReuseInputsOutputs(n, node_has_out_variant);
    for (const auto* v : n->inputs()) {
      if (!seen_values.count(v)) {
        all_values.emplace_back(v);
        seen_values.insert(v);
      }
      if (can_reuse_inputs_outputs) {
        can_reuse.insert(v);
      } else {
        cannot_reuse.insert(v);
      }
    }
    for (const auto* v : n->outputs()) {
      all_values.emplace_back(v);
      seen_values.insert(v);
      if (can_reuse_inputs_outputs) {
        can_reuse.insert(v);
      } else {
        cannot_reuse.insert(v);
      }
    }
  }
  for (const auto* v : cannot_reuse) {
    can_reuse.erase(v);
  }
  // find a deterministic order
  std::vector<const Value*> optimizable;
  for (const auto* v : all_values) {
    if (can_reuse.count(v)) {
      optimizable.emplace_back(v);
      can_reuse.erase(v);
    }
  }
  return std::make_pair(optimizable, all_values);
}

// Equipped with a liveness map we can allocate memory to
// ivalues, reusing memory along the way. However, we are
// constrained by the set of optimizable_values
// (inputs/outputs of out variants). Inputs/outputs of view ops
// can't be reused.
//
// Algorithm:
// # clusters of values sharing the same memory
// # are called "value_to_same_storage_values" in the implementation
// # inserting into a cluster denotes sharing memory.
//
// clusters = {}
// for all v in optimzable_values:
//   for all cluster in clusters: # can we insert into cluster?
//     for all live_v in live_during(v):
//        if cluster.contains(live_v):
//          skip to next custer
//     cluster.add(v)
//     skip to next v
//   if no cluster found:
//     clusters.add(cluster{v})
//
//
// NB: This is a deterministic implementation, which makes it easier to tune
// and debug.
FastMap<const Value*, std::vector<const Value*>> GenerateSameStorageValues(
    const LivenessMap& alive_during,
    const FastSet<const Value*>& always_alive,
    const std::pair<std::vector<const Value*>, std::vector<const Value*>>&
        optimizable,
    AliasDb& db) {
  const auto& optimizable_values = optimizable.first;
  const auto& all_values = optimizable.second;

  // map Value* to a set Value* that can share the same storage with it
  FastMap<const Value*, std::vector<const Value*>> same_storage_values;

  // make new_v and old_v map to the same storage (i.e., add to each other's
  // same_storage_values set)
  auto share_storage_fn = [&](const Value* new_v, const Value* old_v) {
    if (new_v == old_v) {
      return;
    }
    DCHECK(same_storage_values.count(old_v));
    FastSet<const Value*> seen;
    std::vector<const Value*> values;
    for (auto* v : same_storage_values.at(old_v)) {
      if (seen.count(v)) {
        continue;
      }
      seen.insert(v);
      values.emplace_back(v);
    }
    for (auto* v : same_storage_values.at(new_v)) {
      if (seen.count(v)) {
        continue;
      }
      seen.insert(v);
      values.emplace_back(v);
    }
    for (const auto* v : values) {
      same_storage_values[v] = values;
    }
  };

  // initialize with known same_storage_values (aliasing values)
  for (const auto* v : all_values) {
    if (!same_storage_values.count(v)) {
      same_storage_values[v] = {v};
    }
    // skip always alive values (alias inputs/outputs/weights)
    if (always_alive.count(v)) {
      continue;
    }
    for (const auto& p : same_storage_values) {
      // NB: this means we cannot optimize operations that "sometimes alias"
      // TODO: add a more robust check of this behavior at runtime
      // FIXME (penguin): this handling makes v and MayAlias(v) share the
      // same storage, which is not correct.
      if (db.mayAlias(p.first, v)) {
        share_storage_fn(v, p.first);
      }
    }
  }

  // to preserve determinism
  std::vector<const Value*> seen;

  auto compute_liveset_fn =
      [&always_alive, &alive_during, &same_storage_values](
          FastSet<const Value*>& live, const Value* v) {
        for (const auto* sv : same_storage_values.at(v)) {
          const auto& l = alive_during.count(sv) ? alive_during.at(sv)
                                                 : FastSet<const Value*>{};
          live.insert(l.begin(), l.end());
        }
        live.insert(always_alive.begin(), always_alive.end());
      };

  // check if same_storage_values[s] intersects with live
  auto intersect_fn = [&same_storage_values](
                          FastSet<const Value*>& live, const Value* s) {
    bool intersect = false;
    for (const auto* v : same_storage_values.at(s)) {
      if (live.count(v)) {
        intersect = true;
        break;
      }
    }
    return intersect;
  };

  for (const auto* v : optimizable_values) {
    if (always_alive.count(v)) {
      continue;
    }
    // get values that are live during the lifetime of v
    FastSet<const Value*> live;
    compute_liveset_fn(live, v);
    for (const auto* s : seen) {
      // if live(same_storage_values[v]) and same_storage_values[s]
      // do not overlap, then s and v can share the same storage
      if (!intersect_fn(live, s)) {
        share_storage_fn(v, s);
        // since s is added to same_storage_values[v], live needs
        // to be recomputed, so bail out here
        break;
      }
    }
    seen.emplace_back(v);
  }

  return same_storage_values;
}

void PrepareGraphForStaticModule(
    std::shared_ptr<torch::jit::Graph> graph,
    const StaticModuleOptions& opts) {
  TORCH_CHECK(canEnableStaticRuntime(graph));
  OptimizeGraph(graph, opts);
}

std::pair<std::shared_ptr<Graph>, std::shared_ptr<Module>>
PrepareForStaticModule(
    const torch::jit::Module& m,
    bool is_frozen,
    const StaticModuleOptions& opts) {
  VLOG(1) << "StaticModuleOptions: cleanup_activations "
          << opts.cleanup_activations << ", enable_out_variant "
          << opts.enable_out_variant << ", optimize_memory"
          << opts.optimize_memory << ", optimize_graph_output_memory"
          << opts.optimize_graph_output_memory;

  std::shared_ptr<Module> module_ptr;
  if (!is_frozen) {
    auto module = m.copy();
    module.eval();
    module_ptr = std::make_shared<Module>(freeze_module(module));
  } else {
    module_ptr = std::make_shared<Module>(m.copy());
  }

  Method method = module_ptr->get_method("forward");
  auto graph = module_ptr->get_method("forward").graph();

  // graph->dump();
  PrepareGraphForStaticModule(graph, opts);

  return std::make_pair(graph, module_ptr);
}

std::pair<std::shared_ptr<Graph>, std::shared_ptr<Module>>
PrepareForStaticModule(
    std::shared_ptr<torch::jit::Graph> graph,
    const StaticModuleOptions& opts) {
  PrepareGraphForStaticModule(graph, opts);
  return std::make_pair(graph, nullptr);
}

} // namespace

StaticModule::StaticModule(
    std::shared_ptr<torch::jit::Graph> g,
    const StaticModuleOptions& opts)
    : StaticModule(PrepareForStaticModule(g->copy(), opts), opts) {}

StaticModule::StaticModule(
    const torch::jit::Module& m,
    bool is_frozen,
    const StaticModuleOptions& opts)
    : StaticModule(PrepareForStaticModule(m, is_frozen, opts), opts) {}

StaticModule::StaticModule(
    std::pair<std::shared_ptr<torch::jit::Graph>, std::shared_ptr<Module>>
        graph_and_module,
    const StaticModuleOptions& opts)
    : opts_(opts),
      graph_(std::move(graph_and_module.first)),
      module_(std::move(graph_and_module.second)) {
  // check opt flags
  if (opts.optimize_graph_output_memory) {
    TORCH_CHECK(
        opts_.enable_out_variant && opts_.optimize_memory,
        "When optimize_graph_output_memory is true, enable_out_variant and optimize_memory must be set to true");
  }
  if (opts_.optimize_memory) {
    TORCH_CHECK(
        opts_.enable_out_variant,
        "When optimize_memory is true, enable_out_variant must be set to true");
  }

  // handle schema
  if (module_) {
    Method method = module_->get_method("forward");
    schema_ = method.function().getSchema();
    if (RemoveSelfFromGraphInput(graph_)) {
      schema_ = RemoveSelfFromSchema(method.function().getSchema());
    } else {
      first_input_is_self_ = true;
      schema_ = method.function().getSchema();
    }
  }

  // map Value* to IValue (from inputs or prim::Constant) or null
  FastMap<Value*, IValue*> value_to_ivalue;
  // map Value* to its SSA definition IR
  FastMap<Value*, DefInfo> value_to_ssa_def;

  // N inputs map to the first N entries in storage
  for (const auto i : c10::irange(graph_->inputs().size())) {
    Value* input = graph_->inputs()[i];
    value_to_ivalue[input] = nullptr;
    value_to_ssa_def[input] = std::make_pair(INPUT_VALUE, i);
  }

  // NB: before optimizing the order of execution, ensure that the
  // memory optimization pass (LivenessMap) is
  // aware of the new order!

  // Fill constants first, so we have a std::vector<IValue> we can reference
  // later
  for (Node* node : graph_->nodes()) {
    if (node->kind() != prim::Constant) {
      continue;
    }
    auto* v = node->output();
    TORCH_CHECK(v->type()->kind() != FunctionType::Kind);
    constants_.emplace_back(toIValue(v).value());
  }
  {
    // construct SSA definition for constant nodes
    int i = 0;
    for (Node* node : graph_->nodes()) {
      if (node->kind() != prim::Constant) {
        continue;
      }
      auto* v = node->output();
      value_to_ssa_def[v] = std::make_pair(CONSTANT_VALUE, i);
      value_to_ivalue[v] = &(constants_[i++]);
    }
  }

  // construct SSA definition for non-constant nodes
  int node_idx = 0;
  FastMap<Node*, bool> node_has_out_variant;
  for (Node* node : graph_->nodes()) {
    if (node->kind() == prim::Constant) {
      continue;
    }
    std::vector<const IValue*> ivalue_inputs;
    std::vector<DefInfo> input_ssa_defs;
    for (Value* input : node->inputs()) {
      ivalue_inputs.emplace_back(value_to_ivalue.at(input));
      input_ssa_defs.emplace_back(value_to_ssa_def.at(input));
    }
    node_inputs_ssa_def_map_[node_idx] = input_ssa_defs;
    auto pnode =
        ProcessedNode(node, std::move(ivalue_inputs), opts.enable_out_variant);
    node_has_out_variant.emplace(node, pnode.has_out_variant());
    nodes_.emplace_back(std::move(pnode));
    for (const auto i : c10::irange(node->outputs().size())) {
      value_to_ivalue[node->outputs()[i]] = nullptr;
      value_to_ssa_def[node->outputs()[i]] = std::make_pair(node_idx, i);
    }
    node_idx++;
  }
  for (auto& pnode : nodes_) {
    if (pnode.outputs().size() == 1 &&
        isOptimizableContainerType(pnode.node(), node_has_out_variant)) {
      node_is_optimizable_container_type_.emplace(pnode.node());
    }
  }
  for (auto output : graph_->outputs()) {
    output_ssa_defs_.emplace_back(value_to_ssa_def[output]);
  }

  // Prepare for memory planning
  AliasDb alias_db(graph_);
  external_values_ = GetAlwaysAliveValues(graph_, alias_db);

  if (opts_.optimize_memory) {
    auto lm = GetLivenessMap(graph_, external_values_, alias_db);
    auto values = GetMemoryPlanningCandidates(graph_, node_has_out_variant);
    value_to_same_storage_values_ =
        GenerateSameStorageValues(lm, external_values_, values, alias_db);
  }
}

const StaticModuleOptions& StaticModule::opts() const {
  return opts_;
}

size_t StaticModule::num_outputs() const {
  return graph_->outputs().size();
}

size_t StaticModule::num_inputs() const {
  return graph_->inputs().size();
}

StaticRuntime& StaticModule::runtime() {
  if (!cached_runtime_) {
    cached_runtime_ = std::make_unique<StaticRuntime>(*this);
  }
  return *cached_runtime_;
}

std::vector<at::Tensor> StaticModule::operator()(
    const std::vector<at::Tensor>& inps) {
  return runtime()(inps);
}
c10::IValue StaticModule::operator()(
    c10::ArrayRef<c10::IValue> args,
    const std::unordered_map<std::string, c10::IValue>& kwargs) {
  return runtime()(args, kwargs);
}

StaticRuntime::StaticRuntime(const StaticModule& sm) : static_module_(sm) {
  // NB: create unchanging std::vector<IValue>s we can reference
  inputs_.resize(sm.num_inputs());
  nodes_.resize(sm.nodes().size());

  for (const auto idx : c10::irange(sm.nodes().size())) {
    const auto& n_ref = sm.nodes()[idx];
    nodes_[idx] = n_ref; // copy the node
    auto& n = nodes_[idx];
    // hook up the inputs

    for (const auto i : c10::irange(n.inputs().size())) {
      if (n.inputs()[i] == nullptr) {
        int node_idx = 0;
        int out_idx = 0;
        std::tie(node_idx, out_idx) = sm.index_map().at(idx)[i];
        DCHECK(out_idx >= 0);
        // input
        if (node_idx == StaticModule::INPUT_VALUE) {
          n.set_input(i, &inputs_[out_idx]);
        } else if (node_idx == StaticModule::CONSTANT_VALUE) {
          n.set_input(i, &sm.constants()[out_idx]);
        } else {
          DCHECK(node_idx >= 0);
          n.set_input(i, &(nodes_[node_idx].Output(out_idx)));
        }
      }
    }
  }

  for (const auto& index_pair : sm.output_indices()) {
    int node_idx = 0;
    int out_idx = 0;
    std::tie(node_idx, out_idx) = index_pair;
    if (node_idx == StaticModule::INPUT_VALUE) {
      outputs_.emplace_back(&inputs_[out_idx]);
    } else if (node_idx == StaticModule::CONSTANT_VALUE) {
      // This is a very rare case where const correctness
      // breaks -- the user is returning a constant from
      // the graph.
      // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
      outputs_.emplace_back(const_cast<IValue*>(&sm.constants()[out_idx]));
    } else {
      auto* out = &nodes_[node_idx].Output(out_idx);
      outputs_.emplace_back(out);
    }
  }
}

StaticRuntime::~StaticRuntime() = default;

std::vector<at::Tensor> StaticRuntime::operator()(
    const std::vector<at::Tensor>& inps) {
  std::vector<c10::IValue> stack;
  stack.resize(inps.size());
  for (const auto i : c10::irange(inps.size())) {
    stack[i] = inps[i];
  }

  c10::IValue v =
      (*this)(stack, std::unordered_map<std::string, c10::IValue>());

  std::vector<at::Tensor> out;

  if (v.isTuple()) {
    auto t = v.toTuple();
    for (const auto& el : t->elements()) {
      out.emplace_back(el.toTensor());
    }
  } else {
    out.emplace_back(v.toTensor());
  }
  return out;
}

void StaticRuntime::set_inputs(
    c10::ArrayRef<c10::IValue> args,
    const std::unordered_map<std::string, c10::IValue>& kwargs) {
  if (!kwargs.empty()) {
    // This is not ideal
    TORCH_CHECK(
        static_module_.schema(),
        "Schema is not available. Consider creating the Static Runtime "
        "with StaticModule(const torch::jit::Module& m) instead.");
    std::vector<c10::IValue> stack;
    stack.reserve(inputs_.size());
    if (static_module_.first_input_is_self()) {
      stack.emplace_back(static_module_.module()._ivalue());
    }
    stack.insert(stack.end(), args.begin(), args.end());

    static_module_.schema()->checkAndNormalizeInputs(stack, kwargs);
    DCHECK_EQ(inputs_.size(), stack.size());
    for (const auto i : c10::irange(stack.size())) {
      Input(i) = std::move(stack[i]);
    }
  } else {
    if (static_module_.first_input_is_self()) {
      Input(0) = static_module_.module()._ivalue();
      DCHECK_EQ(inputs_.size(), args.size() + 1);
      for (const auto i : c10::irange(args.size())) {
        Input(i + 1) = args[i];
      }
    } else {
      DCHECK_EQ(inputs_.size(), args.size());
      for (const auto i : c10::irange(args.size())) {
        Input(i) = args[i];
      }
    }
  }
}

c10::IValue StaticRuntime::moveOutputsToTuple(size_t num_outputs) {
  switch (num_outputs) {
    case 1:
      return c10::ivalue::Tuple::create(std::move(*outputs_[0]));
    case 2:
      return c10::ivalue::Tuple::create(
          std::move(*outputs_[0]), std::move(*outputs_[1]));
    case 3:
      return c10::ivalue::Tuple::create(
          std::move(*outputs_[0]),
          std::move(*outputs_[1]),
          std::move(*outputs_[2]));
    default: {
      std::vector<c10::IValue> outputs;
      outputs.reserve(static_module_.num_outputs());
      for (const auto i : c10::irange(static_module_.num_outputs())) {
        // use move here. Otherwise, clean up outputs_[i] explicitly
        outputs.emplace_back(std::move(*outputs_[i]));
      }
      return c10::ivalue::Tuple::create(std::move(outputs));
    }
  }
}

c10::IValue StaticRuntime::operator()(
    c10::ArrayRef<c10::IValue> args,
    const std::unordered_map<std::string, c10::IValue>& kwargs) {
  // We assume inference workloads, so we do not need
  // autograd. Enabling this is a significant win on dispatcher
  // overhead because it saves a round of dispatch for at least some
  // functions, such as resize_ and resize_as_.
  c10::InferenceMode mode;

  if (planner_) {
    planner_->allocate();
  }

  set_inputs(args, kwargs);

  // NB: before optimizing the order of execution, ensure that the
  // memory optimization pass (LivenessMap) is
  // aware of the new order!
  for (auto& n : nodes_) {
    // LOG(INFO) << "Running node: " << PrintNode(n.node());
    n.run();
  }

  if (static_module_.opts().cleanup_activations) {
    // MemoryPlanner is created after the first invocation of `run()`. This is
    // done intentionally because MemoryPlanner uses `Tensor` sizes of the
    // previous `run()` for memory planning of subsequent runs
    if (!planner_) {
      planner_ = std::make_unique<MemoryPlanner>(
          this,
          static_module_.values_share_same_storage(),
          static_module_.external_values(),
          static_module_.opts().enable_out_variant,
          static_module_.opts().optimize_graph_output_memory);
    }
    planner_->deallocate();
    // clean up owning refs of input tensors
    clean_up_input_ivalues();
  }

  // no need to keep references of outputs in static runtime anymore
  if (static_module_.num_outputs() > 1) {
    return moveOutputsToTuple(static_module_.num_outputs());
  }

#ifndef NDEBUG
  check_for_memory_leak(false);
#endif

  // use move here. Otherwise, clean up outputs_[0] explicitly
  return std::move(*outputs_[0]);
}

namespace {

std::string generate_latency_json(const std::string& label, double millis) {
#ifdef FBCODE_CAFFE2
  folly::dynamic json = folly::dynamic::object();
  json["type"] = label;
  json["metric"] = "latency";
  json["unit"] = "ms";
  json["value"] = millis;
  return "PyTorchObserver " + folly::toJson(json);
#else
  return "";
#endif
}

} // namespace

void StaticRuntime::benchmark(
    c10::ArrayRef<c10::IValue> args,
    const std::unordered_map<std::string, c10::IValue>& kwargs,
    const int warmup_runs,
    const int main_runs,
    bool print_per_node_time,
    bool generate_ai_pep_output) {
  float time_per_iter = benchmark_model(args, kwargs, warmup_runs, main_runs);
  std::cout << "Static runtime ms per iter: " << time_per_iter
            << ". Iters per second: " << 1000.0 / time_per_iter << std::endl;

  IndividualMetrics results =
      benchmark_individual_ops(args, kwargs, warmup_runs, main_runs);

  if (print_per_node_time) {
    for (const auto i : c10::irange(nodes_.size())) {
      const Node* node = nodes_[i].node();
      std::cout << "Node #" << i << ": " << results.time_per_node[i]
                << " ms/iter, ";
      node->print(std::cout, 0, nullptr, false);
    }
  }

  std::vector<std::pair<std::string, double>> time_per_node_type_vec{
      results.time_per_node_type.begin(), results.time_per_node_type.end()};
  std::sort(
      time_per_node_type_vec.begin(),
      time_per_node_type_vec.end(),
      [](auto& left, auto& right) { return left.second > right.second; });

  std::cout << "Time per node type:" << std::endl;
  for (const auto& p : time_per_node_type_vec) {
    const std::string& kind = p.first;
    const double ms = p.second;
    std::cout << std::setw(15) << ms << " ms. " << std::setw(10)
              << results.percent_per_node_type[kind] << "%. " << kind << " ("
              << results.instances_per_node_type[kind] << " nodes";
    if (results.out_nodes.count(kind)) {
      std::cout << ", out variant)" << std::endl;
    } else if (results.native_nodes.count(kind)) {
      std::cout << ", native)" << std::endl;
    } else {
      std::cout << ")" << std::endl;
    }

    if (generate_ai_pep_output) {
      LOG(INFO) << generate_latency_json(kind, ms);
    }
  }
  if (generate_ai_pep_output) {
    LOG(INFO) << generate_latency_json(
        "static_runtime_first_iter", results.first_iter_time);
  }
  std::cout << std::setw(15) << results.total_time << " ms. in Total"
            << std::endl;
  std::cout << "StaticRuntime setup time: " << results.setup_time << " ms"
            << std::endl;
  std::cout << "Memory allocation time: " << results.memory_alloc_time
            << " ms\n";
  std::cout << "Memory deallocation time: " << results.memory_dealloc_time
            << " ms" << std::endl;
  std::cout << "Outputs deallocation time: " << results.output_dealloc_time
            << " ms" << std::endl;
  std::cout << "First iter time: " << results.first_iter_time << " ms"
            << std::endl;

  if (planner_) {
    std::cout << "Total number of managed tensors: "
              << planner_->total_num_managed_tensors() << std::endl;
    std::cout << "Total number of unmanaged values: "
              << planner_->total_num_unmanaged() << std::endl;
    std::cout << "Total memory managed: " << planner_->total_managed()
              << " bytes" << std::endl;
    if (static_module_.opts().optimize_memory) {
      std::cout << "Total number of reused tensors: "
                << planner_->total_reused_tensors() << std::endl;
    }
    std::cout << "Total number of 'out' variant nodes/total number of nodes: "
              << results.out_nodes_count << "/" << results.total_nodes_count
              << " ("
              << 100.0 * (results.out_nodes_count) /
            static_cast<float>(results.total_nodes_count)
              << "%)" << std::endl;
  }
  check_for_memory_leak();

#ifndef NDEBUG
  display_nodes(args, kwargs);
#endif
}

float StaticRuntime::benchmark_model(
    c10::ArrayRef<c10::IValue> args,
    const std::unordered_map<std::string, c10::IValue>& kwargs,
    const int warmup_runs,
    const int main_runs) {
  TORCH_CHECK(warmup_runs >= 0 && main_runs >= 1);

  for (const auto i : c10::irange(warmup_runs)) {
    (void)i; // Suppress unused variable warning
    operator()(args, kwargs);
  }
  caffe2::Timer timer;
  for (const auto i : c10::irange(main_runs)) {
    (void)i; // Suppress unused variable warning
    operator()(args, kwargs);
  }
  float millis = timer.MilliSeconds();
  return millis / static_cast<float>(main_runs);
}

bool display_ivalue(const IValue& iv) {
  if (iv.isTensor()) {
    std::cout << "Tensor " << iv.toTensor().toString() << " {";
    for (const auto i : c10::irange(iv.toTensor().sizes().size())) {
      std::cout << iv.toTensor().sizes()[i];
      if (iv.toTensor().sizes().size() > i + 1) {
        std::cout << ", ";
      }
    }
    std::cout << "}\n";
    return true;
  } else if (iv.isTensorList()) {
    std::cout << "TensorList {" << iv.toTensorList().size() << "}\n";
    return true;
  } else if (iv.isGenericDict()) {
    std::cout << "Dict {" << iv.toGenericDict().size() << "}\n";
    return true;
  } else if (iv.isTuple()) {
    std::cout << "Tuple {" << iv.toTupleRef().elements().size() << "}\n";
    return true;
  } else if (iv.isInt()) {
    std::cout << "int {" << iv.toInt() << "}\n";
    return true;
  } else if (iv.isBool()) {
    std::cout << "bool {" << iv.toBool() << "}\n";
    return true;
  } else if (iv.isDouble()) {
    std::cout << "double {" << iv.toDouble() << "}\n";
    return true;
  }
  return false;
}

void display_pnode_info(const ProcessedNode& pnode) {
  pnode.node()->print(std::cout, 0, nullptr, false);
  const std::vector<const IValue*>& inputs = pnode.inputs();
  for (const auto i : c10::irange(inputs.size())) {
    std::cout << "\ti" << i << ": ";
    if (!display_ivalue(*inputs[i])) {
      std::cout << *(pnode.node()->inputs()[i]->type()) << '\n';
    }
  }
  const std::vector<IValue>& outputs = pnode.outputs();
  for (const auto i : c10::irange(outputs.size())) {
    std::cout << "\to" << i << ": ";
    if (!display_ivalue(outputs[i])) {
      std::cout << *(pnode.node()->outputs()[i]->type()) << '\n';
    }
  }
}

void StaticRuntime::display_nodes(
    c10::ArrayRef<c10::IValue> args,
    const std::unordered_map<std::string, c10::IValue>& kwargs) {
  c10::InferenceMode mode;
  if (planner_) {
    planner_->allocate();
  }
  set_inputs(args, kwargs);

  for (auto& node : nodes_) {
    node.run();
    display_pnode_info(node);
  }

  if (static_module_.opts().cleanup_activations) {
    // MemoryPlanner is created after the first invocation of `run()`. This is
    // done intentionally because MemoryPlanner uses `Tensor` sizes of the
    // previous `run()` for memory planning of subsequent runs
    if (!planner_) {
      planner_ = std::make_unique<MemoryPlanner>(
          this,
          static_module_.values_share_same_storage(),
          static_module_.external_values(),
          static_module_.opts().enable_out_variant,
          static_module_.opts().optimize_graph_output_memory);
    }
    planner_->deallocate();
    // clean up owning refs of input tensors
    clean_up_input_ivalues();
  }
}

StaticRuntime::IndividualMetrics StaticRuntime::benchmark_individual_ops(
    c10::ArrayRef<c10::IValue> args,
    const std::unordered_map<std::string, c10::IValue>& kwargs,
    const int warmup_runs,
    const int main_runs) {
  TORCH_CHECK(warmup_runs >= 1 && main_runs >= 1);

  // See comment on above use of InferenceMode for
  // explanation.
  c10::InferenceMode mode;

  IndividualMetrics results;
  results.time_per_node.resize(nodes_.size(), 0);

  // setup time
  caffe2::Timer timer;

  set_inputs(args, kwargs);

  results.setup_time = timer.MilliSeconds();

  // The first iteration profiles each node's output Tensors' sizes and
  // initializes the memory planner with the profile information. Folllowing
  // iterations just use the already established memory planning.
  timer.Start();
  operator()(args, kwargs);
  results.first_iter_time = timer.MilliSeconds();

  // warmup runs
  for (const auto i : c10::irange(warmup_runs - 1)) {
    (void)i; // Suppress unused variable warning
    operator()(args, kwargs);
  }

  // main runs
  for (const auto k : c10::irange(main_runs)) {
    (void)k; // Suppress unused variable warning

    set_inputs(args, kwargs);

    timer.Start();
    if (planner_) {
      planner_->allocate();
    }
    float millis = timer.MilliSeconds();
    results.memory_alloc_time += millis;

    for (const auto i : c10::irange(nodes_.size())) {
      timer.Start();
      nodes_[i].run();
      millis = timer.MilliSeconds();
      results.time_per_node[i] += millis;
    }
    timer.Start();
    if (static_module_.opts().cleanup_activations) {
      if (!planner_) {
        planner_ = std::make_unique<MemoryPlanner>(
            this,
            static_module_.values_share_same_storage(),
            static_module_.external_values(),
            static_module_.opts().enable_out_variant,
            static_module_.opts().optimize_graph_output_memory);
      }
      planner_->deallocate();
      // clean up owning refs of input tensors
      clean_up_input_ivalues();
    }
    millis = timer.MilliSeconds();
    results.memory_dealloc_time += millis;

    timer.Start();
    // no need to keep references of outputs in static runtime anymore
    c10::IValue output;
    if (static_module_.num_outputs() > 1) {
      output = moveOutputsToTuple(static_module_.num_outputs());
    }

#ifndef NDEBUG
    check_for_memory_leak(false);
#endif

    // use move here. Otherwise, clean up outputs_[0] explicitly
    output = std::move(*outputs_[0]);
    // release outputs explicitly to measure the time it takes
    output = IValue();
    millis = timer.MilliSeconds();
    results.output_dealloc_time += millis;
  }

  // post processing
  for (const auto i : c10::irange(nodes_.size())) {
    const Node* node = nodes_[i].node();
    std::string kind = std::string(node->kind().toQualString());
    results.time_per_node[i] /= static_cast<float>(main_runs);
    results.time_per_node_type[kind] += results.time_per_node[i];
    results.instances_per_node_type[kind]++;
    if (nodes_[i].has_out_variant()) {
      results.out_nodes.insert(kind);
      results.out_nodes_count++;
    } else if (nodes_[i].has_native()) {
      results.native_nodes.insert(kind);
    }
    results.total_time += results.time_per_node[i];
  }
  results.total_nodes_count = nodes_.size();
  results.memory_alloc_time /= static_cast<float>(main_runs);
  results.memory_dealloc_time /= static_cast<float>(main_runs);
  results.output_dealloc_time /= static_cast<float>(main_runs);
  for (const auto& p : results.time_per_node_type) {
    const std::string& kind = p.first;
    results.percent_per_node_type[kind] = p.second / results.total_time * 100;
  }
  return results;
}

void StaticRuntime::check_for_memory_leak(bool output_returned) {
  if (!static_module_.opts().cleanup_activations) {
    return;
  }

  // check for inputs
  for (const auto i : c10::irange(inputs_.size())) {
    TORCH_CHECK(inputs_[i].isNone(), "Input ", i, " was not cleaned up");
  }

  FastSet<const IValue*> output_ivalues(outputs_.begin(), outputs_.end());
  for (const auto n : c10::irange(nodes_.size())) {
    auto& pnode = nodes_[n];
    for (const auto i : c10::irange(pnode.outputs().size())) {
      const IValue* ival = &pnode.Output(i);
      const Value* val = pnode.node()->output(i);
      const std::string error_msg = "Output " + c10::to_string(i) + ", %" +
          val->debugName() + " of node " + c10::to_string(n) +
          " was not cleaned up";
      if (output_ivalues.count(ival) == 0) {
        // check for intermediates
        if (!ival->isNone()) {
          TORCH_CHECK(
              ival->isTensor() ||
                  static_module_.is_optimizable_container_type(pnode.node()),
              error_msg);
          if (ival->isTensor()) {
            const auto& t = ival->toTensor();
            if (t.defined()) {
              auto* storage_impl = t.storage().unsafeGetStorageImpl();
              TORCH_CHECK(storage_impl->data() == nullptr, error_msg);
            }
          }
        }
      } else {
        // check for outputs
        if (output_returned) {
          TORCH_CHECK(ival->isNone(), error_msg);
        }
      }
    }
  }
  VLOG(1) << "Finished checking for memory leak";
}

ProcessedNode::ProcessedNode(
    Node* node,
    std::vector<const IValue*>&& inputs,
    bool enable_out_variant)
    : node_(node), inputs_(std::move(inputs)) {
  // TODO leverage type information
  outputs_.resize(node->outputs().size());

  if (enable_out_variant && (fn_ = getOutOfPlaceOperation(node))) {
    VLOG(1) << "Switch to out variant for node: " << PrintNode(node);
    return;
  }
  if (!fn_ && (native_fn_ = getNativeOperation(node))) {
    VLOG(1) << "Switch to native impl for node: " << PrintNode(node);
    return;
  }
  {
    const Operator& op = node->getOperator();
    op_ = op.getOperation(node);
    VLOG(1) << "Fallback interpreter for node: " << PrintNode(node);
  }
}

void ProcessedNode::run() {
  DCHECK(verify_no_memory_overlap());
  if (fn_) {
    fn_(this);
  } else if (native_fn_) {
    native_fn_(this);
  } else {
    std::vector<IValue> stack;
    const size_t size = node_->inputs().size();
    stack.reserve(size + 1);
    for (const auto i : c10::irange(size)) {
      stack.emplace_back(Input(i));
    }
    // Need to store the number of inputs in stack for variadic ops.
    if (hasVarArgs(node_)) {
      stack.emplace_back(static_cast<int>(size));
    }

    DCHECK(op_);
    op_->operator()(stack);

    DCHECK_EQ(stack.size(), node_->outputs().size());
    for (const auto i : c10::irange(node_->outputs().size())) {
      Output(i) = std::move(stack[i]);
    }
  }
}

static bool checkNoMemoryOverlap(const at::Tensor& a, const at::Tensor& b) {
  at::MemOverlapStatus status = at::get_overlap_status(a, b);
  if (status == at::MemOverlapStatus::FULL ||
      status == at::MemOverlapStatus::PARTIAL) {
    return false;
  }
  if (status == at::MemOverlapStatus::TOO_HARD) {
    LOG(WARNING) << "Detected TOO_HARD memory overlap status";
  }
  return true;
}

bool ProcessedNode::verify_no_memory_overlap() const {
  for (size_t i = 0; i < outputs_.size(); ++i) {
    if (!outputs_[i].isTensor()) {
      continue;
    }
    const auto& out0_t = outputs_[i].toTensor();
    for (size_t j = i + 1; j < outputs_.size(); ++j) {
      if (!outputs_[j].isTensor()) {
        continue;
      }
      const auto& out1_t = outputs_[j].toTensor();
      if (!checkNoMemoryOverlap(out0_t, out1_t)) {
        return false;
      }
    }
  }

  auto schema = node()->maybeSchema();
  if (!schema || schema->is_mutable()) {
    return true;
  }
  for (const IValue* in : inputs_) {
    if (!in->isTensor()) {
      continue;
    }
    const auto& in_t = in->toTensor();
    for (const IValue& out : outputs_) {
      if (!out.isTensor()) {
        continue;
      }
      const auto& out_t = out.toTensor();
      if (!checkNoMemoryOverlap(in_t, out_t)) {
        return false;
      }
    }
  }
  return true;
}

} // namespace jit
} // namespace torch
