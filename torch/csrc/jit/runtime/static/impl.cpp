#include <torch/csrc/jit/runtime/static/impl.h>

#include <ATen/core/LegacyTypeDispatch.h>
#include <ATen/core/interned_strings.h>
#include <c10/core/CPUAllocator.h>
#include <caffe2/core/scope_guard.h>
#include <caffe2/core/timer.h>
#include <torch/csrc/jit/ir/alias_analysis.h>
#include <torch/csrc/jit/passes/canonicalize.h>
#include <torch/csrc/jit/passes/dead_code_elimination.h>
#include <torch/csrc/jit/passes/freeze_module.h>
#include <torch/csrc/jit/passes/remove_mutation.h>
#include <torch/csrc/jit/passes/subgraph_rewrite.h>
#include <torch/csrc/jit/runtime/static/ops.h>
#include <torch/csrc/jit/runtime/static/passes.h>
#include <torch/csrc/jit/runtime/vararg_functions.h>

namespace torch {
namespace jit {

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

  // TODO: we can avoid this guard by moving operations
  // to exposed folders.
#ifdef FBCODE_CAFFE2
  if (opts.enable_out_variant) {
    ReplaceWithCopy(graph);
    FuseSigridTransformsListUnpack(graph);
  }
#endif
  ConstantPropagation(graph);
}

void CheckGraphEligibility(const std::shared_ptr<torch::jit::Graph>& graph) {
  for (auto n : graph->nodes()) {
    if (n->kind() == c10::Symbol::fromQualString("prim::GetAttr")) {
      throw std::runtime_error("Cannot accelerate unfrozen graphs");
    }
  }
  // check output types
  // Static Runtime supports output types include None, Tensor and List/Tuple
  // of Tensor
  for (Value* output : graph->outputs()) {
    VLOG(1) << "output: %" << output->debugName()
            << " has type: " << output->type()->repr_str();
    auto kind = output->node()->kind();
    if (kind == prim::TupleConstruct || kind == prim::ListConstruct) {
      for (Value* input : output->node()->inputs()) {
        const auto& type = input->type();
        TORCH_CHECK(
            type->cast<TensorType>() != nullptr,
            "Static Runtime expects output type as List or Tuple of Tensor, but got List or Tuple of ",
            type->repr_str());
      }
    } else {
      const auto& type = output->type();
      TORCH_CHECK(
          type->cast<TensorType>() != nullptr ||
              type->cast<NoneType>() != nullptr,
          "Static Runtime expects output type as None or Tensor, but got ",
          type->repr_str());
    }
  }
}

// remove unused input 0 from graph
void RemoveSelfFromGraphInput(std::shared_ptr<torch::jit::Graph>& graph) {
  if (graph->inputs().at(0)->type()->is_module()) {
    TORCH_CHECK(!graph->inputs().at(0)->hasUses());
    graph->eraseInput(0);
  }
}

// remove "self" from function schema
c10::FunctionSchema RemoveSelfFromSchema(const c10::FunctionSchema& s) {
  TORCH_CHECK(s.arguments().size() >= 1 && s.arguments()[0].name() == "self");
  std::vector<Argument> args({s.arguments().begin() + 1, s.arguments().end()});
  return s.cloneWithArguments(args);
}

bool mayContainAlias(AliasDb& db, const Value* a, const Value* b) {
  return db.mayContainAlias(const_cast<Value*>(a), const_cast<Value*>(b));
}

bool mayContainAlias(
    AliasDb& db,
    const std::unordered_set<const Value*>& a,
    const std::unordered_set<const Value*>& b) {
  std::vector<Value*> as;
  std::vector<Value*> bs;
  as.reserve(a.size());
  for (auto* v : a) {
    as.emplace_back(const_cast<Value*>(v));
  }
  bs.reserve(b.size());
  for (auto* v : b) {
    bs.emplace_back(const_cast<Value*>(v));
  }
  return db.mayContainAlias(as, bs);
}

// Returns two useful constructs:
//  first: map each value to all values that are alive
//    at the same time.
//  second: set of all inputs/outputs/constants (always alive)
//    and their aliases
//  The algorithm does a traversal of the execution graph
//  while keeping track of the live values.
using LivenessInformation = std::pair<
    std::unordered_map<const Value*, std::set<const Value*>>,
    std::unordered_set<const Value*>>;

LivenessInformation GetLivenessInformation(
    const std::shared_ptr<torch::jit::Graph>& graph,
    AliasDb& db) {
  // map a Value to a set of Values that overlap live-ranges with the Value's
  std::unordered_map<const Value*, std::set<const Value*>> liveness_map;
  // a set of Values whose live-range exceed current inference
  std::unordered_set<const Value*> always_alive;

  // map Values to its creation order in graph (Note: only traverse top-level
  // nodes such that nodes under control-flows are represented by top-level
  // block nodes)
  std::vector<const Value*> values_in_creation_order;
  std::unordered_map<const Value*, size_t> values_to_idx_in_creation_order;
  for (const auto* node : graph->nodes()) {
    for (const auto* v : node->outputs()) {
      values_to_idx_in_creation_order[v] = values_in_creation_order.size();
      values_in_creation_order.emplace_back(v);
    }
  }

  // presence of a Value in live_values_use_chain means the Value alive
  // Value mapped to set of Nodes that may use the Value (i.e., use-chain of
  // Value)
  std::unordered_map<const Value*, std::set<const Node*>> live_values_use_chain;
  // Node mapped to set of Values that the Node may use (i.e., def-chain of node
  // inputs)
  std::unordered_map<const Node*, std::set<const Value*>> live_nodes_def_chain;

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

  // add v to the current liveness_map
  std::function<void(const Value* v)> add_live_value_fn = [&](const Value* v) {
    if (liveness_map.count(v)) {
      return;
    }
    liveness_map[v] = {};

    for (const auto& live_v : live_values_use_chain) {
      liveness_map.at(v).insert(live_v.first);
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
      if (mayContainAlias(db, ValueSet{v}, always_alive)) {
        always_alive.insert(v);
      } else {
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
    for (const auto* input : node->inputs()) {
      for (const auto* output : node->outputs()) {
        if (liveness_map.count(input) && liveness_map.count(output)) {
          liveness_map.at(input).insert(output);
          liveness_map.at(output).insert(input);
        }
      }
    }
  }

  return std::make_pair(liveness_map, always_alive);
}

// Collect the set of Values that are candidates for memory planning:
//   - Values that are used in in-place operators (i.e., _out variants), and
//   - excluding those that are either inputs or outputs of
//     non in-place operators
//
// Returns
//   first: Values that are candidates for memory planning
//   second: A deterministc order of all values
std::pair<std::vector<const Value*>, std::vector<const Value*>>
GetMemoryPlanningCandidates(const std::shared_ptr<torch::jit::Graph>& graph) {
  // for determinism
  std::unordered_set<const Value*> seen_values;
  std::vector<const Value*> all_values;
  std::unordered_set<const Value*> can_reuse;
  // values used by unsupported ops (as either inputs or outputs)
  // these need to be removed from "can_reuse" after analyzing all nodes
  std::unordered_set<const Value*> cannot_reuse;
  for (auto* n : graph->nodes()) {
    for (const auto* v : n->inputs()) {
      if (!seen_values.count(v)) {
        all_values.emplace_back(v);
        seen_values.insert(v);
      }
      if (canReuseInputsOutputs(n)) {
        can_reuse.insert(v);
      } else {
        cannot_reuse.insert(v);
      }
    }
    for (const auto* v : n->outputs()) {
      all_values.emplace_back(v);
      seen_values.insert(v);
      if (canReuseInputsOutputs(n)) {
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
std::unordered_map<const Value*, std::vector<const Value*>>
GenerateSameStorageValues(
    const LivenessInformation& lm,
    const std::pair<std::vector<const Value*>, std::vector<const Value*>>&
        optimizable,
    AliasDb& db) {
  const auto& alive_during = lm.first;
  const auto& always_alive = lm.second;
  const auto& optimizable_values = optimizable.first;
  const auto& all_values = optimizable.second;

  // map Value* to a set Value* that can share the same storage with it
  std::unordered_map<const Value*, std::vector<const Value*>>
      same_storage_values;

  // make new_v and old_v map to the same storage (i.e., add to each other's
  // same_storage_values set)
  auto share_storage_fn = [&](const Value* new_v, const Value* old_v) {
    if (new_v == old_v) {
      return;
    }
    DCHECK(same_storage_values.count(old_v));
    std::set<const Value*> seen;
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
          std::set<const Value*>& live, const Value* v) {
        for (const auto* sv : same_storage_values.at(v)) {
          const auto& l = alive_during.count(sv) ? alive_during.at(sv)
                                                 : std::set<const Value*>{};
          live.insert(l.begin(), l.end());
        }
        live.insert(always_alive.begin(), always_alive.end());
      };

  // check if same_storage_values[s] intersects with live
  auto intersect_fn = [&same_storage_values](
                          std::set<const Value*>& live, const Value* s) {
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
    std::set<const Value*> live;
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

} // namespace

void PrepareGraphForStaticModule(
    std::shared_ptr<torch::jit::Graph> graph,
    const StaticModuleOptions& opts) {
  OptimizeGraph(graph, opts);
  CheckGraphEligibility(graph);
  RemoveSelfFromGraphInput(graph);
}

std::pair<std::shared_ptr<Graph>, c10::optional<c10::FunctionSchema>>
PrepareForStaticModule(
    const torch::jit::Module& m,
    const StaticModuleOptions& opts) {
  auto module = m.copy();
  module.eval();

  module = freeze_module(module);

  Method method = module.get_method("forward");
  auto graph = module.get_method("forward").graph();
  PrepareGraphForStaticModule(graph, opts);

  c10::FunctionSchema s = RemoveSelfFromSchema(method.function().getSchema());
  return std::make_pair(graph, s);
}

std::pair<std::shared_ptr<Graph>, c10::optional<c10::FunctionSchema>>
PrepareForStaticModule(
    std::shared_ptr<torch::jit::Graph> graph,
    const StaticModuleOptions& opts) {
  PrepareGraphForStaticModule(graph, opts);
  return std::make_pair(graph, c10::nullopt);
}

StaticModule::StaticModule(
    std::shared_ptr<torch::jit::Graph> g,
    const StaticModuleOptions& opts)
    : StaticModule(PrepareForStaticModule(g, opts), opts) {}

StaticModule::StaticModule(
    const torch::jit::Module& m,
    const StaticModuleOptions& opts)
    : StaticModule(PrepareForStaticModule(m, opts), opts) {}

StaticModule::StaticModule(
    std::pair<
        std::shared_ptr<torch::jit::Graph>,
        c10::optional<c10::FunctionSchema>> graph_and_schema,
    const StaticModuleOptions& opts)
    : opts_(opts),
      graph_(std::move(graph_and_schema.first)),
      schema_(std::move(graph_and_schema.second)) {
  // map Value* to IValue (from inputs or prim::Constant) or null
  std::unordered_map<Value*, IValue*> value_to_ivalue;
  // map Value* to its SSA definition IR
  std::unordered_map<Value*, DefInfo> value_to_ssa_def;

  // N inputs map to the first N entries in storage
  for (auto i = 0; i < graph_->inputs().size(); ++i) {
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
    nodes_.emplace_back(
        ProcessedNode(node, std::move(ivalue_inputs), opts.enable_out_variant));
    for (size_t i = 0; i < node->outputs().size(); ++i) {
      value_to_ivalue[node->outputs()[i]] = nullptr;
      value_to_ssa_def[node->outputs()[i]] = std::make_pair(node_idx, i);
    }
    node_idx++;
  }
  for (auto output : graph_->outputs()) {
    output_ssa_defs_.emplace_back(value_to_ssa_def[output]);
  }

  AliasDb alias_db(graph_);
  auto lm = GetLivenessInformation(graph_, alias_db);
  external_values_ = lm.second;
  if (opts_.optimize_memory) {
    auto values = GetMemoryPlanningCandidates(graph_);
    if (!opts_.enable_out_variant) {
      values.first = {};
    }
    value_to_same_storage_values_ =
        GenerateSameStorageValues(lm, values, alias_db);
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
    const std::vector<c10::IValue>& args,
    const std::unordered_map<std::string, c10::IValue>& kwargs) {
  return runtime()(args, kwargs);
}

StaticRuntime::StaticRuntime(const StaticModule& sm) : static_module_(sm) {
  // NB: create unchanging std::vector<IValue>s we can reference
  inputs_.resize(sm.num_inputs());
  nodes_.resize(sm.nodes().size());
  for (auto idx = 0; idx < sm.nodes().size(); ++idx) {
    const auto& n_ref = sm.nodes()[idx];
    nodes_[idx] = n_ref; // copy the node
    auto& n = nodes_[idx];
    // hook up the inputs
    for (auto i = 0; i < n.inputs().size(); ++i) {
      if (n.inputs()[i] == nullptr) {
        int node_idx;
        int out_idx;
        std::tie(node_idx, out_idx) = sm.index_map().at(idx)[i];
        DCHECK(out_idx >= 0);
        // input
        if (node_idx == StaticModule::INPUT_VALUE) {
          n.set_input(i, &inputs_[out_idx]);
        } else if (node_idx == StaticModule::CONSTANT_VALUE) {
          n.set_input(i, &sm.constants()[out_idx]);
        } else {
          n.set_input(i, &(nodes_[node_idx].Output(out_idx)));
        }
      }
    }
  }

  for (const auto& index_pair : sm.output_indices()) {
    int node_idx;
    int out_idx;
    std::tie(node_idx, out_idx) = index_pair;
    if (node_idx == StaticModule::INPUT_VALUE) {
      outputs_.emplace_back(&inputs_[out_idx]);
    } else if (node_idx == StaticModule::CONSTANT_VALUE) {
      // This is a very rare case where const correctness
      // breaks -- the user is returning a constant from
      // the graph.
      outputs_.emplace_back(const_cast<IValue*>(&sm.constants()[out_idx]));
    } else {
      auto& n = nodes_.at(node_idx);
      auto* out = &n.Output(out_idx);
      outputs_.emplace_back(out);
    }
  }
}

std::vector<at::Tensor> StaticRuntime::operator()(
    const std::vector<at::Tensor>& inps) {
  std::vector<c10::IValue> stack;
  stack.resize(inps.size());
  for (size_t i = 0; i < inps.size(); i++) {
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

c10::IValue StaticRuntime::operator()(
    const std::vector<c10::IValue>& args,
    const std::unordered_map<std::string, c10::IValue>& kwargs) {
  // We assume inference workloads, so we do not need
  // autograd. Enabling this is a significant win on dispatcher
  // overhead because it saves a round of dispatch for at least some
  // functions, such as resize_ and resize_as_.
  at::AutoNonVariableTypeMode non_var_type_mode(true);

  if (planner_) {
    planner_->allocate();
  }

  if (!kwargs.empty()) {
    // This is not ideal
    TORCH_CHECK(
        static_module_.schema(),
        "Schema is not available. Consider creating the Static Runtime "
        "with StaticModule(const torch::jit::Module& m) instead.");
    std::vector<c10::IValue> s = args;
    static_module_.schema()->checkAndNormalizeInputs(s, kwargs);
    for (size_t i = 0; i < s.size(); i++) {
      Input(i) = std::move(s[i]);
    }
  } else {
    for (size_t i = 0; i < args.size(); i++) {
      Input(i) = args[i];
    }
  }

  // NB: before optimizing the order of execution, ensure that the
  // memory optimization pass (LivenessMap) is
  // aware of the new order!
  for (auto& n : nodes_) {
    n.run();
  }

  if (static_module_.opts().cleanup_activations) {
    if (!planner_) {
      planner_ = std::make_unique<MemoryPlanner>(
          this,
          static_module_.values_share_same_storage(),
          static_module_.external_values(),
          static_module_.opts().enable_out_variant);
    }
    planner_->deallocate();
    // clean up owning refs of input tensors
    for (IValue& ival : inputs_) {
      ival = IValue();
    }
  }

  // no need to keep references of outputs in static runtime anymore
  if (static_module_.num_outputs() > 1) {
    std::vector<c10::IValue> outputs;
    outputs.reserve(static_module_.num_outputs());
    for (auto i = 0; i < static_module_.num_outputs(); ++i) {
      // use move here. Otherwise, clean up outputs_[i] explicitly
      outputs.emplace_back(std::move(*outputs_[i]));
    }
    return c10::ivalue::Tuple::create(std::move(outputs));
  }

#ifndef NDEBUG
  check_for_memory_leak(false);
#endif

  // use move here. Otherwise, clean up outputs_[0] explicitly
  return std::move(*outputs_[0]);
}

void StaticRuntime::benchmark(
    const std::vector<c10::IValue>& args,
    const std::unordered_map<std::string, c10::IValue>& kwargs,
    const int warmup_runs,
    const int main_runs) {
  float time_per_iter = benchmark_model(args, kwargs, warmup_runs, main_runs);
  std::cout << "Static runtime ms per iter: " << time_per_iter
            << ". Iters per second: " << 1000.0 / time_per_iter << std::endl;

  IndividualMetrics results =
      benchmark_individual_ops(args, kwargs, warmup_runs, main_runs);

  for (size_t i = 0; i < nodes_.size(); i++) {
    const Node* node = nodes_[i].node();
    std::cout << "Node #" << i << ": " << results.time_per_node[i]
              << " ms/iter, ";
    node->print(std::cout, 0, nullptr, false);
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
              << results.instances_per_node_type[kind] << " nodes)"
              << std::endl;
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

  if (planner_) {
    std::cout << "Total memory managed: " << planner_->total_managed()
              << " bytes" << std::endl;
    if (static_module_.opts().optimize_memory) {
      std::cout << "Total number of reused tensors: "
                << planner_->total_reused_tensors() << std::endl;
    }
  }
}

float StaticRuntime::benchmark_model(
    const std::vector<c10::IValue>& args,
    const std::unordered_map<std::string, c10::IValue>& kwargs,
    const int warmup_runs,
    const int main_runs) {
  TORCH_CHECK(warmup_runs >= 0 && main_runs >= 1);

  for (int i = 0; i < warmup_runs; i++) {
    operator()(args, kwargs);
  }
  caffe2::Timer timer;
  for (int i = 0; i < main_runs; i++) {
    operator()(args, kwargs);
  }
  float millis = timer.MilliSeconds();
  return millis / static_cast<float>(main_runs);
}

StaticRuntime::IndividualMetrics StaticRuntime::benchmark_individual_ops(
    const std::vector<c10::IValue>& args,
    const std::unordered_map<std::string, c10::IValue>& kwargs,
    const int warmup_runs,
    const int main_runs) {
  TORCH_CHECK(warmup_runs >= 0 && main_runs >= 1);

  // See comment on above use of AutoNonVariableTypeMode for
  // explanation.
  at::AutoNonVariableTypeMode non_var_type_mode(true);

  IndividualMetrics results;
  results.time_per_node.resize(nodes_.size(), 0);

  // setup time
  caffe2::Timer timer;
  std::vector<IValue> stack(args);
  if (!kwargs.empty()) {
    // This is not ideal
    TORCH_CHECK(
        static_module_.schema(),
        "Schema is not available. Consider creating the Static Runtime "
        "with StaticModule(const torch::jit::Module& m) instead.");
    static_module_.schema()->checkAndNormalizeInputs(stack, kwargs);
  }
  for (size_t i = 0; i < stack.size(); i++) {
    Input(i) = stack[i];
  }
  results.setup_time = timer.MilliSeconds();

  // warmup runs
  for (int i = 0; i < warmup_runs; i++) {
    operator()(args, kwargs);
  }

  // main runs
  for (int k = 0; k < main_runs; k++) {
    for (size_t i = 0; i < stack.size(); i++) {
      Input(i) = stack[i];
    }
    timer.Start();
    if (planner_) {
      planner_->allocate();
    }
    float millis = timer.MilliSeconds();
    results.memory_alloc_time += millis;

    for (size_t i = 0; i < nodes_.size(); i++) {
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
            static_module_.opts().enable_out_variant);
      }
      planner_->deallocate();
      // clean up owning refs of input tensors
      for (IValue& ival : inputs_) {
        ival = IValue();
      }
    }
    millis = timer.MilliSeconds();
    results.memory_dealloc_time += millis;

    timer.Start();
    // no need to keep references of outputs in static runtime anymore
    c10::IValue output;
    if (static_module_.num_outputs() > 1) {
      std::vector<c10::IValue> outputs;
      outputs.reserve(static_module_.num_outputs());
      for (auto i = 0; i < static_module_.num_outputs(); ++i) {
        // use move here. Otherwise, clean up outputs_[i] explicitly
        outputs.emplace_back(std::move(*outputs_[i]));
      }
      output = c10::ivalue::Tuple::create(std::move(outputs));
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
  for (size_t i = 0; i < nodes_.size(); i++) {
    const Node* node = nodes_[i].node();
    std::string kind = std::string(node->kind().toQualString());
    results.time_per_node[i] /= static_cast<float>(main_runs);
    results.time_per_node_type[kind] += results.time_per_node[i];
    results.instances_per_node_type[kind]++;
    results.total_time += results.time_per_node[i];
  }
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
  for (size_t i = 0; i < inputs_.size(); i++) {
    TORCH_CHECK(inputs_[i].isNone(), "Input ", i, " was not cleaned up");
  }

  std::unordered_set<const IValue*> output_ivalues(
      outputs_.begin(), outputs_.end());
  for (size_t n = 0; n < nodes_.size(); n++) {
    auto& pnode = nodes_[n];
    for (size_t i = 0; i < pnode.outputs().size(); i++) {
      const IValue* ival = &pnode.Output(i);
      const std::string error_msg = "Output " + c10::to_string(i) +
          " of node " + c10::to_string(n) + " was not cleaned up";
      if (output_ivalues.count(ival) == 0) {
        // check for intermediates
        if (!ival->isNone()) {
          TORCH_CHECK(
              ival->isTensor() || canOptimizeConstruct(pnode.node()),
              error_msg);
          if (ival->isTensor()) {
            const auto& t = ival->toTensor();
            if (t.defined()) {
              const auto* storage_impl = t.storage().unsafeGetStorageImpl();
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
}

MemoryPlanner::MemoryPlanner(
    StaticRuntime* runtime,
    const std::unordered_map<const Value*, std::vector<const Value*>>&
        value_to_same_storage_values,
    const std::unordered_set<const Value*>& external_values,
    bool out_variants) {
  // collect register indices of outputs of ops with out variant
  std::unordered_set<const Value*> managed_values;
  std::unordered_set<IValue*> unmanaged_ivalues;
  for (ProcessedNode& pnode : runtime->nodes()) {
    if (canReuseInputsOutputs(pnode.node())) {
      for (auto i = 0; i < pnode.outputs().size(); ++i) {
        // Types are stored in the underlying TorchScript IR
        const Value* out_v = pnode.node()->outputs()[i];
        IValue& out = pnode.Output(i);
        const auto& type = out_v->type();
        if (out_variants && !external_values.count(out_v)) {
          if (type->cast<TensorType>()) {
            managed_values.insert(out_v);
          } else if (canOptimizeConstruct(pnode.node())) {
            // We "leak" containers of this type
          } else {
            unmanaged_ivalues.insert(&out);
          }
        } else {
          unmanaged_ivalues.insert(&out);
        }
      }
    } else {
      for (auto i = 0; i < pnode.outputs().size(); ++i) {
        unmanaged_ivalues.insert(&pnode.Output(i));
      }
    }
  }

  // remove model outputs from managed_values and unmanaged_ivalues
  for (const Value* output : runtime->graph().outputs()) {
    managed_values.erase(output);
  }
  for (IValue* output : runtime->outputs()) {
    unmanaged_ivalues.erase(output);
  }

  // unmanaged_ivalues => unmanaged_ivalues_
  for (IValue* out : unmanaged_ivalues) {
    unmanaged_ivalues_.emplace_back(out);
  }

  // map Value to index to managed_storage_, where multiple values can
  // map to the same index (i.e., sharing the same storage)
  std::unordered_map<const Value*, size_t> value_to_storage_idx;
  // the StorageImpls of Tensor views should not be managed
  std::unordered_set<c10::StorageImpl*> managed_storage_impls;

  // Snapshot of the current memory state
  for (const auto& pnode : runtime->nodes()) {
    for (auto i = 0; i < pnode.outputs().size(); ++i) {
      const auto& ival = pnode.outputs()[i];
      const auto* val = pnode.node()->outputs()[i];
      if (managed_values.count(val)) {
        TORCH_CHECK(ival.isTensor());
        auto* impl = ival.toTensor().storage().unsafeGetStorageImpl();

        auto didInsert = managed_storage_impls.insert(impl).second;
        if (!didInsert) {
          continue;
        }

        if (value_to_storage_idx.count(val)) {
          managed_storage_[value_to_storage_idx.at(val)].second.emplace_back(
              impl);
        } else {
          auto p =
              std::make_pair<size_t, std::vector<c10::StorageImpl*>>(0, {impl});
          managed_storage_.emplace_back(std::move(p));
          // first of a group, update the value_to_storage_idx map with the
          // index
          if (value_to_same_storage_values.count(val)) {
            for (const auto* v : value_to_same_storage_values.at(val)) {
              value_to_storage_idx[v] = managed_storage_.size() - 1;
            }
          }
        }
      }
    }
  }
}

// Don't change the size if it is already aligned, otherwise increase the size
// to make it aligned.
size_t MemoryPlanner::compute_aligned_tensor_size(size_t nbytes) {
  // Note: everything below is size_t
  return (nbytes + c10::gAlignment - 1) & (~(c10::gAlignment - 1));
}

at::DataPtr MemoryPlanner::allocate_buffer(size_t size) {
  at::Allocator* allocator = c10::GetCPUCachingAllocator();
  return allocator->allocate(size);
}

void MemoryPlanner::allocate() {
  if (managed_bytes_ == 0) {
    return;
  }
  buffer_ = allocate_buffer(managed_bytes_);

  size_t offset = 0;
  uint8_t* start = static_cast<uint8_t*>(buffer_.get());

  reused_tensors_ = 0;
  for (const auto& ms : managed_storage_) {
    auto tensor_size = ms.first;
    if (tensor_size == 0) {
      continue;
    }
    const auto& impls = ms.second;
    DCHECK_LE(offset + tensor_size, managed_bytes_);
    void* src = static_cast<void*>(start + offset);

    for (auto& impl : impls) {
      impl->set_data_ptr_noswap(at::DataPtr(src, src, nullptr, impl->device()));
      impl->set_nbytes(tensor_size);
      reused_tensors_++;
    }
    reused_tensors_--;

    offset += tensor_size;
  }
  DCHECK_EQ(offset, managed_bytes_);
}

void MemoryPlanner::deallocate() {
  managed_bytes_ = 0;

  // free memory used by outputs of ops in out variants
  // but keep the TensorImpl and StorageImpl around
  for (auto& ms : managed_storage_) {
    const auto& impls = ms.second;
    size_t max = 0;
    for (auto& impl : impls) {
      size_t current_size = compute_aligned_tensor_size(impl->nbytes());
      impl->reset();
      max = std::max(max, current_size);
    }
    ms.first = max;
    managed_bytes_ += max;
  }
  for (auto& iv : unmanaged_ivalues_) {
    *iv = IValue();
  }
  buffer_ = {};
}

ProcessedNode::ProcessedNode(
    Node* node,
    std::vector<const IValue*>&& inputs,
    bool enable_out_variants)
    : node_(node), inputs_(std::move(inputs)) {
  // TODO leverage type information
  outputs_.resize(node->outputs().size());

  if (enable_out_variants && canRunOutOfPlace(node)) {
    fn_ = getOutOfPlaceOperation(node);
    std::ostringstream ss;
    node->print(ss, 0, nullptr, false);
    VLOG(1) << "Switch to out variant for node: " << ss.str();
  } else if (canRunNatively(node)) {
    native_fn_ = getNativeOperation(node);
    std::ostringstream ss;
    node->print(ss, 0, nullptr, false);
    VLOG(1) << "Switch to native impl for node: " << ss.str();
  } else if (
      node->kind() != prim::ListConstruct &&
      node->kind() != prim::TupleConstruct &&
      node->kind() != prim::DictConstruct && node->kind() != prim::ListUnpack) {
    const Operator& op = node->getOperator();
    TORCH_CHECK(op.hasOperation());
    op_ = op.getOperation(node);

    std::ostringstream ss;
    node->print(ss, 0, nullptr, false);
    VLOG(1) << "Fallback interpreter for node: " << ss.str();
  }
}

void ProcessedNode::run() {
  if (fn_) {
    fn_(this);
  } else if (native_fn_) {
    native_fn_(this);
  } else {
    std::vector<IValue> stack;
    const size_t size = node_->inputs().size();
    stack.reserve(size);
    for (size_t i = 0; i < size; i++) {
      stack.emplace_back(Input(i));
    }

    DCHECK(op_);
    op_->operator()(&stack);

    DCHECK_EQ(stack.size(), node_->outputs().size());
    for (auto i = 0; i < node_->outputs().size(); i++) {
      Output(i) = std::move(stack[i]);
    }
  }
}

} // namespace jit
} // namespace torch
