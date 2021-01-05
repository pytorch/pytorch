#include <torch/csrc/jit/runtime/static/impl.h>
#include <ATen/core/LegacyTypeDispatch.h>
#include <ATen/core/interned_strings.h>
#include <c10/core/CPUAllocator.h>
#include <caffe2/core/scope_guard.h>
#include <caffe2/core/timer.h>
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

void RemoveListTupleConstruct(std::shared_ptr<torch::jit::Graph>& graph) {
  auto nodes = graph->nodes();
  for (auto it = nodes.begin(); it != nodes.end(); it++) {
    auto n = *it;

    auto kind = n->kind();
    if (kind == prim::TupleConstruct || kind == prim::ListConstruct) {
      TORCH_CHECK(n->outputs().size() == 1);
      auto out = n->outputs().at(0);
      if (out->uses().size() != 1) {
        continue;
      }
      auto u = out->uses().at(0).user;
      if (u->kind() != aten::cat && u->kind() != aten::stack) {
        continue;
      }
      auto idx = 0;
      // TODO audit
      for (auto i : u->inputs()) {
        if (i == out) {
          break;
        }
        idx++;
      }
      auto is = n->inputs();
      u->replaceInput(idx, is.front());
      for (auto iter = is.rbegin(); iter + 1 != is.rend(); ++iter) {
        u->insertInput(idx + 1, *iter);
      }
      it.destroyCurrent();
    }
  }
}

// remove "self" from function schema
c10::FunctionSchema RemoveSelfFromSchema(const c10::FunctionSchema& s) {
  TORCH_CHECK(s.arguments().size() >= 1 && s.arguments()[0].name() == "self");
  std::vector<Argument> args({s.arguments().begin() + 1, s.arguments().end()});
  return s.cloneWithArguments(args);
}

// Returns two useful constructs:
//  first: map each value to all values that are alive
//    at the same time.
//  second: set of all inputs/outputs/constants (always alive)
std::pair<std::unordered_map<Value*, std::set<Value*>>, std::set<Value*>>
LivenessMap(const std::shared_ptr<torch::jit::Graph>& graph) {
  std::unordered_map<Value*, std::set<Value*>> liveness_map;
  std::set<Value*> always_alive;

  std::vector<Value*> frontier;
  // map live values to their deps, invariant: set.size() > 0
  std::unordered_map<Value*, std::set<Node*>> live_values;
  for (const auto& input : graph->inputs()) {
    frontier.emplace_back(input);
    always_alive.insert(input);
  }
  for (const auto& output : graph->outputs()) {
    always_alive.insert(output);
  }

  auto add_live_value = [&](Value* v) {
    liveness_map[v] = {};

    for (const auto& live_v : live_values) {
      liveness_map.at(v).insert(live_v.first);
      liveness_map.at(live_v.first).insert(v);
    }

    // only add values to the live set if they
    // have deps, otherwise they die immediately
    if (v->uses().size()) {
      live_values[v] = {};
    }

    for (const auto& u : v->uses()) {
      const auto& node = u.user;
      // track deps of this value
      live_values.at(v).insert(node);
    }
  };

  auto traverse_node = [&](Node* node, std::vector<Value*>& dead) {
    for (const auto& input : node->inputs()) {
      // ignore constant values
      if (input->node()->kind() == prim::Constant) {
        always_alive.insert(input);
        continue;
      }
      if (live_values.count(input)) {
        live_values.at(input).erase(node);
        if (!live_values.at(input).size()) {
          dead.emplace_back(input);
        }
      }
    }
  };

  for (const auto& node : graph->nodes()) {
    for (const auto& v : node->outputs()) {
      add_live_value(v);
    }

    std::vector<Value*> dead;
    traverse_node(node, dead);
    for (const auto& dead_value : dead) {
      live_values.erase(dead_value);
    }
  }

  for (const auto& v : live_values) {
    TORCH_CHECK(always_alive.count(v.first));
  }

  for (const auto& node : graph->nodes()) {
    for (const auto& input : node->inputs()) {
      for (const auto& output : node->outputs()) {
        if (liveness_map.count(input) && liveness_map.count(output)) {
          liveness_map.at(input).insert(output);
          liveness_map.at(output).insert(input);
        }
      }
    }
  }

  return std::make_pair(liveness_map, always_alive);
}

std::vector<Value*> GetOptimizableValues(
    const std::shared_ptr<torch::jit::Graph>& graph) {
  // for determinism
  std::vector<Value*> all_values;
  std::unordered_set<Value*> can_reuse;
  // values used by unsupported ops (as either inputs or outputs)
  // these need to be removed from "can_reuse" after analyzing all nodes
  std::unordered_set<Value*> cannot_reuse;
  for (const auto& input : graph->inputs()) {
    cannot_reuse.insert(input);
  }
  for (const auto& output : graph->outputs()) {
    cannot_reuse.insert(output);
  }
  for (const auto& n : graph->nodes()) {
    for (const auto& v : n->inputs()) {
      all_values.emplace_back(v);
      if (canRunOutOfPlace(n)) {
        can_reuse.insert(v);
      } else {
        cannot_reuse.insert(v);
      }
    }
    for (const auto& v : n->outputs()) {
      all_values.emplace_back(v);
      if (canRunOutOfPlace(n)) {
        can_reuse.insert(v);
      } else {
        cannot_reuse.insert(v);
      }
    }
  }
  for (auto v : cannot_reuse) {
    can_reuse.erase(v);
  }
  std::vector<Value*> out;
  for (auto v : all_values) {
    if (can_reuse.count(v)) {
      out.emplace_back(v);
      can_reuse.erase(v);
    }
  }
  return out;
}

std::unordered_map<Value*, std::vector<Value*>> FindShared(
    std::pair<std::unordered_map<Value*, std::set<Value*>>, std::set<Value*>>
        lm,
    std::vector<Value*> optimizable_values,
    bool optimize) {
  if (!optimize) {
    return {};
  }

  std::unordered_map<Value*, std::vector<Value*>> shared;
  // to preserve determinism
  std::vector<Value*> seen;

  // make these two values share memory
  auto share = [&](Value* new_v, Value* old_v) {
    DCHECK(shared.count(old_v));
    auto values = shared.at(old_v);
    values.emplace_back(new_v);
    for (auto* v : values) {
      shared[v] = values;
    }
  };

  for (auto v : optimizable_values) {
    // get values that are live during the lifetime of v
    auto live = lm.first.count(v) ? lm.first.at(v) : std::set<Value*>{};
    live.insert(lm.second.begin(), lm.second.end());

    for (auto s : seen) {
      // check if any values in this set of shared
      // are alive at the time of v
      // effectively finding | set_intersection(live, set(s.second)) | > 0
      bool intersects = false;
      for (auto candidate_v : shared.at(s)) {
        if (live.count(candidate_v)) {
          intersects = true;
          break;
        }
      }
      // we can share memory if there's no overlap
      if (!intersects) {
        share(v, s);
        break;
      }
    }
    // Couldn't share this value, give it its own storage
    if (!shared.count(v)) {
      shared[v] = {v};
    }
    seen.emplace_back(v);
  }

  return shared;
}

} // namespace

std::pair<std::shared_ptr<Graph>, c10::FunctionSchema> PrepareForStaticRuntime(
    const torch::jit::Module& m) {
  auto module = m.copy();
  module.eval();
  module = freeze_module(module);

  Method method = module.get_method("forward");
  auto graph = method.graph();
  PrepareGraphForStaticRuntime(graph);

  const c10::FunctionSchema& s =
      RemoveSelfFromSchema(method.function().getSchema());
  return std::make_pair(graph, s);
}

void PrepareGraphForStaticRuntime(std::shared_ptr<torch::jit::Graph> graph) {
  Inline(*graph);
  ConstantPropagation(graph);
  Canonicalize(graph);
  ConstantPropagation(graph);
  RemoveTensorMutation(graph);
  ConstantPropagation(graph);
  EliminateDeadCode(graph);

  FuseInferenceOpsForSparseNN(graph);
  ConstantPropagation(graph);
  CheckGraphEligibility(graph);
  RemoveSelfFromGraphInput(graph);
  RemoveListTupleConstruct(graph);
}

StaticRuntime::StaticRuntime(
    std::shared_ptr<torch::jit::Graph> g,
    const StaticRuntimeOptions& opts)
    : StaticRuntime(g, c10::nullopt, opts) {}

StaticRuntime::StaticRuntime(
    std::shared_ptr<torch::jit::Graph> g,
    const c10::optional<c10::FunctionSchema>& schema,
    const StaticRuntimeOptions& opts)
    : opts_(opts), graph_(g), schema_(schema) {
  std::unordered_map<Value*, IValue*> val_to_ival;

  // NB: create an unchanging std::vector<IValue> we can reference
  inputs_.resize(graph_->inputs().size());
  for (auto i = 0; i < graph_->inputs().size(); ++i) {
    Value* input = graph_->inputs().at(i);
    val_to_ival[input] = &(inputs_[i]);
  }

  // fill workspace_ with constants and create ProcessedNodes
  // NB: before optimizing the order of execution, ensure that the
  // memory optimization pass (LivenessMap + AssignRegisters) is
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
    int i = 0;
    for (Node* node : graph_->nodes()) {
      if (node->kind() != prim::Constant) {
        continue;
      }
      auto* v = node->output();
      val_to_ival[v] = &(constants_[i++]);
    }
  }
  for (Node* node : graph_->nodes()) {
    if (node->kind() == prim::Constant) {
      continue;
    }
    std::vector<const IValue*> inputs;
    for (Value* input : node->inputs()) {
      inputs.emplace_back(val_to_ival.at(input));
    }
    nodes_.emplace_back(
        ProcessedNode(node, std::move(inputs), opts.enable_out_variant));
    for (auto i = 0; i < node->outputs().size(); ++i) {
      val_to_ival[node->outputs().at(i)] = &nodes_.back().outputs().at(i);
    }
  }
  for (auto output : graph_->outputs()) {
    outputs_.emplace_back(val_to_ival.at(output));
  }
}

size_t StaticRuntime::num_outputs() const {
  return outputs_.size();
}

std::vector<at::Tensor> StaticRuntime::run(
    const std::vector<at::Tensor>& inps) {
  std::vector<c10::IValue> stack;
  stack.resize(inps.size());
  for (size_t i = 0; i < inps.size(); i++) {
    stack[i] = inps[i];
  }

  c10::IValue v = run(stack, std::unordered_map<std::string, c10::IValue>());

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

c10::IValue StaticRuntime::run(
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
        schema_,
        "Schema is not available. Consider creating the Static Runtime "
        "with StaticRuntime(const torch::jit::Module& m) instead.");
    std::vector<c10::IValue> s = args;
    schema_->checkAndNormalizeInputs(s, kwargs);
    for (size_t i = 0; i < s.size(); i++) {
      Input(i) = s[i];
    }
  } else {
    for (size_t i = 0; i < args.size(); i++) {
      Input(i) = args[i];
    }
  }

  // NB: before optimizing the order of execution, ensure that the
  // memory optimization pass (LivenessMap + AssignRegisters) is
  // aware of the new order!
  for (auto& n : nodes_) {
    n.run();
  }

  if (opts_.cleanup_activations) {
    if (!planner_) {
      auto lm = LivenessMap(graph_);
      auto optimizable_values = GetOptimizableValues(graph_);
      auto shared = FindShared(lm, optimizable_values, opts_.optimize_memory);
      planner_ = std::make_unique<MemoryPlanner>(this, shared);
    }
    planner_->deallocate();
  }

  // no need to keep references of outputs in static runtime anymore
  if (num_outputs() > 1) {
    std::vector<c10::IValue> outputs;
    outputs.reserve(num_outputs());
    for (auto i = 0; i < num_outputs(); ++i) {
      outputs.emplace_back(Output(i));
    }
    return c10::ivalue::Tuple::create(outputs);
  }
  return Output(0);
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
  std::cout << "Setting up took " << results.setup_time << " ms" << std::endl;

  for (size_t i = 0; i < nodes_.size(); i++) {
    const Node* node = nodes_[i].get_node();
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

  if (planner_) {
    std::cout << "Total memory managed: " << planner_->total_managed()
              << " bytes" << std::endl;
  }
  if (opts_.optimize_memory) {
    std::cout << "Total number of reused tensors: "
              << planner_->total_reused_tensors() << std::endl;
  }
}

float StaticRuntime::benchmark_model(
    const std::vector<c10::IValue>& args,
    const std::unordered_map<std::string, c10::IValue>& kwargs,
    const int warmup_runs,
    const int main_runs) {
  TORCH_CHECK(warmup_runs >= 0 && main_runs >= 1);

  for (int i = 0; i < warmup_runs; i++) {
    run(args, kwargs);
  }
  caffe2::Timer timer;
  for (int i = 0; i < main_runs; i++) {
    run(args, kwargs);
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
  results.total_time = 0.0;
  results.time_per_node.resize(nodes_.size(), 0);

  // setup time
  caffe2::Timer timer;
  std::vector<IValue> stack(args);
  if (!kwargs.empty()) {
    // This is not ideal
    TORCH_CHECK(
        schema_,
        "Schema is not available. Consider creating the Static Runtime "
        "with StaticRuntime(const torch::jit::Module& m) instead.");
    schema_->checkAndNormalizeInputs(stack, kwargs);
  }
  for (size_t i = 0; i < stack.size(); i++) {
    Input(i) = stack[i];
  }
  results.setup_time = timer.MilliSeconds();

  // warmup runs
  for (int i = 0; i < warmup_runs; i++) {
    run(args, kwargs);
  }

  // main runs
  for (int i = 0; i < main_runs; i++) {
    if (planner_) {
      planner_->allocate();
    }
    for (size_t j = 0; j < nodes_.size(); j++) {
      timer.Start();
      nodes_[j].run();
      float millis = timer.MilliSeconds();
      results.time_per_node[j] += millis;
    }
    if (opts_.cleanup_activations) {
      if (!planner_) {
        auto lm = LivenessMap(graph_);
        auto optimizable_values = GetOptimizableValues(graph_);
        auto shared = FindShared(lm, optimizable_values, opts_.optimize_memory);
        planner_ = std::make_unique<MemoryPlanner>(this, shared);
      }
      planner_->deallocate();
    }
  }

  // post processing
  for (size_t i = 0; i < nodes_.size(); i++) {
    const Node* node = nodes_[i].get_node();
    std::string kind = std::string(node->kind().toQualString());
    results.time_per_node[i] /= static_cast<float>(main_runs);
    results.time_per_node_type[kind] += results.time_per_node[i];
    results.instances_per_node_type[kind]++;
    results.total_time += results.time_per_node[i];
  }
  for (const auto& p : results.time_per_node_type) {
    const std::string& kind = p.first;
    results.percent_per_node_type[kind] = p.second / results.total_time * 100;
  }
  return results;
}

MemoryPlanner::MemoryPlanner(
    StaticRuntime* runtime,
    std::unordered_map<Value*, std::vector<Value*>> should_share) {
  // collect register indices of outputs of ops with out variant
  std::unordered_set<IValue*> unmanaged_value_set;
  for (ProcessedNode& node : runtime->get_nodes()) {
    if (node.has_out_variant()) {
      for (auto i = 0; i < node.outputs().size(); ++i) {
        Value* out_v = node.get_node()->outputs().at(i);
        IValue& out = node.outputs().at(i);
        if (out_v->type()->cast<TensorType>()) {
          managed_values_.insert(out_v);
        } else {
          unmanaged_value_set.insert(&out);
        }
      }
    } else {
      for (IValue& out : node.outputs()) {
        unmanaged_value_set.insert(&out);
      }
    }
  }

  // remove model outputs from managed_values_
  for (Value* output : runtime->graph()->outputs()) {
    managed_values_.erase(output);
  }
  for (IValue* output : runtime->outputs()) {
    unmanaged_value_set.erase(output);
  }
  for (IValue* out : unmanaged_value_set) {
    unmanaged_values_.emplace_back(out);
  }

  // remove tensors in output List/Tuple from managed_values_
  for (Value* output : runtime->graph()->outputs()) {
    Node* output_node = output->node();
    if (output_node->kind() == prim::TupleConstruct ||
        output_node->kind() == prim::ListConstruct) {
      for (Value* input : output_node->inputs()) {
        managed_values_.erase(input);
      }
    }
  }

  // some Values should share storage, this map will
  // keep track of the index into managed_storage_
  std::unordered_map<Value*, size_t> shared;

  // Snapshot of the current memory state
  for (const auto& node : runtime->get_nodes()) {
    for (auto i = 0; i < node.outputs().size(); ++i) {
      const auto& ival = node.outputs().at(i);
      const auto& val = node.get_node()->outputs().at(i);
      if (managed_values_.count(val)) {
        TORCH_CHECK(ival.isTensor());
        auto* impl = ival.toTensor().storage().unsafeGetStorageImpl();
        if (shared.count(val)) {
          managed_storage_[shared.at(val)].second.emplace_back(impl);
        } else {
          auto p =
              std::make_pair<size_t, std::vector<c10::StorageImpl*>>(0, {impl});
          managed_storage_.emplace_back(std::move(p));
          // first of a group, update the shared map with the index
          if (should_share.count(val)) {
            for (auto v : should_share.at(val)) {
              shared[v] = managed_storage_.size() - 1;
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
      impl->set_data_ptr(at::DataPtr(src, src, nullptr, impl->device()));
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
  for (auto& iv : unmanaged_values_) {
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
  for (const auto& output : node->outputs()) {
    outputs_.emplace_back();
  }
  if ((enable_out_variants && canRunOutOfPlace(node)) ||
      mustRunOutOfPlace(node)) {
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
      node->kind() != prim::ListUnpack) {
    const Operator& op = node->getOperator();
    TORCH_CHECK(op.hasOperation());
    op_ = op.getOperation(node);
    std::ostringstream ss;
    node->print(ss, 0, nullptr, false);
    VLOG(1) << "Fallback interpreter for node: " << ss.str();
  } else {
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
