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

void PrepareGraphForStaticRuntime(std::shared_ptr<torch::jit::Graph> graph) {
  Inline(*graph);
  ConstantPropagation(graph);
  Canonicalize(graph);
  ConstantPropagation(graph);
  RemoveTensorMutation(graph);
  ConstantPropagation(graph);
  EliminateDeadCode(graph);
}

namespace {
void OptimizeGraph(std::shared_ptr<torch::jit::Graph>& graph) {
  PrepareGraphForStaticRuntime(graph);
  FuseInferenceOpsForSparseNN(graph);
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
std::unique_ptr<c10::FunctionSchema> RemoveSelfFromSchema(
    const c10::FunctionSchema& s) {
  TORCH_CHECK(s.arguments().size() >= 1 && s.arguments()[0].name() == "self");
  std::vector<Argument> args({s.arguments().begin() + 1, s.arguments().end()});
  return std::make_unique<c10::FunctionSchema>(s.cloneWithArguments(args));
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

std::unordered_set<Value*> GetOptimizableValues(
    const std::shared_ptr<torch::jit::Graph>& graph) {
  std::unordered_set<Value*> can_reuse;
  // values used by unsupported ops (as either inputs or outputs)
  // these need to be removed from "can_reuse" after analyzing all nodes
  std::unordered_set<Value*> cannot_reuse;
  for (const auto& n : graph->nodes()) {
    for (const auto& v : n->inputs()) {
      if (canRunOutOfPlace(n) && canReuseInputsOutputs(n) &&
          canReuseInputs(n)) {
        can_reuse.insert(v);
      } else {
        cannot_reuse.insert(v);
      }
    }
    for (const auto& v : n->outputs()) {
      if (canRunOutOfPlace(n) && canReuseInputsOutputs(n) &&
          canReuseOutputs(n)) {
        can_reuse.insert(v);
      } else {
        cannot_reuse.insert(v);
      }
    }
  }
  for (auto v : cannot_reuse) {
    can_reuse.erase(v);
  }
  return can_reuse;
}

size_t AssignRegisters(
    const std::shared_ptr<torch::jit::Graph>& graph,
    std::unordered_map<Value*, size_t>& value_to_reg,
    std::vector<Value*>& values,
    std::vector<size_t>& input_regs,
    std::vector<size_t>& output_regs,
    bool optimize_memory) {
  auto lm = LivenessMap(graph);
  auto optimizable_values = GetOptimizableValues(graph);

  size_t num_regs = 0;
  size_t reused_regs = 0;
  std::unordered_map<size_t, std::set<Value*>> reg_to_val;
  auto getReg = [&](Value* v) -> size_t {
    if (!optimize_memory) {
      return num_regs++;
    }
    TORCH_CHECK(!value_to_reg.count(v));
    auto iter = lm.first.find(v);
    if (iter == lm.first.end()) {
      return num_regs++;
    }
    if (!optimizable_values.count(v)) {
      return num_regs++;
    }
    if (lm.second.count(v)) {
      return num_regs++;
    }
    const auto& live_values = iter->second;
    // iterate through all the allocated registers
    // and check for potential re-use, greedily
    for (const auto& v2r : value_to_reg) {
      auto candidate_v = v2r.first;

      if (!optimizable_values.count(candidate_v)) {
        continue;
      }
      if (lm.second.count(candidate_v)) {
        continue;
      }

      // Only re-use float* tensors
      auto t = candidate_v->type()->cast<TensorType>();
      if (!t) {
        continue;
      }
      // TODO audit this assumption (passes tests, but is scary)
      if (t->scalarType() && *(t->scalarType()) != at::kFloat) {
        continue;
      }
      // TODO
      // if (*(t->scalarType()) != at::kFloat) {
      //  continue;
      //}
      if (!live_values.count(candidate_v)) {
        bool already_used = false;
        for (auto use : reg_to_val.at(v2r.second)) {
          if (live_values.count(use)) {
            already_used = true;
          }
        }
        if (already_used) {
          continue;
        }
        reused_regs++;
        return v2r.second;
      }
    }
    return num_regs++;
  };

  // assign register to Value*
  for (Value* input : graph->inputs()) {
    TORCH_CHECK(value_to_reg.count(input) == 0);
    auto reg = getReg(input);
    value_to_reg[input] = reg;
    reg_to_val[reg].insert(input);
    input_regs.push_back(reg);
  }
  for (Node* node : graph->nodes()) {
    for (Value* input : node->inputs()) {
      TORCH_CHECK(value_to_reg.count(input) > 0);
    }
    for (Value* output : node->outputs()) {
      TORCH_CHECK(
          value_to_reg.count(output) == 0, "the graph needs to be in SSA form");
      auto reg = getReg(output);
      value_to_reg[output] = reg;
      reg_to_val[reg].insert(output);
    }
  }
  TORCH_CHECK(graph->outputs().size() > 0);
  for (Value* output : graph->outputs()) {
    TORCH_CHECK(value_to_reg.count(output) > 0);
    output_regs.push_back(value_to_reg[output]);
  }

  values.resize(value_to_reg.size());
  for (const auto& p : value_to_reg) {
    values[p.second] = p.first;
  }
  return reused_regs;
}

// Internal values are discarded after run if
// opts_.cleanup_activations is true.
void DeduceInternalValues(
    const std::shared_ptr<torch::jit::Graph>& graph,
    const std::unordered_map<Value*, size_t>& value_to_reg,
    std::vector<size_t>& internals) {
  std::unordered_set<Value*> outputs{
      graph->outputs().begin(), graph->outputs().end()};
  for (Node* node : graph->nodes()) {
    if (node->kind() != prim::Constant) {
      for (Value* output : node->outputs()) {
        if (outputs.count(output) == 0) {
          internals.push_back(value_to_reg.at(output));
        }
      }
    }
  }
}
} // namespace

void InferenceModule::init() {
  OptimizeGraph(graph);
  CheckGraphEligibility(graph);
  RemoveSelfFromGraphInput(graph);
  reused_regs = AssignRegisters(
      graph,
      value_to_reg,
      values,
      input_regs,
      output_regs,
      opts.optimize_memory);
  DeduceInternalValues(graph, value_to_reg, internals);
}

InferenceModule::InferenceModule(
    const torch::jit::Module& m,
    InferenceModuleOptions opts_)
    : module(m.copy()), graph(nullptr), schema(nullptr), opts(opts_) {
  module.eval();
  module = freeze_module(module);

  Method method = module.get_method("forward");
  graph = method.graph();

  const c10::FunctionSchema& s = method.function().getSchema();
  schema = RemoveSelfFromSchema(s);

  init();
}

InferenceModule::InferenceModule(
    std::shared_ptr<torch::jit::Graph> g,
    InferenceModuleOptions opts_)
    : module(), graph(std::move(g)), schema(nullptr), opts(opts_) {
  init();
}

StaticRuntime::StaticRuntime(
    const torch::jit::Module& m,
    const StaticRuntimeOptions& opts)
    : StaticRuntime(PrepareForStaticRuntime(m), opts) {}

StaticRuntime::StaticRuntime(
    std::shared_ptr<InferenceModule> m,
    const StaticRuntimeOptions& opts)
    : module_(m), opts_(opts) {
  TORCH_CHECK(
      module_ != nullptr,
      "std::shared_ptr<InferenceModule> module_ cannot be nullptr")

  Graph* graph = module_->graph.get();
  std::unordered_map<Value*, IValue*> val_to_ival;

  // NB: create an unchanging std::vector<IValue> we can reference
  for (auto input : graph->inputs()) {
    inputs_.emplace_back();
  }
  for (auto i = 0; i < graph->inputs().size(); ++i) {
    Value* input = graph->inputs()[i];
    val_to_ival[input] = &(inputs_[i]);
  }

  // fill workspace_ with constants and create ProcessedNodes
  // NB: before optimizing the order of execution, ensure that the
  // memory optimization pass (LivenessMap + AssignRegisters) is
  // aware of the new order!

  // Fill constants first, so we have a std::vector<IValue> we can reference
  // later
  for (Node* node : graph->nodes()) {
    if (node->kind() != prim::Constant) {
      continue;
    }
    auto* v = node->output();
    TORCH_CHECK(v->type()->kind() != FunctionType::Kind);
    constants_.emplace_back(toIValue(v).value());
  }
  {
    int i = 0;
    for (Node* node : graph->nodes()) {
      if (node->kind() != prim::Constant) {
        continue;
      }
      auto* v = node->output();
      val_to_ival[v] = &(constants_[i++]);
    }
  }
  for (Node* node : graph->nodes()) {
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
      val_to_ival[node->outputs()[i]] = &nodes_.back().Output(i);
    }
  }
  for (auto output : graph->outputs()) {
    outputs_.emplace_back(val_to_ival.at(output));
  }
}

size_t StaticRuntime::num_outputs() const {
  return module_->output_regs.size();
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
        module_->schema != nullptr,
        "Schema is not available. Consider creating the Static Runtime "
        "with StaticRuntime(const torch::jit::Module& m) instead.");
    std::vector<c10::IValue> s = args;
    module_->schema->checkAndNormalizeInputs(s, kwargs);
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
      std::unordered_map<Value*, std::vector<Value*>> shared;
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
  if (module_->opts.optimize_memory) {
    std::cout << "Total number of reused registers: " << module_->reused_regs
              << std::endl;
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
        module_->schema != nullptr,
        "Schema is not available. Consider creating the Static Runtime "
        "with StaticRuntime(const torch::jit::Module& m) instead.");
    module_->schema->checkAndNormalizeInputs(stack, kwargs);
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
        std::unordered_map<Value*, std::vector<Value*>> shared;
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
  // get input Value*
  at::ArrayRef<Value*> inputs =
      runtime->get_inference_module()->graph->inputs();
  std::unordered_set<Value*> graph_input_values(inputs.begin(), inputs.end());

  // collect register indices of outputs of ops with out variant
  std::unordered_set<Value*> managed_values;
  std::unordered_set<IValue*> unmanaged_value_set;
  for (ProcessedNode& pnode : runtime->get_nodes()) {
    bool should_manage = pnode.has_out_variant();
    if (should_manage && isViewOp(pnode.get_node())) {
      // outputs of view ops with inputs as the graph inputs shouldn't be
      // managed by the MemoryPlanner. It may release the storage of the graph
      // inputs.
      for (Value* in : pnode.get_node()->inputs()) {
        if (graph_input_values.count(in) > 0) {
          should_manage = false;
          break;
        }
      }
    }
    if (should_manage) {
      // Types are stored in the underlying TorchScript IR
      for (Value* out : pnode.get_node()->outputs()) {
        if (out->type()->cast<TensorType>()) {
          managed_values.insert(out);
        }
      }
    } else {
      for (auto i = 0; i < pnode.outputs().size(); ++i) {
        unmanaged_value_set.insert(&pnode.Output(i));
      }
    }
  }

  const InferenceModule* module = runtime->get_inference_module();

  // remove model outputs from managed_values
  for (Value* output : module->graph->outputs()) {
    managed_values.erase(output);
  }
  for (IValue* output : runtime->outputs()) {
    unmanaged_value_set.erase(output);
  }
  for (IValue* out : unmanaged_value_set) {
    unmanaged_values_.emplace_back(out);
  }

  // remove tensors in output List/Tuple from managed_values
  for (Value* output : module->graph->outputs()) {
    Node* output_node = output->node();
    if (output_node->kind() == prim::TupleConstruct ||
        output_node->kind() == prim::ListConstruct) {
      for (Value* input : output_node->inputs()) {
        managed_values.erase(input);
      }
    }
  }

  // some Values should share storage, this map will
  // keep track of the index into managed_storage_
  std::unordered_map<Value*, size_t> shared;
  // the StorageImpls of Tensor views should not be managed
  std::unordered_set<c10::StorageImpl*> managed_storage_impls;

  // Snapshot of the current memory state
  for (const auto& pnode : runtime->get_nodes()) {
    for (auto i = 0; i < pnode.outputs().size(); ++i) {
      const auto& ival = pnode.outputs()[i];
      auto* val = pnode.get_node()->outputs()[i];
      if (managed_values.count(val)) {
        TORCH_CHECK(ival.isTensor());
        auto* impl = ival.toTensor().storage().unsafeGetStorageImpl();

        auto didInsert = managed_storage_impls.insert(impl).second;
        if (!didInsert) {
          continue;
        }

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
    }

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
  outputs_.resize(node->outputs().size());
  if (node->kind() != prim::ListConstruct &&
      node->kind() != prim::TupleConstruct &&
      node->kind() != prim::ListUnpack) {
    const Operator& op = node->getOperator();
    TORCH_CHECK(op.hasOperation());
    op_ = op.getOperation(node);
  }
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
