#include <torch/csrc/jit/runtime/static/impl.h>
#include <ATen/core/interned_strings.h>
#include <caffe2/core/scope_guard.h>
#include <caffe2/core/timer.h>
#include <torch/csrc/jit/passes/canonicalize.h>
#include <torch/csrc/jit/passes/freeze_module.h>
#include <torch/csrc/jit/passes/remove_mutation.h>
#include <torch/csrc/jit/passes/subgraph_rewrite.h>
#include <torch/csrc/jit/runtime/static/ops.h>
#include <torch/csrc/jit/runtime/vararg_functions.h>

namespace torch {
namespace jit {
namespace {
void OptimizeGraph(std::shared_ptr<torch::jit::Graph>& graph) {
  Inline(*graph);
  ConstantPropagation(graph);
  Canonicalize(graph);
  ConstantPropagation(graph);
  RemoveTensorMutation(graph);
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

void AssignRegisters(
    const std::shared_ptr<torch::jit::Graph>& graph,
    std::unordered_map<Value*, size_t>& value_to_reg,
    std::vector<Value*>& values,
    std::vector<size_t>& input_regs,
    std::vector<size_t>& output_regs) {
  // assign register to Value*
  for (Value* input : graph->inputs()) {
    TORCH_CHECK(value_to_reg.count(input) == 0);
    size_t index = value_to_reg.size();
    value_to_reg[input] = index;
    input_regs.push_back(index);
  }
  for (Node* node : graph->nodes()) {
    for (Value* input : node->inputs()) {
      TORCH_CHECK(value_to_reg.count(input) > 0);
    }
    for (Value* output : node->outputs()) {
      TORCH_CHECK(
          value_to_reg.count(output) == 0, "the graph needs to be in SSA form");
      size_t index = value_to_reg.size();
      value_to_reg[output] = index;
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
}

// Internal blobs (IValues) are discarded after run if
// opts_.cleanup_activations is true.
void DeduceInternalBlobs(
    const std::shared_ptr<torch::jit::Graph>& graph,
    const std::unordered_map<Value*, size_t>& value_to_reg,
    std::vector<size_t>& internals) {
  std::unordered_set<Value*> outputs{graph->outputs().begin(),
                                     graph->outputs().end()};
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
  AssignRegisters(graph, value_to_reg, values, input_regs, output_regs);
  DeduceInternalBlobs(graph, value_to_reg, internals);
}

InferenceModule::InferenceModule(const torch::jit::Module& m)
    : module(m.copy()), graph(nullptr), schema(nullptr) {
  module.eval();
  module = freeze_module(module);

  Method method = module.get_method("forward");
  graph = method.graph();

  const c10::FunctionSchema& s = method.function().getSchema();
  schema = RemoveSelfFromSchema(s);

  init();
}

InferenceModule::InferenceModule(std::shared_ptr<torch::jit::Graph> g)
    : module(), graph(g), schema(nullptr) {
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
  // initialize registers
  reg_.resize(module_->value_to_reg.size());

  Graph* graph = module_->graph.get();
  const auto& value_to_reg = module_->value_to_reg;

  // fill workspace_ with constants and create ProcessedNodes
  for (Node* node : graph->nodes()) {
    if (node->kind() == prim::Constant) {
      TORCH_CHECK(node->output()->type()->kind() != FunctionType::Kind);
      reg_[value_to_reg.at(node->output())] = toIValue(node->output()).value();
    } else {
      std::vector<size_t> input_regs, output_regs;
      for (Value* input : node->inputs()) {
        input_regs.push_back(value_to_reg.at(input));
      }
      for (Value* output : node->outputs()) {
        output_regs.push_back(value_to_reg.at(output));
      }
      nodes_.emplace_back(
          node,
          std::move(input_regs),
          std::move(output_regs),
          opts.enable_out_variant);
    }
  }
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
  if (planner_) {
    planner_->allocate();
  }

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

  for (const auto& n : nodes_) {
    n.run(reg_);
  }

  if (opts_.cleanup_activations) {
    if (!planner_) {
      planner_ = std::make_unique<MemoryPlanner>(this);
    }
    planner_->deallocate();
    deallocate_registers(module_->internals);
  }

  // no need to keep references of outputs in static runtime anymore
  DCHECK(module_->output_regs.size() == 1);
  return std::move(reg_[module_->output_regs[0]]);
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
      nodes_[j].run(reg_);
      float millis = timer.MilliSeconds();
      results.time_per_node[j] += millis;
    }
    if (opts_.cleanup_activations) {
      if (!planner_) {
        planner_ = std::make_unique<MemoryPlanner>(this);
      }
      planner_->deallocate();
      deallocate_registers(module_->internals);
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

void StaticRuntime::deallocate_registers(const std::vector<size_t>& internals) {
  // discard Tensor objects to reduce memory usage
  // they will be re-created in the next iteration regardless
  for (auto i : internals) {
    if (reg_[i].isTensor()) {
      if (reg_[i].toTensor().storage().nbytes() > 0) {
        reg_[i] = IValue();
      }
    } else {
      // TensorLists and Tuples
      // TODO: cache the List and Tuple objects but release what's inside
      reg_[i] = IValue();
    }
  }
}

MemoryPlanner::MemoryPlanner(StaticRuntime* runtime)
    : reg_(runtime->get_registers()) {
  // collect register indices of outputs of ops with out variant
  for (const ProcessedNode& node : runtime->get_nodes()) {
    if (node.has_out_variant()) {
      for (auto out : node.output_regs()) {
        reg_out_variant_.insert(out);
      }
    }
  }

  const InferenceModule* module = runtime->get_inference_module();

  // remove model outputs from reg_out_variant_
  for (size_t output : module->output_regs) {
    reg_out_variant_.erase(output);
  }

  // remove tensors in output List/Tuple from reg_out_variant_
  for (Value* output : module->graph->outputs()) {
    Node* output_node = output->node();
    if (output_node->kind() == prim::TupleConstruct ||
        output_node->kind() == prim::ListConstruct) {
      for (Value* input : output_node->inputs()) {
        reg_out_variant_.erase(module->value_to_reg.at(input));
      }
    }
  }

  // debug only
  for (auto reg : reg_out_variant_) {
    VLOG(1) << "reg_out_variant_: %" << module->values[reg]->debugName();
  }

  // dedup tensor storages (tensor views share the same tensor storage)
  auto internal_storages_set = reg_to_storage_impls();
  internal_storages_.assign(
      internal_storages_set.begin(), internal_storages_set.end());

  internal_blob_max_sizes_.resize(internal_storages_.size());
}

// Don't change the size if it is already aligned, otherwise increase the size
// to make it aligned.
size_t MemoryPlanner::compute_aligned_tensor_size(size_t nbytes) {
  // Note: everything below is size_t
  return (nbytes + c10::gAlignment - 1) & (~(c10::gAlignment - 1));
}

at::DataPtr MemoryPlanner::allocate_buffer(size_t size) {
  at::Allocator* allocator = c10::GetCPUAllocator();
  return allocator->allocate(size);
}

std::unordered_set<c10::StorageImpl*> MemoryPlanner::reg_to_storage_impls() {
  std::unordered_set<c10::StorageImpl*> internal_storages_set;
  for (auto i : reg_out_variant_) {
    internal_storages_set.insert(
        reg_[i].toTensor().storage().unsafeGetStorageImpl());
  }
  return internal_storages_set;
}

void MemoryPlanner::allocate() {
  if (internal_blob_max_sizes_sum_ == 0) {
    return;
  }

  buffer_ = allocate_buffer(internal_blob_max_sizes_sum_);

  size_t offset = 0;
  uint8_t* start = static_cast<uint8_t*>(buffer_.get());

  for (auto i = 0; i < internal_storages_.size(); i++) {
    auto tensor_size = internal_blob_max_sizes_[i];
    if (tensor_size == 0) {
      continue;
    }
    DCHECK_LE(offset + tensor_size, internal_blob_max_sizes_sum_);
    void* src = static_cast<void*>(start + offset);

    c10::StorageImpl* impl = internal_storages_[i];
    impl->set_data_ptr(at::DataPtr(src, src, nullptr, impl->device()));
    impl->set_nbytes(tensor_size);

    offset += tensor_size;
  }
  DCHECK_EQ(offset, internal_blob_max_sizes_sum_);
}

void MemoryPlanner::verify_internal_storages() {
  auto internal_storages_set = reg_to_storage_impls();
  for (auto* storage_impl : internal_storages_) {
    TORCH_CHECK(
        internal_storages_set.count(storage_impl) > 0,
        "Found internal_storage mismatch");
  }
}

void MemoryPlanner::deallocate() {
#ifndef NDEBUG
  verify_internal_storages();
#endif
  internal_blob_max_sizes_sum_ = 0;
  // free memory used by outputs of ops in out variants
  // but keep the TensorImpl and StorageImpl around
  for (auto i = 0; i < internal_storages_.size(); i++) {
    c10::StorageImpl* impl = internal_storages_[i];
    size_t current_size = compute_aligned_tensor_size(impl->nbytes());
    size_t& max_size = internal_blob_max_sizes_[i];
    max_size = std::max(max_size, current_size);
    internal_blob_max_sizes_sum_ += max_size;
    impl->reset();
  }
  buffer_ = {};
}

ProcessedNode::ProcessedNode(
    Node* node,
    std::vector<size_t>&& input_regs,
    std::vector<size_t>&& output_regs,
    bool enable_out_variants)
    : node_(node),
      input_regs_(std::move(input_regs)),
      output_regs_(std::move(output_regs)) {
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
  }
  if (canRunNatively(node)) {
    native_fn_ = getNativeOperation(node);
    std::ostringstream ss;
    node->print(ss, 0, nullptr, false);
    VLOG(1) << "Switch to native impl for node: " << ss.str();
  }
}

void ProcessedNode::run(std::vector<IValue>& reg) const {
  if (fn_) {
    fn_->operator()(this, reg);
  } else if (native_fn_) {
    native_fn_->operator()(this, reg);
  } else {
    std::vector<IValue> stack;
    const size_t size = node_->inputs().size();
    stack.reserve(size);
    for (size_t i = 0; i < size; i++) {
      stack.emplace_back(Input(i, reg));
    }
    if (op_) {
      op_->operator()(&stack);
    } else {
      if (node_->kind() == prim::ListConstruct) {
        listConstruct(
            stack,
            node_->output()->type()->expect<ListType>(),
            node_->inputs().size());
      } else if (node_->kind() == prim::TupleConstruct) {
        bool named =
            node_->output()->type()->expect<TupleType>()->name().has_value();
        if (named) {
          namedTupleConstruct(
              stack,
              node_->output()->type()->expect<TupleType>(),
              node_->inputs().size());
        } else {
          tupleConstruct(stack, node_->inputs().size());
        }
      } else if (node_->kind() == prim::ListUnpack) {
        size_t num_outputs = node_->outputs().size();
        listUnpack(stack, num_outputs);
      } else {
        TORCH_CHECK(0, "Unhandled operation!", node_->kind().toQualString());
      }
    }
    DCHECK_EQ(stack.size(), node_->outputs().size());
    for (auto i = 0; i < node_->outputs().size(); i++) {
      Output(i, reg) = std::move(stack[i]);
    }
  }
}

} // namespace jit
} // namespace torch
