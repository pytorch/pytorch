#include <torch/csrc/jit/runtime/static/impl.h>
#include <ATen/core/interned_strings.h>
#include <caffe2/core/timer.h>
#include <torch/csrc/jit/passes/canonicalize.h>
#include <torch/csrc/jit/passes/freeze_module.h>
#include <torch/csrc/jit/passes/remove_mutation.h>
#include <torch/csrc/jit/passes/subgraph_rewrite.h>
#include <torch/csrc/jit/runtime/static/ops.h>
#include <torch/csrc/jit/runtime/vararg_functions.h>

namespace torch {
namespace jit {

std::shared_ptr<torch::jit::Graph> PrepareForStaticRuntime(
    std::shared_ptr<torch::jit::Graph> g) {
  Inline(*g);
  ConstantPropagation(g);
  Canonicalize(g);
  ConstantPropagation(g);
  RemoveTensorMutation(g);
  ConstantPropagation(g);

  for (auto n : g->nodes()) {
    if (n->kind() == c10::Symbol::fromQualString("prim::GetAttr")) {
      throw std::runtime_error("Cannot accelerate unfrozen graphs");
    }
  }

  // remove unused input 0 from graph
  if (g->inputs().at(0)->type()->is_module()) {
    TORCH_CHECK(!g->inputs().at(0)->hasUses());
    g->eraseInput(0);
  }

  return g;
}

std::shared_ptr<torch::jit::Graph> PrepareForStaticRuntime(
    const torch::jit::Module& m) {
  auto module = m.copy();
  module.eval();
  module = freeze_module(module);
  auto g = module.get_method("forward").graph();
  return PrepareForStaticRuntime(g);
}

StaticRuntime::StaticRuntime(std::shared_ptr<torch::jit::Graph> g)
    : StaticRuntime(g, c10::nullopt) {}

StaticRuntime::StaticRuntime(const torch::jit::Module& m)
    : StaticRuntime(PrepareForStaticRuntime(m), m) {}

StaticRuntime::StaticRuntime(
    std::shared_ptr<torch::jit::Graph> g,
    c10::optional<torch::jit::Module> m)
    : graph_(g) {
  // assign register to Value*
  std::unordered_map<Value*, size_t> value_to_reg;
  for (Value* input : g->inputs()) {
    TORCH_CHECK(value_to_reg.count(input) == 0);
    size_t index = value_to_reg.size();
    value_to_reg[input] = index;
    input_regs_.push_back(index);
  }
  for (Node* node : graph_->nodes()) {
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

  TORCH_CHECK(g->outputs().size() > 0);
  for (Value* output : g->outputs()) {
    TORCH_CHECK(value_to_reg.count(output) > 0);
    output_regs_.push_back(value_to_reg[output]);
  }

  // initialize registers
  reg_.resize(value_to_reg.size());

  // fill workspace_ with constants
  for (Node* node : graph_->nodes()) {
    if (node->kind() == prim::Constant) {
      TORCH_CHECK(node->output()->type()->kind() != FunctionType::Kind);
      reg_[value_to_reg[node->output()]] = toIValue(node->output()).value();
    } else {
      std::vector<size_t> input_regs, output_regs;
      for (Value* input : node->inputs()) {
        input_regs.push_back(value_to_reg[input]);
      }
      for (Value* output : node->outputs()) {
        output_regs.push_back(value_to_reg[output]);
      }
      nodes_.emplace_back(node, std::move(input_regs), std::move(output_regs));
    }
  }

  if (m) {
    Method method = m->get_method("forward");
    const c10::FunctionSchema& schema = method.function().getSchema();

    // remove "self" from function schema
    TORCH_CHECK(
        schema.arguments().size() >= 1 &&
        schema.arguments()[0].name() == "self");
    std::vector<Argument> args(
        {schema.arguments().begin() + 1, schema.arguments().end()});
    schema_ =
        std::make_unique<c10::FunctionSchema>(schema.cloneWithArguments(args));
  }
}

std::vector<at::Tensor> StaticRuntime::run(
    const std::vector<at::Tensor>& inps) const {
  for (size_t i = 0; i < inps.size(); i++) {
    Input(i) = inps[i];
  }

  for (const auto& n : nodes_) {
    n.run(reg_);
  }

  std::vector<at::Tensor> out;
  for (size_t i = 0; i < graph_->outputs().size(); i++) {
    const IValue& v = Output(i);
    if (v.isTuple()) {
      auto t = v.toTuple();
      for (const auto& el : t->elements()) {
        out.emplace_back(el.toTensor());
      }
    } else {
      out.emplace_back(v.toTensor());
    }
  }
  return out;
}

c10::IValue StaticRuntime::run(
    const std::vector<c10::IValue>& args,
    const std::unordered_map<std::string, c10::IValue>& kwargs) const {
  std::vector<IValue> stack(args);
  if (!kwargs.empty()) {
    // This is not ideal
    TORCH_CHECK(
        schema_ != nullptr,
        "Schema is not available. Consider creating the Static Runtime "
        "with StaticRuntime(const torch::jit::Module& m) instead.");
    schema_->checkAndNormalizeInputs(stack, kwargs);
  }
  for (size_t i = 0; i < stack.size(); i++) {
    Input(i) = stack[i];
  }

  for (const auto& n : nodes_) {
    n.run(reg_);
  }

  return Output(0);
}

void StaticRuntime::benchmark(
    const std::vector<c10::IValue>& args,
    const std::unordered_map<std::string, c10::IValue>& kwargs,
    const int warmup_runs,
    const int main_runs) const {
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
    const int main_runs) const {
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
    const int main_runs) const {
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
        schema_ != nullptr,
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
    for (size_t j = 0; j < nodes_.size(); j++) {
      timer.Start();
      nodes_[j].run(reg_);
      float millis = timer.MilliSeconds();
      results.time_per_node[j] += millis;
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

ProcessedNode::ProcessedNode(
    Node* node,
    std::vector<size_t>&& input_regs,
    std::vector<size_t>&& output_regs)
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
  if (canRunOutOfPlace(node)) {
    fn_ = getOutOfPlaceOperation(node);
  }
}

void ProcessedNode::run(std::vector<IValue>& reg) const {
  if (!fn_) {
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
  } else {
    fn_->operator()(this, reg);
  }
}

} // namespace jit
} // namespace torch
