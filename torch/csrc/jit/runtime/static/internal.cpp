#include <torch/csrc/jit/runtime/static/internal.h>

#include <ATen/MemoryOverlap.h>
#include <ATen/core/interned_strings.h>
#include <ATen/record_function.h>
#include <c10/core/CPUAllocator.h>
#include <c10/core/InferenceMode.h>
#include <c10/util/irange.h>
#include <caffe2/core/scope_guard.h>
#include <caffe2/core/timer.h>
#include <torch/csrc/jit/ir/alias_analysis.h>
#include <torch/csrc/jit/jit_log.h>
#include <torch/csrc/jit/passes/canonicalize.h>
#include <torch/csrc/jit/passes/dead_code_elimination.h>
#include <torch/csrc/jit/passes/eliminate_no_ops.h>
#include <torch/csrc/jit/passes/freeze_module.h>
#include <torch/csrc/jit/passes/remove_mutation.h>
#include <torch/csrc/jit/passes/subgraph_rewrite.h>
#include <torch/csrc/jit/passes/variadic_ops.h>
#include <torch/csrc/jit/runtime/static/memory_planner.h>
#include <torch/csrc/jit/runtime/static/ops.h>
#include <torch/csrc/jit/runtime/static/passes.h>
#include <torch/csrc/jit/runtime/vararg_functions.h>
#include <iterator>
#include <sstream>
#include <stdexcept>

// used in test only
C10_DEFINE_bool(
    static_runtime_disable_debug_memory_overlap_check,
    false,
    "If true, disable the memory overlap check in debug mode in ProcessedNode::run()");

namespace torch {
namespace jit {

std::string dumpValueSet(
    const FastSet<const Value*>& value_set,
    const char* set_name) {
  std::ostringstream oss;
  oss << set_name << ": {";
  for (const auto* val : value_set) {
    oss << "%" << val->debugName() << ", ";
  }
  oss << "}";
  return oss.str();
}

bool mayContainAlias(AliasDb& db, const Value* a, const Value* b) {
  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
  return db.mayContainAlias(const_cast<Value*>(a), const_cast<Value*>(b));
}

namespace {

std::vector<Value*> valueVecFromFastSet(const FastSet<const Value*>& s) {
  std::vector<Value*> result;
  result.reserve(s.size());
  for (auto* v : s) {
    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
    result.emplace_back(const_cast<Value*>(v));
  }
  return result;
}

} // namespace

bool mayContainAlias(
    AliasDb& db,
    const FastSet<const Value*>& a,
    const FastSet<const Value*>& b) {
  return db.mayContainAlias(valueVecFromFastSet(a), valueVecFromFastSet(b));
}

ValueGroup::ValueGroup(
    const std::shared_ptr<torch::jit::Graph>& graph,
    AliasDb& db) {
  external_aliases_.clear();
  output_aliases_.clear();
  // Build `input_or_constant_aliases` as we look through nodes forwardly from
  // the graph's inputs and add aliases of the inputs being created by the
  // nodes.
  external_aliases_.insert(graph->inputs().begin(), graph->inputs().end());
  for (const auto* node : graph->nodes()) {
    if (node->kind() == prim::Constant) {
      for (const auto* output : node->outputs()) {
        external_aliases_.insert(output);
      }
    }
  }
  for (const auto* node : graph->nodes()) {
    if (node->kind() == prim::Constant) {
      // Constants are already in `input_or_constant_aliases`.
      continue;
    }
    for (const auto* v : node->outputs()) {
      if (mayContainAlias(db, {v}, external_aliases_)) {
        external_aliases_.insert(v);
      }
    }
  }

  // Build `output_aliases` as we look through nodes reversely so that we can
  // start from the output values, and follow the flows backwardly from there.
  output_aliases_.insert(graph->outputs().begin(), graph->outputs().end());
  for (const auto* node : graph->nodes().reverse()) {
    if (node->kind() == prim::Constant) {
      // Constants cannot create any aliases.
      continue;
    }
    for (const auto* v : node->outputs()) {
      // Add values that can aliase input/constant values. Note some output
      // aliases may end up in this category via collection objects (e.g.,
      // Tuple).
      if (mayContainAlias(db, {v}, external_aliases_)) {
        external_aliases_.insert(v);
        continue;
      }
      if (mayContainAlias(db, {v}, output_aliases_)) {
        output_aliases_.insert(v);
      }
    }
  }
}

namespace {

bool mayContainAlias(const Value* v1, const Value* v2, const AliasDb& db) {
  // AliasDb is not const-correct here, so we have to const_cast
  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
  return db.mayContainAlias(const_cast<Value*>(v1), const_cast<Value*>(v2));
}

bool isPureFunction(const Node* node) {
  auto* schema = node->maybeSchema();
  return schema &&
      schema->aliasAnalysis() == c10::AliasAnalysisKind::PURE_FUNCTION;
}

} // namespace

ManagedTensorRanges::ManagedTensorRanges(
    const std::shared_ptr<Graph>& graph,
    const FastSet<const Value*>& managed_tensor_values) {
  AliasDb alias_db(graph);
  const std::vector<Node*> nodes(graph->nodes().begin(), graph->nodes().end());
  const FastSet<const Value*> graph_inputs(
      graph->inputs().begin(), graph->inputs().end());

  auto isUntrackedValue = [&alias_db, &graph_inputs](const Value* value) {
    return !alias_db.isMutableType(value) ||
        graph_inputs.find(value) != graph_inputs.end();
  };

  const auto num_nodes = nodes.size();
  for (const auto i : c10::irange(num_nodes)) {
    auto* node = nodes[i];
    for (auto* input : node->inputs()) {
      auto* lifetime = getLifetime(input);
      if (!lifetime) {
        DCHECK(isUntrackedValue(input));
        continue;
      }
      DCHECK(lifetime->end <= i);
      lifetime->end = i;
    }
    for (auto* output : node->outputs()) {
      if (!alias_db.isMutableType(output)) {
        continue;
      }
      value_lifetimes_.emplace(output, Lifetime(i, i));
    }
  }
  for (auto* graph_output : graph->outputs()) {
    auto* lifetime = getLifetime(graph_output);
    if (!lifetime) {
      DCHECK(isUntrackedValue(graph_output));
      continue;
    }
    lifetime->end = num_nodes;
  }

  // Handle aliases. Aliases may extend a Value*'s lifetime. If a node
  // has an input and output that may alias each other, set the input's
  // lifetime end to max(input.lifetime_end, output.lifetime_end). Iterate
  // backwards to handle chains of aliases.
  for (const auto* node : graph->nodes().reverse()) {
    if (isPureFunction(node)) {
      // If the node is a pure function, it doesn't create any aliases,
      // so we can safely skip it.
      continue;
    }

    auto inputs = collectValuesWithTrackedLifetimes(node->inputs());
    auto outputs = collectValuesWithTrackedLifetimes(node->outputs());
    for (auto* input : inputs) {
      auto* input_lifetime = getLifetime(input);
      DCHECK(input_lifetime != nullptr);
      for (auto* output : outputs) {
        if (mayContainAlias(input, output, alias_db)) {
          auto* output_lifetime = getLifetime(output);
          DCHECK(output_lifetime != nullptr);
          input_lifetime->end =
              std::max(output_lifetime->end, input_lifetime->end);
        }
      }
    }
  }
  for (auto* managed_tensor : managed_tensor_values) {
    auto* lifetime = getLifetime(managed_tensor);
    DCHECK(lifetime && lifetime->end <= num_nodes);
    // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
    Node* freeing_node;
    if (lifetime->end == num_nodes) {
      freeing_node = graph->return_node();
    } else {
      freeing_node = nodes[lifetime->end];
    }
    node_to_newly_free_tensors_[freeing_node].emplace_back(managed_tensor);
  }
}

bool ManagedTensorRanges::isUnusedValue(const Value* value) const {
  auto* lifetime = getLifetime(value);
  return lifetime && lifetime->start == lifetime->end;
}

bool ManagedTensorRanges::nodeFreesManagedTensors(Node* node) const {
  auto it = node_to_newly_free_tensors_.find(node);
  return it != node_to_newly_free_tensors_.end() && !it->second.empty();
}

const std::vector<const Value*>& ManagedTensorRanges::availableTensorsAfterNode(
    Node* node) const {
  return node_to_newly_free_tensors_.at(node);
}

bool ManagedTensorRanges::lifetimesOverlap(const Value* v1, const Value* v2)
    const {
  const auto* v1_lifetime = getLifetime(v1);
  const auto* v2_lifetime = getLifetime(v2);
  if (!v1_lifetime || !v2_lifetime) {
    return false;
  }

  if (v1_lifetime->start < v2_lifetime->start) {
    return v1_lifetime->end >= v2_lifetime->start;
  }
  return v2_lifetime->end >= v1_lifetime->start;
}

const ManagedTensorRanges::Lifetime* ManagedTensorRanges::getLifetime(
    const Value* value) const {
  auto it = value_lifetimes_.find(value);
  if (it != value_lifetimes_.end()) {
    return &it->second;
  }
  return nullptr;
}

ManagedTensorRanges::Lifetime* ManagedTensorRanges::getLifetime(
    const Value* value) {
  // const_cast is safe here, this is just a way to avoid code duplication
  // between the const/non-const versions of getLifetime.

  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
  const auto* const_this = const_cast<const ManagedTensorRanges*>(this);

  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
  return const_cast<ManagedTensorRanges::Lifetime*>(
      const_this->getLifetime(value));
}

std::vector<const Value*> ManagedTensorRanges::
    collectValuesWithTrackedLifetimes(at::ArrayRef<const Value*> values) {
  std::vector<const Value*> mutable_values;
  mutable_values.reserve(values.size());
  std::copy_if(
      values.begin(),
      values.end(),
      std::back_inserter(mutable_values),
      [this](const Value* value) { return getLifetime(value) != nullptr; });
  return mutable_values;
}

ProcessedFunction::ProcessedFunction(
    Node* node,
    bool enable_out_variant,
    bool check_memory_overlap)
    : check_memory_overlap_(check_memory_overlap) {
  if (enable_out_variant) {
    f_ = getOutOfPlaceOperation(node);
    if (f_) {
      kind_ = ProcessedFunction::Kind::kOutVariant;
      // do not check memory overlap for out variants
      check_memory_overlap_ = false;
      VLOG(1) << "Switch to out variant for node: " << PrintNode(node);
      return;
    }
  }
  {
    f_ = getNativeOperation(node);
    if (f_) {
      kind_ = ProcessedFunction::Kind::kNativeFunction;
#ifdef NDEBUG
      // skip this check in opt mode because these ops are better vetted
      check_memory_overlap_ = false;
#endif
      VLOG(1) << "Switch to native impl for node: " << PrintNode(node);
      return;
    }
  }
  {
    const Operator& op = node->getOperator();
    f_ = [node_op = op.getOperation(node)](ProcessedNode* pnode) mutable {
      std::vector<IValue> stack;
      Node* node = pnode->node();
      const size_t size = node->inputs().size();
      stack.reserve(size + (hasVarArgs(node) ? 1 : 0));
      for (const auto i : c10::irange(size)) {
        stack.emplace_back(pnode->Input(i));
      }
      // Need to store the number of inputs in stack for variadic ops.
      if (hasVarArgs(node)) {
        stack.emplace_back(static_cast<int>(size));
      }

      node_op(stack);

      DCHECK_EQ(stack.size(), node->outputs().size());
      for (const auto i : c10::irange(node->outputs().size())) {
        pnode->Output(i) = std::move(stack[i]);
      }
    };
    kind_ = ProcessedFunction::Kind::kInterpreterFallback;
    VLOG(1) << "Fallback interpreter for node: " << PrintNode(node);
  }
}

ProcessedNode::ProcessedNode(
    Node* node,
    ProcessedFunction* fn,
    ProcessedNodeInputs inputs,
    uint16_t outputs_offset)
    : node_(node),
      fn_(fn),
      inputs_(std::move(inputs)),
      outputs_offset_(outputs_offset)
#ifndef PYTORCH_DISABLE_PER_OP_PROFILING
      ,
      op_name_(node->kind().toQualString())
#endif
{
  TORCH_CHECK(
      node->outputs().size() < (1 << (sizeof(num_outputs_) * 8)),
      node->outputs().size(),
      " outputs to ProcessedNode ",
      node->kind().toQualString(),
      " is too many to use 2-byte indexing");
  num_outputs_ = node->outputs().size();
}

std::vector<IValue> ProcessedNode::inputs_ivalue_vec() const {
  std::vector<IValue> result;
  result.reserve(inputs_.size());
  for (const auto idx : c10::irange(num_inputs())) {
    result.emplace_back(Input(idx));
  }
  return result;
}

void ProcessedNode::run() {
#ifndef PYTORCH_DISABLE_PER_OP_PROFILING
  bool pre_sampled = false;
  if (C10_UNLIKELY(at::shouldRunRecordFunction(&pre_sampled))) {
    at::RecordFunction guard(at::RecordScope::FUNCTION, pre_sampled);
    if (guard.isActive()) {
      if (guard.needsInputs()) {
        guard.before(get_op_name(), inputs_ivalue_vec());
      } else {
        guard.before(get_op_name());
      }
    }
    fn_->f()(this);
  } else {
    fn_->f()(this);
  }
#else
  fn_->f()(this);
#endif
#ifndef NDEBUG
  if (FLAGS_static_runtime_disable_debug_memory_overlap_check) {
    // run check but do not enforce
    verify_no_memory_overlap();
  } else {
    DCHECK(verify_no_memory_overlap());
  }
#endif
}

static bool checkNoMemoryOverlap(const at::Tensor& a, const at::Tensor& b) {
  at::MemOverlapStatus status = at::get_overlap_status(a, b);
  if (status == at::MemOverlapStatus::FULL ||
      status == at::MemOverlapStatus::PARTIAL) {
    return false;
  }
  if (status == at::MemOverlapStatus::TOO_HARD) {
    VLOG(1) << "Detected TOO_HARD memory overlap status";
  }
  return true;
}

bool ProcessedNode::verify_no_memory_overlap(bool force_check) const {
  const static std::array<c10::Symbol, 3> special_case_ops = {
      fromQualString("prim::TypeCheck"),
      fromQualString("static_runtime::VarTupleUnpack"),
      fromQualString("static_runtime::dict_unpack")};
  if (!force_check &&
      std::find(
          begin(special_case_ops), end(special_case_ops), node()->kind()) !=
          end(special_case_ops)) {
    return true;
  }

  return verify_outputs_dont_overlap_each_other() &&
      verify_inputs_dont_overlap_outputs(force_check);
}

bool ProcessedNode::verify_outputs_dont_overlap_each_other() const {
  for (const auto i : c10::irange(num_outputs_)) {
    if (!Output(i).isTensor()) {
      continue;
    }
    const auto& out0_t = Output(i).toTensor();
    for (const auto j : c10::irange(i + 1, num_outputs_)) {
      if (!Output(j).isTensor()) {
        continue;
      }
      const auto& out1_t = Output(j).toTensor();
      if (!checkNoMemoryOverlap(out0_t, out1_t)) {
        LOG(INFO) << "Node output " << i << " overlaps with output " << j
                  << ", " << PrintNode(node_);
        return false;
      }
    }
  }
  return true;
}

bool ProcessedNode::verify_inputs_dont_overlap_outputs(bool force_check) const {
  auto schema = node()->maybeSchema();
  // skip memory overlap check for mutable or view ops with only one output
  bool skip_check = !schema ||
      ((schema->is_mutable() || !fn_->checkMemoryOverlap()) &&
       num_outputs_ == 1);
  if (!force_check && skip_check) {
    if (!schema) {
      VLOG(2) << "Detected that op schema is null";
      return true;
    }
    VLOG(2) << "schema->is_mutable: " << schema->is_mutable()
            << ", fn_->checkMemoryOverlap: " << fn_->checkMemoryOverlap()
            << ", num_outputs_: " << num_outputs_;
    return true;
  }

  for (const auto i : c10::irange(inputs_.size())) {
    const IValue* in = &Input(i);
    if (!in->isTensor()) {
      continue;
    }
    const auto& in_t = in->toTensor();
    for (const auto j : c10::irange(num_outputs_)) {
      const IValue& out = Output(j);
      if (!out.isTensor()) {
        continue;
      }
      const auto& out_t = out.toTensor();
      if (!checkNoMemoryOverlap(in_t, out_t)) {
        LOG(INFO) << "Node input " << i << " overlaps with output " << j << ", "
                  << PrintNode(node_);
        LOG(INFO) << *schema;
        return false;
      }
    }
  }
  return true;
}

void ProcessedNode::verify_and_correct_memory_overlap() {
  for (const auto i : c10::irange(inputs_.size())) {
    const IValue& in = Input(i);
    if (!in.isTensor()) {
      continue;
    }
    const auto& in_t = in.toTensor();
    for (const auto j : c10::irange(num_outputs_)) {
      const auto& out_t = Output(j).toTensor();
      if (!checkNoMemoryOverlap(in_t, out_t)) {
        DLOG(INFO) << "Detected alias for node: " << PrintNode(node());
        Output(i) = at::native::clone(out_t, c10::nullopt);
        set_outputs_memory_overlap_detected();
      }
    }
  }
}

} // namespace jit
} // namespace torch
