#include "lazy_tensor_core/csrc/ir.h"

#include <functional>
#include <sstream>

#include "lazy_tensors/computation_client/cache.h"
#include "lazy_tensors/computation_client/debug_macros.h"
#include "lazy_tensors/computation_client/sys_util.h"
#include "lazy_tensors/computation_client/util.h"
#include "lazy_tensors/str_cat.h"

namespace torch_lazy_tensors {
namespace ir {
namespace {

struct ScopeEntry {
  std::string name;
  size_t saved_next_id = 1;
};

struct ScopeContext {
  std::vector<ScopeEntry> scopes;
  size_t next_id = 1;
};

thread_local ScopeContext g_scope_context;

void PushScope(const std::string& name) {
  size_t id = g_scope_context.next_id;
  g_scope_context.scopes.push_back(
      {lazy_tensors::StrCat(name, ".", id), g_scope_context.next_id + 1});
  g_scope_context.next_id = 1;
}

void PopScope() {
  LTC_CHECK(!g_scope_context.scopes.empty());
  g_scope_context.next_id = g_scope_context.scopes.back().saved_next_id;
  g_scope_context.scopes.pop_back();
}

void ResetScopeContext() {
  LTC_CHECK_EQ(g_scope_context.scopes.size(), 0);
  g_scope_context.next_id = 1;
}

std::string GetCurrentScope() {
  std::string scope;
  for (auto& scope_entry : g_scope_context.scopes) {
    if (scope.empty()) {
      lazy_tensors::StrAppend(&scope, scope_entry.name);
    } else {
      lazy_tensors::StrAppend(&scope, "/", scope_entry.name);
    }
  }
  return scope;
}

}  // namespace

void EmitShortFrameInfo(std::ostream& stream,
                        const std::vector<SourceLocation>& frames) {
  if (!frames.empty()) {
    const SourceLocation& frame = frames.front();
    std::string::size_type pos = frame.file.find_last_of('/');
    if (pos == std::string::npos) {
      pos = 0;
    } else {
      ++pos;
    }
    stream << ", location=" << frame.function << "@" << frame.file.substr(pos)
           << ":" << frame.line;
  }
}

bool Use::operator<(const Use& rhs) const {
  if (node->op() != rhs.node->op()) {
    return node->op() < rhs.node->op();
  }
  if (operand_index != rhs.operand_index) {
    return operand_index < rhs.operand_index;
  }
  return index < rhs.index;
}

std::string Use::ToString() const {
  std::stringstream ss;
  ss << node->ToString() << ", operand_index=" << operand_index
     << ", index=" << index;
  return ss.str();
}

size_t Output::Hasher::operator()(const Output& output) const {
  return torch::lazy::StdHashCombine(
      reinterpret_cast<std::ptrdiff_t>(output.node), output.index);
}

torch::lazy::hash_t Output::hash() const {
  return torch::lazy::HashCombine(node->hash(), index);
}

std::string Output::ToString() const {
  std::stringstream ss;
  ss << node->ToString() << ", index=" << index;
  return ss.str();
}

torch::lazy::hash_t Value::hash() const {
  return torch::lazy::HashCombine(node->hash(), index);
}

OpKind OpKind::Get(const std::string& name) {
  return OpKind(c10::Symbol::fromQualString(name));
}

torch::lazy::hash_t OpKind::hash() const {
  return torch::lazy::StringHash(op.toQualString());
}

Node::Node(OpKind op, OpList operands,
           const std::vector<at::ScalarType>& at_dtypes,
           const std::vector<std::vector<int64_t>>& at_shapes,
           size_t num_outputs,
           torch::lazy::hash_t node_hash, torch::lazy::hash_t dag_hash)
    : op_(std::move(op)),
      at_dtypes_(at_dtypes),
      at_shapes_(at_shapes),
      num_outputs_(num_outputs),
      node_hash_(node_hash),
      dag_hash_(dag_hash) {
  LTC_CHECK_EQ(at_dtypes.size(), at_shapes.size());
  metadata_.scope = GetCurrentScope();
  metadata_.frame_info = GetFrameInfo();
  for (auto& operand : operands) {
    // Ideally, optional operands should be filtered by the leaf node classes,
    // but it's just much easier to do it here.
    // TODO(alanwaketan): Find a way to move the below logic to the leaf node
    // classes.
    if (!operand) {
      continue;
    }

    AddOperand(operand.node, operand.index);
  }
}

Node::Node(OpKind op, const std::vector<at::ScalarType>& at_dtypes,
           const std::vector<std::vector<int64_t>>& at_shapes,
           size_t num_outputs,
           torch::lazy::hash_t node_hash)
    : op_(std::move(op)),
      at_dtypes_(at_dtypes),
      at_shapes_(at_shapes),
      num_outputs_(num_outputs),
      node_hash_(node_hash),
      dag_hash_(node_hash) {
  LTC_CHECK_EQ(at_dtypes.size(), at_shapes.size());
  metadata_.scope = GetCurrentScope();
  metadata_.frame_info = GetFrameInfo();
}

Node::~Node() {
  for (size_t i = 0; i < operands_as_outputs_.size(); ++i) {
    operands_[i]->RemoveUse(Use(this, i, operands_as_outputs_[i].index));
  }
}

void Node::AddOperand(NodePtr node, size_t index) {
  LTC_CHECK_LT(index, node->num_outputs());
  operands_.push_back(std::move(node));
  operands_as_outputs_.push_back(Output(operands_.back().get(), index));
  operands_.back()->AddUse(Use(this, operands_.size() - 1, index));
}

void Node::ReplaceOperand(size_t operand_no, NodePtr node, size_t index) {
  LTC_CHECK_LT(index, node->num_outputs());
  Output* output = &operands_as_outputs_.at(operand_no);
  operands_[operand_no]->RemoveUse(Use(this, operand_no, output->index));
  node->AddUse(Use(this, operand_no, index));
  *output = Output(node.get(), index);
  operands_[operand_no] = std::move(node);
}

void Node::ReplaceAllUsesWith(NodePtr node, size_t index) {
  // A call to ReplaceOperand() will end up calling RemoveUse() into the
  // current node, so snapshot the current uses and iterate over them.
  std::vector<Use> current_uses(uses_.begin(), uses_.end());
  for (auto& use : current_uses) {
    use.node->ReplaceOperand(use.operand_index, node, index);
  }
}

std::string Node::ToString() const {
  std::stringstream ss;
  ss << "TODO reimplement Node::ToString with aten shape in base class, for "
        "now it's solely implemented in TsNode";
  // ss << shape() << " " << op();
  // if (num_outputs() > 1) {
  //   ss << ", num_outputs=" << num_outputs();
  // }
  // if (!metadata_.scope.empty()) {
  //   ss << ", scope=" << metadata_.scope;
  // }
  // EmitShortFrameInfo(ss, metadata_.frame_info);
  return ss.str();
}

NodePtr Node::Clone(OpList operands) const {
  LTC_ERROR() << "Cloning not implemented for node: " << *this;
}

std::vector<SourceLocation> Node::GetFrameInfo() {
  // At the time of writing, retrieving Python frames costs from 1us up to 20us.
  // This per IR Node. Since it is not unreasonable to have a many hundreds of
  // IR Node, this can be a multi-millisecond cost, which is not negligible.
  static bool wants_frames =
      lazy_tensors::sys_util::GetEnvBool("LTC_IR_DEBUG", false);
  return wants_frames ? GetPythonFrames() : std::vector<SourceLocation>();
}

ScopePusher::ScopePusher(const std::string& name) { PushScope(name); }

ScopePusher::~ScopePusher() { PopScope(); }

void ScopePusher::ResetScopes() { ResetScopeContext(); }

}  // namespace ir
}  // namespace torch_lazy_tensors
