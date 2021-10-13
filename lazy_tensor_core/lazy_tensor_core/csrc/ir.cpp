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

Node::Node(OpKind op, size_t num_outputs, torch::lazy::hash_t node_hash,
           torch::lazy::hash_t dag_hash)
    : op_(std::move(op)),
      num_outputs_(num_outputs),
      node_hash_(node_hash),
      dag_hash_(dag_hash) {
  metadata_.scope = GetCurrentScope();
  metadata_.frame_info = GetFrameInfo();
}

Node::Node(OpKind op, size_t num_outputs, torch::lazy::hash_t node_hash)
    : op_(std::move(op)),
      num_outputs_(num_outputs),
      node_hash_(node_hash),
      dag_hash_(node_hash) {
  metadata_.scope = GetCurrentScope();
  metadata_.frame_info = GetFrameInfo();
}

Node::~Node() {}

std::string Node::ToString() const {
  std::stringstream ss;
  ss << op();
  if (num_outputs() > 1) {
    ss << ", num_outputs=" << num_outputs();
  }
  if (!metadata_.scope.empty()) {
    ss << ", scope=" << metadata_.scope;
  }
  EmitShortFrameInfo(ss, metadata_.frame_info);
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
