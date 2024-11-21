#include <torch/csrc/lazy/core/config.h>
#include <torch/csrc/lazy/core/debug_util.h>
#include <torch/csrc/lazy/core/ir_metadata.h>
#include <functional>

namespace torch::lazy {

void EmitShortFrameInfo(
    std::ostream& stream,
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

std::ostream& operator<<(
    std::ostream& stream,
    const std::vector<SourceLocation>& frames) {
  stream << "Frames:\n";
  for (auto& location : frames) {
    stream << "  " << location.function << " (" << location.file << ":"
           << location.line << ")\n";
  }
  return stream;
}

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

std::string GetCurrentScope() {
  std::string scope;
  for (auto& scope_entry : g_scope_context.scopes) {
    if (scope.empty()) {
      scope = scope_entry.name;
    } else {
      scope += "/" + scope_entry.name;
    }
  }
  return scope;
}

void PushScope(const std::string& name) {
  size_t id = g_scope_context.next_id;
  g_scope_context.scopes.push_back(
      {c10::str(name, ".", id), g_scope_context.next_id + 1});
  g_scope_context.next_id = 1;
}

void PopScope() {
  TORCH_CHECK(!g_scope_context.scopes.empty());
  g_scope_context.next_id = g_scope_context.scopes.back().saved_next_id;
  g_scope_context.scopes.pop_back();
}

void ResetScopeContext() {
  if (!g_scope_context.scopes.empty()) {
    TORCH_CHECK(
        false, "Expecting scope to be empty but it is " + GetCurrentScope());
  }
  g_scope_context.next_id = 1;
}
} // namespace

ScopePusher::ScopePusher(const std::string& name) {
  PushScope(name);
}

ScopePusher::~ScopePusher() {
  PopScope();
}

void ScopePusher::ResetScopes() {
  ResetScopeContext();
}

MetaData GetMetaDataIfDebugging() {
  if (!FLAGS_torch_lazy_ir_debug) {
    return MetaData();
  }
  MetaData meta;
  meta.scope = GetCurrentScope();
  meta.frame_info = torch::lazy::GetPythonFramesFunction()();
  return meta;
}

} // namespace torch::lazy
