#include <torch/csrc/jit/script/error_report.h>
#include <c10/util/Optional.h>
#include <torch/csrc/jit/script/tree.h>
#include <torch/csrc/utils/memory.h>

namespace torch {
namespace jit {
namespace script {


struct Call {
  std::string fn_name;
  std::unique_ptr<SourceRange> caller_range;
};

thread_local std::unique_ptr<SourceRange> pending_range;
thread_local std::vector<Call> calls;

void ErrorReport::CallStack::update_pending_range(const SourceRange& range) {
  pending_range = torch::make_unique<SourceRange>(range);
}

void ErrorReport::CallStack::push_function(const std::string& name) {
  if (pending_range != nullptr) {
    calls.push_back({name, std::move(pending_range)});
    pending_range = nullptr;
  } else {
    calls.push_back({name, nullptr});
  }
}

void ErrorReport::CallStack::pop_function() {
  calls.pop_back();
}

const char* ErrorReport::what() const noexcept {
  std::stringstream msg;
  msg << "\n" << ss.str();
  if (context) {
    msg << ":\n";
    context->highlight(msg);
  } else {
    msg << ".\n";
  }

  if (calls.size() > 0) {
    for (auto it = calls.rbegin(); it != calls.rend() - 1; ++it) {
      msg << "'" << it->fn_name
          << "' is being compiled since it was called from '"
          << (it + 1)->fn_name << "'\n";
      if (it->caller_range == nullptr) {
        msg << "<no range>\n";
      } else {
        it->caller_range->highlight(msg);
      }
    }
  }

  the_message = msg.str();
  return the_message.c_str();
}

} // namespace script
} // namespace jit
} // namespace torch
