#include "torch/csrc/jit/assertions.h"
#include "torch/csrc/jit/script/module.h"
#include "torch/csrc/jit/script/compiler.h"
#include "torch/csrc/jit/script/error_report.h"
#include "torch/csrc/jit/export.h"
#include "torch/csrc/jit/operator.h"

namespace torch { namespace jit { namespace script {


struct RecursiveMethodCallError : public std::exception {};
void placeholderCreator(Method&) {
  throw RecursiveMethodCallError();
}

c10::optional<std::vector<Value*>> try_emit_call_to(
    Graph& graph,
    SourceRange loc,
    Method& callee,
    c10::optional<NamedValue> self,
    ArrayRef<NamedValue> args,
    ArrayRef<NamedValue> kwargs,
    std::stringstream& failure_messages,
    Method* caller,
    bool conv_tensors_to_nums) {
  try {
    callee.ensure_defined();
  } catch (RecursiveMethodCallError&) {
    throw ErrorReport(loc) << " method '" << callee.name()
        << "' is called recursively involving this call site. Recursive calls are not supported";
  }
  auto fn = callee.graph();

  auto matched_schema = tryMatchSchema(
    callee.getSchema(),
    loc, graph, self, args, kwargs, failure_messages, conv_tensors_to_nums);
  if(!matched_schema)
    return c10::nullopt;

  // parameters to callee method (which become parameters to _this_ method
  // if they were not already)
  for(at::Tensor* member : callee.params()) {
    if(!caller) {
      throw ErrorReport(loc) << " attempting to call a method with parameters from a raw graph. File a bug report";
    }
    matched_schema->inputs.push_back(caller->get_or_add_parameter(member));
  }
  return inlineCallTo(graph, *callee.graph(), matched_schema->inputs);
}

std::vector<Value*> Method::emit_call_to(SourceRange loc, Method & callee, ArrayRef<NamedValue> args, ArrayRef<NamedValue> kwargs) {
  JIT_ASSERT(!executor);
  std::stringstream failure_messages;
  if (auto result = try_emit_call_to(
          *graph(),
          loc,
          callee,
          c10::nullopt,
          args,
          kwargs,
          failure_messages,
          this,
          /*conv_tensors_to_nums=*/true)) {
    return *result;
  }
  throw ErrorReport(loc) << failure_messages.str();
}

void Method::ensure_defined() {
  if(method_creator) {
    auto creator = method_creator;
    method_creator = placeholderCreator;
    creator(*this);
    method_creator = nullptr;
  }
}

void Module::save(std::ostream& out) {
  ExportModule(*this, out);
}

void Module::save(const std::string& filename) {
  ExportModule(*this, filename);
}

}}}
