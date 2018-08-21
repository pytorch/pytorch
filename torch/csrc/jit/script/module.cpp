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

static FunctionSchema defaultSchemaFor(Method& method) {
  std::vector<Argument> args;
  std::vector<Argument> returns;
  Graph& g = *method.graph();
  size_t num_inputs = method.num_inputs();
  for(size_t i = 0; i < num_inputs; ++i) {
    const Value* v = g.inputs().at(i);
    std::string name = v->hasUniqueName() ? v->uniqueName() : ("argument_"  + std::to_string(i));
    args.push_back({std::move(name), DynamicType::get()});
  }
  for(size_t i = 0; i < g.outputs().size(); ++i) {
    returns.push_back({"", DynamicType::get()});
  }
  return { method.name(), std::move(args), std::move(returns) };
}

std::vector<Value*> Method::emit_call_to(SourceRange loc, Method & callee, ArrayRef<NamedValue> args, ArrayRef<NamedValue> kwargs) {
  JIT_ASSERT(!executor);
  try {
    callee.ensure_defined();
  } catch (RecursiveMethodCallError&) {
    throw ErrorReport(loc) << " method '" << callee.name()
        << "' is called recursively involving this call site. Recursive calls are not supported";
  }
  auto fn = callee.graph();

  std::stringstream failure_messages;
  auto all_inputs = tryMatchSchema(
    callee.schema ? *callee.schema : defaultSchemaFor(callee),
    loc, *graph(), args, kwargs, failure_messages);
  if(!all_inputs)
    throw ErrorReport(loc) << failure_messages.str();

  // parameters to callee method (which become parameters to _this_ method
  // if they were not already)
  for(at::Tensor* member : callee.member_inputs) {
    all_inputs->push_back(get_or_add_parameter(member));
  }
  return inlineCallTo(*graph(), *callee.graph(), *all_inputs);
}

void Method::ensure_defined() {
  if(method_creator) {
    auto creator = method_creator;
    method_creator = placeholderCreator;
    creator(*this);
    method_creator = nullptr;
  }
}

void Module::save(const std::string& filename) {
  ExportModule(*this, filename);
}

}}}
