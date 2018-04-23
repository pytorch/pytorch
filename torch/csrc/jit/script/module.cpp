#include "torch/csrc/jit/script/module.h"
#include "torch/csrc/jit/script/compiler.h"
#include "torch/csrc/jit/script/error_report.h"

namespace torch { namespace jit { namespace script {


struct RecursiveMethodCallError : public std::exception {};
void placeholderCreator(Method&) {
  throw RecursiveMethodCallError();
}

std::vector<Value*> Method::emit_call_to(SourceRange loc, Method & callee, ArrayRef<Value*> inputs) {
  ensureTensors(loc, inputs);
  JIT_ASSERT(!executor);
  try {
    callee.ensure_defined();
  } catch (RecursiveMethodCallError&) {
    throw ErrorReport(loc) << " method '" << callee.name()
        << "' is called recursively involving this call site. Recursive calls are not supported";
  }
  auto fn = callee.graph();
  ensureSizeMatches(loc, callee.num_inputs(), inputs.size(), "inputs");
  std::vector<Value*> all_inputs = inputs;
  // parameters to callee method (which become parameters to _this_ method
  // if they were not already)
  for(at::Tensor* member : callee.member_inputs) {
    all_inputs.push_back(get_or_add_parameter(member));
  }
  return inlineCallTo(*graph(), *callee.graph(), all_inputs);
}

void Method::ensure_defined() {
  if(method_creator) {
    auto creator = method_creator;
    method_creator = placeholderCreator;
    creator(*this);
    method_creator = nullptr;
  }
}

}}}
