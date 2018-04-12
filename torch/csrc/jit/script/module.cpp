#include "torch/csrc/jit/script/module.h"
#include "torch/csrc/jit/script/compiler.h"

namespace torch { namespace jit { namespace script {

std::vector<Value*> Method::emit_call_to(Method & callee, ArrayRef<Value*> inputs) {
  JIT_ASSERT(!executor);
  auto fn = callee.graph();
  JIT_ASSERT(inputs.size() == callee.num_inputs());
  std::vector<Value*> all_inputs = inputs;
  // parameters to callee method (which become parameters to _this_ method
  // if they were not already)
  for(at::Tensor* member : callee.member_inputs) {
    all_inputs.push_back(get_or_add_parameter(member));
  }
  return inlineCallTo(*graph(), *callee.graph(), all_inputs);
}

}}}
