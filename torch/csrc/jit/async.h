#pragma once

#include "torch/csrc/jit/ivalue.h"
#include "torch/csrc/jit/script/module.h"

namespace torch { namespace jit { namespace async {

struct Future {
  explicit Future(IValue result) : result(result), ready(true) {}
  IValue result;
  bool ready = false;
  IValue get() const {
    JIT_ASSERT(ready);
    return result;
  }
};

Future fork(script::Module &sm, std::vector<IValue> inputs) {
  // TODO: actually fork
  return Future(sm.forward(inputs));
}

IValue wait(Future &fut) {
  // TODO: actually wait
  JIT_ASSERT(fut.ready);
  return fut.get();
}

}}}  // namespace torch::jit::async
