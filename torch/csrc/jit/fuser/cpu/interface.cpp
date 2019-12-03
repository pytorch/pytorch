#include <torch/csrc/jit/fuser/interface.h>
#include <torch/csrc/jit/fuser/cpu/interface.h>
#include <torch/csrc/jit/fuser/cpu/ir.h>
#include <torch/csrc/jit/fuser/common/utils.h>
#include <torch/csrc/jit/fuser/common/tensor_meta.h>
#include <torch/csrc/jit/fuser/common/management.h>

#include <c10/util/Exception.h>

#include <asmjit/asmjit.h>

#include <iostream>
#include <unordered_map>
#include <unordered_set>
#include <functional>

namespace torch {
namespace jit {
namespace fuser {
namespace cpu {

using namespace torch::jit::fuser;

static JitRuntime rt;
typedef int (*FusionFn)(void);
typedef void (*FusionWrapper)(void);

const std::unordered_set<Symbol> fusibleNodes{
  aten::add
};

bool isFusibleOnCPU(const Node* const node) {
  const auto it = fusibleNodes.find(node->kind());
  if (it == fusibleNodes.end()) {
    return false;
  }

  return true;
}

int fuseOnCPU(const Node* const node) {
  TORCH_CHECK(isFusibleOnCPU(node), "Trying to fuse nonfusible node!");

  return getAndIncrementGlobalFusionCounter();
}

void executeFusion(FusionFn fn) {
  const auto result = fn();
}

void compileFusionOnCPU(Node* fusion) {
  CodeHolder code;
  code.init(rt.codeInfo());
  Compiler cc{&code};
  cc.addFunc(FuncSignatureT<int>());      // Begin a function of `int fn(void)` signature.

  Gp vReg = cc.newGpd();                  // Create a 32-bit general purpose register.
  cc.mov(vReg, 1);                        // Move one to our virtual register `vReg`.
  cc.ret(vReg);                           // Return `vReg` from the function.

  cc.endFunc();                           // End of the function body.
  cc.finalize();                          // Translate and assemble the whole `cc` content.

  FusionFn fn;
  Error err = rt.add(&fn, &code);
  TORCH_CHECK(err == 0, "Error jitting on CPU!");

  // Sets (std::)function pointer
  auto wrapped_fn = [fn](){
    executeFusion(fn);
  };
  auto* p = new std::function<void()>(wrapped_fn);
  fusion->v_(attr::value, static_cast<void*>(p));
}

void callFusionOnCPU(
  const Node* const fusion
, std::vector<at::Tensor>& outputs
, at::ArrayRef<IValue> inputs) {
  auto* p = static_cast<std::function<void()>*>(fusion->v(attr::value));
  (*p)();
}


}}}} // namespace torch::jit::fuser::cpu
