#pragma once

#include <torch/csrc/jit/fuser/cpu/ir.h>

#include <c10/util/Exception.h>

#include <asmjit/asmjit.h>

namespace torch {
namespace jit {
namespace fuser {
namespace cpu {

typedef void (*FusedFunc)();
using namespace asmjit;
using namespace asmjit::x86;

struct Kernel {

  Kernel(
    Compiler& _cc
  , FusedFunc& _fn
  , JitRuntime& _rt)
  : cc_{_cc}
  , fn_{_fn}
  , rt_{_rt} { }

  void print(std::ostream& stream) const {
    stream << "Kernel{}";
  }

  int getRegisterName() { return register_name_counter_++; }
  int getSnippetName() { return snippet_name_counter_++; }
  Compiler& cc() { return cc_; }

  FusedFunc& compile () {
    CodeHolder code;
    code.init(rt_.codeInfo());

    // TODO: emit preamble unpack

    for (const auto& snippet : snippets_) {
      snippet.lower();
    }

    // emit return (postamble)
    auto& cc = cc_;
    cc.ret();

    // compile
    Error err = rt_.add(&fn_, &code);
    TORCH_CHECK(!err, "Error while jitting!");
    return fn_;
  }

  Compiler& cc_;
  FusedFunc& fn_;
  JitRuntime& rt_;
  int register_name_counter_ = 0;
  int snippet_name_counter_ = 0;
  std::vector<KernelSnippet> snippets_;
};


}}}} // namespace torch::jit::fuser::cpu
