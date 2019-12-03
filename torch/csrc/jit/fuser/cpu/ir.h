#pragma once

#include <c10/util/Exception.h>

#include <iostream>

// #include <torch/csrc/jit/fuser/cpu/kernel.h>

#include <asmjit/asmjit.h>

namespace torch {
namespace jit {
namespace fuser {
namespace cpu {

typedef void (*FusedFunc)();
using namespace asmjit;
using namespace asmjit::x86;

/*
* Registers.
*/

// Registers
enum class RegisterKind {
  Generic
, Abstract
, Pointer
, Scalar
, Constant
};

void printRegisterKind(std::ostream& stream, const RegisterKind kind) {
  if (kind == RegisterKind::Generic) {
    stream << "Generic";
  } else if (kind == RegisterKind::Abstract) {
    stream << "Abstract";
  } else if (kind == RegisterKind::Pointer) {
    stream << "Pointer";
  } else if (kind == RegisterKind::Scalar) {
    stream << "Scalar";
  } else if (kind == RegisterKind::Constant) {
    stream << "Constant";
  }
}

struct Register {
  Register() : name_{-1} { }
  Register(const int _name) : name_{_name} { }

  virtual RegisterKind kind() const {
    return RegisterKind::Generic;
  }

  virtual void print(std::ostream& stream) const {
    stream << "%r" << name_;
  }

  const int name_;
};

struct AbstractRegister : Register {

  AbstractRegister(
    const int _name
  , const Value* const _value)
  : Register{_name}
  , value_{_value} { }

  RegisterKind kind() const override {
    return RegisterKind::Abstract;
  }

  void print(std::ostream& stream) const override {
    stream << "%a" << name_;
  }

  const Value* value_;
};

// TODO: add scalar type to pointer
struct PointerRegister : Register {
  PointerRegister(
    const int _name)
  : Register{_name} { }

  RegisterKind kind() const override {
    return RegisterKind::Pointer;
  }

  void print(std::ostream& stream) const override {
    stream << "%p" << name_;
  }
};

// TODO: add scalar type
struct ScalarRegister : Register {

  ScalarRegister (
    const int _name)
  : Register{_name} { }

  RegisterKind kind() const override {
    return RegisterKind::Scalar;
  }

  void print(std::ostream& stream) const override {
    stream << "%s" << name_;
  }
};

// TODO: add scalar type
struct ConstantRegister : Register {

  ConstantRegister (
    const int _name
  , const float _value)
  : Register{_name}
  , value_{_value} { }

  RegisterKind kind() const override {
    return RegisterKind::Constant;
  }

  void print(std::ostream& stream) const override {
    stream << "%c" << name_ << "=" << value_;
  }

  const float value_;
};

/*
* Kernels and kernel snippets.
*/

enum class SnippetKind {
  Generic
, LoopNest
, Loop
};

struct KernelSnippet {

  KernelSnippet(
    const int _name
  , Label _label
  , Compiler& _cc)
  : name_{_name}
  , label_{_label}
  , cc_{_cc} { }

  virtual SnippetKind kind() const {
    return SnippetKind::Generic;
  }

  virtual void print(std::ostream& stream) const {
    stream << "KernelSnippet{}";
  }

  virtual void lower() const;

  Label& label() { return label_; }
  const Label& label() const { return label_; }

  const int name_;
  Label label_;
  Compiler& cc_;
};

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

// start, stop, inc
// inc_snippet, body_snippet, exit_jmp
// TODO: handle loops that require 64-bit indexing
struct Loop : KernelSnippet {

  Loop(
    const int _name
  , Label _label
  , Compiler& _cc
  , const int _start
  , const int _stop
  , const int _inc)
  : KernelSnippet{_name, _label, _cc}
  , start_{_start}
  , stop_{_stop}
  , inc_{_inc} { }

  virtual SnippetKind kind() const {
    return SnippetKind::Generic;
  }

  virtual void print(std::ostream& stream) const {
    stream << "KernelSnippet{}";
  }

  void lower() const override {
    auto& cc = cc_;

    Mem cStart = cc.newInt32Const(ConstPool::kScopeLocal, start_);
    Mem cStop = cc.newInt32Const(ConstPool::kScopeLocal, stop_ - 1);
    Mem cInc = cc.newInt32Const(ConstPool::kScopeLocal, inc_);

    Gp loop_counter = cc.newGpd();

    cc.bind(label());
    cc.xor_(loop_counter, loop_counter);
    cc.jmp(body_->label());

    cc.bind(counters_->label());
    cc.add(loop_counter, cInc);

    cc.bind(body_->label());
    body_->lower();

    cc.cmp(loop_counter, cStop);
    cc.jne(counters_->label());
  }

  int start_;
  int stop_;
  int inc_;

  KernelSnippet* counters_ = nullptr;
  KernelSnippet* body_ = nullptr;
};

// // Holds loops
// struct LoopNest : KernelSnippet {

//   LoopNest(
//     Kernel& _k)
//   : KernelSnippet{_k} { }

//   std::vector<Loop> loops;
// };



// enum class FusionNodeKind {
//   GenericFusionNode
// , UnpackPointer
// , UnpackScalar
// , Loop
// , Add
// , Move
// };

// struct FusionNode {
//   FusionNode() { }

//   virtual FusionNodeKind kind() const {
//     return FusionNodeKind::GenericFusionNode;
//   }

//   virtual void print(std::ostream& stream) const {
//     stream << "FusionNode{}";
//   }

//   template <typename T>
//   T* expect() {
//     TORCH_CHECK(T::Kind == kind(), "Expected a different kind!");
//     return static_cast<T*>(this);
//   }
// };

// struct Kernel {
//   void print(std::ostream& stream) const {
//     stream << "Kernel{" << std::endl;
//     for (const auto* node : nodes_) {
//       node->print(stream);
//       stream << std::endl;
//     }
//     stream << "}" << std::endl;
//   }

//   void addNode(FusionNode* node) {
//     nodes_.push_back(node);
//   }

//   int getRegisterName() { return register_name_counter_++; }

//   std::vector<FusionNode*> nodes_;
//   int register_name_counter_ = 0;
// };

// // Register defintions

// // Fusion Node definitions

// struct UnpackPointer : FusionNode {

//   UnpackPointer(
//     const PointerRegister* const _dst
//   , const int _unpack_from)
//   : dst_{_dst}
//   , unpack_from_{_unpack_from} { }

//   FusionNodeKind kind() const override {
//     return FusionNodeKind::UnpackPointer;
//   }

//   void print(std::ostream& stream) const override {
//     stream << "UnpackPointer*{";
//     stream << "dst: ";
//     dst_->print(stream);
//     stream << ", unpack_from: " << unpack_from_;
//     stream << "}";
//   }

//   const PointerRegister* dst_;
//   int unpack_from_;
// };

// struct UnpackScalar : FusionNode {

//   UnpackScalar(
//     const ScalarRegister* const _dst
//   , const int _unpack_from)
//   : dst_{_dst}
//   , unpack_from_{_unpack_from} { }

//   FusionNodeKind kind() const override {
//     return FusionNodeKind::UnpackScalar;
//   }

//   void print(std::ostream& stream) const override {
//     stream << "UnpackScalar*{";
//     stream << "dst: ";
//     dst_->print(stream);
//     stream << ", unpack_from: " << unpack_from_;
//     stream << "}";
//   }

//   const ScalarRegister* dst_;
//   int unpack_from_;
// };

// struct Loop : FusionNode {

//   Loop(
//     const int _start
//   , const int _end
//   , const int _inc)
//   : start_{_start}
//   , end_{_end}
//   , inc_{_inc} { }

//   FusionNodeKind kind() const override {
//     return FusionNodeKind::Loop;
//   }

//   void print(std::ostream& stream) const override {
//     stream << "Loop*{";
//     stream << "start: " << start_;
//     stream << ", end: " << end_;
//     stream << ", inc: " << inc_;
//     stream << "}";
//   }

//   int start_;
//   int end_;
//   int inc_;
// };

// struct Add : FusionNode {

//   Add(
//     const Register* const _dst
//   , const Register* const _lhs
//   , const Register* const _rhs)
//   : dst_{_dst}
//   , lhs_{_lhs}
//   , rhs_{_rhs} { }

//   FusionNodeKind kind() const override {
//     return FusionNodeKind::Add;
//   }

//   void print(std::ostream& stream) const override {
//     stream << "Add{";
//     stream << "dst: ";
//     dst_->print(stream);
//     stream << ", lhs: ";
//     lhs_->print(stream);
//     stream << ", rhs: ";
//     rhs_->print(stream);
//     stream << "}";
//   }

//   const Register* dst_;
//   const Register* lhs_;
//   const Register* rhs_;
// };

// struct Move : FusionNode {

//   Move(
//     const Register* const _dst
//   , const Register* const _src)
//   : dst_{_dst}
//   , src_{_src} { }

//   FusionNodeKind kind() const override {
//     return FusionNodeKind::Move;
//   }

//   void print(std::ostream& stream) const override {
//     stream << "Move{";
//     stream << "dst: ";
//     dst_->print(stream);
//     stream << ", src: ";
//     src_->print(stream);
//     stream << "}";
//   }

//   const Register* dst_;
//   const Register* src_;
// };

}}}} // namespace torch::jit::fuser::cpu
