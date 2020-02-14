#include "torch/csrc/jit/tensorexpr/asmjit_codegen.h"
#include "torch/csrc/jit/tensorexpr/ir.h"

#include <cassert>
#include <memory>

namespace torch {
namespace jit {
namespace tensorexpr {

static void dumpCode(asmjit::BaseBuilder& cb, const char* phase) {
  asmjit::String sb;
  cb.dump(sb);
  printf("%s:\n%s\n", phase, sb.data());
}

using GPD = asmjit::x86::Gpd;

ASMJITCodeGen::ASMJITCodeGen() {
  jit_.reset(new asmjit::JitRuntime());
  code_.reset(new asmjit::CodeHolder());
  code_->init(jit_->codeInfo());
  cc_.reset(new asmjit::x86::Compiler(code_.get()));

  cc_->addFunc(asmjit::FuncSignatureT<int>());
}

void ASMJITCodeGen::visit(const Add* v) {
  v->lhs().accept(this);
  auto lhs = this->value_.as<GPD>();
  v->rhs().accept(this);
  auto rhs = this->value_.as<GPD>();

  value_ = cc_->newGpd("add_val");
  cc_->lea(value_.as<GPD>(), asmjit::x86::ptr(lhs, rhs));
}

void ASMJITCodeGen::visit(const Sub* v) {
  v->lhs().accept(this);
  auto lhs = this->value_.as<GPD>();
  v->rhs().accept(this);
  auto rhs = this->value_.as<GPD>();

  value_ = cc_->newGpd("sub_val");
  cc_->mov(value_.as<GPD>(), lhs);
  cc_->sub(value_.as<GPD>(), rhs);
}

void ASMJITCodeGen::visit(const Mul* v) {
  v->lhs().accept(this);
  auto lhs = this->value_.as<GPD>();
  v->rhs().accept(this);
  auto rhs = this->value_.as<GPD>();

  value_ = cc_->newGpd("mul_val");
  cc_->mov(value_.as<GPD>(), lhs);
  cc_->imul(value_.as<GPD>(), rhs);
}

void ASMJITCodeGen::visit(const Div* v) {
  v->lhs().accept(this);
  auto lhs = this->value_.as<GPD>();
  v->rhs().accept(this);
  auto rhs = this->value_.as<GPD>();

  value_ = asmjit::x86::eax;
  cc_->mov(value_.as<GPD>(), lhs);

  cc_->mov(asmjit::x86::edx, 0);
  cc_->idiv(asmjit::x86::edx, value_.as<GPD>(), rhs);
}

void ASMJITCodeGen::visit(const IntImm* v) {
  asmjit::x86::Mem const_mem =
      cc_->newInt32Const(asmjit::ConstPool::kScopeGlobal, v->value());

  value_ = cc_->newGpd("const");
  cc_->mov(value_.as<GPD>(), const_mem);
}

void ASMJITCodeGen::visit(const FloatImm* v) {
  assert(false && "Integer only now sorry");
}

int ASMJITCodeGen::value() {
  cc_->ret(value_);
  cc_->endFunc();
  cc_->finalize();

  typedef int (*Func)(void);

  Func fn;
  asmjit::Error err = jit_->add(&fn, code_.get());
  if (err) {
    std::stringstream ss;
    ss << "asmjit encountered error " << err;
    throw std::runtime_error(ss.str());
  }
  return fn();
}

} // namespace tensorexpr
} // namespace jit
} // namespace torch
