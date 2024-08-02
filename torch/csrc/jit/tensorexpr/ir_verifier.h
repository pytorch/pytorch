#pragma once

#include <torch/csrc/jit/tensorexpr/fwd_decls.h>
#include <torch/csrc/jit/tensorexpr/ir_visitor.h>

namespace torch {
namespace jit {
namespace tensorexpr {

class Expr;
class ExprHandle;
class Mod;
class And;
class Or;
class Xor;
class Lshift;
class Rshift;
class CompareSelect;
class Ramp;
class Load;
class IfThenElse;
class Intrinsics;

class Stmt;
class ExternalCall;
class Store;
class For;
class Block;

class TORCH_API IRVerifier : public IRVisitor {
 public:
  IRVerifier() = default;

  void visit(const ModPtr& v) override;
  void visit(const AndPtr& v) override;
  void visit(const OrPtr& v) override;
  void visit(const XorPtr& v) override;
  void visit(const LshiftPtr& v) override;
  void visit(const RshiftPtr& v) override;
  void visit(const CompareSelectPtr& v) override;
  void visit(const RampPtr& v) override;
  void visit(const LoadPtr& v) override;
  void visit(const IfThenElsePtr& v) override;
  void visit(const IntrinsicsPtr& v) override;

  void visit(const ExternalCallPtr& v) override;
  void visit(const StorePtr& v) override;
  void visit(const ForPtr& v) override;
  void visit(const BlockPtr& v) override;
};

TORCH_API void verify(StmtPtr);
TORCH_API void verify(ExprPtr);
TORCH_API void verify(ExprHandle);

} // namespace tensorexpr
} // namespace jit
} // namespace torch
