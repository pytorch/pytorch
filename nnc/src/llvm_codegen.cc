#include "ir_visitor.h"
#include "llvm_jit.h"

using namespace nnc;

class LLVMEmitter : public IRVisitor {
 public:
  void visit(const Add *v) override {
  }

  void visit(const Sub *v) override {
  }

  void visit(const Mul *v) override {
  }

  void visit(const Div *v) override {
  }

  void visit(const IntImm *v) override {
  }

  void visit(const FloatImm *v) override {
  }
};
