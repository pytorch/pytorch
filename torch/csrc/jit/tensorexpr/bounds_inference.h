#pragma once

#include <unordered_map>
#include <vector>

#include <torch/csrc/WindowsTorchApiMacro.h>
#include <torch/csrc/jit/tensorexpr/ir_visitor.h>

namespace torch {
namespace jit {
namespace tensorexpr {

class Expr;
class Var;
class Stmt;
class For;
class Store;
class Block;

enum TensorAccessKind { kLoad, kStore };

// represent a range [start, stop)
class Range {
 public:
  Range(){};
  Range(const Expr* start, const Expr* stop) : start_(start), stop_(stop) {}
  const Expr* start() const {
    return start_;
  }
  const Expr* stop() const {
    return stop_;
  }

 private:
  const Expr* start_;
  const Expr* stop_;
};

struct TensorAccess {
  const Var* var;
  TensorAccessKind kind;
  std::vector<const Expr*> start;
  std::vector<const Expr*> stop;
};

TORCH_API std::unordered_map<const Var*, Range> inferBounds(Stmt* s);

class TORCH_API BoundsInference {
 public:
  std::vector<TensorAccess> inferBoundsForLoop(For* f);
  std::vector<TensorAccess> inferBoundsForBlock(Block* b);
  std::vector<TensorAccess> inferBoundsForStore(Store* st);
  std::vector<TensorAccess> mergeBufVectors(
      std::vector<TensorAccess> a,
      std::vector<TensorAccess> b);
  std::unordered_map<Stmt*, std::vector<TensorAccess>> accesses;
};

class AccessFinder : public IRVisitor {
 public:
  AccessFinder() {}

  void visit(const FunctionCall* v);
  void visit(const Load* v) override;
  void visit(const Store* v) override;
  std::vector<TensorAccess> accesses;
};

// TODO: remove/cleanup this
TORCH_API void printBufVector(const std::vector<TensorAccess>& v);

} // namespace tensorexpr
} // namespace jit
} // namespace torch
