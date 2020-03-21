#pragma once

#include <vector>
#include <unordered_map>

#include <torch/csrc/WindowsTorchApiMacro.h>

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
   Range() {};
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
  Range range;
};


TORCH_API std::unordered_map<const Var*, Range> inferBounds(Stmt* s);

class TORCH_API BoundsInference {
 public:
  std::unordered_map<const Var*, Range> inferBoundsForLoop(For* f);
  std::unordered_map<const Var*, Range> inferBoundsForBlock(Block* b);
  std::unordered_map<const Var*, Range> inferBoundsForStore(Store* st);
  std::unordered_map<const Var*, Range> mergeBufVectors(
      std::unordered_map<const Var*, Range> a,
      std::unordered_map<const Var*, Range> b);
};

} // namespace tensorexpr
} // namespace jit
} // namespace torch

