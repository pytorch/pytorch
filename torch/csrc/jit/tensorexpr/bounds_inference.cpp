#include <torch/csrc/jit/tensorexpr/bounds_inference.h>
#include <torch/csrc/jit/tensorexpr/stmt.h>
#include <torch/csrc/jit/tensorexpr/expr.h>
#include <torch/csrc/jit/tensorexpr/ir.h>
#include <torch/csrc/jit/tensorexpr/ir_printer.h>
#include <torch/csrc/jit/tensorexpr/ir_mutator.h>
#include <torch/csrc/jit/tensorexpr/eval.h>

namespace torch {
namespace jit {
namespace tensorexpr {

/*
 * TODO:
 * [ ] Save found accesses in a vector
 * [ ] Run simplification in the beginning
 * [ ] Compute accesses at an outer scope (replace index with its min/max)
 * [ ] Implement merging
 * [ ] Implement compute_at
 * [ ] Take into account how index is used when replacing (if with '-', then it
 * should be replaced with max/min instead of min/max)
 * [ ] Generally handle signed multiplication
 */

class AccessFinder : public IRVisitor {
 public:
  AccessFinder() {}

  void visit(const Load* v) override {
    std::cerr << "Load:" << *v << "\n";
  };
  void visit(const Store* v) override {
    std::cerr << "Store:" << *v << "\n";
    IRVisitor::visit(v);
  };
};


static void printBufVector(const std::unordered_map<const Var*, Range>& v) {
  std::cerr << "Access vector {\n";
  for (const auto& b : v) {
    std::cerr << *b.first << " in (" << *b.second.start() << "; " << *b.second.stop() << ")\n";
  }
  std::cerr << "}\n";
}

std::unordered_map<const Var*, Range> BoundsInference::inferBoundsForLoop(For *f) {
  std::cerr << "Analyzing For loop:\n" << *f << "\n";
  auto res = inferBoundsForBlock(f->body());
//   body_inner = Substitute(body_inner, {{f->var(), combined_index}});
//   ConstantFolder constant_folder;
  for (const auto& p : res) {
    const Expr* old_start = p.second.start();
    const Expr* old_stop = p.second.stop();
//     const Expr* new_start = Substitute(old_start, {{f->var(),f->start()}})->accept_mutator(&constant_folder);
//     const Expr* new_stop = Substitute(old_stop, {{f->var(),new Sub(f->stop(), new IntImm(1))}})->accept_mutator(&constant_folder);
//     const Expr* new_start = Substitute(&a, {{f->var(),f->start()}});
//     const Expr* new_stop = Substitute(&b, {{f->var(),f->stop()}});
//
    const Expr* new_start = Substitute(old_start, {{f->var(), f->start()}});
    const Expr* new_stop = Substitute(old_stop, {{f->var(),new Sub(f->stop(), new IntImm(1))}});
    res[p.first] = Range(new_start, new_stop);
  }
  std::cerr << "Analyzed For loop:\n" << *f << "\n";
  printBufVector(res);
  return res;
}


std::unordered_map<const Var*, Range> BoundsInference::inferBoundsForBlock(Block *b) {
  std::cerr << "Analyzing block:\n" << *b << "\n";
  std::unordered_map<const Var*, Range> res;
  for (auto s : b->stmts()) {
    std::unordered_map<const Var*, Range> stmt_bufs;;
    if (auto* f = dynamic_cast<For*>(s)) {
      stmt_bufs = inferBoundsForLoop(f);
    } else if (auto* st = dynamic_cast<Store*>(s)) {
      stmt_bufs = inferBoundsForStore(st);
    } else {
      std::cerr << "Not analyzing stmt:\n" << *s << "\n";
    }
    res = mergeBufVectors(res, stmt_bufs);
  }
  std::cerr << "Analyzed block:\n" << *b << "\nResult:\n";
  printBufVector(res);
  return res;
}

std::unordered_map<const Var*, Range> BoundsInference::inferBoundsForStore(Store *st) {
  std::cerr << "Analyzing Store:\n" << *st << "\n";
  std::unordered_map<const Var*, Range> res;
  res[st->base_handle()] = Range(st->index(), new Add(st->index(), new IntImm(1)));
  AccessFinder ac;
  st->accept(&ac);
  return res;
}

std::unordered_map<const Var*, Range> BoundsInference::mergeBufVectors(
    std::unordered_map<const Var*, Range> a,
    std::unordered_map<const Var*, Range> b) {
  std::unordered_map<const Var*, Range> res(a);
  for (const auto& p : b) {
    res[p.first] = p.second;
  }
  return res;
}

std::unordered_map<const Var*, Range> inferBounds(Stmt* s) {
  BoundsInference bi;
  std::cerr << "Given stmt:\n" << *s << "\n";
  if (auto *b = dynamic_cast<Block*>(s)) {
    return bi.inferBoundsForBlock(b);
  }
  auto *f = dynamic_cast<For*>(s);
  CHECK(f);
  return bi.inferBoundsForLoop(f);
}

} // namespace tensorexpr
} // namespace jit
} // namespace torch

