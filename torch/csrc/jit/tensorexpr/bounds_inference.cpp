#include <torch/csrc/jit/tensorexpr/bounds_inference.h>
#include <torch/csrc/jit/tensorexpr/eval.h>
#include <torch/csrc/jit/tensorexpr/expr.h>
#include <torch/csrc/jit/tensorexpr/ir.h>
#include <torch/csrc/jit/tensorexpr/ir_mutator.h>
#include <torch/csrc/jit/tensorexpr/ir_printer.h>
#include <torch/csrc/jit/tensorexpr/ir_simplifier.h>
#include <torch/csrc/jit/tensorexpr/stmt.h>

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
    accesses.push_back({v->base_handle(), kLoad, v->index(), v->index()});
  };
  void visit(const Store* v) override {
    std::cerr << "Store:" << *v << "\n";
    accesses.push_back({v->base_handle(), kStore, v->index(), v->index()});
    IRVisitor::visit(v);
  };
  std::vector<TensorAccess> accesses;
};


static void printBufVector(const std::vector<TensorAccess>& v) {
  std::cerr << "Access vector {\n";
  for (const auto& b : v) {
    std::cerr << *b.var << " in (" << *b.start << "; " << *b.stop << ")\n";
  }
  std::cerr << "}\n";
}

std::vector<TensorAccess> BoundsInference::inferBoundsForLoop(For *f) {
  std::cerr << "Analyzing For loop:\n" << *f << "\n";
  auto res = inferBoundsForBlock(f->body());
//   body_inner = Substitute(body_inner, {{f->var(), combined_index}});
//   ConstantFolder constant_folder;
  for (size_t i = 0; i < res.size(); i++) {
    const Expr* old_start = res[i].start;
    const Expr* old_stop = res[i].stop;
//     const Expr* new_start = Substitute(old_start, {{f->var(),f->start()}})->accept_mutator(&constant_folder);
//     const Expr* new_stop = Substitute(old_stop, {{f->var(),new Sub(f->stop(), new IntImm(1))}})->accept_mutator(&constant_folder);
//
    const Expr* new_start = Substitute(old_start, {{f->var(), f->start()}});
    const Expr* new_stop = Substitute(old_stop, {{f->var(),new Sub(f->stop(), new IntImm(1))}});
    res[i].start = IRSimplifier::simplify(new_start);
    res[i].stop = IRSimplifier::simplify(new_stop);
//     res[p.first] = Range(new_start, new_stop);
  }
  std::cerr << "Analyzed For loop:\n" << *f << "\n";
  printBufVector(res);
  return res;
}


std::vector<TensorAccess> BoundsInference::inferBoundsForBlock(Block *b) {
  std::cerr << "Analyzing block:\n" << *b << "\n";
  std::vector<TensorAccess> res;
  for (auto s : b->stmts()) {
    std::vector<TensorAccess> stmt_bufs;;
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

std::vector<TensorAccess> BoundsInference::inferBoundsForStore(Store *st) {
  std::cerr << "Analyzing Store:\n" << *st << "\n";
  std::vector<TensorAccess> res;
//   res[st->base_handle()] = Range(st->index(), new Add(st->index(), new IntImm(1)));
  AccessFinder ac;
  st->accept(&ac);
  res = ac.accesses;
  printBufVector(res);
  return res;
}

std::vector<TensorAccess> BoundsInference::mergeBufVectors(
    std::vector<TensorAccess> a,
    std::vector<TensorAccess> b) {
  std::vector<TensorAccess> res(a);
  res.insert(a.end(), b.begin(), b.end());
  for (const auto& p : b) {
//     res[p.first] = p.second;
  }
  return res;
}

std::unordered_map<const Var*, Range> convert(
    const std::vector<TensorAccess>& v) {
  std::unordered_map<const Var*, Range> r;
  for (auto ta : v) {
    r[ta.var] = Range(ta.start, ta.stop);
  }
  return r;
}

std::unordered_map<const Var*, Range> inferBounds(Stmt* s) {
  BoundsInference bi;
  std::cerr << "Given stmt:\n" << *s << "\n";
  if (auto *b = dynamic_cast<Block*>(s)) {
    return convert(bi.inferBoundsForBlock(b));
  }
  auto *f = dynamic_cast<For*>(s);
  CHECK(f);
  return convert(bi.inferBoundsForLoop(f));
}

} // namespace tensorexpr
} // namespace jit
} // namespace torch

