#include <torch/csrc/jit/tensorexpr/bounds_inference.h>
#include <torch/csrc/jit/tensorexpr/expr.h>
#include <torch/csrc/jit/tensorexpr/ir.h>
#include <torch/csrc/jit/tensorexpr/ir_printer.h>
#include <torch/csrc/jit/tensorexpr/ir_simplifier.h>
#include <torch/csrc/jit/tensorexpr/ir_visitor.h>
#include <torch/csrc/jit/tensorexpr/stmt.h>

namespace torch {
namespace jit {
namespace tensorexpr {

class BoundsInference : public IRVisitor {
 public:
  void visit(const FunctionCall* v) override;
  void visit(const Load* v) override;
  void visit(const Store* v) override;
  void visit(const For* v) override;
  void visit(const Block* v) override;

  BoundsInfo accesses() const {
    return accesses_;
  }

 private:
  BoundsInfo accesses_;
};

void BoundsInference::visit(const Load* v) {
  accesses_.push_back({v->buf(), kLoad, v->indices(), v->indices()});
}

void BoundsInference::visit(const FunctionCall* v) {
  accesses_.push_back(
      {v->tensor()->func_var(), kLoad, v->params(), v->params()});
}

void BoundsInference::visit(const Store* v) {
  accesses_.push_back({v->buf(), kStore, v->indices(), v->indices()});
  IRVisitor::visit(v);
}

void BoundsInference::visit(const For* v) {
  v->body()->accept(this);
  for (TensorAccessBoundsInfo& access : accesses_) {
    for (size_t j = 0; j < access.start.size(); j++) {
      // TODO: This function assumes that all indices grow monotonically and
      // thus for the loop:
      //   for i in A..B:
      //     buf[i] = i
      // the range for i is [A, B). It should be generalized to correctly handle
      // all cases.
      const Expr* old_start = access.start[j];
      const Expr* old_stop = access.stop[j];
      const Expr* new_start = Substitute(old_start, {{v->var(), v->start()}});
      const Expr* new_stop =
          Substitute(old_stop, {{v->var(), new Sub(v->stop(), new IntImm(1))}});
      access.start[j] = IRSimplifier::simplify(new_start);
      access.stop[j] = IRSimplifier::simplify(new_stop);
    }
  }
}

void BoundsInference::visit(const Block* v) {
  BoundsInfo res;
  for (auto s : *v) {
    s->accept(this);
    res.insert(res.end(), accesses_.begin(), accesses_.end());
  }
  accesses_ = res;
}

void printBoundsInfo(const BoundsInfo& v) {
  std::cerr << "Access vector {\n";
  for (const auto& b : v) {
    std::cerr << *b.buf << " in (";
    int i = 0;
    for (const auto& s : b.start) {
      if (i != 0) {
        std::cerr << ", ";
      }
      std::cerr << *s;
      i++;
    }
    std::cerr << "; ";
    i = 0;
    for (const auto& s : b.stop) {
      if (i != 0) {
        std::cerr << ", ";
      }
      std::cerr << *s;
      i++;
    }
    std::cerr << ")\n";
  }
  std::cerr << "}\n";
}

// TODO: This probably should be done as a part of IR simplifier.
static const Expr* simplifyMin(const Expr* a, const Expr* b) {
  const Expr* diff = IRSimplifier::simplify(new Sub(a, b));
  if (auto diff_imm = dynamic_cast<const IntImm*>(diff)) {
    if (diff_imm->value() < 0) {
      return a;
    } else {
      return b;
    }
  }
  return new Min(a, b, true);
}

static const Expr* simplifyMax(const Expr* a, const Expr* b) {
  const Expr* diff = IRSimplifier::simplify(new Sub(a, b));
  if (auto diff_imm = dynamic_cast<const IntImm*>(diff)) {
    if (diff_imm->value() > 0) {
      return a;
    } else {
      return b;
    }
  }
  return new Max(a, b, true);
}

/*
 * Go through the given BoundsInfo vector and merge entries corresponding to
 * the same buf. E.g. given
 *    [{a, kLoad, 0, 100}, {b, kStore, 0, 100}, {a, kLoad, 10, 110}]
 * produce:
 *    [{a, kLoad, 0, 110}, {b, kStore, 0, 100}]
 */
static BoundsInfo mergeTensorAccesses(const BoundsInfo& unmerged) {
  BoundsInfo res;
  std::unordered_map<const Buf*, TensorAccessBoundsInfo> merged;
  for (const auto& t : unmerged) {
    if (!merged.count(t.buf)) {
      merged[t.buf] = t;
      continue;
    }

    // We already have some range for this buf, try to merge them
    TensorAccessBoundsInfo old_t = merged.at(t.buf);
    TensorAccessBoundsInfo new_t = t;

    for (size_t i = 0; i < old_t.start.size(); i++) {
      new_t.start[i] = simplifyMin(old_t.start[i], new_t.start[i]);
    }
    for (size_t i = 0; i < old_t.stop.size(); i++) {
      new_t.stop[i] = simplifyMax(old_t.stop[i], new_t.stop[i]);
    }
    merged[t.buf] = new_t;
  }

  // Do the merge in two passes so that the original order of elements in
  // BoundsInfo vector is preserved
  std::unordered_set<const Buf*> added;
  for (const auto& t : unmerged) {
    if (added.insert(t.buf).second) {
      res.push_back(merged.at(t.buf));
    }
  }
  return res;
}

BoundsInfo inferBounds(Stmt* s) {
  BoundsInference ac;
  s->accept(&ac);
  return mergeTensorAccesses(ac.accesses());
}

} // namespace tensorexpr
} // namespace jit
} // namespace torch
