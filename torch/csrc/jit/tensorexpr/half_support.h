#include <torch/csrc/jit/tensorexpr/ir.h>
#include <torch/csrc/jit/tensorexpr/ir_printer.h>
#include <torch/csrc/jit/tensorexpr/ir_visitor.h>
#include <torch/csrc/jit/tensorexpr/tensor.h>

namespace torch {
namespace jit {
namespace tensorexpr {

// Walk the Statment looking for Half size loads/stores.
class HalfChecker : public IRMutator {
 public:
  bool hasHalf() {
    return hasHalf_;
  }

  const Expr* mutate(const Load* v) override {
    const Expr* child = IRMutator::mutate(v);
    if (child->dtype().scalar_type() != ScalarType::Half) {
      return child;
    }

    hasHalf_ = true;

    return new Cast(Dtype(kFloat, child->dtype().lanes()), child);
  }

  Stmt* mutate(const Store* v) override {
    const Expr* new_val = v->value()->accept_mutator(this);

    if (v->value()->dtype().scalar_type() == ScalarType::Half) {
      // TODO discards lanes.
      new_val = new Cast(Dtype(kHalf, v->value()->dtype().lanes()), new_val);
      inserted_half_casts_.insert(new_val);
      hasHalf_ = true;
    }

    return new Store(v->buf(), v->indices(), new_val, v->mask());
  }

  const Expr* mutate(const HalfImm* v) override {
    hasHalf_ = true;
    return new Cast(kFloat, v);
  }

  const Expr* mutate(const Cast* v) override {
    const Expr* child = v->src_value()->accept_mutator(this);

    // just don't allow half casts we didn't insert.
    if (v->dtype().scalar_type() == ScalarType::Half) {
      if (inserted_half_casts_.count(v) < 1) {
        return new Cast(Dtype(kFloat, child->dtype().lanes()), child);
      }
    }

    if (child == v->src_value()) {
      return v;
    }

    return new Cast(v->dtype(), child);
  }

 private:
  bool hasHalf_{false};
  std::unordered_set<const Expr*> inserted_half_casts_;
};


}
}
}
