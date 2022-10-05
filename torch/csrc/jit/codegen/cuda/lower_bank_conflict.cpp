#include <torch/csrc/jit/codegen/cuda/lower_bank_conflict.h>

#include <torch/csrc/jit/codegen/cuda/expr_evaluator.h>
#include <torch/csrc/jit/codegen/cuda/kernel_ir.h>
#include <torch/csrc/jit/codegen/cuda/kernel_ir_dispatch.h>
#include <torch/csrc/jit/codegen/cuda/type.h>

#include <unordered_set>

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {

namespace {

bool isSmemTensorIndex(Val* value) {
  return value->isA<kir::TensorIndex>() &&
      value->as<kir::TensorIndex>()->view()->getMemoryType() ==
      MemoryType::Shared;
}

int64_t getVectorizeSize(kir::TensorIndex* ti) {
  for (auto id : ti->view()->domain()->domain()) {
    if (!isParallelTypeVectorize(id->getParallelType())) {
      continue;
    }

    ExpressionEvaluator expr_eval(id->fusion());
    auto vector_size_optional = expr_eval.evaluate(id->extent());

    TORCH_INTERNAL_ASSERT(
        vector_size_optional.has_value(),
        "Could not evaluate constant value bound to vectorized dim.");

    return vector_size_optional->as<int64_t>();
  }
  return 1;
}

inline int64_t getPhaseSize(int64_t word_size_bytes) {
  if (word_size_bytes == 16) {
    return 8;
  }
  if (word_size_bytes == 8) {
    return 16;
  }
  return 32;
}

std::vector<int64_t> evaluateAddressesOnFirstPhase(
    kir::TensorIndex* ti,
    const std::vector<kir::ForLoop*>& for_loops) {
  std::vector<int64_t> addresses;
  const auto word_size_bytes =
      dataTypeSize(*(ti->getDataType())) * getVectorizeSize(ti);
  int64_t phase_size = getPhaseSize(word_size_bytes);

  for (auto tidx : c10::irange(phase_size)) {
    int64_t index = 0;
    ExpressionEvaluator expr_eval(ti->fusion());
    for (auto fl : for_loops) {
      if (fl->index()->isA<NamedScalar>() &&
          fl->index()->as<NamedScalar>()->name() == "threadIdx.x") {
        expr_eval.bind(fl->index(), tidx);
      } else {
        expr_eval.bind(fl->index(), 0);
      }
    }
    for (auto ind : ti->indices()) {
      index += expr_eval.evaluate(ind)->as<int64_t>();
    }
    addresses.emplace_back(index * word_size_bytes);
  }
  return addresses;
}

int getConflictWays(const std::vector<int64_t>& addresses) {
  std::unordered_set<int64_t> words_by_bank[32];
  for (auto addr : addresses) {
    int64_t word = addr / 4;
    int64_t bank = word % 32;
    words_by_bank[bank].insert(word);
  }
  int conflict = 1;
  for (const auto& words : words_by_bank) {
    conflict = std::max<int>(conflict, words.size());
  }
  return conflict;
}

} // namespace

class BankConflictInfo : public kir::IrVisitor {
 public:
  static std::unordered_map<const Expr*, std::pair<int, int>> get(
      const std::vector<Expr*>& exprs) {
    return BankConflictInfo(exprs).bank_conflict_info_;
  }

 private:
  BankConflictInfo(const std::vector<Expr*>& exprs) {
    handle(exprs);
  }

  using kir::IrVisitor::handle;

  void handle(Expr* expr) final {
    if (expr->isA<kir::ForLoop>() || expr->isA<kir::IfThenElse>()) {
      kir::IrVisitor::handle(expr);
      return;
    }

    if (expr->isA<UnaryOp>()) {
      auto uop = expr->as<UnaryOp>();
      if (uop->getUnaryOpType() != UnaryOpType::Set) {
        return;
      }
      std::pair<int, int> conflict_ways{0, 0};
      if (isSmemTensorIndex(uop->in())) {
        conflict_ways.first = getConflictWays(evaluateAddressesOnFirstPhase(
            uop->in()->as<kir::TensorIndex>(), for_loops_));
      }
      if (isSmemTensorIndex(uop->out())) {
        conflict_ways.second = getConflictWays(evaluateAddressesOnFirstPhase(
            uop->out()->as<kir::TensorIndex>(), for_loops_));
      }
      if (conflict_ways.first > 1 || conflict_ways.second > 1) {
        bank_conflict_info_[expr] = conflict_ways;
      }
    } else if (expr->isA<LoadStoreOp>()) {
      auto ldst = expr->as<LoadStoreOp>();
      std::pair<int, int> conflict_ways{0, 0};
      if (isSmemTensorIndex(ldst->in())) {
        conflict_ways.first = getConflictWays(evaluateAddressesOnFirstPhase(
            ldst->in()->as<kir::TensorIndex>(), for_loops_));
      }
      if (isSmemTensorIndex(ldst->out())) {
        conflict_ways.second = getConflictWays(evaluateAddressesOnFirstPhase(
            ldst->out()->as<kir::TensorIndex>(), for_loops_));
      }
      if (conflict_ways.first > 1 || conflict_ways.second > 1) {
        bank_conflict_info_[expr] = conflict_ways;
      }
    }
  }

  std::unordered_map<const Expr*, std::pair<int, int>> bank_conflict_info_;
};

std::unordered_map<const Expr*, std::pair<int, int>> getBankConflictInfo(
    kir::Kernel* kernel) {
  return BankConflictInfo::get(kernel->topLevelExprs());
}

} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
