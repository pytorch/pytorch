#include <torch/csrc/jit/codegen/cuda/lower_bank_conflict.h>

#include <torch/csrc/jit/codegen/cuda/dynamic_type.h>
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

bool isThreadIdx(const std::string& name) {
  return name == "threadIdx.x" || name == "threadIdx.y" ||
      name == "threadIdx.z";
}

bool isBlockIdx(const std::string& name) {
  return name == "blockIdx.x" || name == "blockIdx.y" || name == "blockIdx.z";
}

bool isBlockDim(const std::string& name) {
  return name == "blockDim.x" && name == "blockDim.y" && name == "blockDim.z";
}

bool isGridDim(const std::string& name) {
  return name == "gridDim.x" && name == "gridDim.y" && name == "gridDim.z";
}

ParallelType getParallelType(const std::string& name) {
  if (name == "threadIdx.x") {
    return ParallelType::TIDx;
  } else if (name == "threadIdx.y") {
    return ParallelType::TIDy;
  } else if (name == "threadIdx.z") {
    return ParallelType::TIDz;
  } else if (name == "blockIdx.x") {
    return ParallelType::BIDx;
  } else if (name == "blockIdx.y") {
    return ParallelType::BIDy;
  } else if (name == "blockIdx.z") {
    return ParallelType::BIDz;
  }
  TORCH_INTERNAL_ASSERT(false, "Not a parallel type");
}

std::vector<int64_t> evaluateAddressesOnFirstPhase(
    kir::TensorIndex* ti,
    const std::vector<kir::ForLoop*>& for_loops,
    c10::optional<LaunchParams> launch_params,
    const ExpressionEvaluator& expr_eval_common) {
  std::vector<int64_t> addresses;
  const auto word_size_bytes =
      dataTypeSize(*(ti->getDataType())) * getVectorizeSize(ti);
  int64_t phase_size = getPhaseSize(word_size_bytes);

  if (launch_params.has_value()) {
    phase_size = std::min<int64_t>(phase_size, launch_params->nThreads());
  }

  for (int64_t linear_tidx : c10::irange(phase_size)) {
    int64_t tidx = linear_tidx;
    int64_t tidy = 0;
    int64_t tidz = 0;
    if (launch_params.has_value()) {
      tidy = tidx / launch_params->bdimx();
      tidx = tidx % launch_params->bdimx();
      tidz = tidy / launch_params->bdimy();
      tidy = tidy % launch_params->bdimy();
    }
    int64_t index = 0;
    // make a copy of the expression evaluator
    ExpressionEvaluator expr_eval = expr_eval_common;
    expr_eval.bind("threadIdx.x", tidx);
    expr_eval.bind("threadIdx.y", tidy);
    expr_eval.bind("threadIdx.z", tidz);
    for (auto fl : for_loops) {
      if (fl->index()->isA<NamedScalar>()) {
        auto name = fl->index()->as<NamedScalar>()->name();
        TORCH_INTERNAL_ASSERT(
            isThreadIdx(name) || isBlockIdx(name), "unknow loop index");
      } else {
        auto start = expr_eval.evaluate(fl->start())->as<int64_t>();
        expr_eval.bind(fl->index(), start);
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

class InferLaunchParams : public kir::IrVisitor {
 public:
  static c10::optional<LaunchParams> get(
      const std::vector<Expr*>& exprs,
      const std::unordered_map<std::string, IntOrDouble>& known_values) {
    if (exprs.empty()) {
      return c10::nullopt;
    }
    return InferLaunchParams(exprs, known_values).launch_params_;
  }

 private:
  InferLaunchParams(
      const std::vector<Expr*>& exprs,
      const std::unordered_map<std::string, IntOrDouble>& known_values)
      : expr_eval_(exprs[0]->fusion()) {
    for (auto pair : known_values) {
      expr_eval_.bind(pair.first, pair.second);
    }
    handle(exprs);
  }

  using kir::IrVisitor::handle;

  void handle(Expr* expr) final {
    if (expr->isA<kir::ForLoop>() || expr->isA<kir::IfThenElse>()) {
      kir::IrVisitor::handle(expr);
      return;
    }

    for (auto fl : for_loops_) {
      if (fl->index()->isA<NamedScalar>()) {
        auto name = fl->index()->as<NamedScalar>()->name();
        if (isThreadIdx(name) || isBlockIdx(name)) {
          auto ptype = getParallelType(name);
          auto stop = expr_eval_.evaluate(fl->stop());
          if (stop.has_value()) {
            if (!launch_params_.has_value()) {
              launch_params_ = LaunchParams();
            }
            if (launch_params_->getRawVal(ptype) ==
                LaunchParams::UNINITIALIZED_VAL) {
              launch_params_->bind(stop->as<int64_t>(), ptype);
            } else {
              TORCH_INTERNAL_ASSERT(
                  launch_params_->getDim(ptype) == stop,
                  "Unable to infer launch parameters");
            }
          }
        }
      }
    }
  }

  ExpressionEvaluator expr_eval_;
  c10::optional<LaunchParams> launch_params_;
};

class BankConflictInfo : public kir::IrVisitor {
 public:
  static std::unordered_map<const Expr*, std::pair<int, int>> get(
      const std::vector<Expr*>& exprs,
      c10::optional<LaunchParams> launch_params,
      const std::unordered_map<std::string, IntOrDouble>& known_values) {
    if (exprs.empty()) {
      return {};
    }
    return BankConflictInfo(exprs, launch_params, known_values)
        .bank_conflict_info_;
  }

 private:
  BankConflictInfo(
      const std::vector<Expr*>& exprs,
      c10::optional<LaunchParams> launch_params,
      const std::unordered_map<std::string, IntOrDouble>& known_values)
      : launch_params_(launch_params), expr_eval_common_(exprs[0]->fusion()) {
    expr_eval_common_.bind("blockIdx.x", 0);
    expr_eval_common_.bind("blockIdx.y", 0);
    expr_eval_common_.bind("blockIdx.z", 0);
    if (launch_params.has_value()) {
      expr_eval_common_.bind("blockDim.x", launch_params->bdimx());
      expr_eval_common_.bind("blockDim.y", launch_params->bdimy());
      expr_eval_common_.bind("blockDim.z", launch_params->bdimz());
      expr_eval_common_.bind("gridDim.x", launch_params->gdimx());
      expr_eval_common_.bind("gridDim.y", launch_params->gdimy());
      expr_eval_common_.bind("gridDim.z", launch_params->gdimz());
    }
    for (auto pair : known_values) {
      expr_eval_common_.bind(pair.first, pair.second);
    }
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
            uop->in()->as<kir::TensorIndex>(),
            for_loops_,
            launch_params_,
            expr_eval_common_));
      }
      if (isSmemTensorIndex(uop->out())) {
        conflict_ways.second = getConflictWays(evaluateAddressesOnFirstPhase(
            uop->out()->as<kir::TensorIndex>(),
            for_loops_,
            launch_params_,
            expr_eval_common_));
      }
      if (conflict_ways.first > 1 || conflict_ways.second > 1) {
        bank_conflict_info_[expr] = conflict_ways;
      }
    } else if (expr->isA<LoadStoreOp>()) {
      auto ldst = expr->as<LoadStoreOp>();
      std::pair<int, int> conflict_ways{0, 0};
      if (isSmemTensorIndex(ldst->in())) {
        conflict_ways.first = getConflictWays(evaluateAddressesOnFirstPhase(
            ldst->in()->as<kir::TensorIndex>(),
            for_loops_,
            launch_params_,
            expr_eval_common_));
      }
      if (isSmemTensorIndex(ldst->out())) {
        conflict_ways.second = getConflictWays(evaluateAddressesOnFirstPhase(
            ldst->out()->as<kir::TensorIndex>(),
            for_loops_,
            launch_params_,
            expr_eval_common_));
      }
      if (conflict_ways.first > 1 || conflict_ways.second > 1) {
        bank_conflict_info_[expr] = conflict_ways;
      }
    }
  }

  std::unordered_map<const Expr*, std::pair<int, int>> bank_conflict_info_;
  c10::optional<LaunchParams> launch_params_;
  ExpressionEvaluator expr_eval_common_;
};

} // namespace

std::unordered_map<const Expr*, std::pair<int, int>> getBankConflictInfo(
    kir::Kernel* kernel,
    c10::optional<LaunchParams> launch_params,
    const std::unordered_map<std::string, IntOrDouble>& known_values) {
  for (auto pair : known_values) {
    TORCH_CHECK(
        !isThreadIdx(pair.first),
        "threadIdx.{x,y,z} should be computed instead of provided");
    TORCH_CHECK(
        !isBlockIdx(pair.first),
        "blockIdx.{x,y,z} should not be provided (they are always zero)");
    TORCH_CHECK(
        !isBlockDim(pair.first),
        "blockDim.{x,y,z} should be provided by launch_params");
    TORCH_CHECK(
        !isGridDim(pair.first),
        "gridDim.{x,y,z} should be provided by launch_params");
  }
  if (!launch_params.has_value()) {
    launch_params =
        InferLaunchParams::get(kernel->topLevelExprs(), known_values);
  }
  return BankConflictInfo::get(
      kernel->topLevelExprs(), launch_params, known_values);
}

} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
