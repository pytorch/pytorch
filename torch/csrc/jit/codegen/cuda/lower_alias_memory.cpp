#include <torch/csrc/jit/codegen/cuda/lower_alias_memory.h>

#include <torch/csrc/jit/codegen/cuda/expr_evaluator.h>
#include <torch/csrc/jit/codegen/cuda/instrumentation.h>
#include <torch/csrc/jit/codegen/cuda/ir_utils.h>
#include <torch/csrc/jit/codegen/cuda/kernel_ir_builder.h>
#include <torch/csrc/jit/codegen/cuda/kernel_ir_printer.h>
#include <torch/csrc/jit/codegen/cuda/lower2device.h>
#include <torch/csrc/jit/codegen/cuda/lower_utils.h>

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {

namespace {

//! Get string representation of Allocate size for symbolic comparison
//!
class SymbolicSizePrinter final : private OptOutConstDispatch {
 public:
  static std::string print_size(const kir::Allocate* alloc) {
    SymbolicSizePrinter printer;
    printer.handle(alloc->size());
    return printer.os_.str();
  }

 private:
  void handle(const Val* v) final {
    OptOutConstDispatch::handle(v);
  }

  void handle(const Expr* e) final {
    OptOutConstDispatch::handle(e);
  }

  void handle(const kir::Int* node) final {
    if (auto def = FusionGuard::getCurFusion()->origin(node)) {
      os_ << "( ";
      handle(def);
      os_ << " )";
      return;
    } else if (node->isSymbolic()) {
      os_ << "i" << node->name();
    } else {
      os_ << *node->value();
    }
  }

  void handle(const kir::NamedScalar* node) final {
    os_ << node->name();
  }

  void handle(const kir::BinaryOp* node) final {
    if (auto inline_bop = inline_op_str(node->getBinaryOpType())) {
      handle(node->lhs());
      os_ << " " << inline_bop.value() << " ";
      handle(node->rhs());
    } else {
      os_ << node->getBinaryOpType() << "(";
      handle(node->lhs());
      os_ << ", ";
      handle(node->rhs());
      os_ << ")";
    }
  }

 private:
  std::stringstream os_;
};

//! Reuse Allocation nodes via pointer aliasing
//!
class AllocateReuseModifier final : private OptOutDispatch {
 public:
  explicit AllocateReuseModifier(Fusion* fusion, size_t register_size_threshold)
      : eval_evaluator_(fusion),
        register_size_threshold_(register_size_threshold) {}

  void modify(const std::vector<Expr*>& exprs) {
    // Find candidate TensorViews and collect analysis information
    for (auto expr : exprs) {
      handle(expr);
    }

    // Iterate over candidates to find match
    for (auto tv : candidate_alias_tv_) {
      TORCH_INTERNAL_ASSERT(
          map_tv_to_origin_expr_.find(tv) != map_tv_to_origin_expr_.end());

      const auto& expr = map_tv_to_origin_expr_[tv];
      const auto output = expr->output(0)->as<TensorView>();

      TORCH_INTERNAL_ASSERT(
          map_tv_to_allocations_.find(output->name()) !=
          map_tv_to_allocations_.end());

      auto output_alloc = map_tv_to_allocations_[output->name()];

      auto input_alloc = findCompatibleInputAllocate(
          SymbolicSizePrinter::print_size(output_alloc), expr);
      if (input_alloc != nullptr) {
        // std::cout << "Alias Match\t" << output->getMemoryType() << std::endl;
        output_alloc->setAlias(input_alloc);
      }
    }
  }

 private:
  // Check if we are a Pointwise TensorView op.
  bool isPwiseTVOp(const Expr* expr) {
    // Ignore set operations
    if (expr->outputs().size() == 1 && ir_utils::isTV(expr->output(0)) &&
        ((expr->getExprType().value() == ExprType::UnaryOp &&
          expr->as<UnaryOp>()->getUnaryOpType() != UnaryOpType::Set) ||
         expr->getExprType().value() == ExprType::BinaryOp ||
         expr->getExprType().value() == ExprType::TernaryOp))
      return true;
    return false;
  }

  // Find an Input Allocate that is compatible with the Output Allocate
  kir::Allocate* findCompatibleInputAllocate(
      const std::string& output_size_str,
      Expr* expr) {
    // Stop searching if current op is not point-wise
    if (!isPwiseTVOp(expr)) {
      return nullptr;
    }

    const auto& expr_inputs_iter =
        ir_utils::filterByType<TensorView>(expr->inputs());

    std::vector<TensorView*> expr_inputs(
        expr_inputs_iter.begin(), expr_inputs_iter.end());

    for (const auto input : expr_inputs) {
      auto input_alloc = map_tv_to_allocations_[input->name()];

      // input_allocation == nullptr implies that input_tv is a fusion input.
      if (input_alloc != nullptr) {
        if (candidate_alias_tv_.find(input) != candidate_alias_tv_.end() &&
            output_size_str == SymbolicSizePrinter::print_size(input_alloc) &&
            map_tv_to_last_usage_[input] <= map_expr_to_pos_[expr]) {
          return input_alloc;
        }
      }
    }

    // Assume the first argument contains the primary variable
    // Follow path along point-wise operations
    if (!expr_inputs.empty()) {
      auto first_input_argument_tv = expr_inputs.front()->getOrigin();
      if (first_input_argument_tv != nullptr) {
        return findCompatibleInputAllocate(
            output_size_str, first_input_argument_tv);
      }
    }
    return nullptr;
  }

  void handle(Expr* expr) final {
    size_t expr_index = map_expr_to_pos_.size();
    map_expr_to_pos_[expr] = expr_index;

    if (ir_utils::isTVOp(expr)) {
      const auto output = expr->output(0)->as<TensorView>();
      map_tv_to_origin_expr_[output] = expr;

      bool has_allocation = map_tv_to_allocations_.find(output->name()) !=
          map_tv_to_allocations_.end();

      if (has_allocation) {
        bool smem_valid = output->getMemoryType() == MemoryType::Shared;

        bool local_valid = false;
        if (output->getMemoryType() == MemoryType::Local) {
          auto allocation = map_tv_to_allocations_[output->name()];
          auto inferred_register_size =
              eval_evaluator_.inferValue(allocation->size());
          if (inferred_register_size.has_value()) {
            local_valid = inferred_register_size.value() >
                static_cast<int64_t>(register_size_threshold_);
          }
        }

        // For the output TV to be an alias candidate,
        // its allocation size must exceed the threshold
        // OR be in shared memory
        if (smem_valid || local_valid) {
          candidate_alias_tv_.insert(output);
        }
      }

      const auto& expr_inputs =
          ir_utils::filterByType<TensorView>(expr->inputs());
      for (const auto input : expr_inputs) {
        map_tv_to_last_usage_[input] = expr_index;
      }
    } else {
      OptOutDispatch::handle(expr);
    }
  }

  void handle(kir::Allocate* a) final {
    if (a->buffer()->getValType().value() == ValType::KirTensorView) {
      auto tv = a->buffer()->as<kir::TensorView>()->fuserTv();
      map_tv_to_allocations_[tv->name()] = a;
    }
  }

  void handle(kir::ForLoop* fl) final {
    for (auto expr : fl->body().exprs()) {
      handle(expr);
    }
  }

  void handle(kir::IfThenElse* ite) final {
    for (auto expr : ite->thenBody().exprs()) {
      handle(expr);
    }
    for (auto expr : ite->elseBody().exprs()) {
      handle(expr);
    }
  }

 private:
  // Expression Evaluator to infer size of register allocation
  StatefulExpressionEvaluator eval_evaluator_;

  // Alias local memory if it exceeds this threshold
  const size_t register_size_threshold_;

  // Map expression to unique position
  std::unordered_map<Expr*, size_t> map_expr_to_pos_;

  // Map TensorView to origin expression
  std::unordered_map<const TensorView*, Expr*> map_tv_to_origin_expr_;

  // Map TensorView to last usage expression position
  std::unordered_map<const TensorView*, size_t> map_tv_to_last_usage_;

  // Map TensorView name to Allocate node
  std::unordered_map<unsigned int, kir::Allocate*> map_tv_to_allocations_;

  // Track candidate TensorViews whose Allocate nodes
  // could potentially alias another Allocate node
  std::unordered_set<const TensorView*> candidate_alias_tv_;
};

} // namespace

std::vector<Expr*> reuseMemoryAllocations(
    Fusion* fusion,
    const std::vector<Expr*>& exprs) {
  FUSER_PERF_SCOPE("reuseMemoryAllocations");
  FusionGuard fg(fusion);

  // Alias local memory if it exceeds this threshold
  const size_t register_size_threshold = 1;
  AllocateReuseModifier arm(fusion, register_size_threshold);
  arm.modify(exprs);
  return exprs;
}

} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
