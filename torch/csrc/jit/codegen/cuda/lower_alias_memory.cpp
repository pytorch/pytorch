#include <torch/csrc/jit/codegen/cuda/lower_alias_memory.h>

#include <torch/csrc/jit/codegen/cuda/instrumentation.h>
#include <torch/csrc/jit/codegen/cuda/ir_utils.h>
#include <torch/csrc/jit/codegen/cuda/kernel_expr_evaluator.h>
#include <torch/csrc/jit/codegen/cuda/kernel_ir.h>
#include <torch/csrc/jit/codegen/cuda/lower2device.h>
#include <torch/csrc/jit/codegen/cuda/lower_utils.h>

#include <sstream>
#include <unordered_map>
#include <unordered_set>

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {

namespace {

//! Get string representation of Allocate size for symbolic comparison
//!
class SymbolicSizePrinter : private kir::IrVisitor {
 public:
  static std::string printSize(const kir::Allocate* allocate) {
    SymbolicSizePrinter printer;
    allocate->size()->accept(&printer);
    return printer.os_.str();
  }

 private:
  void visit(const kir::Int* node) final {
    if (auto def = node->definition()) {
      def->accept(this);
    } else if (node->isConst()) {
      os_ << *node->value();
    } else {
      os_ << "ki" << node->id();
    }
  }

  void visit(const kir::NamedScalar* named_scalar) final {
    os_ << "@" << named_scalar->name();
  }

  void visit(const kir::UnaryOp* unary_op) final {
    os_ << unary_op->operation() << "(";
    unary_op->accept(this);
    os_ << ")";
  }

  void visit(const kir::BinaryOp* binary_op) final {
    os_ << binary_op->operation() << "(";
    binary_op->lhs()->accept(this);
    os_ << ",";
    binary_op->rhs()->accept(this);
    os_ << ")";
  }

 private:
  std::stringstream os_;
};

//! Reuse Allocation nodes via pointer aliasing
//!
class AllocateReuseModifier {
  // Alias local memory if it exceeds this threshold
  static constexpr size_t kRegisterSizeThreshold = 1;

 public:
  void modify(const std::vector<kir::Expr*>& exprs) {
    // Find candidate TensorViews and collect analysis information
    for (auto expr : exprs) {
      handle(expr);
    }

    // Iterate over candidates to find match
    for (auto tv : candidate_alias_tv_) {
      const auto def = tv->definition();
      TORCH_INTERNAL_ASSERT(def != nullptr);

      const auto alloc_it = map_tv_to_allocations_.find(tv->name());
      TORCH_INTERNAL_ASSERT(alloc_it != map_tv_to_allocations_.end());
      const auto output_alloc = alloc_it->second;

      const auto input_alloc = findCompatibleInputAllocate(
          tv->dtype(), SymbolicSizePrinter::printSize(output_alloc), def);

      if (input_alloc != nullptr) {
        output_alloc->setAlias(input_alloc);
      }
    }
  }

 private:
  // Do we have a true pointwise op?
  // (ie. a TV op, excluding direct assignments and reductions)
  static bool isPointwiseTvOp(const kir::Expr* expr) {
    if (ir_utils::isTVOp(expr)) {
      if (auto unary_op = dynamic_cast<const kir::UnaryOp*>(expr)) {
        return unary_op->operation() != UnaryOpType::Set;
      } else {
        return expr->isA<kir::BinaryOp>() || expr->isA<kir::TernaryOp>();
      }
    }
    return false;
  }

  // Find an Input Allocate that is compatible with the Output Allocate
  const kir::Allocate* findCompatibleInputAllocate(
      const DataType output_dtype,
      const std::string& output_size_str,
      const kir::Expr* expr) {
    // Stop searching if current op is not point-wise
    if (!isPointwiseTvOp(expr)) {
      return nullptr;
    }

    const kir::TensorView* first_tv_input = nullptr;
    for (const auto input : expr->inputs()) {
      if (auto input_tv = dynamic_cast<const kir::TensorView*>(input)) {
        if (first_tv_input == nullptr) {
          first_tv_input = input_tv;
        }

        // input_alloc == nullptr implies that input_tv is a kernel input
        const auto input_alloc = map_tv_to_allocations_[input_tv->name()];
        if (input_alloc != nullptr) {
          if (candidate_alias_tv_.find(input_tv) != candidate_alias_tv_.end() &&
              output_size_str == SymbolicSizePrinter::printSize(input_alloc) &&
              output_dtype == input_tv->dtype() &&
              map_tv_to_last_usage_[input_tv] <= map_expr_to_pos_[expr]) {
            return input_alloc;
          }
        }
      }
    }

    // Assume the first argument contains the primary variable
    // Follow path along point-wise operations
    if (first_tv_input != nullptr &&
        map_tv_to_last_usage_[first_tv_input] <= map_expr_to_pos_[expr]) {
      if (const auto def = first_tv_input->definition()) {
        return findCompatibleInputAllocate(output_dtype, output_size_str, def);
      }
    }

    return nullptr;
  }

  void handle(kir::Expr* expr) {
    const size_t expr_index = map_expr_to_pos_.size();
    map_expr_to_pos_[expr] = expr_index;

    if (ir_utils::isTVOp(expr)) {
      const auto output_tv = expr->outputs()[0]->as<kir::TensorView>();

      const auto alloc_it = map_tv_to_allocations_.find(output_tv->name());
      if (alloc_it != map_tv_to_allocations_.end()) {
        const bool smem_valid = (output_tv->memoryType() == MemoryType::Shared);

        bool local_valid = false;
        if (output_tv->memoryType() == MemoryType::Local) {
          const auto allocation = alloc_it->second;
          const auto register_size =
              expr_evaluator_.evaluate(allocation->size());
          if (register_size.has_value()) {
            local_valid = size_t(*register_size) > kRegisterSizeThreshold;
          }
        }

        // For the output TV to be an alias candidate,
        // its allocation size must exceed the threshold
        // OR be in shared memory
        if (smem_valid || local_valid) {
          candidate_alias_tv_.insert(output_tv);
        }
      }

      for (auto input_tv :
           ir_utils::filterByType<kir::TensorView>(expr->inputs())) {
        map_tv_to_last_usage_[input_tv] = expr_index;
      }
    } else if (auto ite = dynamic_cast<kir::IfThenElse*>(expr)) {
      handle(ite);
    } else if (auto for_loop = dynamic_cast<kir::ForLoop*>(expr)) {
      handle(for_loop);
    } else if (auto allocate = dynamic_cast<kir::Allocate*>(expr)) {
      handle(allocate);
    }
  }

  void handle(kir::Allocate* allocate) {
    if (auto tv = dynamic_cast<const kir::TensorView*>(allocate->buffer())) {
      map_tv_to_allocations_[tv->name()] = allocate;
    }
  }

  void handle(const kir::ForLoop* for_loop) {
    for (auto expr : for_loop->body().exprs()) {
      handle(expr);
    }
  }

  void handle(const kir::IfThenElse* ite) {
    for (auto expr : ite->thenBody().exprs()) {
      handle(expr);
    }
    for (auto expr : ite->elseBody().exprs()) {
      handle(expr);
    }
  }

 private:
  // Expression Evaluator to infer size of register allocation
  kir::ExpressionEvaluator expr_evaluator_;

  // Map expression to unique position
  // TODO: elaborate - position relative to what?
  std::unordered_map<const kir::Expr*, size_t> map_expr_to_pos_;

  // Map TensorView to last usage expression position
  std::unordered_map<const kir::TensorView*, size_t> map_tv_to_last_usage_;

  // Map TensorView name to Allocate node
  std::unordered_map<StmtNameType, kir::Allocate*> map_tv_to_allocations_;

  // Track candidate TensorViews whose Allocate nodes
  // could potentially alias another Allocate node
  std::unordered_set<const kir::TensorView*> candidate_alias_tv_;
};

} // namespace

std::vector<kir::Expr*> reuseMemoryAllocations(
    const std::vector<kir::Expr*>& exprs) {
  FUSER_PERF_SCOPE("GpuLower::Lower::reuseMemoryAllocations");
  AllocateReuseModifier arm;
  arm.modify(exprs);
  return exprs;
}

} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
