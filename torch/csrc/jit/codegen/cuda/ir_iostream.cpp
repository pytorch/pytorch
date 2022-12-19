#include <torch/csrc/jit/codegen/cuda/ir_iostream.h>
#include <torch/csrc/jit/codegen/cuda/ir_printer.h>

#include <torch/csrc/jit/codegen/cuda/fusion.h>
#include <torch/csrc/jit/codegen/cuda/instrumentation.h>
#include <torch/csrc/jit/codegen/cuda/ir_all_nodes.h>
#include <torch/csrc/jit/codegen/cuda/ir_utils.h>
#include <torch/csrc/jit/codegen/cuda/kernel.h>
#include <torch/csrc/jit/codegen/cuda/lower_utils.h>

#include <c10/util/irange.h>

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {

// Make sure we can inline something, before we attempt to.
void checkInlineable(const Expr* expr) {
  for (auto input : expr->inputs()) {
    TORCH_CHECK(
        input->isScalar() || input->isA<kir::TensorIndex>() ||
            (expr->isA<UnaryOp>() &&
             expr->as<UnaryOp>()->getUnaryOpType() == UnaryOpType::Address),
        "Printing inline computations involving values other than scalars is not currently supported.");
  }
  TORCH_CHECK(
      expr->outputs().size() == 1,
      "Cannot print inline computations if there's more than one output.");
  TORCH_CHECK(
      expr->output(0)->isScalar() || expr->output(0)->isA<NamedScalar>(),
      "Printing inline computations involving values other than scalars is not currently supported.");
}

void IrPrinter::handle(Fusion* fusion) {
  FUSER_PERF_SCOPE("IrPrinter");
  resetIndent();
  for (const Expr* expr : fusion->exprs()) {
    os_ << expr->toString();
  }
}

void IrPrinter::handle(const kir::Kernel* kernel) {
  TORCH_CHECK(kernel != nullptr);

  // kernel declaration
  os_ << "\nKERNEL (";
  for (auto in : kernel->inputs()) {
    os_ << in->toString();
    if (in != kernel->inputs().back()) {
      os_ << ", ";
    }
  }
  os_ << ") -> (";
  for (auto out : kernel->outputs()) {
    os_ << out->toString();
    if (out != kernel->outputs().back()) {
      os_ << ", ";
    }
  }
  os_ << ") :\n";

  // kernel body
  indent_size_++;
  for (auto expr : kernel->topLevelExprs()) {
    os_ << expr->toString();
  }
  indent_size_--;
  os_ << "END.\n\n";
}

void IrPrinter::handle(kir::Kernel& kernel) {
  handle(&kernel);
}

void IrTransformPrinter::handle(Fusion* f) {
  auto all_vals = f->usedMathVals();

  for (auto tv : ir_utils::filterByType<TensorView>(all_vals)) {
    os() << tv->toString();
    os() << "\n";
    printTransforms(tv);
  }
}

void IrTransformPrinter::printTransforms(TensorView* tv) {
  auto root_domain = tv->domain()->getRootDomain();
  os() << " root domain : (";
  for (const auto root_idx : c10::irange(root_domain.size())) {
    os() << root_domain[root_idx]->toString();
    if (root_idx + 1 < root_domain.size()) {
      os() << ",";
    }
  }
  os() << ")\n";

  if (tv->hasRFactor()) {
    auto rfactor_domain = tv->domain()->getRFactorDomain();

    auto all_exp = DependencyCheck::getAllExprsBetween(
        {root_domain.begin(), root_domain.end()},
        {rfactor_domain.begin(), rfactor_domain.end()});

    for (auto exp : all_exp) {
      os() << "  " << exp->toString();
    }

    os() << " rfactor domain : (";
    for (const auto root_idx : c10::irange(rfactor_domain.size())) {
      os() << rfactor_domain[root_idx]->toString();
      if (root_idx + 1 < rfactor_domain.size()) {
        os() << ",";
      }
    }
    os() << ")\n";
  }

  os() << " contiguity: " << tv->domain()->contiguity() << "\n";

  auto from = tv->getMaybeRFactorDomain();
  auto all_exp = DependencyCheck::getAllExprsBetween(
      {from.begin(), from.end()},
      {tv->domain()->domain().begin(), tv->domain()->domain().end()});

  for (auto exp : all_exp) {
    os() << "  " << exp->toString();
  }
}

std::ostream& operator<<(std::ostream& os, const Statement* stmt) {
  return os << stmt->toString();
}

std::ostream& operator<<(std::ostream& os, Fusion* f) {
  IrPrinter p(os);
  FusionGuard guard(f);
  p.handle(f);
  return os;
}

std::ostream& operator<<(std::ostream& os, Fusion& f) {
  return os << &f;
}

} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
