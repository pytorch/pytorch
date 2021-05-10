#include <torch/csrc/jit/codegen/cuda/kernel_ir_printer.h>

#include <torch/csrc/jit/codegen/cuda/instrumentation.h>
#include <torch/csrc/jit/codegen/cuda/type.h>

#include <sstream>

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {
namespace kir {

static std::string boolLiteral(bool value) {
  return value ? "true" : "false";
}

void IrPrinter::printNode(const Statement* stmt) {
  handle(stmt);
}

void IrPrinter::printKernel(const Kernel* kernel) {
  TORCH_CHECK(kernel != nullptr);

  // kernel declaration
  os_ << "\nKERNEL (";
  for (auto in : kernel->inputs()) {
    os_ << gen(in);
    if (in != kernel->inputs().back()) {
      os_ << ", ";
    }
  }
  os_ << ") -> (";
  for (auto out : kernel->outputs()) {
    os_ << gen(out);
    if (out != kernel->outputs().back()) {
      os_ << ", ";
    }
  }
  os_ << ") :\n";

  // kernel body
  startBlock();
  for (auto expr : kernel->topLevelExprs()) {
    handle(expr);
  }
  endBlock();
  os_ << "END.\n\n";
}

std::ostream& IrPrinter::indent() {
  for (int i = 0; i < indent_level_; ++i) {
    os_ << kTab;
  }
  return os_;
}

std::string IrPrinter::gen(const Statement* stmt) {
  std::stringstream ss;
  IrPrinter ir_printer(ss);
  ir_printer.handle(stmt);
  return ss.str();
}

void IrPrinter::startBlock() {
  ++indent_level_;
}

void IrPrinter::endBlock() {
  TORCH_CHECK(indent_level_ > 0);
  --indent_level_;
}

void IrPrinter::handleBlock(const kir::Scope& scope) {
  startBlock();
  for (auto expr : scope.exprs()) {
    handle(expr);
  }
  endBlock();
}

void IrPrinter::handle(const Statement* s) {
  OptInConstDispatch::handle(s);
}

void IrPrinter::handle(const Val* v) {
  OptInConstDispatch::handle(v);
}

void IrPrinter::handle(const Expr* e) {
  OptInConstDispatch::handle(e);
}

void IrPrinter::handle(const kir::Bool* node) {
  if (node->isSymbolic()) {
    os_ << "b" << node->name();
  } else {
    os_ << boolLiteral(*node->value());
  }
}

void IrPrinter::handle(const kir::Float* node) {
  if (node->isSymbolic()) {
    os_ << "f" << node->name();
  } else {
    const int digits = std::numeric_limits<Float::ScalarType>::max_digits10;
    os_ << "float(" << std::setprecision(digits) << *node->value() << ")";
  }
}

void IrPrinter::handle(const kir::Half* node) {
  if (node->isSymbolic()) {
    os_ << "h" << node->name();
  } else {
    os_ << "half(" << *node->value() << ")";
  }
}

void IrPrinter::handle(const kir::Int* node) {
  if (node->isSymbolic()) {
    os_ << "i" << node->name();
  } else {
    os_ << *node->value();
  }
}

void IrPrinter::handle(const kir::NamedScalar* node) {
  os_ << node->name();
}

void IrPrinter::handle(const kir::TensorIndex* node) {
  os_ << gen(node->view()) << "[";
  for (auto index : node->indices()) {
    os_ << gen(index);
    if (index != node->indices().back()) {
      os_ << ", ";
    }
  }
  os_ << "]";
}

void IrPrinter::handle(const kir::IterDomain* node) {
  if (node->isRFactorProduct()) {
    os_ << "rfactor.";
  }
  os_ << node->getParallelType() << "." << node->getIterType() << "("
      << gen(node->start()) << " .. " << gen(node->rawExtent()) << ")";
}

void IrPrinter::handle(const kir::TensorDomain*) {
  // TODO(kir): print Tensor shapes?
  os_ << "kir::TensorDomain";
}

void IrPrinter::handle(const kir::TensorView* node) {
  // TODO(KIR): print memory type too?
  os_ << "T" << node->name();
}

void IrPrinter::handle(const kir::UnaryOp* node) {
  indent() << gen(node->out()) << " = ";

  if (auto op = inline_op_str(node->getUnaryOpType())) {
    os_ << *op << gen(node->in());
  } else {
    if (node->getUnaryOpType() == UnaryOpType::Cast) {
      const auto cast_str = cast_func_str(
          {node->in()->getDataType().value(),
           node->out()->getDataType().value()});
      os_ << cast_str.value();
    } else {
      os_ << node->getUnaryOpType();
    }

    os_ << "(";
    if (node->getUnaryOpType() == UnaryOpType::RandLike) {
      os_ << "RND";
    } else {
      os_ << gen(node->in());
    }
    os_ << ")";
  }

  os_ << "\n";
}

void IrPrinter::handle(const kir::BinaryOp* node) {
  indent() << gen(node->out()) << " = ";

  const auto op_type = node->getBinaryOpType();
  const auto lhs = gen(node->lhs());
  const auto rhs = gen(node->rhs());

  if (auto op = inline_op_str(op_type)) {
    os_ << lhs << " " << *op << " " << rhs;
  } else {
    os_ << op_type << "(" << lhs << ", " << rhs << ")";
  }

  os_ << "\n";
}

void IrPrinter::handle(const kir::TernaryOp* node) {
  indent() << gen(node->out()) << " = " << node->getTernaryOpType() << "("
           << gen(node->in1()) << ", " << gen(node->in2()) << ", "
           << gen(node->in3()) << ")\n";
}

void IrPrinter::handle(const kir::ReductionOp* node) {
  indent() << gen(node->out()) << " = "
           << "REDUCTION(op='" << node->getReductionOpType() << "'"
           << ", in=" << gen(node->in()) << ", init=" << gen(node->init())
           << ", pred=" << gen(node->pred()) << ")\n";
}

void IrPrinter::handle(const kir::GridReduction* node) {
  const auto* reduction_op = node->reduction_op();
  indent() << gen(reduction_op->out()) << " = "
           << "GRID_REDUCTION(op='" << reduction_op->getReductionOpType() << "'"
           << ", in=" << gen(reduction_op->in())
           << ", init=" << gen(reduction_op->init())
           << ", pred=" << gen(reduction_op->pred()) << ")\n";
  indent() << kTab << ".reduction_buffer=" << gen(node->reduction_buffer())
           << "\n";
  indent() << kTab << ".sync_buffer=" << gen(node->sync_buffer()) << "\n";
  indent() << kTab << ".grid_pred=" << gen(node->pred()) << "\n";
}

void IrPrinter::handle(const kir::BroadcastOp* node) {
  indent() << gen(node->out()) << " = BROADCAST(" << gen(node->in()) << ")\n";
}

void IrPrinter::handle(const kir::ForLoop* node) {
  indent() << "FOR " << gen(node->index()) << " in " << gen(node->iter_domain())
           << ":\n";
  handleBlock(node->body());
}

void IrPrinter::handle(const kir::IfThenElse* node) {
  indent() << "IF " << gen(node->cond()) << ":\n";
  handleBlock(node->thenBody());
  if (node->hasElse()) {
    indent() << "ELSE:\n";
    handleBlock(node->elseBody());
  }
}

void IrPrinter::handle(const kir::Allocate* node) {
  indent() << gen(node->buffer()) << " = ALLOCATE("
           << "mem_type=" << node->getMemoryType() << ", "
           << "size=" << gen(node->size()) << ", "
           << "zero_init=" << boolLiteral(node->zeroInit()) << ")\n";
}

void IrPrinter::handle(const kir::Sync* node) {
  indent() << "SYNC(war_hazard=" << boolLiteral(node->isWarHazardSync())
           << ")\n";
}

std::string toString(const Statement* stmt) {
  std::stringstream ss;
  IrPrinter ir_printer(ss);
  ir_printer.printNode(stmt);
  return ss.str();
}

} // namespace kir
} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
