#include <torch/csrc/jit/codegen/cuda/kernel_ir_printer.h>
#include <torch/csrc/jit/codegen/cuda/instrumentation.h>
#include <torch/csrc/jit/codegen/cuda/type.h>

#include <sstream>

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {
namespace kir {

namespace {

std::string boolLiteral(bool value) {
  return value ? "true" : "false";
}

std::string varName(const kir::Val* val, const char* prefix) {
  std::stringstream value_name;
  if (val == nullptr) {
    value_name << "$nullptr";
  } else if (val->name() != kInvalidStmName) {
    value_name << prefix << val->name();
  } else {
    value_name << "k" << prefix << val->id();
  }
  return value_name.str();
}

} // namespace

void IrPrinter::printNode(const kir::Node* stmt) {
  stmt->accept(this);
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
    expr->accept(this);
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

std::string IrPrinter::gen(const kir::Node* stmt) {
  if (stmt != nullptr) {
    std::stringstream ss;
    IrPrinter ir_printer(ss);
    ir_printer.printNode(stmt);
    return ss.str();
  } else {
    return "$nullptr";
  }
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
    expr->accept(this);
  }
  endBlock();
}

void IrPrinter::visit(const kir::Bool* node) {
  if (node->isConst()) {
    os_ << boolLiteral(*node->value());
  } else {
    os_ << varName(node, "b");
  }
}

void IrPrinter::visit(const kir::Float* node) {
  if (node->isConst()) {
    const int digits = std::numeric_limits<Float::ScalarType>::max_digits10;
    os_ << "float(" << std::setprecision(digits) << *node->value() << ")";
  } else {
    os_ << varName(node, "f");
  }
}

void IrPrinter::visit(const kir::Half* node) {
  if (node->isConst()) {
    os_ << "half(" << *node->value() << ")";
  } else {
    os_ << varName(node, "h");
  }
}

void IrPrinter::visit(const kir::Int* node) {
  if (node->isConst()) {
    os_ << *node->value();
  } else {
    os_ << varName(node, "i");
  }
}

void IrPrinter::visit(const kir::NamedScalar* node) {
  os_ << node->name();
}

void IrPrinter::visit(const kir::TensorIndex* node) {
  os_ << gen(node->view()) << "[";
  for (auto index : node->indices()) {
    os_ << gen(index);
    if (index != node->indices().back()) {
      os_ << ", ";
    }
  }
  os_ << "]";
}

void IrPrinter::visit(const kir::IterDomain* node) {
  if (node->isRFactorProduct()) {
    os_ << "rfactor.";
  }
  os_ << node->getParallelType() << "." << node->getIterType() << "("
      << gen(node->start()) << " .. " << gen(node->rawExtent()) << ")";
}

void IrPrinter::visit(const kir::TensorDomain*) {
  // TODO(kir): print Tensor shapes?
  os_ << "kir::TensorDomain";
}

void IrPrinter::visit(const kir::TensorView* node) {
  // TODO(KIR): print memory type too?
  os_ << varName(node, "T");
}

void IrPrinter::visit(const kir::UnaryOp* node) {
  indent() << gen(node->out()) << " = ";

  if (auto op = inline_op_str(node->operation())) {
    os_ << *op << gen(node->in());
  } else {
    if (node->operation() == UnaryOpType::Cast) {
      const auto cast_str =
          cast_func_str({node->in()->dtype(), node->out()->dtype()});
      os_ << cast_str.value();
    } else {
      os_ << node->operation();
    }

    os_ << "(";
    if (node->operation() == UnaryOpType::RandLike) {
      os_ << "RND";
    } else {
      os_ << gen(node->in());
    }
    os_ << ")";
  }

  os_ << "\n";
}

void IrPrinter::visit(const kir::BinaryOp* node) {
  indent() << gen(node->out()) << " = ";

  const auto operation = node->operation();
  const auto lhs = gen(node->lhs());
  const auto rhs = gen(node->rhs());

  if (auto op = inline_op_str(operation)) {
    os_ << lhs << " " << *op << " " << rhs;
  } else {
    os_ << operation << "(" << lhs << ", " << rhs << ")";
  }

  os_ << "\n";
}

void IrPrinter::visit(const kir::TernaryOp* node) {
  indent() << gen(node->out()) << " = " << node->operation() << "("
           << gen(node->in1()) << ", " << gen(node->in2()) << ", "
           << gen(node->in3()) << ")\n";
}

void IrPrinter::visit(const kir::ReductionOp* node) {
  indent() << gen(node->out()) << " = "
           << "REDUCTION(op='" << node->operation() << "'"
           << ", in=" << gen(node->in()) << ", init=" << gen(node->init())
           << ", pred=" << gen(node->predicate()) << ")\n";
}

void IrPrinter::visit(const kir::GridReduction* node) {
  const auto* reduction_op = node->reduction_op();
  indent() << gen(reduction_op->out()) << " = "
           << "GRID_REDUCTION(op='" << reduction_op->operation() << "'"
           << ", in=" << gen(reduction_op->in())
           << ", init=" << gen(reduction_op->init())
           << ", pred=" << gen(reduction_op->predicate()) << ")\n";
  indent() << kTab << kTab
           << ".reduction_buffer=" << gen(node->reduction_buffer()->buffer())
           << "\n";
  indent() << kTab << kTab
           << ".sync_buffer=" << gen(node->sync_buffer()->buffer()) << "\n";
  indent() << kTab << kTab << ".grid_pred=" << gen(node->predicate()) << "\n";
}

void IrPrinter::visit(const kir::BroadcastOp* node) {
  indent() << gen(node->out()) << " = BROADCAST(" << gen(node->in()) << ")\n";
}

void IrPrinter::visit(const kir::ForLoop* node) {
  indent() << "FOR " << gen(node->index()) << " in " << gen(node->iter_domain())
           << ":\n";
  handleBlock(node->body());
}

void IrPrinter::visit(const kir::IfThenElse* node) {
  indent() << "IF " << gen(node->cond()) << ":\n";
  handleBlock(node->thenBody());
  if (node->hasElse()) {
    indent() << "ELSE:\n";
    handleBlock(node->elseBody());
  }
}

void IrPrinter::visit(const kir::Allocate* node) {
  indent() << gen(node->buffer()) << " = ALLOCATE("
           << "mem_type=" << node->memoryType() << ", "
           << "size=" << gen(node->size()) << ", "
           << "zero_init=" << boolLiteral(node->zeroInit()) << ")\n";
  if (node->alias() != nullptr) {
    indent() << kTab << kTab << ".alias=" << gen(node->alias()->buffer())
             << "\n";
  }
}

void IrPrinter::visit(const kir::Sync* node) {
  indent() << "SYNC(war_hazard=" << boolLiteral(node->isWarHazardSync())
           << ")\n";
}

std::string toString(const kir::Node* stmt) {
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
