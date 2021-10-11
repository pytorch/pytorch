#include <c10/util/irange.h>
#include <torch/csrc/jit/codegen/cuda/kernel_ir_printer.h>

#include <torch/csrc/jit/codegen/cuda/instrumentation.h>
#include <torch/csrc/jit/codegen/cuda/type.h>

#include <utility>

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {
namespace kir {

namespace {

const char* boolLiteral(bool value) {
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

void IrPrinter::printNode(const kir::Node* node) {
  os_ << gen(node, true);
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
    os_ << gen(expr, true);
  }
  endBlock();
  os_ << "END.\n\n";
}

std::ostream& IrPrinter::indent() {
  for (const auto i : c10::irange(indent_level_)) {
    (void)i; // Suppress unused variable warning
    ir_str_ << kTab;
  }
  ir_str_ << margin_;
  return ir_str_;
}

std::string IrPrinter::gen(const kir::Node* node, bool top_level) {
  if (node == nullptr) {
    return "$nullptr";
  }

  // If we're generatign a top level statement we expect to start
  // with an empty set of uses
  TORCH_INTERNAL_ASSERT(!implicit_definition_ || uses_.empty() || !top_level);

  // Mark the node as generated
  visited_.insert(node);

  // Generate the node itself
  std::stringstream node_str;
  std::swap(node_str, ir_str_);
  node->accept(this);
  std::swap(node_str, ir_str_);

  if (!implicit_definition_) {
    return node_str.str();
  }

  if (top_level) {
    // Implicitly mark top level nodes as used, so we
    // get their definitions printed (useful for debugging)
    if (auto val = dynamic_cast<const kir::Val*>(node)) {
      uses_.insert(val);
    }

    // Make a copy of the node uses (and reset global state)
    const auto node_uses = uses_;
    uses_.clear();

    std::stringstream top_level_str;

    // Hoist implicit definitions
    for (auto use : node_uses) {
      const auto def = use->definition();
      if (def && visited_.find(def) == visited_.end()) {
        margin_ = "~ ";
        top_level_str << gen(def, true);
        margin_ = "";
      }
    }

    top_level_str << node_str.str();
    return top_level_str.str();
  } else {
    return node_str.str();
  }
}

std::string IrPrinter::use(const kir::Val* val) {
  if (val != nullptr) {
    uses_.insert(val);
  }
  return gen(val);
}

void IrPrinter::startBlock() {
  ++indent_level_;
}

void IrPrinter::endBlock() {
  TORCH_CHECK(indent_level_ > 0);
  --indent_level_;
}

void IrPrinter::handleBlock(const kir::Scope& scope) {
  // Save the uses of the parent scope
  decltype(uses_) outer_uses;
  std::swap(uses_, outer_uses);

  startBlock();
  for (auto expr : scope.exprs()) {
    ir_str_ << gen(expr, true);
  }
  endBlock();

  // Restore parent's uses
  std::swap(uses_, outer_uses);
}

void IrPrinter::visit(const kir::Bool* node) {
  if (node->isConst()) {
    ir_str_ << boolLiteral(*node->value());
  } else {
    ir_str_ << varName(node, "b");
  }
}

void IrPrinter::visit(const kir::Double* node) {
  if (node->isConst()) {
    const int digits = std::numeric_limits<Double::ScalarType>::max_digits10;
    ir_str_ << "double(" << std::setprecision(digits) << *node->value() << ")";
  } else {
    ir_str_ << varName(node, "d");
  }
}

void IrPrinter::visit(const kir::Int* node) {
  if (node->isConst()) {
    ir_str_ << *node->value();
  } else {
    ir_str_ << varName(node, "i");
  }
}

void IrPrinter::visit(const kir::NamedScalar* node) {
  ir_str_ << node->name();
}

void IrPrinter::visit(const kir::Predicate* node) {
  switch (node->predicate_type()) {
    case PredicateType::Inline: {
      ir_str_ << "Inline";
      break;
    }
    case PredicateType::Manual: {
      ir_str_ << node->value();
      break;
    }
    case PredicateType::Misaligned: {
      ir_str_ << "Misaligned";
      break;
    }
    case PredicateType::Padding: {
      ir_str_ << "Padding";
      break;
    }
    case PredicateType::Shift: {
      ir_str_ << "Shift";
      break;
    }
    case PredicateType::Unswitch: {
      ir_str_ << "Unswitch";
      break;
    }
    case PredicateType::Vectorize: {
      ir_str_ << "Vectorize";
      break;
    }
    default:
      break;
  }
}

void IrPrinter::visit(const kir::TensorIndex* node) {
  ir_str_ << gen(node->view()) << "[";
  for (auto index : node->indices()) {
    ir_str_ << use(index);
    if (index != node->indices().back()) {
      ir_str_ << ", ";
    }
  }
  ir_str_ << "]";
}

void IrPrinter::visit(const kir::IterDomain* node) {
  ir_str_ << varName(node, "id") << "[";
  if (node->isRFactorProduct()) {
    ir_str_ << "rfactor.";
  }
  ir_str_ << node->parallelType() << "." << node->iterType() << "("
          << use(node->start()) << " .. " << use(node->extent()) << ")]";
}

void IrPrinter::visit(const kir::TensorDomain*) {
  // TODO(kir): print Tensor shapes?
  ir_str_ << "kir::TensorDomain";
}

void IrPrinter::visit(const kir::TensorView* node) {
  // TODO(kir): print memory type too?
  ir_str_ << varName(node, "T");
}

void IrPrinter::visit(const kir::UnaryOp* node) {
  indent() << gen(node->out()) << " = ";

  auto op_type = node->operation();

  if (auto op = inline_op_str(op_type)) {
    if (alsoBooleanOperator(op_type) &&
        node->out()->dtype() == DataType::Bool) {
      ir_str_ << stringifyBooleanOp(op_type) << gen(node->in());
    } else {
      ir_str_ << *op << gen(node->in());
    }
  } else {
    if (op_type == UnaryOpType::Cast) {
      const auto cast_str =
          cast_func_str({node->in()->dtype(), node->out()->dtype()});
      ir_str_ << cast_str.value();
    } else {
      ir_str_ << op_type;
      if (needFloatSuffix(op_type) && node->out()->dtype() == DataType::Float) {
        ir_str_ << "f";
      }
    }

    if (op_type == UnaryOpType::RandLike) {
      ir_str_ << "(RND";
    } else {
      ir_str_ << "(";
      ir_str_ << use(node->in());
    }
    ir_str_ << ")";
  }

  ir_str_ << "\n";
}

void IrPrinter::visit(const kir::BinaryOp* node) {
  indent() << gen(node->out()) << " = ";

  const auto op_type = node->operation();
  const auto lhs = use(node->lhs());
  const auto rhs = use(node->rhs());

  if (auto op = inline_op_str(op_type)) {
    ir_str_ << lhs << " ";
    if (alsoBooleanOperator(op_type) &&
        node->out()->dtype() == DataType::Bool) {
      ir_str_ << stringifyBooleanOp(op_type);
    } else {
      ir_str_ << *op;
    }
    ir_str_ << " " << rhs;
  } else {
    ir_str_ << op_type;
    if (needFloatSuffix(op_type) && node->out()->dtype() == DataType::Float) {
      ir_str_ << "f";
    }
    ir_str_ << "(" << lhs << ", " << rhs << ")";
  }

  ir_str_ << "\n";
}

void IrPrinter::visit(const kir::TernaryOp* node) {
  indent() << gen(node->out()) << " = " << node->operation() << "("
           << use(node->in1()) << ", " << use(node->in2()) << ", "
           << use(node->in3()) << ")\n";
}

void IrPrinter::visit(const kir::ReductionOp* node) {
  indent() << gen(node->out()) << " = "
           << "REDUCTION(op='" << node->operation() << "'"
           << ", in=" << use(node->in()) << ", init=" << use(node->init())
           << ", pred=" << use(node->predicate()) << ")\n";
}

void IrPrinter::visit(const kir::WelfordOp* node) {
  indent() << gen(node->outVar()) << "," << gen(node->outAvg()) << ","
           << gen(node->outN()) << " = "
           << "Welford( inAvg=" << use(node->inAvg());
  if (!node->inN()->isOneInt()) {
    indent() << " inVar=" << use(node->inVar());
  }
  indent() << " inN=" << use(node->inN());
  if (!node->initN()->isZeroInt()) {
    indent() << ", initVar=" << use(node->initVar())
             << " initAvg=" << use(node->initAvg())
             << " initN=" << use(node->initN());
  }
  indent() << ", pred=" << use(node->predicate()) << ")\n";
}

void IrPrinter::visit(const kir::GridReduction* node) {
  const auto* reduction_op = node->reduction_op();
  indent() << gen(reduction_op->out()) << " = "
           << "GRID_REDUCTION(op='" << reduction_op->operation() << "'"
           << ", in=" << use(reduction_op->in())
           << ", init=" << use(reduction_op->init())
           << ", pred=" << use(reduction_op->predicate()) << ")\n";
  indent() << kTab << kTab
           << ".reduction_buffer=" << use(node->reduction_buffer()->buffer())
           << "\n";
  indent() << kTab << kTab
           << ".sync_buffer=" << use(node->sync_buffer()->buffer()) << "\n";
  indent() << kTab << kTab << ".grid_pred=" << use(node->predicate()) << "\n";
}

void IrPrinter::visit(const kir::GridWelford* node) {
  const auto* welford_op = node->welford_op();
  indent() << gen(welford_op->outVar()) << "," << gen(welford_op->outAvg())
           << "," << gen(welford_op->outN()) << " = "
           << "GRID_WELFORD("
           << "inAvg=" << use(welford_op->inAvg());
  if (!welford_op->inN()->isOneInt()) {
    indent() << ", inVar=" << use(welford_op->inVar());
  }
  indent() << ", inN=" << use(welford_op->inN());
  if (!welford_op->initN()->isZeroInt()) {
    indent() << ", initVar=" << use(welford_op->initVar())
             << " initAvg=" << use(welford_op->initAvg())
             << " initN=" << use(welford_op->initN());
  }
  indent() << ", pred=" << use(welford_op->predicate()) << ")\n";
  indent() << kTab << kTab
           << ".var_buffer=" << use(node->var_buffer()->buffer())
           << ".avg_buffer=" << use(node->avg_buffer()->buffer())
           << ".n_buffer=" << use(node->N_buffer()->buffer()) << "\n";
  indent() << kTab << kTab
           << ".sync_buffer=" << use(node->sync_buffer()->buffer()) << "\n";
  indent() << kTab << kTab << ".grid_pred=" << use(node->predicate()) << "\n";
}

void IrPrinter::visit(const kir::BroadcastOp* node) {
  indent() << gen(node->out()) << " = BROADCAST(" << use(node->in()) << ")\n";
}

void IrPrinter::visit(const kir::ForLoop* node) {
  indent() << "FOR " << gen(node->index()) << " in " << gen(node->iter_domain())
           << ":\n";
  handleBlock(node->body());
}

void IrPrinter::visit(const kir::IfThenElse* node) {
  indent() << "IF " << use(node->predicate()) << ":\n";
  handleBlock(node->thenBody());
  if (node->hasElse()) {
    indent() << "ELSE:\n";
    handleBlock(node->elseBody());
  }
}

void IrPrinter::visit(const kir::Allocate* node) {
  indent() << gen(node->buffer()) << " = ALLOCATE("
           << "mem_type=" << node->memoryType() << ", "
           << "size=" << use(node->size()) << ", "
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

void IrPrinter::visit(const kir::InitMagicZero* node) {
  indent() << "NVFUSER_DEFINE_MAGIC_ZERO\n";
}

void IrPrinter::visit(const kir::UpdateMagicZero* node) {
  indent() << "NVFUSER_UPDATE_MAGIC_ZERO\n";
}

std::string toString(const kir::Node* stmt, bool implicit_definitions) {
  std::stringstream ss;
  IrPrinter ir_printer(ss, implicit_definitions);
  ir_printer.printNode(stmt);
  return ss.str();
}

std::string toString(
    const std::vector<kir::Expr*>& exprs,
    bool implicit_definitions) {
  std::stringstream ss;
  IrPrinter ir_printer(ss, implicit_definitions);
  for (auto expr : exprs) {
    ir_printer.printNode(expr);
  }
  return ss.str();
}

} // namespace kir
} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
