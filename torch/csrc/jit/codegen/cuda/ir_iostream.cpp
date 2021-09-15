#include <torch/csrc/jit/codegen/cuda/ir_iostream.h>
#include <torch/csrc/jit/codegen/cuda/ir_printer.h>

#include <torch/csrc/jit/codegen/cuda/fusion.h>
#include <torch/csrc/jit/codegen/cuda/instrumentation.h>
#include <torch/csrc/jit/codegen/cuda/ir_all_nodes.h>
#include <torch/csrc/jit/codegen/cuda/ir_utils.h>
#include <torch/csrc/jit/codegen/cuda/lower_utils.h>

#include <c10/util/irange.h>

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {

// Make sure we can inline something, before we attempt to.
static void checkInlineable(const Expr* expr) {
  for (auto input : expr->inputs()) {
    TORCH_CHECK(
        input->isScalar(),
        "Printing inline computations involving values other than scalars is not currently supported.");
  }
  TORCH_CHECK(
      expr->outputs().size() == 1,
      "Cannot print inline computations if there's more than one output.");
  TORCH_CHECK(
      expr->output(0)->isScalar(),
      "Printing inline computations involving values other than scalars is not currently supported.");
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

void IrPrinter::handle(Fusion* fusion) {
  FUSER_PERF_SCOPE("IrPrinter");
  resetIndent();
  for (const Expr* expr : fusion->exprs()) {
    handle(expr);
  }
}

void IrPrinter::handle(const TensorDomain* td) {
  if (td->nDims() == 0) {
    os_ << "[ 0 ]";
    return;
  }
  os_ << "[ ";
  for (const auto i : c10::irange(td->nDims())) {
    handle(td->axis(i));
    if (i != td->nDims() - 1)
      os_ << ", ";
  }
  os_ << " ]";
}

void IrPrinter::handle(const TensorView* tv) {
  if (tv->nDims() == 0) {
    os_ << typePrefix(tv->getDataType().value()) << tv->name();
  } else {
    os_ << "T" << tv->name();
    switch (tv->getMemoryType()) {
      case MemoryType::Global:
        os_ << "_g";
        break;
      case MemoryType::Shared:
        os_ << "_s";
        break;
      case MemoryType::Local:
        os_ << "_l";
        break;
    }
    handle(tv->domain());

    if (tv->getComputeAtPosition() > 0) {
      os_ << " ca_pos( ";
      os_ << tv->getComputeAtPosition();
      os_ << " )";
    }
    if (tv->getMaxProducerPosition() > 0) {
      os_ << " produce_pos( ";
      os_ << tv->getMaxProducerPosition();
      os_ << ")";
    }
  }
}

void IrPrinter::handle(const IterDomain* id) {
  os_ << id->getIterType();
  os_ << id->getParallelType();
  os_ << id->name();
  os_ << "{";
  if (!id->start()->isZeroInt()) {
    print_inline(id->start());
    os_ << " : ";
  }
  print_inline(id->extent());
  os_ << "}";
  if (id->isRFactorProduct())
    os_ << "rf";
}

void IrPrinter::handle(const Bool* b) {
  if (print_inline_ && b->definition() != nullptr) {
    os_ << "( ";
    handle(b->definition());
    os_ << " )";
    return;
  }

  if (b->isSymbolic()) {
    os_ << "b" << b->name();
  } else {
    os_ << "bool(" << *(b->value()) << ")";
  }
}

void IrPrinter::handle(const Double* d) {
  if (print_inline_ && d->definition() != nullptr) {
    os_ << "( ";
    handle(d->definition());
    os_ << " )";
    return;
  }

  if (d->isSymbolic()) {
    os_ << "d" << d->name();
  } else {
    os_ << "double("
        << std::setprecision(
               std::numeric_limits<Double::ScalarType>::max_digits10)
        << *(d->value()) << ")";
  }
}

void IrPrinter::handle(const Int* i) {
  if (print_inline_) {
    if (auto def = i->definition()) {
      os_ << "( ";
      handle(def);
      os_ << " )";
      return;
    }
  }

  if (i->isSymbolic()) {
    os_ << "i" << i->name();
  } else {
    os_ << *(i->value());
  }
}

void IrPrinter::handle(const NamedScalar* i) {
  os_ << i->name();
}

static bool isTV(const Val* val) {
  return val->getValType().value() == ValType::TensorView;
}

// Check if we're a TensorView op that we can generate code for.
static bool isTVOp(const Expr* expr) {
  return expr->outputs().size() == 1 && isTV(expr->outputs().front());
}

void IrPrinter::handle(const UnaryOp* uop) {
  bool istvop = isTVOp(uop);
  if (!print_inline_) {
    indent();
    os_ << uop->out();
    if (istvop) {
      os_ << "\n";
      indent_size_++;
      indent();
    }
    os_ << " = ";
  } else {
    checkInlineable(uop);
  }

  auto op_type = uop->getUnaryOpType();

  if (auto inline_uop = inline_op_str(op_type)) {
    os_ << inline_uop.value();
    handle(uop->in());
  } else {
    if (op_type == UnaryOpType::Cast) {
      c10::optional<std::string> cast_str = cast_func_str(std::make_pair(
          uop->in()->getDataType().value(), uop->out()->getDataType().value()));
      TORCH_INTERNAL_ASSERT(cast_str != c10::nullopt, "Unsupported Cast");
      os_ << cast_str.value();
    } else {
      if (alsoBooleanOperator(op_type) &&
          uop->out()->getDataType().value() == DataType::Bool) {
        os_ << stringifyBooleanOp(op_type);
      } else {
        os_ << op_type;
      }
      if (uop->out()->getDataType().value() == DataType::Float &&
          needFloatSuffix(op_type)) {
        os_ << "f";
      }
    }
    if (op_type == UnaryOpType::RandLike) {
      os_ << "(";
      handle(uop->in());
    } else {
      os_ << "(";
      handle(uop->in());
    }
    os_ << ")";
  }

  if (istvop)
    indent_size_--;

  if (!print_inline_)
    os_ << ";\n";
}

void IrPrinter::handle(const BinaryOp* bop) {
  bool istvop = isTVOp(bop);
  if (!print_inline_) {
    indent();
    os_ << bop->out();

    // tensor operations tend to be long, break them up into multiple lines
    if (istvop) {
      os_ << "\n";
      indent_size_++;
      indent();
    }

    os_ << " = ";
  } else {
    checkInlineable(bop);
  }

  auto op_type = bop->getBinaryOpType();
  if (auto inline_bop = inline_op_str(op_type)) {
    handle(bop->lhs());
    if (istvop) {
      os_ << "\n";
      indent();
    }
    os_ << " " << inline_bop.value() << " ";
    handle(bop->rhs());
  } else {
    if (alsoBooleanOperator(op_type) &&
        bop->out()->getDataType().value() == DataType::Bool) {
      os_ << stringifyBooleanOp(op_type);
    } else {
      os_ << op_type;
    }
    if (bop->out()->getDataType().value() == DataType::Float &&
        needFloatSuffix(op_type)) {
      os_ << "f";
    }
    os_ << "(";
    handle(bop->lhs());
    if (istvop) {
      os_ << "\n";
      indent();
    }
    os_ << ", ";
    handle(bop->rhs());
    os_ << ")";
  }

  if (istvop)
    indent_size_--;

  if (!print_inline_)
    os_ << ";\n";
}

void IrPrinter::handle(const TernaryOp* top) {
  bool istvop = isTVOp(top);
  if (!print_inline_) {
    indent();
    os_ << top->out();

    // tensor operations tend to be long, break them up into multiple lines
    if (istvop) {
      os_ << "\n";
      indent_size_++;
      indent();
    }

    os_ << " = ";
  } else {
    checkInlineable(top);
  }

  os_ << top->getTernaryOpType() << "(";
  handle(top->in1());
  if (istvop) {
    os_ << "\n";
    indent();
  }
  os_ << ", ";
  handle(top->in2());
  if (istvop) {
    os_ << "\n";
    indent();
  }
  os_ << ", ";
  handle(top->in3());
  os_ << ")";

  if (istvop)
    indent_size_--;

  if (!print_inline_)
    os_ << ";\n";
}

void IrPrinter::handle(const ReductionOp* rop) {
  indent();
  os_ << rop->out() << " = reduction( " << rop->in()
      << ", op = " << rop->getReductionOpType()
      << ", initial value = " << rop->init() << " )\n";
}

void IrPrinter::handle(const WelfordOp* wop) {
  indent();
  os_ << wop->outAvg() << "(Avg),\n"
      << wop->outVar() << "(Var),\n"
      << wop->outN() << "(Count)"
      << "\n = Welford ( ";
  if (wop->singleValue()) {
    os_ << wop->inAvg() << "(Avg), ";
  } else {
    os_ << wop->inAvg() << "(Avg)\n  " << wop->inVar() << "(Var)\n  "
        << wop->inN() << "(Count)";
  }
  if (wop->hasInit()) {
    os_ << "\n  initial value = " << wop->initAvg() << "(Avg)\n  "
        << wop->initVar() << "(Var)\n  " << wop->initN() << "(N)";
  }
  os_ << " )\n";
}

void IrPrinter::handle(const BroadcastOp* bop) {
  indent();
  os_ << bop->out() << " = broadcast( " << bop->in() << " )\n";
}

void IrPrinter::handle(const TransposeOp* top) {
  indent();
  os_ << top->out() << " = transpose( " << top->in() << " )\n";
}

void IrPrinter::handle(const ShiftOp* sop) {
  indent();
  os_ << sop->out() << " = shift( " << sop->in() << ", {" << sop->offsets()
      << "} )\n";
}

void IrPrinter::handle(const GatherOp* op) {
  indent();
  os_ << op->out() << " = gather( " << op->in() << ", {";
  bool no_comma = true;
  for (const auto& s : op->windowShape()) {
    if (!no_comma) {
      os_ << ", ";
    }
    os_ << s;
    no_comma = false;
  }
  os_ << "}, {";
  no_comma = true;
  for (const auto& pad : op->padWidth()) {
    if (!no_comma) {
      os_ << ", ";
    }
    os_ << "{" << pad[0] << ", " << pad[1] << "}";
    no_comma = false;
  }
  os_ << "} )\n";
}

void IrPrinter::handle(const Split* s) {
  os_ << (s->innerSplit() ? "Split: " : "Outer split: ");
  handle(s->in());
  os_ << " by factor " << s->factor() << " -> ";
  handle(s->outer());
  os_ << ", ";
  handle(s->inner());
  os_ << "\n";
}

void IrPrinter::handle(const Merge* m) {
  os_ << "Merge: ";
  handle(m->outer());
  os_ << " and ";
  handle(m->inner());
  os_ << " -> ";
  handle(m->out());
  os_ << "\n";
}

void IrTransformPrinter::handle(Fusion* f) {
  auto all_vals = f->usedMathVals();

  for (auto tv : ir_utils::filterByType<TensorView>(all_vals)) {
    IrPrinter::handle(tv);
    os() << "\n";
    printTransforms(tv);
  }
}

void IrTransformPrinter::printTransforms(TensorView* tv) {
  auto root_domain = tv->getMaybeRFactorDomain();
  auto all_exp = DependencyCheck::getAllExprsBetween(
      {root_domain.begin(), root_domain.end()},
      {tv->domain()->domain().begin(), tv->domain()->domain().end()});

  os() << " root domain : (";
  for (size_t root_idx = 0; root_idx < root_domain.size(); root_idx++) {
    IrPrinter::handle(root_domain[root_idx]);
    if (root_idx + 1 < root_domain.size()) {
      os() << ",";
    }
  }
  os() << ")\n";

  for (auto exp : all_exp) {
    os() << "    ";
    IrPrinter::handle(exp);
  }
}

std::ostream& operator<<(std::ostream& os, const Statement* stmt) {
  IrPrinter p(os);
  p.handle(stmt);
  return os;
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
