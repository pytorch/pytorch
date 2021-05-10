#include <torch/csrc/jit/codegen/cuda/ir_iostream.h>

#include <torch/csrc/jit/codegen/cuda/fusion.h>
#include <torch/csrc/jit/codegen/cuda/instrumentation.h>
#include <torch/csrc/jit/codegen/cuda/ir_all_nodes.h>
#include <torch/csrc/jit/codegen/cuda/lower_utils.h>

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
  for (size_t i = 0; i < td->nDims(); i++) {
    handle(td->axis(i));
    if (i != td->nDims() - 1)
      os_ << ", ";
  }
  os_ << " ]";
}

void IrPrinter::handle(const TensorView* tv) {
  if (tv->nDims() == 0) {
    switch (tv->getDataType().value()) {
      case DataType::Bool:
        os_ << "b";
        break;
      case DataType::Float:
        os_ << "f";
        break;
      case DataType::Half:
        os_ << "h";
        break;
      case DataType::Int:
        os_ << "i";
        break;
      default:
        TORCH_INTERNAL_ASSERT(
            false, "Did not recognize type ", tv->getDataType().value());
    }
    os_ << tv->name();

  } else {
    os_ << "T" << tv->name();
    handle(tv->domain());

    if (tv->getComputeAtView() != nullptr) {
      os_ << " compute_at( ";
      os_ << "T" << tv->getComputeAtView()->name();
      os_ << ", " << tv->getRelativeComputeAtAxis() << " )";
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
  if (print_inline_ && FusionGuard::getCurFusion()->origin(b) != nullptr) {
    os_ << "( ";
    handle(FusionGuard::getCurFusion()->origin(b));
    os_ << " )";
    return;
  }

  if (b->isSymbolic()) {
    os_ << "b" << b->name();
  } else {
    os_ << "bool(" << *(b->value()) << ")";
  }
}

void IrPrinter::handle(const Float* f) {
  if (print_inline_ && FusionGuard::getCurFusion()->origin(f) != nullptr) {
    os_ << "( ";
    handle(FusionGuard::getCurFusion()->origin(f));
    os_ << " )";
    return;
  }

  if (f->isSymbolic()) {
    os_ << "f" << f->name();
  } else {
    os_ << "float("
        << std::setprecision(
               std::numeric_limits<Float::ScalarType>::max_digits10)
        << *(f->value()) << ")";
  }
}

void IrPrinter::handle(const Half* h) {
  if (print_inline_ && FusionGuard::getCurFusion()->origin(h) != nullptr) {
    os_ << "( ";
    handle(FusionGuard::getCurFusion()->origin(h));
    os_ << " )";
    return;
  }

  if (h->isSymbolic()) {
    os_ << "h" << h->name();
  } else {
    os_ << "__float2half(" << *(h->value()) << ")";
  }
}

void IrPrinter::handle(const Int* i) {
  if (print_inline_) {
    if (auto def = FusionGuard::getCurFusion()->origin(i)) {
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

void IrPrinter::handle(const kir::Bool* b) {
  os_ << "kir::Bool (use kir::toString() to print Kernel IR nodes)";
}

void IrPrinter::handle(const kir::Float* f) {
  os_ << "kir::Float (use kir::toString() to print Kernel IR nodes)";
}

void IrPrinter::handle(const kir::Half* h) {
  os_ << "kir::Half (use kir::toString() to print Kernel IR nodes)";
}

void IrPrinter::handle(const kir::Int* i) {
  os_ << "kir::Int (use kir::toString() to print Kernel IR nodes)";
}

void IrPrinter::handle(const kir::NamedScalar*) {
  os_ << "kir::NamedScalar (use kir::toString() to print Kernel IR nodes)";
}

void IrPrinter::handle(const kir::TensorIndex*) {
  os_ << "kir::TensorIndex (use kir::toString() to print Kernel IR nodes)";
}

void IrPrinter::handle(const kir::IterDomain*) {
  os_ << "kir::IterDomain (use kir::toString() to print Kernel IR nodes)";
}

void IrPrinter::handle(const kir::TensorDomain*) {
  os_ << "kir::TensorDomain (use kir::toString() to print Kernel IR nodes)";
}

void IrPrinter::handle(const kir::TensorView*) {
  os_ << "kir::TensorView (use kir::toString() to print Kernel IR nodes)";
}

static bool isTV(const Val* val) {
  return val->getValType().value() == ValType::TensorView ||
      val->getValType().value() == ValType::TensorIndex;
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

  if (auto inline_uop = inline_op_str(uop->getUnaryOpType())) {
    os_ << inline_uop.value();
    handle(uop->in());
  } else {
    if (uop->getUnaryOpType() == UnaryOpType::Cast) {
      c10::optional<std::string> cast_str = cast_func_str(std::make_pair(
          uop->in()->getDataType().value(), uop->out()->getDataType().value()));
      TORCH_INTERNAL_ASSERT(cast_str != c10::nullopt, "Unsupported Cast");
      os_ << cast_str.value();
    } else {
      os_ << uop->getUnaryOpType();
    }
    os_ << "(";
    if (uop->getUnaryOpType() == UnaryOpType::RandLike)
      os_ << "rnd";
    else
      handle(uop->in());
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

  if (auto inline_bop = inline_op_str(bop->getBinaryOpType())) {
    handle(bop->lhs());
    if (istvop) {
      os_ << "\n";
      indent();
    }
    os_ << " " << inline_bop.value() << " ";
    handle(bop->rhs());
  } else {
    os_ << bop->getBinaryOpType() << "(";
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

void IrPrinter::handle(const kir::UnaryOp* uop) {
  os_ << "kir::UnaryOp (use kir::toString() to print Kernel IR nodes)";
}

void IrPrinter::handle(const kir::BinaryOp* bop) {
  os_ << "kir::BinaryOp (use kir::toString() to print Kernel IR nodes)";
}

void IrPrinter::handle(const kir::TernaryOp* top) {
  os_ << "kir::TernaryOp (use kir::toString() to print Kernel IR nodes)";
}

void IrPrinter::handle(const ReductionOp* rop) {
  TORCH_CHECK(rop->out()->getValType() != ValType::TensorIndex);
  indent();
  os_ << rop->out() << " = reduction( " << rop->in()
      << ", op = " << rop->getReductionOpType()
      << ", initial value = " << rop->init() << " )\n";
}

void IrPrinter::handle(const kir::ReductionOp* rop) {
  os_ << "kir::ReductionOp (use kir::toString() to print Kernel IR nodes)";
}

void IrPrinter::handle(const kir::GridReduction* gr) {
  os_ << "kir::GridReduction (use kir::toString() to print Kernel IR nodes)";
}

void IrPrinter::handle(const BroadcastOp* bop) {
  TORCH_CHECK(bop->out()->getValType() != ValType::TensorIndex);
  indent();
  os_ << bop->out() << " = broadcast( " << bop->in() << " )\n";
}

void IrPrinter::handle(const kir::BroadcastOp*) {
  os_ << "kir::BroadcastOp (use kir::toString() to print Kernel IR nodes)";
}

void IrPrinter::handle(const kir::ForLoop* fl) {
  os_ << "kir::ForLoop (use kir::toString() to print Kernel IR nodes)";
}

void IrPrinter::handle(const kir::IfThenElse* ite) {
  os_ << "kir::IfThenElse (use kir::toString() to print Kernel IR nodes)";
}

void IrPrinter::handle(const kir::Allocate* a) {
  os_ << "kir::Allocate (use kir::toString() to print Kernel IR nodes)";
}

void IrPrinter::handle(const kir::Sync* a) {
  os_ << "kir::Sync (use kir::toString() to print Kernel IR nodes)";
}

void IrPrinter::handle(const Split* s) {
  os_ << "Split: ";
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
