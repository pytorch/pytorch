#include <torch/csrc/jit/codegen/cuda/ir_iostream.h>
#include <torch/csrc/jit/codegen/cuda/fusion.h>
#include <torch/csrc/jit/codegen/cuda/ir_all_nodes.h>
#include <torch/csrc/jit/codegen/cuda/tensor.h>

#include <iostream>

namespace torch {
namespace jit {
namespace fuser {

void IRPrinter::handle(Fusion* fusion) {
  for (const Expr* expr : fusion->exprs()) {
    handle(expr);
  }
}

void IRPrinter::handle(const TensorDomain* const td) {
  os << "[ ";
  for (std::vector<const IterDomain*>::size_type i = 0; i < td->size(); i++) {
    handle(td->axis(i));
    if (i != td->size() - 1)
      os << ", ";
  }
  os << " ]";
}

void IRPrinter::handle(const TensorView* const tv) {
  os << "T" << tv->name();
  handle(tv->domain());

  if (tv->getComputeAtView() != nullptr) {
    os << " compute_at( ";
    os << "T" << tv->getComputeAtView()->name();
    os << ", " << tv->getComputeAtAxis() << " )";
  }
}

void IRPrinter::handle(const IterDomain* const id) {
  if (id->isReduction())
    os << "r";
  else
    os << "i";
  switch (id->parallel_method()) {
    case (ParallelType::Vectorize):
      os << "V";
      break;
    case (ParallelType::Unroll):
      os << "U";
      break;
    case (ParallelType::Serial):
      os << "S";
      break;
    default:
      os << id->parallel_method();
  }
  os << "{";
  print_inline(id->size());
  os << "}";
}

void IRPrinter::handle(const TensorContiguity* const t) {
  os << "format_tag: " << t->getContiguityTag();
}

void IRPrinter::handle(const Float* const f) {
  if (print_inline_ && FusionGuard::getCurFusion()->origin(f) != nullptr) {
    os << "( ";
    handle(FusionGuard::getCurFusion()->origin(f));
    os << " )";
    return;
  }

  if (f->isSymbolic()) {
    os << "f" << f->name();
  } else {
    os << "float(" << *(f->value()) << ")";
  }
}

void IRPrinter::handle(const Int* const i) {
  if (print_inline_ && FusionGuard::getCurFusion()->origin(i) != nullptr) {
    os << "( ";
    handle(FusionGuard::getCurFusion()->origin(i));
    os << " )";
    return;
  }

  if (i->isSymbolic()) {
    os << "i" << i->name();
  } else {
    os << *(i->value());
  }
}
namespace {
// Make sure we can inline something, before we attempt to.
void check_inlineable(const IRInputOutput* const irio) {
  for (auto inp : irio->inputs())
    TORCH_CHECK(
        inp->getValType().value() == ValType::Scalar,
        "Printing inline computations involving values other than scalars is not currently supported.");
  TORCH_CHECK(
      irio->nOutputs() == 1,
      "Cannot print inline computations if there's more than one output.");
  TORCH_CHECK(
      irio->output(0)->getValType().value() == ValType::Scalar,
      "Printing inline computations involving values other than scalars is not currently supported.");
}
} // namespace

void IRPrinter::handle(const UnaryOp* const uop) {
  if (print_inline_)
    check_inlineable(uop);

  if (!print_inline_)
    os << uop->out() << " = ";
  if (auto inline_uop = inline_op_str(uop->getUnaryOpType())) {
    os << inline_uop.value();
    handle(uop->in());
  } else {
    os << uop->getUnaryOpType() << "(";
    handle(uop->in());
    os << ")";
  }

  if (!print_inline_)
    os << "\n";
}

void IRPrinter::handle(const BinaryOp* const bop) {
  if (print_inline_)
    check_inlineable(bop);

  if (!print_inline_)
    os << bop->out() << " = ";
  if (auto inline_bop = inline_op_str(bop->getBinaryOpType())) {
    handle(bop->lhs());
    os << " " << inline_bop.value() << " ";
    handle(bop->rhs());
  } else {
    os << bop->getBinaryOpType() << "(";
    handle(bop->lhs());
    os << ", ";
    handle(bop->rhs());
    os << ")";
  }

  if (!print_inline_)
    os << "\n";
}

void IRPrinter::handle(const Split* const s) {
  os << "Split: ";
  handle(s->in());
  os << " axis " << s->axis() << " by factor " << s->factor() << " -> ";
  handle(s->out());
  os << "\n";
}

void IRPrinter::handle(const Merge* const m) {
  os << "Merge: " << m->in() << " axis " << m->axis()
     << " with the following -> ";
  handle(m->out());
  os << "\n";
}

void IRPrinter::handle(const Reorder* const ro) {
  os << "Reorder: ";
  handle(ro->in());
  os << " -> ";
  handle(ro->out());
  os << "\n";
}

std::ostream& operator<<(std::ostream& os, const Statement* const stmt) {
  IRPrinter p(os);
  p.handle(stmt);
  return os;
}

std::ostream& operator<<(std::ostream& os, Fusion* f) {
  IRPrinter p(os);
  FusionGuard guard(f);
  p.handle(f);
  return os;
}

std::ostream& operator<<(std::ostream& os, Fusion& f) {
  return os << &f;
}

} // namespace fuser
} // namespace jit
} // namespace torch
