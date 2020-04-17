#include <torch/csrc/jit/codegen/cuda/ir_iostream.h>
#include <torch/csrc/jit/codegen/cuda/fusion.h>
#include <torch/csrc/jit/codegen/cuda/ir_all_nodes.h>

#include <iostream>

namespace torch {
namespace jit {
namespace fuser {

namespace {
// Make sure we can inline something, before we attempt to.
void check_inlineable(const IRInputOutput* const irio) {
  for (auto inp : irio->inputs())
    TORCH_CHECK(
        inp->isScalar(),
        "Printing inline computations involving values other than scalars is not currently supported.");
  TORCH_CHECK(
      irio->nOutputs() == 1,
      "Cannot print inline computations if there's more than one output.");
  TORCH_CHECK(
      irio->output(0)->isScalar(),
      "Printing inline computations involving values other than scalars is not currently supported.");
}
} // namespace

void IRPrinter::printHeader(Fusion* fusion, const std::string& kernel_name_) {
  // ceilDiv Helper funtion
  os << "__device__ int ceilDiv(const int a, const int b) {\n"
     << "  return (a + b - 1) / b;\n"
     << "}\n\n";

  os << "__global__ void " << kernel_name_ << "(";

  std::deque<Val*> vals;
  for (decltype(fusion->nInputs()) i{0}; i < fusion->nInputs(); i++)
    vals.push_back(fusion->input(i));
  for (decltype(fusion->nOutputs()) i{0}; i < fusion->nOutputs(); i++)
    vals.push_back(fusion->output(i));

  for (Val* val : vals) {
    switch (val->getValType().value()) {
      case (ValType::TensorView):
        os << "Tensor<" << val->getDataType().value() << ", "
           << static_cast<TensorView*>(val)->getRootDomain()->nDims() << "> T"
           << val->name();
        break;
      case (ValType::Scalar):
        os << val->getDataType().value() << " " << val;
        break;
      default:
        TORCH_CHECK(
            false,
            "printHeader() found an input to the fusion of unexpected data type.");
    }

    if (val != vals.back())
      os << ", ";
  }

  os << "){\n";
  indent_size++;
}

void IRPrinter::handle(Fusion* fusion) {
  resetIndent();
  for (const Expr* expr : fusion->exprs()) {
    handle(expr);
  }
}

void IRPrinter::handle(const TensorDomain* const td) {
  os << "[ ";
  for (std::vector<const IterDomain*>::size_type i = 0; i < td->nDims(); i++) {
    handle(td->axis(i));
    if (i != td->nDims() - 1)
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
  if (!id->start()->isZeroInt()) {
    print_inline(id->start());
    os << " : ";
  }
  print_inline(id->extent());
  os << "}";
}

void IRPrinter::handle(const TensorIndex* const ti) {
  os << "T" << ti->view()->name() << "[ ";

  bool first = true;
  for (auto* ind : ti->indices()) {
    if (!first)
      os << " + ";
    print_inline(ind);
    first = false;
  }
  os << " ]";
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

void IRPrinter::handle(const NamedScalar* const i) {
  os << i->name();
}

namespace {

bool isTV(const Val* const val) {
  return (
      val->getValType().value() == ValType::TensorView ||
      val->getValType().value() == ValType::TensorIndex);
}

// Check if we're a TensorView op that we can generate code for.
bool isTVOp(const Expr* const expr) {
  if (expr->nOutputs() == 1 && isTV(expr->output(0)))
    return true;
  return false;
}
} // namespace

void IRPrinter::handle(const UnaryOp* const uop) {
  bool istvop = isTVOp(uop);
  if (!print_inline_) {
    indent();
    os << uop->out();
    if (istvop) {
      os << "\n";
      indent_size++;
      indent();
    }
    os << " = ";
  } else {
    check_inlineable(uop);
  }

  if (auto inline_uop = inline_op_str(uop->getUnaryOpType())) {
    os << inline_uop.value();
    handle(uop->in());
  } else {
    os << uop->getUnaryOpType() << "(";
    handle(uop->in());
    os << ")";
  }

  if (istvop)
    indent_size--;

  if (!print_inline_)
    os << ";\n";
}

void IRPrinter::handle(const BinaryOp* const bop) {
  bool istvop = isTVOp(bop);
  if (!print_inline_) {
    indent();
    os << bop->out();

    // tensor operations tend to be long, break them up into multiple lines
    if (istvop) {
      os << "\n";
      indent_size++;
      indent();
    }

    os << " = ";
  } else {
    check_inlineable(bop);
  }

  if (auto inline_bop = inline_op_str(bop->getBinaryOpType())) {
    handle(bop->lhs());
    if (istvop) {
      os << "\n";
      indent();
    }
    os << " " << inline_bop.value() << " ";
    handle(bop->rhs());
  } else {
    os << bop->getBinaryOpType() << "(";
    handle(bop->lhs());
    if (istvop) {
      os << "\n";
      indent();
    }
    os << ", ";
    handle(bop->rhs());
    os << ")";
  }

  if (istvop)
    indent_size--;

  if (!print_inline_)
    os << ";\n";
}

void IRPrinter::handle(const ForLoop* const fl) {
  if (fl->iter_domain()->isThread()) {
    for (auto& expr : fl->constBody().exprs())
      handle(expr);
    return;
  }

  indent();
  os << "for(size_t ";
  handle(fl->index());
  os << " = ";
  print_inline(fl->iter_domain()->start());
  os << "; ";
  handle(fl->index());
  os << " < ";
  print_inline(fl->iter_domain()->extent());
  os << "; ++";
  handle(fl->index());
  os << " ) {\n";
  indent_size++;
  for (auto& expr : fl->constBody().exprs())
    handle(expr);

  indent_size--;
  indent();
  os << "}\n";
}

void IRPrinter::handle(const IfThenElse* const ite) {
  indent();

  // IF
  os << "if ( ";
  print_inline(ite->cond());
  os << " ) { \n";

  indent_size++;
  for (auto& expr : ite->constBody().exprs()) {
    handle(expr);
  }
  indent_size--;

  // ELSE
  if (ite->hasElse()) {
    indent();
    os << "} else { \n";
    indent_size++;
    for (auto& expr : ite->constElseBody().exprs()) {
      handle(expr);
    }
    indent_size--;
  }
  indent();
  os << "}\n";
}

void IRPrinter::handle(const Allocate* const a) {
  indent();
  os << a->buf_type() << " T" << a->buffer()->name() << "[";
  print_inline(a->extent());
  os << "];" << std::endl;
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

void IRPrinter::printKernel(
    const std::vector<Expr*>& exprs,
    const std::string& kernel_name) {
  Fusion* fusion = FusionGuard::getCurFusion();

  printHeader(fusion, kernel_name);
  for (auto* expr : exprs) {
    handle(expr);
  }
  os << "}\n";
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
