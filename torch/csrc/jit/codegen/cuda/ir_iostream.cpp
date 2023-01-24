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

namespace {
const char* boolLiteral(bool value) {
  return value ? "true" : "false";
}

std::string varName(const Val* val) {
  std::stringstream value_name;
  if (val == nullptr) {
    value_name << "$nullptr";
  } else {
    value_name << val->name();
  }
  return value_name.str();
}

} // namespace

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

void IrPrinter::handle(const kir::Kernel* kernel) {
  TORCH_CHECK(kernel != nullptr);

  // kernel declaration
  os_ << "\nKERNEL (";
  for (auto in : kernel->inputs()) {
    handle(in);
    if (in != kernel->inputs().back()) {
      os_ << ", ";
    }
  }
  os_ << ") -> (";
  for (auto out : kernel->outputs()) {
    handle(out);
    if (out != kernel->outputs().back()) {
      os_ << ", ";
    }
  }
  os_ << ") :\n";

  // kernel body
  indent_size_++;
  for (auto expr : kernel->topLevelExprs()) {
    handle(expr);
  }
  indent_size_--;
  os_ << "END.\n\n";
}

void IrPrinter::handle(kir::Kernel& kernel) {
  handle(&kernel);
}

void IrPrinter::handleScope(const kir::Scope& scope) {
  // Save the uses of the parent scope
  indent_size_++;
  for (auto expr : scope.exprs()) {
    handle(expr);
  }
  indent_size_--;
}

void IrPrinter::handle(const IterDomain* id) {
  os_ << id->getIterType();
  os_ << id->getParallelType();
  os_ << varName(id);
  os_ << "{";
  if (!id->start()->isZeroInt()) {
    print_inline(id->start());
    os_ << " : ";
  }
  if (id->stop() != id->extent()) {
    print_inline(id->stop());
    os_ << " : ";
  }
  if (id->isBroadcast() && id->hasExpandedExtent()) {
    print_inline(id->expandedExtent());
  } else {
    print_inline(id->extent());
  }
  os_ << "}";
  if (id->isRFactorProduct())
    os_ << "rf";
  if (id->hasPaddingToMultipleOfWarp()) {
    os_ << "_p";
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
  os_ << "T" << varName(tv);
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

void IrPrinter::handle(const Bool* b) {
  if (print_inline_ && b->definition() != nullptr) {
    os_ << "( ";
    handle(b->definition());
    os_ << " )";
    return;
  }

  os_ << "b" << varName(b);
  if (b->isConst()) {
    os_ << "(" << (b->value().value() ? "true" : "false") << ")";
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
    os_ << "d" << varName(d);
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
    os_ << "i" << varName(i);
  } else {
    os_ << *(i->value());
  }
}

void IrPrinter::handle(const ComplexDouble* c) {
  if (print_inline_) {
    if (auto def = c->definition()) {
      os_ << "( ";
      handle(def);
      os_ << " )";
      return;
    }
  }

  if (c->isSymbolic()) {
    os_ << "c" << varName(c);
  } else {
    os_ << "std::complex<double>"
        << std::setprecision(std::numeric_limits<double>::max_digits10)
        << *(c->value());
  }
}

void IrPrinter::handle(const NamedScalar* ns) {
  os_ << ns->name();
}

void IrPrinter::handle(const FullOp* fop) {
  if (!print_inline_) {
    indent();
    os_ << fop->output(0) << "\n";
    indent_size_++;
    indent();
    os_ << " = ";
  } else {
    checkInlineable(fop);
  }

  os_ << "full({";
  for (auto i : c10::irange(fop->inputs().size())) {
    if (i == fop->inputs().size() - 1) {
      os_ << "}";
    }
    if (i > 0) {
      os_ << ", ";
    }
    handle(fop->input(i));
  }
  os_ << ", " << fop->dtype() << ")";

  indent_size_--;

  if (!print_inline_)
    os_ << ";\n";
}

void IrPrinter::handle(const ARangeOp* aop) {
  if (!print_inline_) {
    indent() << aop->output(0);
    os_ << "\n";
    indent_size_++;
    indent();
    os_ << " = ";
  } else {
    checkInlineable(aop);
  }

  os_ << "arange(";
  handle(aop->start());
  os_ << ", ";
  handle(aop->end());
  os_ << ", ";
  handle(aop->step());
  os_ << ", " << aop->dtype() << ")";

  indent_size_--;

  if (!print_inline_)
    os_ << ";\n";
}

void IrPrinter::handle(const EyeOp* eop) {
  if (!print_inline_) {
    indent();
    os_ << eop->output(0) << "\n";
    indent_size_++;
    indent();
    os_ << " = ";
  } else {
    checkInlineable(eop);
  }

  os_ << "eye(";
  handle(eop->input(0));
  os_ << ", " << eop->dtype() << ")";

  indent_size_--;

  if (!print_inline_)
    os_ << ";\n";
}

void IrPrinter::handle(const UnaryOp* uop) {
  bool istvop = ir_utils::isTvOp(uop);
  if (!print_inline_) {
    indent() << uop->out();
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
    os_ << "(";
    handle(uop->in());
    os_ << ")";
  }

  if (istvop)
    indent_size_--;

  if (!print_inline_)
    os_ << ";\n";
}

void IrPrinter::handle(const BinaryOp* bop) {
  bool istvop = ir_utils::isTvOp(bop);
  if (!print_inline_) {
    indent() << bop->out();

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
  bool istvop = ir_utils::isTvOp(top);
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

void IrPrinter::handle(const RNGOp* rop) {
  if (!print_inline_) {
    indent();
    os_ << rop->output(0) << "\n";
    indent_size_++;
    indent();
    os_ << " = ";
  } else {
    checkInlineable(rop);
  }

  os_ << rop->getRNGOpType() << "({";
  bool first = true;
  for (auto i : rop->getShape()) {
    if (!first) {
      os_ << ", ";
    }
    handle(i);
    first = false;
  }
  os_ << "}";
  for (auto i : rop->getParameters()) {
    os_ << ", ";
    handle(i);
  }
  os_ << ", " << rop->dtype() << ")";

  indent_size_--;

  if (!print_inline_) {
    os_ << ";\n";
  }
}

void IrPrinter::handle(const ReductionOp* rop) {
  indent() << rop->out() << "\n";
  indent() << "   = reduction( " << rop->in()
           << ", op = " << rop->getReductionOpType()
           << ", initial value = " << rop->init()
           << ", allreduce = " << (rop->isAllreduce() ? "true" : "false")
           << " )\n";
}

void IrPrinter::handle(const GroupedReductionOp* grouped_rop) {
  indent() << "GroupedReductionOp(\n";
  ++indent_size_;
  for (const auto i : c10::irange(grouped_rop->numExprs())) {
    indent() << grouped_rop->output(i) << " = reduction( "
             << grouped_rop->input(i)
             << ", op = " << grouped_rop->getReductionOpType(i)
             << ", initial value = " << grouped_rop->initVal(i) << " )\n";
  }
  indent() << "allreduce = " << (grouped_rop->isAllreduce() ? "true" : "false")
           << " )\n";
  --indent_size_;
}

void IrPrinter::handle(const WelfordOp* wop) {
  indent() << wop->outAvg() << "(Avg),\n"
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
  os_ << "\n  allreduce = " << (wop->isAllreduce() ? "true" : "false");
  os_ << " )\n";
}

void IrPrinter::handle(const GroupedWelfordOp* grouped_wop) {
  indent() << "GroupedWelford(\n";
  ++indent_size_;
  for (const auto i : c10::irange(grouped_wop->numExprs())) {
    indent() << grouped_wop->outAvg(i) << " (Avg),\n";
    indent() << grouped_wop->outVar(i) << " (Var),\n";
    indent() << grouped_wop->outN(i) << " (Count)\n";
    indent() << " = Welford ( ";
    ++indent_size_;
    indent() << grouped_wop->inAvg(i) << " (Avg),\n";
    indent() << grouped_wop->inVar(i) << " (Var),\n";
    indent() << grouped_wop->inN(i) << " (Count)\n";
    indent() << "initial value =\n";
    ++indent_size_;
    indent() << grouped_wop->initAvg(i) << " (Avg),\n";
    indent() << grouped_wop->initVar(i) << " (Var),\n";
    indent() << grouped_wop->initN(i) << " (Count) )\n";
    indent_size_ -= 2;
  }
  indent() << "allreduce = " << (grouped_wop->isAllreduce() ? "true" : "false")
           << " )\n";
  --indent_size_;
}

void IrPrinter::handle(const LoadStoreOp* ldst) {
  indent() << ldst->out() << " = " << ldst->opType() << "( " << ldst->in()
           << " )\n";
}

void IrPrinter::handle(const BroadcastOp* bop) {
  indent() << bop->out() << "\n";
  indent() << "   = broadcast( " << bop->in() << " )\n";
}

void IrPrinter::handle(const Split* s) {
  os_ << (s->innerSplit() ? "Split: " : "Outer split: ");
  handle(s->in());
  os_ << " by factor " << s->factor() << " -> ";
  handle(s->outer());
  os_ << ", ";
  handle(s->inner());
  if (s->startOffset()) {
    os_ << ", start offset: ";
    handle(s->startOffset());
  }
  if (s->stopOffset()) {
    os_ << ", stop offset: ";
    handle(s->stopOffset());
  }
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

void IrPrinter::handle(const Swizzle2D* s) {
  os_ << s->swizzleType() << "(2D): ";
  handle(s->inX());
  os_ << " , ";
  handle(s->inY());
  os_ << " -> ";
  handle(s->outX());
  os_ << " , ";
  handle(s->outY());
  os_ << "\n";
}

void IrPrinter::handle(const TransposeOp* top) {
  indent() << top->out() << " = transpose( " << top->in() << " )\n";
}

void IrPrinter::handle(const ExpandOp* eop) {
  indent() << eop->out() << " = expand( " << eop->in() << ", {";
  std::stringstream ss;
  for (auto expanded_extent : eop->expanded_extents()) {
    if (ss.tellp()) {
      ss << ", ";
    }
    ss << expanded_extent;
  }
  os_ << ss.str() << "} )\n";
}

void IrPrinter::handle(const ShiftOp* sop) {
  indent() << sop->out() << " = shift( " << sop->in() << ", {" << sop->offsets()
           << "}, {" << sop->padWidth() << "} )\n";
}

void IrPrinter::handle(const MmaOp* mma) {
  indent() << mma->out() << " = mma(" << mma->inA() << "," << mma->inB();
  os_ << ")\n";
}

void IrPrinter::handle(const GatherOp* op) {
  indent() << op->out() << " = gather( " << op->in() << ", {";
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

void IrPrinter::handle(const ViewAsScalar* top) {
  indent() << top->out() << " = view_as_scalar( " << top->in() << ", "
           << top->vector_id() << " )\n";
}

void IrPrinter::handle(const ViewOp* top) {
  indent() << top->out() << " = view( " << top->in() << " )\n";
}

void IrPrinter::handle(const kir::Predicate* node) {
  switch (node->predicate_type()) {
    case PredicateType::Manual: {
      os_ << node->value();
      break;
    }
    default:
      os_ << node->predicate_type();
      break;
  }
}

void IrPrinter::handle(const kir::TensorIndex* ti) {
  os_ << "T" << varName(ti);
  switch (ti->view()->getMemoryType()) {
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
  os_ << "[";
  for (auto index : ti->indices()) {
    print_inline(index);
    if (index != ti->indices().back()) {
      os_ << ", ";
    }
  }
  os_ << "]";
  os_ << " view( T" << varName(ti->view()) << " )";
}

void IrPrinter::handle(const kir::Allocate* node) {
  indent();
  handle(node->buffer());
  os_ << " = ALLOCATE("
      << "mem_type=" << node->memoryType() << ", "
      << "size=";
  print_inline(node->size());
  os_ << ", "
      << "zero_init=" << boolLiteral(node->zeroInit()) << ")\n";
  if (node->alias() != nullptr) {
    indent() << kTab << ".alias=";
    handle(node->alias()->buffer());
    os_ << "\n";
  }
}

void IrPrinter::handle(const kir::BlockSync* node) {
  indent() << "BLOCKSYNC(war_hazard=" << boolLiteral(node->isWarHazardSync())
           << ")\n";
}

void IrPrinter::handle(const kir::CpAsyncWait* node) {
  indent() << "CPASYNC_WAIT(" << node->keepStages() << ")\n";
}

void IrPrinter::handle(const kir::CpAsyncCommit* node) {
  indent() << "CPASYNC_WAIT()\n";
}

void IrPrinter::handle(const kir::GridSync* node) {
  indent() << "GRIDSYNC(" << node->syncDims().toString() << ", ";
  handle(node->syncBuffer());
  os_ << ")\n";
}

void IrPrinter::handle(const kir::ForLoop* node) {
  indent() << "FOR ";
  handle(node->index());
  os_ << " in ";
  handle(node->iter_domain());
  os_ << ":\n";
  handleScope(node->body());
}

void IrPrinter::handle(const kir::IfThenElse* node) {
  indent() << "IF ";
  handle(node->predicate());
  os_ << ":\n";
  handleScope(node->thenBody());
  if (node->hasElse()) {
    indent() << "ELSE:\n";
    handleScope(node->elseBody());
  }
}

void IrPrinter::handle(const kir::GridBroadcast* node) {
  const auto* broadcast_op = node->broadcast_op();
  indent();
  handle(broadcast_op->out());
  os_ << " = "
      << "GRID_BROADCAST(in=";
  handle(broadcast_op->in());
  os_ << ")\n";
  indent() << kTab << ".broadcast_buffer=";
  handle(node->broadcast_buffer()->buffer());
  os_ << "\n";
  indent() << kTab << ".sync_buffer=";
  handle(node->sync_buffer()->buffer());
  os_ << "\n";
}

void IrPrinter::handle(const kir::GridReduction* node) {
  indent() << node->out() << " = reduction( " << node->in()
           << ", op = " << node->getReductionOpType()
           << ", initial value = " << node->init() << ",\n";
  ++indent_size_;
  indent() << "reduction buffer = " << node->reduction_buffer()->buffer()
           << ",\n";
  indent() << "sync buffer = " << node->sync_buffer()->buffer() << ",\n";
  indent() << "read predicate = ";
  if (node->predicate() != nullptr) {
    os_ << node->predicate();
  } else {
    os_ << "nullptr";
  }
  os_ << ",\n";
  indent() << "write predicate = ";
  if (node->writePredicate() != nullptr) {
    os_ << node->writePredicate();
  } else {
    os_ << "nullptr";
  }
  os_ << ",\n";
  indent() << "thread predicate = " << node->threadPredicate().toString()
           << ",\n";
  indent() << "allreduce = " << (node->isAllreduce() ? "true" : "false")
           << " )\n";
  --indent_size_;
}

void IrPrinter::handle(const kir::GroupedGridReduction* node) {
  indent() << "GroupedGridReduction(\n";
  ++indent_size_;
  for (const auto i : c10::irange(node->numExprs())) {
    indent() << node->output(i) << " = reduction( " << node->input(i)
             << ", op = " << node->getReductionOpType(i)
             << ", initial value = " << node->initVal(i)
             << ", reduction buffer = "
             << node->reduction_buffers().at(i)->buffer() << " )\n";
  }
  indent() << "sync buffer = " << node->sync_buffer()->buffer() << ",\n";
  indent() << "read predicate = ";
  if (node->predicate() != nullptr) {
    os_ << node->predicate();
  } else {
    os_ << "nullptr";
  }
  os_ << ",\n";
  indent() << "write predicate = ";
  if (node->writePredicate() != nullptr) {
    os_ << node->writePredicate();
  } else {
    os_ << "nullptr";
  }
  os_ << ",\n";
  indent() << "thread predicate = " << node->threadPredicate().toString()
           << ",\n";
  indent() << "allreduce = " << (node->isAllreduce() ? "true" : "false")
           << " )\n";
  --indent_size_;
}

void IrPrinter::handle(const kir::GridWelford* node) {
  std::cerr << "current indent size: " << indent_size_ << std::endl;
  const auto* welford_op = node->welford_op();
  indent() << welford_op->outAvg() << " (Avg),\n";
  indent() << welford_op->outVar() << " (Var),\n";
  indent() << welford_op->outN() << " (Count)\n";
  indent() << " = Welford (\n";
  ++indent_size_;
  indent() << welford_op->inAvg() << " (Avg),\n";
  indent() << welford_op->inVar() << " (Var),\n";
  indent() << welford_op->inN() << " (Count)\n";
  indent() << "initial value =\n";
  ++indent_size_;
  indent() << welford_op->initAvg() << " (Avg),\n";
  indent() << welford_op->initVar() << " (Var),\n";
  indent() << welford_op->initN() << " (Count),\n";
  --indent_size_;
  indent() << "reduction buffer =\n";
  ++indent_size_;
  indent() << node->avg_buffer()->buffer() << " (Avg),\n";
  indent() << node->var_buffer()->buffer() << " (Var),\n";
  indent() << node->N_buffer()->buffer() << " (Count),\n";
  --indent_size_;
  indent() << "sync buffer = " << node->sync_buffer()->buffer() << ",\n";
  indent() << "read predicate = ";
  if (welford_op->predicate() != nullptr) {
    os_ << welford_op->predicate();
  } else {
    os_ << "nullptr";
  }
  os_ << ",\n";
  indent() << "write predicate = ";
  if (welford_op->writePredicate() != nullptr) {
    os_ << welford_op->writePredicate();
  } else {
    os_ << "nullptr";
  }
  os_ << ",\n";
  indent() << "grid read predicate = ";
  if (node->predicate() != nullptr) {
    os_ << node->predicate();
  } else {
    os_ << "nullptr";
  }
  os_ << ",\n";
  indent() << "grid write predicate = ";
  if (node->writePredicate() != nullptr) {
    os_ << node->writePredicate();
  } else {
    os_ << "nullptr";
  }
  os_ << ",\n";
  indent() << "thread predicate = " << node->threadPredicate().toString()
           << ",\n";
  indent() << "allreduce = " << (welford_op->isAllreduce() ? "true" : "false")
           << " )\n";
  --indent_size_;
  std::cerr << "Ending indent size: " << indent_size_ << std::endl;
}

void IrPrinter::handle(const kir::GroupedGridWelford* node) {
  indent() << "GroupedGridWelford(\n";
  ++indent_size_;
  for (const auto i : c10::irange(node->numExprs())) {
    indent() << node->outAvg(i) << " (Avg),\n";
    indent() << node->outVar(i) << " (Var),\n";
    indent() << node->outN(i) << " (Count)\n";
    indent() << " = Welford (\n";
    ++indent_size_;
    indent() << node->inAvg(i) << " (Avg),\n";
    indent() << node->inVar(i) << " (Var),\n";
    indent() << node->inN(i) << " (Count)\n";
    indent() << "initial value =\n";
    ++indent_size_;
    indent() << node->initAvg(i) << " (Avg),\n";
    indent() << node->initVar(i) << " (Var),\n";
    indent() << node->initN(i) << " (Count),\n";
    --indent_size_;
    indent() << "reduction buffer =\n";
    ++indent_size_;
    indent() << node->reduction_buffers()[0].at(i)->buffer() << " (Avg),\n";
    indent() << node->reduction_buffers()[1].at(i)->buffer() << " (Var),\n";
    indent() << node->reduction_buffers()[2].at(i)->buffer() << " (Count) )\n";
    indent_size_ -= 2;
  }
  indent() << "sync buffer = " << node->sync_buffer()->buffer() << ",\n";
  indent() << "read predicate = ";
  if (node->predicate() != nullptr) {
    os_ << node->predicate();
  } else {
    os_ << "nullptr";
  }
  os_ << ",\n";
  indent() << "write predicate = ";
  if (node->writePredicate() != nullptr) {
    os_ << node->writePredicate();
  } else {
    os_ << "nullptr";
  }
  os_ << ",\n";
  indent() << "thread predicate = " << node->threadPredicate().toString()
           << ",\n";
  indent() << "allreduce = " << (node->isAllreduce() ? "true" : "false")
           << " )\n";
  --indent_size_;
}

void IrPrinter::handle(const kir::InitMagicZero* node) {
  indent() << "NVFUSER_DEFINE_MAGIC_ZERO\n";
}

void IrPrinter::handle(const kir::UpdateMagicZero* node) {
  indent() << "NVFUSER_UPDATE_MAGIC_ZERO\n";
}

void IrPrinter::handle(const kir::AllocateFusedReduction* node) {
  indent() << "AllocateFusedReduction(reduction buffer=";
  handle(node->out());
  os_ << ")\n";
}

void IrPrinter::handle(const kir::IntPair* node) {
  if (print_inline_) {
    if (node->definition()) {
      handle(node->definition());
      return;
    }
  }
  os_ << "iPair" << varName(node);
}

void IrPrinter::handle(const kir::Swizzle2DInt* node) {
  if (!print_inline_) {
    indent();
    handle(node->out());
    os_ << " = ";
  }

  os_ << node->swizzleType() << "2D(";
  handle(node->inX());
  os_ << ",";
  handle(node->inY());
  os_ << ")";
}

void IrPrinter::handle(const kir::PairSelect* node) {
  if (!print_inline_) {
    indent();
    handle(node->out());
    os_ << " = ";
  }

  handle(node->in());

  switch (node->selection()) {
    case kir::PairSelect::Selection::X:
      os_ << ".x";
      break;
    case kir::PairSelect::Selection::Y:
      os_ << ".y";
      break;
    default:
      break;
  }
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
  auto root_domain = tv->domain()->getRootDomain();
  os() << " root domain : (";
  for (const auto root_idx : c10::irange(root_domain.size())) {
    IrPrinter::handle(root_domain[root_idx]);
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
      os() << "  ";
      IrPrinter::handle(exp);
    }

    os() << " rfactor domain : (";
    for (const auto root_idx : c10::irange(rfactor_domain.size())) {
      IrPrinter::handle(rfactor_domain[root_idx]);
      if (root_idx + 1 < rfactor_domain.size()) {
        os() << ",";
      }
    }
    os() << ")\n";
  }

  auto from = tv->getMaybeRFactorDomain();
  auto all_exp = DependencyCheck::getAllExprsBetween(
      {from.begin(), from.end()},
      {tv->domain()->domain().begin(), tv->domain()->domain().end()});

  for (auto exp : all_exp) {
    os() << "  ";
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
