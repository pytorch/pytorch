#include <torch/csrc/jit/codegen/cuda/ir_iostream.h>
#include <torch/csrc/jit/codegen/cuda/fusion.h>
#include <torch/csrc/jit/codegen/cuda/ir_all_nodes.h>
#include <torch/csrc/jit/codegen/cuda/lower_thread_predicate.h>
#include <torch/csrc/jit/codegen/cuda/lower_utils.h>

#include <iostream>

namespace torch {
namespace jit {
namespace fuser {

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

void IRPrinter::handle(const Statement* s) {
  OptInConstDispatch::handle(s);
}

void IRPrinter::handle(const Val* v) {
  OptInConstDispatch::handle(v);
}

void IRPrinter::handle(const Expr* e) {
  OptInConstDispatch::handle(e);
}

void IRPrinter::printHeader(
    Fusion* fusion,
    const std::string& kernel_name_,
    const std::vector<Val*>& global_buffers) {
  os << "__global__ void " << kernel_name_ << "(";

  std::vector<Val*> vals;

  for (auto val : fusion->inputs()) {
    vals.push_back(val);
  }
  for (auto val : fusion->outputs()) {
    vals.push_back(val);
  }

  for (auto val : global_buffers) {
    vals.push_back(val);
  }

  for (Val* val : vals) {
    switch (val->getValType().value()) {
      case ValType::TensorView:
        os << "Tensor<" << val->getDataType().value() << ", "
           << TensorDomain::noReductions(val->as<TensorView>()->getRootDomain())
                  .size()
           << "> T" << val->name();
        break;
      case ValType::KirTensorView:
        os << "Tensor<" << val->getDataType().value() << ", "
           << kir::TensorDomain::noReductions(
                  val->as<kir::TensorView>()->domain()->rootDomain())
                  .size()
           << "> T" << val->name();
        break;
      case ValType::Scalar:
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

  if (fusion->hasRNG())
    os << ", unsigned long long seed, unsigned long long offset";

  os << "){\n";
  indent_size++;
  if (fusion->hasRNG()) {
    indent();
    os << "int idx = blockIdx.x*blockDim.x + threadIdx.x;\n";
    indent();
    os << "Philox rnd(seed, idx, offset);\n";
  }
  if (fusion->hasBlockReduction() || fusion->hasGridReduction()) {
    indent();
    // TODO: Dynamic sizing possible? blockReduce originally used 1024
    // values of a given type
    os << "__shared__ float shared_mem[1024];\n";
  }
}

void IRPrinter::handle(Fusion* fusion) {
  resetIndent();
  for (const Expr* expr : fusion->exprs()) {
    handle(expr);
  }
}

void IRPrinter::handle(const TensorDomain* td) {
  if (td->nDims() == 0) {
    os << "[ 0 ]";
    return;
  }
  os << "[ ";
  for (size_t i = 0; i < td->nDims(); i++) {
    handle(td->axis(i));
    if (i != td->nDims() - 1)
      os << ", ";
  }
  os << " ]";
}

void IRPrinter::handle(const TensorView* tv) {
  if (tv->nDims() == 0) {
    switch (tv->getDataType().value()) {
      case DataType::Bool:
        os << "b";
        break;
      case DataType::Float:
        os << "f";
        break;
      case DataType::Half:
        os << "h";
        break;
      case DataType::Int:
        os << "i";
        break;
      default:
        TORCH_INTERNAL_ASSERT(
            false, "Did not recognize type ", tv->getDataType().value());
    }
    os << tv->name();

  } else {
    os << "T" << tv->name();
    handle(tv->domain());

    if (tv->getComputeAtView() != nullptr) {
      os << " compute_at( ";
      os << "T" << tv->getComputeAtView()->name();
      os << ", " << tv->getRelativeComputeAtAxis() << " )";
    }
  }
}

void IRPrinter::handle(const IterDomain* id) {
  os << id->getIterType();
  os << id->getParallelType();
  os << "{";
  if (!id->start()->isZeroInt()) {
    print_inline(id->start());
    os << " : ";
  }
  print_inline(id->extent());
  os << "}";
  if (id->isRFactorProduct())
    os << "rf";
}

void IRPrinter::handle(const kir::TensorIndex* ti) {
  os << "T" << ti->view()->name();
  std::vector<Val*> non_zero_inds;
  for (auto* ind : ti->indices()) {
    if (!ind->isZeroInt()) {
      non_zero_inds.push_back(ind);
    }
  }

  if (non_zero_inds.size() == 0) {
    os << "[ 0 ]";
    return;
  }

  os << "[ ";
  bool first = true;
  for (auto* ind : non_zero_inds) {
    if (!first)
      os << " + ";
    print_inline(ind);
    first = false;
  }
  os << " ]";
}

void IRPrinter::handle(const Bool* b) {
  if (print_inline_ && FusionGuard::getCurFusion()->origin(b) != nullptr) {
    os << "( ";
    handle(FusionGuard::getCurFusion()->origin(b));
    os << " )";
    return;
  }

  if (b->isSymbolic()) {
    os << "b" << b->name();
  } else {
    os << "bool(" << *(b->value()) << ")";
  }
}

void IRPrinter::handle(const Float* f) {
  if (print_inline_ && FusionGuard::getCurFusion()->origin(f) != nullptr) {
    os << "( ";
    handle(FusionGuard::getCurFusion()->origin(f));
    os << " )";
    return;
  }

  if (f->isSymbolic()) {
    os << "f" << f->name();
  } else {
    os << "float("
       << std::setprecision(
              std::numeric_limits<Float::ScalarType>::max_digits10)
       << *(f->value()) << ")";
  }
}

void IRPrinter::handle(const Half* h) {
  if (print_inline_ && FusionGuard::getCurFusion()->origin(h) != nullptr) {
    os << "( ";
    handle(FusionGuard::getCurFusion()->origin(h));
    os << " )";
    return;
  }

  if (h->isSymbolic()) {
    os << "h" << h->name();
  } else {
    os << "__float2half(" << *(h->value()) << ")";
  }
}

void IRPrinter::handle(const Int* i) {
  if (print_inline_) {
    if (auto def = FusionGuard::getCurFusion()->origin(i)) {
      os << "( ";
      handle(def);
      os << " )";
      return;
    }
  }

  if (i->isSymbolic()) {
    os << "i" << i->name();
  } else {
    os << *(i->value());
  }
}

void IRPrinter::handle(const NamedScalar* i) {
  os << i->name();
}

void IRPrinter::handle(const kir::Bool* b) {
  if (print_inline_ && FusionGuard::getCurFusion()->origin(b) != nullptr) {
    os << "( ";
    handle(FusionGuard::getCurFusion()->origin(b));
    os << " )";
    return;
  }

  if (b->isSymbolic()) {
    os << "b" << b->name();
  } else {
    os << "bool(" << *(b->value()) << ")";
  }
}

void IRPrinter::handle(const kir::Float* f) {
  if (print_inline_ && FusionGuard::getCurFusion()->origin(f) != nullptr) {
    os << "( ";
    handle(FusionGuard::getCurFusion()->origin(f));
    os << " )";
    return;
  }

  if (f->isSymbolic()) {
    os << "f" << f->name();
  } else {
    os << "float("
       << std::setprecision(
              std::numeric_limits<Float::ScalarType>::max_digits10)
       << *(f->value()) << ")";
  }
}

void IRPrinter::handle(const kir::Half* h) {
  if (print_inline_ && FusionGuard::getCurFusion()->origin(h) != nullptr) {
    os << "( ";
    handle(FusionGuard::getCurFusion()->origin(h));
    os << " )";
    return;
  }

  if (h->isSymbolic()) {
    os << "h" << h->name();
  } else {
    os << "__float2half(" << *(h->value()) << ")";
  }
}

void IRPrinter::handle(const kir::Int* i) {
  if (print_inline_) {
    if (auto def = FusionGuard::getCurFusion()->origin(i)) {
      os << "( ";
      handle(def);
      os << " )";
      return;
    }
  }

  if (i->isSymbolic()) {
    os << "i" << i->name();
  } else {
    os << *(i->value());
  }
}

void IRPrinter::handle(const kir::NamedScalar* i) {
  os << i->name();
}

void IRPrinter::handle(const kir::IterDomain* id) {
  os << id->getIterType();
  os << id->getParallelType();
  os << "{";
  if (!id->start()->isZeroInt()) {
    print_inline(id->start());
    os << " : ";
  }
  print_inline(id->extent());
  os << "}";
  if (id->isRFactorProduct())
    os << "rf";
}

void IRPrinter::handle(const kir::TensorDomain*) {
  TORCH_INTERNAL_ASSERT(false, "Unreachable");
}

void IRPrinter::handle(const kir::TensorView*) {
  TORCH_INTERNAL_ASSERT(false, "Unreachable");
}

static bool isTV(const Val* val) {
  return val->getValType().value() == ValType::TensorView ||
      val->getValType().value() == ValType::TensorIndex;
}

// Check if we're a TensorView op that we can generate code for.
static bool isTVOp(const Expr* expr) {
  return expr->outputs().size() == 1 && isTV(expr->outputs().front());
}

void IRPrinter::handle(const UnaryOp* uop) {
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
    checkInlineable(uop);
  }

  if (auto inline_uop = inline_op_str(uop->getUnaryOpType())) {
    os << inline_uop.value();
    handle(uop->in());
  } else {
    if (uop->getUnaryOpType() == UnaryOpType::Cast) {
      c10::optional<std::string> cast_str = cast_func_str(std::make_pair(
          uop->in()->getDataType().value(), uop->out()->getDataType().value()));
      TORCH_INTERNAL_ASSERT(cast_str != c10::nullopt, "Unsupported Cast");
      os << cast_str.value();
    } else {
      os << uop->getUnaryOpType();
    }
    os << "(";
    if (uop->getUnaryOpType() == UnaryOpType::RandLike)
      os << "rnd";
    else
      handle(uop->in());
    os << ")";
  }

  if (istvop)
    indent_size--;

  if (!print_inline_)
    os << ";\n";
}

void IRPrinter::handle(const BinaryOp* bop) {
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
    checkInlineable(bop);
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

void IRPrinter::handle(const TernaryOp* top) {
  bool istvop = isTVOp(top);
  if (!print_inline_) {
    indent();
    os << top->out();

    // tensor operations tend to be long, break them up into multiple lines
    if (istvop) {
      os << "\n";
      indent_size++;
      indent();
    }

    os << " = ";
  } else {
    checkInlineable(top);
  }

  os << top->getTernaryOpType() << "(";
  handle(top->in1());
  if (istvop) {
    os << "\n";
    indent();
  }
  os << ", ";
  handle(top->in2());
  if (istvop) {
    os << "\n";
    indent();
  }
  os << ", ";
  handle(top->in3());
  os << ")";

  if (istvop)
    indent_size--;

  if (!print_inline_)
    os << ";\n";
}

void IRPrinter::handle(const kir::UnaryOp* uop) {
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
    checkInlineable(uop);
  }

  if (auto inline_uop = inline_op_str(uop->getUnaryOpType())) {
    os << inline_uop.value();
    handle(uop->in());
  } else {
    if (uop->getUnaryOpType() == UnaryOpType::Cast) {
      c10::optional<std::string> cast_str = cast_func_str(std::make_pair(
          uop->in()->getDataType().value(), uop->out()->getDataType().value()));
      TORCH_INTERNAL_ASSERT(cast_str != c10::nullopt, "Unsupported Cast");
      os << cast_str.value();
    } else {
      os << uop->getUnaryOpType();
    }
    os << "(";
    if (uop->getUnaryOpType() == UnaryOpType::RandLike)
      os << "rnd";
    else
      handle(uop->in());
    os << ")";
  }

  if (istvop)
    indent_size--;

  if (!print_inline_)
    os << ";\n";
}

void IRPrinter::handle(const kir::BinaryOp* bop) {
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
    checkInlineable(bop);
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

void IRPrinter::handle(const kir::TernaryOp* top) {
  bool istvop = isTVOp(top);
  if (!print_inline_) {
    indent();
    os << top->out();

    // tensor operations tend to be long, break them up into multiple lines
    if (istvop) {
      os << "\n";
      indent_size++;
      indent();
    }

    os << " = ";
  } else {
    checkInlineable(top);
  }

  os << top->getTernaryOpType() << "(";
  handle(top->in1());
  if (istvop) {
    os << "\n";
    indent();
  }
  os << ", ";
  handle(top->in2());
  if (istvop) {
    os << "\n";
    indent();
  }
  os << ", ";
  handle(top->in3());
  os << ")";

  if (istvop)
    indent_size--;

  if (!print_inline_)
    os << ";\n";
}

void IRPrinter::handle(const ReductionOp* rop) {
  TORCH_CHECK(rop->out()->getValType() != ValType::TensorIndex);
  indent();
  os << rop->out() << " = reduction( " << rop->in()
     << ", op = " << rop->getReductionOpType()
     << ", initial value = " << rop->init() << " )\n";
}

void IRPrinter::handle(const kir::ReductionOp* rop) {
  TORCH_CHECK(rop->out()->getValType() == ValType::TensorIndex);

  const auto out = rop->out()->as<kir::TensorIndex>();
  const auto domain = out->view()->domain();

  const bool has_block_reduce = domain->hasBlockReduction();
  const bool has_grid_reduce = domain->hasGridReduction();

  if (!has_block_reduce && !has_grid_reduce) {
    FusionGuard fg(rop->fusion());
    handle(new BinaryOp(rop->getReductionOpType(), out, out, rop->in()));
    return;
  }

  auto par_domains = rop->getParallelReductionDomains();
  bool tidx = par_domains.find(ParallelType::TIDx) != par_domains.end();
  bool tidy = par_domains.find(ParallelType::TIDy) != par_domains.end();
  bool tidz = par_domains.find(ParallelType::TIDz) != par_domains.end();

  auto d_type = rop->out()->getDataType().value();
  auto op_type = rop->getReductionOpType();
  const std::string block_result = "block_result";
  if (has_block_reduce) {
    if (has_grid_reduce) {
      indent();
      os << d_type << " " << block_result << ";\n";
    }
    indent();
    // Thread all reduce.
    os << "blockReduce< " << (tidx ? "true" : "false") << ", "
       << (tidy ? "true" : "false") << ", " << (tidz ? "true" : "false") << " >"
       << " ( ";
    if (has_grid_reduce) {
      os << block_result;
    } else {
      handle(rop->out());
    }
    os << ", ";
    handle(rop->in());
    os << ", ";
    os << "reduction_" << op_type << "_" << d_type;
    os << ", threadIdx, blockDim";
    os << ", reinterpret_cast<" << d_type << "*>(shared_mem)";
    os << ");\n";
  }
}

void IRPrinter::handle(const kir::GridReduction* gr) {
  // Check if we've lowered yet.
  const auto rop = gr->reduction_op();
  TORCH_INTERNAL_ASSERT(
      rop->out()->getValType() == ValType::TensorIndex,
      "GridReduction node is a lowered node but did not find the output to be a TensorIndex.");

  const auto out = rop->out()->as<kir::TensorIndex>();
  const auto domain = out->view()->domain();
  TORCH_INTERNAL_ASSERT(domain->hasGridReduction());

  const auto par_domains = rop->getParallelReductionDomains();
  const bool tidx = par_domains.find(ParallelType::TIDx) != par_domains.end();
  const bool tidy = par_domains.find(ParallelType::TIDy) != par_domains.end();
  const bool tidz = par_domains.find(ParallelType::TIDz) != par_domains.end();
  const bool bidx = par_domains.find(ParallelType::BIDx) != par_domains.end();
  const bool bidy = par_domains.find(ParallelType::BIDy) != par_domains.end();
  const bool bidz = par_domains.find(ParallelType::BIDz) != par_domains.end();

  const auto d_type = rop->out()->getDataType().value();
  const auto op_type = rop->getReductionOpType();
  TORCH_INTERNAL_ASSERT(
      gr->reduction_buffer()->buffer()->getValType().value() ==
      ValType::KirTensorView);
  TORCH_INTERNAL_ASSERT(
      gr->sync_buffer()->buffer()->getValType().value() ==
      ValType::KirTensorView);
  const auto work_buffer =
      gr->reduction_buffer()->buffer()->as<kir::TensorView>();
  const auto sync_buffer = gr->sync_buffer()->buffer()->as<kir::TensorView>();
  indent();
  // Since block-level reduction is already done, those dimensions
  // with tidx/y/z being true do not participate in the grid reduction.
  os << kir::GridReduction::getPredicateFlagName(out->view()) << " = "
     << "reduction::gridReduce< " << (bidx ? "true" : "false") << ", "
     << (bidy ? "true" : "false") << ", " << (bidz ? "true" : "false") << ", "
     << (!tidx ? "true" : "false") << ", " << (!tidy ? "true" : "false") << ", "
     << (!tidz ? "true" : "false") << " >"
     << " ( ";
  handle(rop->out());
  os << ", ";
  if (domain->hasBlockReduction()) {
    os << "block_result";
  } else {
    handle(rop->in());
  }
  os << ", ";
  os << "reduction_" << op_type << "_" << d_type;
  os << ", &T" << work_buffer->name() << "[0]";
  os << ", T" << sync_buffer->name() << "";
  os << ", reinterpret_cast<" << d_type << "*>(shared_mem)";
  os << ");\n";
}

void IRPrinter::handle(const BroadcastOp* bop) {
  TORCH_CHECK(bop->out()->getValType() != ValType::TensorIndex);
  indent();
  os << bop->out() << " = broadcast( " << bop->in() << " )\n";
}

void IRPrinter::handle(const kir::BroadcastOp* bop) {
  TORCH_CHECK(bop->out()->getValType() == ValType::TensorIndex);

  const ir_utils::ParallelTypeBitmap domains =
      ir_utils::getParallelBroadcastDomains(
          bop->out(), getThreadPredicateMap());
  const bool thread_x = domains.get(ParallelType::TIDx);
  const bool thread_y = domains.get(ParallelType::TIDy);
  const bool thread_z = domains.get(ParallelType::TIDz);
  const bool block_x = domains.get(ParallelType::BIDx);
  const bool block_y = domains.get(ParallelType::BIDy);
  const bool block_z = domains.get(ParallelType::BIDz);

  const bool grid_broadcast_needed = block_x || block_y || block_z;
  const bool block_broadcast_needed = thread_x || thread_y || thread_z;

  TORCH_INTERNAL_ASSERT(
      !grid_broadcast_needed, "Parallel broadcast across blocks not supported");

  if (block_broadcast_needed) {
    indent();
    os << "broadcast::blockBroadcast<";
    os << (thread_x ? "true" : "false") << ", ";
    os << (thread_y ? "true" : "false") << ", ";
    os << (thread_z ? "true" : "false");
    os << ">(";
    handle(bop->out());
    os << ", ";
    handle(bop->in());
    os << ");\n";
  } else {
    indent();
    handle(bop->out());
    os << "\n";
    indent_size++;
    indent();
    os << " = ";
    handle(bop->in());
    indent_size--;
    os << ";\n";
  }
}

void IRPrinter::handle(const kir::ForLoop* fl) {
  if (fl->iter_domain()->isThread() || fl->iter_domain()->isBroadcast()) {
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

void IRPrinter::handle(const kir::IfThenElse* ite) {
  indent();

  // IF
  os << "if ( ";
  print_inline(ite->cond());
  os << " ) {\n";

  indent_size++;
  for (auto& expr : ite->constBody().exprs()) {
    handle(expr);
  }
  indent_size--;

  // ELSE
  if (ite->hasElse()) {
    indent();
    os << "} else {\n";
    indent_size++;
    for (auto& expr : ite->constElseBody().exprs()) {
      handle(expr);
    }
    indent_size--;
  }
  indent();
  os << "}\n";
}

void IRPrinter::handle(const kir::Allocate* a) {
  indent();
  if (a->buffer()->getValType().value() == ValType::KirTensorView) {
    const auto tv = a->buffer()->as<kir::TensorView>();
    TORCH_INTERNAL_ASSERT(tv->domain()->nDims() > 0);
    TORCH_INTERNAL_ASSERT(a->size() != nullptr);
    switch (tv->getMemoryType()) {
      case MemoryType::Global:
        os << "// Allocate global tensor ";
        break;
      case MemoryType::Shared:
        os << "__shared__ ";
        break;
      case MemoryType::Local:
        break;
    }
    os << a->buffer_type();
    os << " T" << tv->name() << "[";
    print_inline(a->size());
    os << "];\n";
  } else {
    os << a->buffer_type() << " ";
    handle(a->buffer());
    os << ";\n";
  }
}

void IRPrinter::handle(const kir::Sync* a) {
  indent();
  os << "__syncthreads();\n";
}

void IRPrinter::handle(const Split* s) {
  os << "Split: ";
  handle(s->in());
  os << " by factor " << s->factor() << " -> ";
  handle(s->outer());
  os << ", ";
  handle(s->inner());
  os << "\n";
}

void IRPrinter::handle(const Merge* m) {
  os << "Merge: ";
  handle(m->outer());
  os << " and ";
  handle(m->inner());
  os << " -> ";
  handle(m->out());
  os << "\n";
}

namespace {

class ReductionOps : OptOutDispatch {
 public:
  std::set<std::pair<BinaryOpType, DataType>> rops;
  void handle(ReductionOp* rop) override {
    rops.emplace(std::pair<BinaryOpType, DataType>{
        rop->getReductionOpType(), rop->in()->getDataType().value()});
  }

  using OptOutDispatch::handle;

  static std::set<std::pair<BinaryOpType, DataType>> get(Fusion* fusion) {
    ReductionOps ROPs;
    for (auto expr : fusion->exprs(true)) {
      ROPs.handle(expr);
    }
    return ROPs.rops;
  }
};

} // namespace

void IRPrinter::printReductionOps(Fusion* fusion) {
  FusionGuard fg(fusion);
  auto a = new NamedScalar("a", DataType::Null);
  auto b = new NamedScalar("b", DataType::Null);
  for (auto rop_pair : ReductionOps::get(fusion)) {
    auto op_type = rop_pair.first;
    auto d_type = rop_pair.second;

    indent();
    os << "__device__ void reduction_" << op_type << "_" << d_type << "("
       << d_type << "& a, "
       << "const " << d_type << " b) {\n";
    indent_size++;

    handle(new BinaryOp(op_type, a, a, b));
    indent_size--;
    indent();
    os << "}\n";
  }
}

void IRPrinter::printKernel(
    const std::vector<Expr*>& exprs,
    const std::string& kernel_name,
    const std::vector<Val*>& global_buffers) {
  Fusion* fusion = FusionGuard::getCurFusion();
  if (exprs.empty())
    return;
  TORCH_INTERNAL_ASSERT(
      exprs[0]->fusion() == FusionGuard::getCurFusion(),
      "Incorrect fusion set during printKernel.");

  printReductionOps(fusion);
  printHeader(fusion, kernel_name, global_buffers);

  for (auto* expr : exprs) {
    handle(expr);
  }
  os << "}\n";
}

const ThreadPredicateMap& IRPrinter::getThreadPredicateMap() {
  if (thread_predicates_ == nullptr) {
    Fusion* fusion = FusionGuard::getCurFusion();
    thread_predicates_ = std::make_unique<ThreadPredicateMap>(fusion);
  }
  return *thread_predicates_;
}

std::ostream& operator<<(std::ostream& os, const Statement* stmt) {
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
