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

void IrPrinter::handle(const Statement* s) {
  OptInConstDispatch::handle(s);
}

void IrPrinter::handle(const Val* v) {
  OptInConstDispatch::handle(v);
}

void IrPrinter::handle(const Expr* e) {
  OptInConstDispatch::handle(e);
}

void IrPrinter::printHeader(
    Fusion* fusion,
    const std::string& kernel_name_,
    const std::vector<Val*>& global_buffers,
    bool hasDynamicSmem) {
  os_ << "__global__ void " << kernel_name_ << "(";

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
        os_ << "Tensor<" << val->getDataType().value() << ", "
            << TensorDomain::noReductions(
                   val->as<TensorView>()->getRootDomain())
                   .size()
            << "> T" << val->name();
        break;
      case ValType::KirTensorView:
        os_ << "Tensor<" << val->getDataType().value() << ", "
            << TensorDomain::noReductions(val->as<kir::TensorView>()
                                              ->fuserTv()
                                              ->getMaybeRFactorDomain())
                   .size()
            << "> T" << val->name();
        break;
      case ValType::Scalar:
        os_ << val->getDataType().value() << " " << val;
        break;
      default:
        TORCH_CHECK(
            false,
            "printHeader() found an input to the fusion of unexpected data type.");
    }

    if (val != vals.back())
      os_ << ", ";
  }

  if (fusion->hasRNG())
    os_ << ", unsigned long long seed, unsigned long long offset";

  os_ << "){\n";
  indent_size_++;

  if (fusion->hasRNG()) {
    indent();
    os_ << "int idx = blockIdx.x*blockDim.x + threadIdx.x;\n";
    indent();
    os_ << "Philox rnd(seed, idx, offset);\n";
  }

  // Dynamic Shared Memory
  const bool hasWorkspace =
      fusion->hasBlockReduction() || fusion->hasGridReduction();
  if (hasDynamicSmem || hasWorkspace) {
    indent();
    os_ << "alignas(";
    os_ << dataTypeSize(fusion->getMaximumSmemDataType());
    os_ << ") extern __shared__ char array[];\n";
  }

  if (hasDynamicSmem) {
    indent();
    os_ << "unsigned offset = 0;\n";
  }

  if (hasWorkspace) {
    indent();
    os_ << "void* shared_mem = array;\n";
    if (hasDynamicSmem) {
      indent();
      os_ << "offset += ((blockDim.x * blockDim.y * blockDim.z) * sizeof(";
      os_ << fusion->getMaximumSmemDataType();
      os_ << "));\n";
    }
  }
}

void IrPrinter::handle(Fusion* fusion) {
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

void IrPrinter::handle(const kir::TensorIndex* ti) {
  os_ << "T" << ti->view()->name();
  std::vector<Val*> non_zero_inds;
  for (auto* ind : ti->indices()) {
    if (!ind->isZeroInt()) {
      non_zero_inds.push_back(ind);
    }
  }

  if (non_zero_inds.size() == 0) {
    os_ << "[ 0 ]";
    return;
  }

  os_ << "[ ";
  bool first = true;
  for (auto* ind : non_zero_inds) {
    if (!first)
      os_ << " + ";
    print_inline(ind);
    first = false;
  }
  os_ << " ]";
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

void IrPrinter::handle(const kir::Float* f) {
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

void IrPrinter::handle(const kir::Half* h) {
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

void IrPrinter::handle(const kir::Int* i) {
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

void IrPrinter::handle(const kir::NamedScalar* i) {
  os_ << i->name();
}

void IrPrinter::handle(const kir::IterDomain* id) {
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

void IrPrinter::handle(const kir::TensorDomain*) {
  TORCH_INTERNAL_ASSERT(false, "Unreachable");
}

void IrPrinter::handle(const kir::TensorView* tv) {
  // This should never be reachable, but the current codebase assumes
  // kir::TensorView can be printable for debugging messages.
  os_ << "KT" << tv->name();
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

void IrPrinter::handle(const kir::BinaryOp* bop) {
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

void IrPrinter::handle(const kir::TernaryOp* top) {
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
  TORCH_CHECK(rop->out()->getValType() != ValType::TensorIndex);
  indent();
  os_ << rop->out() << " = reduction( " << rop->in()
      << ", op = " << rop->getReductionOpType()
      << ", initial value = " << rop->init() << " )\n";
}

void IrPrinter::handle(const kir::ReductionOp* rop) {
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
      os_ << d_type << " " << block_result << ";\n";
    }
    indent();
    // Thread all reduce.
    os_ << "blockReduce< " << (tidx ? "true" : "false") << ", "
        << (tidy ? "true" : "false") << ", " << (tidz ? "true" : "false")
        << " >"
        << " ( ";
    if (has_grid_reduce) {
      os_ << block_result;
    } else {
      handle(rop->out());
    }
    os_ << ", ";
    handle(rop->in());
    os_ << ", ";
    os_ << "reduction_" << op_type << "_" << d_type;
    os_ << ", threadIdx, blockDim";
    os_ << ", static_cast<" << d_type << "*>(shared_mem)";
    if (rop->pred() == nullptr) {
      os_ << ", true";
    } else {
      os_ << ", ";
      print_inline(rop->pred());
    }
    os_ << ", ";
    print_inline(rop->init());
    os_ << ");\n";
  }
}

void IrPrinter::handle(const kir::GridReduction* gr) {
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
  os_ << kir::GridReduction::getPredicateFlagName(out->view()) << " = "
      << "reduction::gridReduce< " << (bidx ? "true" : "false") << ", "
      << (bidy ? "true" : "false") << ", " << (bidz ? "true" : "false") << ", "
      << (!tidx ? "true" : "false") << ", " << (!tidy ? "true" : "false")
      << ", " << (!tidz ? "true" : "false") << " >"
      << " ( ";
  handle(rop->out());
  os_ << ", ";
  if (domain->hasBlockReduction()) {
    os_ << "block_result";
  } else {
    handle(rop->in());
  }
  os_ << ", ";
  os_ << "reduction_" << op_type << "_" << d_type;
  os_ << ", &T" << work_buffer->name() << "[0]";
  os_ << ", T" << sync_buffer->name() << "";
  os_ << ", static_cast<" << d_type << "*>(shared_mem)";
  if (gr->pred() == nullptr) {
    os_ << ", true";
  } else {
    os_ << ", ";
    print_inline(gr->pred());
  }
  os_ << ", ";
  print_inline(gr->reduction_op()->init());
  os_ << ");\n";
}

void IrPrinter::handle(const BroadcastOp* bop) {
  TORCH_CHECK(bop->out()->getValType() != ValType::TensorIndex);
  indent();
  os_ << bop->out() << " = broadcast( " << bop->in() << " )\n";
}

void IrPrinter::handle(const kir::BroadcastOp* bop) {
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
    auto d_type = bop->out()->getDataType().value();
    indent();
    os_ << "broadcast::blockBroadcast<";
    os_ << (thread_x ? "true" : "false") << ", ";
    os_ << (thread_y ? "true" : "false") << ", ";
    os_ << (thread_z ? "true" : "false");
    os_ << ">(";
    handle(bop->out());
    os_ << ", ";
    handle(bop->in());
    os_ << ", static_cast<" << d_type << "*>(shared_mem)";
    os_ << ");\n";
  } else {
    indent();
    handle(bop->out());
    os_ << "\n";
    indent_size_++;
    indent();
    os_ << " = ";
    handle(bop->in());
    indent_size_--;
    os_ << ";\n";
  }
}

void IrPrinter::handle(const kir::ForLoop* fl) {
  if (fl->iter_domain()->isThread() || fl->iter_domain()->isBroadcast()) {
    for (auto& expr : fl->constBody().exprs())
      handle(expr);
    return;
  }

  indent();
  os_ << "for(size_t ";
  handle(fl->index());
  os_ << " = ";
  print_inline(fl->iter_domain()->start());
  os_ << "; ";
  handle(fl->index());
  os_ << " < ";
  print_inline(fl->iter_domain()->extent());
  os_ << "; ++";
  handle(fl->index());
  os_ << " ) {\n";
  indent_size_++;
  for (auto& expr : fl->constBody().exprs())
    handle(expr);

  indent_size_--;
  indent();
  os_ << "}\n";
}

void IrPrinter::handle(const kir::IfThenElse* ite) {
  indent();

  // IF
  os_ << "if ( ";
  print_inline(ite->cond());
  os_ << " ) {\n";

  indent_size_++;
  for (auto& expr : ite->constBody().exprs()) {
    handle(expr);
  }
  indent_size_--;

  // ELSE
  if (ite->hasElse()) {
    indent();
    os_ << "} else {\n";
    indent_size_++;
    for (auto& expr : ite->constElseBody().exprs()) {
      handle(expr);
    }
    indent_size_--;
  }
  indent();
  os_ << "}\n";
}

void IrPrinter::handle(const kir::Allocate* a) {
  indent();
  if (a->buffer()->getValType().value() == ValType::KirTensorView) {
    const auto tv = a->buffer()->as<kir::TensorView>();
    TORCH_INTERNAL_ASSERT(tv->domain()->nDims() > 0);
    TORCH_INTERNAL_ASSERT(a->size() != nullptr);
    switch (tv->getMemoryType()) {
      case MemoryType::Global:
        os_ << "// Allocate global tensor ";
        break;
      case MemoryType::Shared:
        if (a->size()->isConstScalar()) {
          // Static Shared Memory
          os_ << "__shared__ ";
        }
        break;
      case MemoryType::Local:
        break;
    }

    // Dynamic Shared Memory
    if (tv->getMemoryType() == MemoryType::Shared &&
        !a->size()->isConstScalar()) {
      // Align Offset Position
      os_ << "offset = alignBufferSize(offset,";
      os_ << dataTypeSize(a->buffer_type());
      os_ << ");\n";
      // Shared Memory Pointer
      indent();
      os_ << a->buffer_type() << "* ";
      os_ << "T" << tv->name();
      os_ << " = reinterpret_cast<" << a->buffer_type() << "*>";
      os_ << "(array + offset);\n";
      // Increment Offset Position
      indent();
      os_ << "offset += (";
      print_inline(a->size());
      os_ << " * sizeof(";
      os_ << a->buffer_type();
      os_ << "));\n";
    } else {
      os_ << a->buffer_type();
      os_ << " T" << tv->name() << "[";
      print_inline(a->size());
      os_ << "];\n";
    }

  } else {
    os_ << a->buffer_type() << " ";
    handle(a->buffer());
    os_ << ";\n";
  }
}

void IrPrinter::handle(const kir::Sync* a) {
  indent();
  os_ << "__syncthreads();\n";
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

void IrPrinter::printReductionOps(Fusion* fusion) {
  FusionGuard fg(fusion);

  // TODO(kir): we shouldn't be creating new nodes during printing
  auto a = new NamedScalar("a", DataType::Null);
  auto b = new NamedScalar("b", DataType::Null);
  for (auto rop_pair : ReductionOps::get(fusion)) {
    auto op_type = rop_pair.first;
    auto d_type = rop_pair.second;

    indent();
    os_ << "__device__ void reduction_" << op_type << "_" << d_type << "("
        << d_type << "& a, "
        << "const " << d_type << " b) {\n";
    indent_size_++;

    handle(new BinaryOp(op_type, a, a, b));
    indent_size_--;
    indent();
    os_ << "}\n";
  }
}

void IrPrinter::printKernel(
    const std::vector<Expr*>& exprs,
    const std::string& kernel_name,
    const std::vector<Val*>& global_buffers,
    bool hasDynamicSmem) {
  Fusion* fusion = FusionGuard::getCurFusion();
  if (exprs.empty())
    return;
  TORCH_INTERNAL_ASSERT(
      exprs[0]->fusion() == FusionGuard::getCurFusion(),
      "Incorrect fusion set during printKernel.");

  printReductionOps(fusion);
  printHeader(fusion, kernel_name, global_buffers, hasDynamicSmem);

  for (auto* expr : exprs) {
    handle(expr);
  }
  os_ << "}\n";
}

const ThreadPredicateMap& IrPrinter::getThreadPredicateMap() {
  if (thread_predicates_ == nullptr) {
    Fusion* fusion = FusionGuard::getCurFusion();
    thread_predicates_ = std::make_unique<ThreadPredicateMap>(fusion);
  }
  return *thread_predicates_;
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

} // namespace fuser
} // namespace jit
} // namespace torch
