#include <expr_evaluator.h>
#include <expr_simplifier.h>
#include <ir_builder.h>
#include <ir_cloner.h>
#include <ir_iostream.h>
#include <kernel.h>
#include <kernel_ir.h>
#include <lower2device.h>
#include <lower_utils.h>
#include <type.h>

#include <iostream>

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {
namespace kir {

namespace {

inline const char* boolLiteral(bool value) {
  return value ? "true" : "false";
}

} // namespace

Predicate::Predicate(
    IrBuilderPasskey passkey,
    PredicateType ptype,
    const Expr* expr,
    Bool* thread_pred)
    : Val(passkey, ValType::Predicate, DataType::Bool),
      ptype_(ptype),
      expr_(expr),
      thread_pred_(thread_pred) {
  TORCH_INTERNAL_ASSERT(
      passkey.ir_container_->isA<kir::Kernel>(),
      "IR type only valid for Kernel container.");
  TORCH_INTERNAL_ASSERT(
      ptype != PredicateType::Unswitch && ptype != PredicateType::Manual);
}

Predicate::Predicate(IrBuilderPasskey passkey, ForLoop* unrolled_loop)
    : Val(passkey, ValType::Predicate, DataType::Bool),
      ptype_(PredicateType::Unswitch),
      unrolled_loop_(unrolled_loop) {
  TORCH_INTERNAL_ASSERT(
      passkey.ir_container_->isA<kir::Kernel>(),
      "IR type only valid for Kernel container.");
  TORCH_INTERNAL_ASSERT(unrolled_loop != nullptr);
}

Predicate::Predicate(IrBuilderPasskey passkey, Bool* value)
    : Val(passkey, ValType::Predicate, DataType::Bool),
      ptype_(PredicateType::Manual),
      value_(value) {
  TORCH_INTERNAL_ASSERT(
      passkey.ir_container_->isA<kir::Kernel>(),
      "IR type only valid for Kernel container.");
  TORCH_INTERNAL_ASSERT(value != nullptr);
}

std::string Predicate::toString(int indent_size) const {
  if (predicate_type() == PredicateType::Manual) {
    return value()->toString();
  }
  return predicate_type2string(predicate_type());
}

std::string Predicate::toInlineString(int indent_size) const {
  if (predicate_type() == PredicateType::Manual) {
    return value()->toInlineString();
  }
  return predicate_type2string(predicate_type());
}

TensorIndex::TensorIndex(
    IrBuilderPasskey passkey,
    const TensorView* view,
    Val* index)
    : Val(passkey, ValType::TensorIndex, view->getDataType().value()),
      view_(view),
      index_(index) {
  TORCH_INTERNAL_ASSERT(
      passkey.ir_container_->isA<kir::Kernel>(),
      "IR type only valid for Kernel container.");
  TORCH_INTERNAL_ASSERT(
      isIntegralType(index->dtype()),
      "Cannot index with a value other than an int.");
}

std::string TensorIndex::toString(int indent_size) const {
  std::stringstream ss;
  ss << ir_utils::varName(this);
  switch (view()->getMemoryType()) {
    case MemoryType::Global:
      ss << "_g";
      break;
    case MemoryType::Shared:
      ss << "_s";
      break;
    case MemoryType::Local:
      ss << "_l";
      break;
    default:
      TORCH_INTERNAL_ASSERT(false, "Unknown tensor memory type.");
  }
  ss << "[";
  ss << index()->toInlineString(indent_size);
  ss << "]";
  ss << " view( " << ir_utils::varName(view()) << " )";
  return ss.str();
}

std::string TensorIndex::toInlineString(int indent_size) const {
  return toString(indent_size);
}

Allocate::Allocate(
    IrBuilderPasskey passkey,
    Val* buffer,
    MemoryType memory_type,
    std::vector<Val*> shape,
    bool zero_init,
    Allocate* alias)
    : Expr(passkey) {
  TORCH_INTERNAL_ASSERT(
      passkey.ir_container_->isA<kir::Kernel>(),
      "IR type only valid for Kernel container.");
  if (!shape.empty()) {
    TORCH_INTERNAL_ASSERT(
        (shape.size() == 1 && shape[0]->isOneInt()) ||
        buffer->isA<TensorView>());
  } else {
    TORCH_INTERNAL_ASSERT(buffer->isA<TensorView>());
    TORCH_INTERNAL_ASSERT(
        buffer->as<TensorView>()->getMemoryType() == memory_type);
    const auto domain = buffer->as<TensorView>()->domain();
    for (auto axis : domain->noReductions()) {
      shape.push_back(axis->extent());
    }
  }

  Val* size = nullptr;
  for (auto s : shape) {
    if (size == nullptr) {
      size = s;
    } else {
      size = IrBuilder::mulExpr(size, s);
    }
  }

  if (size == nullptr) {
    size = FusionGuard::getCurFusion()->oneVal();
  }

  if (alias != nullptr) {
    TORCH_INTERNAL_ASSERT(alias != this, "Invalid alias");
    TORCH_INTERNAL_ASSERT(alias->memoryType() == memory_type, "Invalid alias");
  }

  addInput(size);
  addAttribute(buffer);
  addAttribute(IrBuilder::create<Attribute<MemoryType>>(
      passkey.ir_container_, memory_type));
  addAttribute(
      IrBuilder::create<Attribute<bool>>(passkey.ir_container_, zero_init));

  addAttribute(alias);

  for (auto s : shape) {
    addAttribute(s);
  }
}

Allocate::Allocate(
    IrBuilderPasskey passkey,
    Val* buffer,
    MemoryType memory_type,
    Val* size,
    bool zero_init)
    : Allocate(
          passkey,
          buffer,
          memory_type,
          size == nullptr ? std::vector<Val*>{} : std::vector<Val*>{size},
          zero_init) {
  TORCH_INTERNAL_ASSERT(
      passkey.ir_container_->isA<kir::Kernel>(),
      "IR type only valid for Kernel container.");
}

std::string Allocate::toString(int indent_size) const {
  std::stringstream ss;
  indent(ss, indent_size) << buffer()->toString();
  ss << " = ALLOCATE("
     << "buffer=" << buffer()->toString() << ", "
     << "mem_type=" << memoryType() << ", "
     << "size=" << size()->toInlineString();
  ss << ", "
     << "zero_init=" << boolLiteral(zeroInit()) << ")\n";
  if (alias() != nullptr) {
    indent(ss, indent_size) << kTab << ".alias=";
    ss << alias()->buffer()->toString() << "\n";
  }
  return ss.str();
}

std::string Allocate::toInlineString(int indent_size) const {
  TORCH_CHECK(false, "Tensor op can not be printed inline");
}

NVFUSER_DEFINE_CLONE_AND_CREATE(Allocate)

BlockSync::BlockSync(IrBuilderPasskey passkey, bool war_sync) : Expr(passkey) {
  TORCH_INTERNAL_ASSERT(
      passkey.ir_container_->isA<kir::Kernel>(),
      "IR type only valid for Kernel container.");
  addAttribute(
      IrBuilder::create<Attribute<bool>>(passkey.ir_container_, war_sync));
}

std::string BlockSync::toString(int indent_size) const {
  std::stringstream ss;
  indent(ss, indent_size) << "BLOCKSYNC(war_hazard="
                          << boolLiteral(isWarHazardSync()) << ")\n";
  return ss.str();
}

std::string BlockSync::toInlineString(int indent_size) const {
  TORCH_CHECK(false, "Tensor op can not be printed inline");
}

NVFUSER_DEFINE_CLONE_AND_CREATE(BlockSync)

GridSync::GridSync(
    IrBuilderPasskey passkey,
    ParallelTypeBitmap sync_dims,
    Val* sync_buffer)
    : Expr(passkey) {
  addAttribute(IrBuilder::create<Attribute<ParallelTypeBitmap>>(
      passkey.ir_container_, sync_dims));
  addAttribute(sync_buffer);
}

std::string GridSync::toString(int indent_size) const {
  std::stringstream ss;
  indent(ss, indent_size) << "GRIDSYNC(" << syncDims().toString() << ", "
                          << syncBuffer()->toString() << ")\n";
  return ss.str();
}

std::string GridSync::toInlineString(int indent_size) const {
  TORCH_CHECK(false, "Tensor op can not be printed inline");
}

NVFUSER_DEFINE_CLONE_AND_CREATE(GridSync)

CpAsyncWait::CpAsyncWait(IrBuilderPasskey passkey, unsigned int keep_stages)
    : Expr(passkey) {
  TORCH_INTERNAL_ASSERT(
      passkey.ir_container_->isA<kir::Kernel>(),
      "IR type only valid for Kernel container.");
  addAttribute(IrBuilder::create<Attribute<unsigned int>>(
      passkey.ir_container_, keep_stages));
}

std::string CpAsyncWait::toString(int indent_size) const {
  std::stringstream ss;
  indent(ss, indent_size) << "CPASYNC_WAIT(" << keepStages() << ")\n";
  return ss.str();
}

std::string CpAsyncWait::toInlineString(int indent_size) const {
  TORCH_CHECK(false, "Tensor op can not be printed inline");
}

NVFUSER_DEFINE_CLONE_AND_CREATE(CpAsyncWait)

CpAsyncCommit::CpAsyncCommit(IrBuilderPasskey passkey) : Expr(passkey) {
  TORCH_INTERNAL_ASSERT(
      passkey.ir_container_->isA<kir::Kernel>(),
      "IR type only valid for Kernel container.");
}

std::string CpAsyncCommit::toString(int indent_size) const {
  std::stringstream ss;
  indent(ss, indent_size) << "CPASYNC_WAIT()\n";
  return ss.str();
}

std::string CpAsyncCommit::toInlineString(int indent_size) const {
  TORCH_CHECK(false, "Tensor op can not be printed inline");
}

NVFUSER_DEFINE_CLONE_AND_CREATE(CpAsyncCommit)

InitMagicZero::InitMagicZero(IrBuilderPasskey passkey) : Expr(passkey) {
  TORCH_INTERNAL_ASSERT(
      passkey.ir_container_->isA<kir::Kernel>(),
      "IR type only valid for Kernel container.");
}

std::string InitMagicZero::toString(int indent_size) const {
  std::stringstream ss;
  indent(ss, indent_size) << "NVFUSER_DEFINE_MAGIC_ZERO\n";
  return ss.str();
}

std::string InitMagicZero::toInlineString(int indent_size) const {
  TORCH_CHECK(false, "Tensor op can not be printed inline");
}

NVFUSER_DEFINE_CLONE_AND_CREATE(InitMagicZero)

UpdateMagicZero::UpdateMagicZero(IrBuilderPasskey passkey) : Expr(passkey) {
  TORCH_INTERNAL_ASSERT(
      passkey.ir_container_->isA<kir::Kernel>(),
      "IR type only valid for Kernel container.");
}

std::string UpdateMagicZero::toString(int indent_size) const {
  std::stringstream ss;
  indent(ss, indent_size) << "NVFUSER_UPDATE_MAGIC_ZERO\n";
  return ss.str();
}

std::string UpdateMagicZero::toInlineString(int indent_size) const {
  TORCH_CHECK(false, "Tensor op can not be printed inline");
}

NVFUSER_DEFINE_CLONE_AND_CREATE(UpdateMagicZero)

std::string Scope::toString(int indent_size) const {
  std::stringstream ss;
  for (auto expr : exprs()) {
    ss << expr->toString(indent_size);
  }
  return ss.str();
}

void Scope::insert(std::vector<Expr*>::const_iterator pos, Expr* expr) {
  exprs_.insert(pos, expr);
}

void Scope::insert_before(Expr* ref, Expr* expr) {
  const auto it = std::find(exprs_.begin(), exprs_.end(), ref);
  TORCH_INTERNAL_ASSERT(
      it != exprs_.end(),
      "Tried to insert ",
      expr,
      " before the reference: ",
      ref,
      " however the reference was not found in this scope.");
  insert(it, expr);
}

void Scope::insert_after(Expr* ref, Expr* expr) {
  const auto it = std::find(exprs_.begin(), exprs_.end(), ref);
  TORCH_INTERNAL_ASSERT(
      it != exprs_.end(),
      "Tried to insert ",
      expr,
      " after the reference: ",
      ref,
      " however the reference was not found in this scope.");
  insert(it + 1, expr);
}

void Scope::insert(size_t pos, Expr* expr) {
  const auto it = exprs_.begin() + pos;
  insert(it, expr);
}

void Scope::erase(std::vector<Expr*>::const_iterator pos) {
  // Remove the scope of the expr if this is the scope
  C10_UNUSED auto expr = *pos;
  exprs_.erase(pos);
}

void Scope::erase(Expr* ref) {
  const auto it = std::find(exprs_.begin(), exprs_.end(), ref);
  if (it != exprs_.end()) {
    erase(it);
  }
}

void Scope::erase(size_t pos) {
  TORCH_INTERNAL_ASSERT(pos < size());
  erase(exprs_.begin() + pos);
}

bool Scope::contains(Expr* expr) const {
  const auto it = std::find(exprs_.begin(), exprs_.end(), expr);
  return it != exprs_.end();
}

void Scope::clear() {
  exprs_.clear();
}

ForLoop::ForLoop(
    IrBuilderPasskey passkey,
    IterDomain* iter_domain,
    Val* index,
    Val* start,
    Val* stop,
    Val* step,
    bool vectorize,
    Val* vectorize_shift,
    bool unroll_required,
    DoubleBufferLoopStage double_buffer_loop_stage)
    : Expr(passkey) {
  TORCH_INTERNAL_ASSERT(
      passkey.ir_container_->isA<kir::Kernel>(),
      "IR type only valid for Kernel container.");
  TORCH_INTERNAL_ASSERT(index->dtype() == DataType::Int);
  addInput(index);
  addInput(iter_domain);
  if (start == nullptr && iter_domain->isThread()) {
    start = NamedScalar::getParallelIndex(iter_domain->getParallelType());
  }
  if (step == nullptr) {
    if (iter_domain->isThread()) {
      step = NamedScalar::getParallelDim(iter_domain->getParallelType());
    } else {
      step = FusionGuard::getCurFusion()->oneVal();
    }
  }
  addAttribute(start);
  addAttribute(stop);
  addAttribute(step);
  addAttribute(
      IrBuilder::create<Attribute<bool>>(passkey.ir_container_, vectorize));
  addAttribute(vectorize_shift);
  addAttribute(IrBuilder::create<Attribute<bool>>(
      passkey.ir_container_, unroll_required));
  addAttribute(IrBuilder::create<Attribute<DoubleBufferLoopStage>>(
      passkey.ir_container_, double_buffer_loop_stage));
  // Storing IR nodes as Attribute is not safe with IrCloner, but fortunately
  // kernel IR does not need this feature.
  addAttribute(
      IrBuilder::create<Attribute<Scope>>(passkey.ir_container_, this));
}

ForLoop::ForLoop(IrBuilderPasskey passkey, IterDomain* iter_domain)
    : ForLoop(
          passkey,
          iter_domain,
          GpuLower::current()->caMap()->getIndexVariable(iter_domain),
          nullptr,
          nullptr,
          nullptr,
          !iter_domain->isBroadcast() &&
              isParallelTypeVectorize(iter_domain->getParallelType()),
          nullptr,
          false,
          DoubleBufferLoopStage::NotApplicable) {
  TORCH_INTERNAL_ASSERT(
      passkey.ir_container_->isA<kir::Kernel>(),
      "IR type only valid for Kernel container.");
}

ForLoop::ForLoop(IrBuilderPasskey passkey, const ForLoop* other)
    : ForLoop(
          passkey,
          other->iter_domain(),
          other->index(),
          other->start(),
          other->stop(),
          other->step(),
          other->vectorize(),
          other->vectorize_shift(),
          other->isUnrollRequired(),
          other->doubleBufferLoopStage()) {
  TORCH_INTERNAL_ASSERT(
      passkey.ir_container_->isA<kir::Kernel>(),
      "IR type only valid for Kernel container.");
}

std::string ForLoop::toString(int indent_size) const {
  std::stringstream ss;
  indent(ss, indent_size) << "FOR " << index()->toString() << " in "
                          << iter_domain()->toString() << ":\n"
                          << body().toString(indent_size + 1);
  return ss.str();
}

std::string ForLoop::toInlineString(int indent_size) const {
  TORCH_CHECK(false, "Tensor op can not be printed inline");
}

bool ForLoop::isUnrollable() const {
  // Start and stop must be constant, must not be a broadcast
  // dimension, cannot be bound to a parallel dimension, must not be
  // vectorized.
  return start()->isConstScalar() && stop()->isConstScalar() &&
      !iter_domain()->isThread() && !iter_domain()->isBroadcast() &&
      !vectorize();
}

bool ForLoop::isUnrolled() const {
  if (isUnrollRequired() && !isUnrollable()) {
    TORCH_WARN(
        "Unroll required but not possible. Register allocation disabled. Loop index: ",
        index()->toString());
    return false;
  }

  // Size-one loop will not be materialized as a loop, so return false
  if (start()->isZeroInt() && stop()->isOneInt()) {
    return false;
  }

  // Unroll if required.
  if (isUnrollRequired()) {
    return true;
  }

  // Don't unroll if not possible
  if (!isUnrollable()) {
    return false;
  }

  // Unrolling is technically possible but avoided
  if (iter_domain()->getParallelType() == ParallelType::Unswitch) {
    // Use ParallelType::Unroll if unrolling is desired. Note that
    // unswitched size-one loops are not unrolled as they are not
    // materialized as actual for-loops.
    return false;
  }

  return true;
}

Val* ForLoop::start() const {
  if (attributeVal(0) != nullptr) {
    return attributeVal(0);
  } else {
    // clang-tidy complains without this
    TORCH_INTERNAL_ASSERT(iter_domain() != nullptr);
    return iter_domain()->start();
  }
}

Val* ForLoop::stop() const {
  if (attributeVal(1) != nullptr) {
    return attributeVal(1);
  } else {
    // clang-tidy complains without this
    TORCH_INTERNAL_ASSERT(iter_domain() != nullptr);
    return iter_domain()->extent();
  }
}

Val* ForLoop::step() const {
  TORCH_INTERNAL_ASSERT(attributeVal(2) != nullptr);
  return attributeVal(2);
}

Val* ForLoop::simplifiedStop() const {
  return simplifyExpr(stop());
}

bool ForLoop::isTrivial() const {
  // These loops are not materialized
  if (vectorize() || iter_domain()->isBroadcast() ||
      iter_domain()->isStride() || iter_domain()->isMma()) {
    return true;
  }

  // By default, a parallelized loop would look like:
  //
  //   for (int x = threadIdx.x; x < stop; x += blockDim.x) {
  //     do_some_comp(x);
  //   }
  //
  // When stop is guaranteed to be smaller or equal to the number of
  // threads, the for-loop is not necessary. In the above case, we
  // would just generate the loop body without the for clause but
  // references to the loop index replaced by the loop start value.
  //
  // When the loop end is the same as the IterDomain extent, the
  // assumption can be safely made. This is more conservative than
  // necessary since the loop stop value just needs to be <= the
  // IterDomain extent. However, at this point, this conservative
  // analysis seems sufficient.
  if (stop() == iter_domain()->extent() && iter_domain()->isThread()) {
    return true;
  }

  // Extent-1 loop: for (int i = 0; i < 1; ++i) {
  if (start()->isZeroInt() && stop()->isOneInt() && step()->isOneInt()) {
    return true;
  }

  // Another extent-1 loop: for (int i = N - 1; i < N; ++i) {
  if (start()->definition() != nullptr &&
      start()->definition()->isA<BinaryOp>() &&
      start()->definition()->as<BinaryOp>()->getBinaryOpType() ==
          BinaryOpType::Sub &&
      start()->definition()->as<BinaryOp>()->lhs() == stop() &&
      start()->definition()->as<BinaryOp>()->rhs()->isOneInt()) {
    return true;
  }

  return false;
}

namespace {

//! A utility class to check if an expression of a particular type exists
class ExprFinder : kir::ConstIrVisitor {
 public:
  //! True if expr or any of its nested expressions is a type included in
  //! expr_types
  static bool exists(
      const Expr* expr,
      const std::unordered_set<std::type_index>& expr_types) {
    ExprFinder finder(expr_types);
    finder.handle(std::vector<const Expr*>{expr});
    return finder.is_found_;
  }

 private:
  ExprFinder(const std::unordered_set<std::type_index>& expr_types)
      : expr_types_(expr_types) {}

  using kir::ConstIrVisitor::handle;

  void handle(const Expr* expr) final {
    if (expr_types_.find(typeid(*expr)) != expr_types_.end()) {
      is_found_ = true;
      return;
    }
    kir::ConstIrVisitor::handle(expr);
  }

 private:
  const std::unordered_set<std::type_index>& expr_types_;
  bool is_found_ = false;
};

} // namespace

bool ForLoop::isGroup() const {
  //! True if loop is grouped. The IterDomain of the loop must have
  //! ParallelType::Group, but it isn't sufficient as the loop may be
  //! for an initialization expression, for which the loop shold not
  //! be grouped. Make sure a GroupedGridReduction is found.
  if (iter_domain()->getParallelType() != ParallelType::Group) {
    return false;
  }

  return ExprFinder::exists(
      this,
      {typeid(kir::GroupedGridReduction), typeid(kir::GroupedGridWelford)});
}

NVFUSER_DEFINE_CLONE_AND_CREATE(ForLoop)

IfThenElse::IfThenElse(IrBuilderPasskey passkey, Predicate* cond)
    : Expr(passkey) {
  setPredicate(cond);
  addInput(cond);
  // Storing IR nodes as Attribute is not safe with IrCloner, but fortunately
  // kernel IR does not need this feature.
  addAttribute(
      IrBuilder::create<Attribute<Scope>>(passkey.ir_container_, this));
  addAttribute(
      IrBuilder::create<Attribute<Scope>>(passkey.ir_container_, this));
}

std::string IfThenElse::toString(int indent_size) const {
  std::stringstream ss;
  indent(ss, indent_size) << "IF " << predicate()->toString() << ":\n"
                          << thenBody().toString(indent_size + 1);
  if (hasElse()) {
    indent(ss, indent_size) << "ELSE:\n"
                            << elseBody().toString(indent_size + 1);
  }
  return ss.str();
}

std::string IfThenElse::toInlineString(int indent_size) const {
  TORCH_CHECK(false, "Tensor op can not be printed inline");
}

NVFUSER_DEFINE_CLONE_AND_CREATE(IfThenElse)

GridReduction::GridReduction(
    IrBuilderPasskey passkey,
    BinaryOpType reduction_op_type,
    Val* init,
    Val* out,
    Val* in,
    Allocate* reduction_buffer,
    Allocate* sync_buffer,
    Val* entrance_index,
    Val* entrances,
    bool is_allreduce)
    : ReductionOp(passkey, reduction_op_type, init, out, in, is_allreduce) {
  TORCH_INTERNAL_ASSERT(
      passkey.ir_container_->isA<kir::Kernel>(),
      "IR type only valid for Kernel container.");
  TORCH_INTERNAL_ASSERT(
      attributes().size() == num_reduction_op_attr,
      "The num_reduction_op_attr does not match the number of attributes ReductionOp has."
      "If you changed ReductionOp, please change num_reduction_op_attr accordingly.");
  addAttribute(reduction_buffer);
  addAttribute(sync_buffer);
  addAttribute(entrance_index);
  addAttribute(entrances);
  addAttribute(
      IrBuilder::create<Attribute<ParallelTypeBitmap>>(passkey.ir_container_));
}

std::string GridReduction::toString(int indent_size) const {
  std::stringstream ss;
  indent(ss, indent_size) << out()->toString() << " = reduction( "
                          << in()->toString()
                          << ", op = " << getReductionOpType()
                          << ", initial value = " << init()->toString()
                          << ",\n";
  ++indent_size;
  indent(ss, indent_size) << "reduction buffer = "
                          << reduction_buffer()->buffer()->toString() << ",\n";
  indent(ss, indent_size) << "sync buffer = "
                          << sync_buffer()->buffer()->toString() << ",\n";
  indent(ss, indent_size) << "read predicate = ";
  if (predicate() != nullptr) {
    ss << predicate()->toString();
  } else {
    ss << "nullptr";
  }
  ss << ",\n";
  indent(ss, indent_size) << "write predicate = ";
  if (writePredicate() != nullptr) {
    ss << writePredicate()->toString();
  } else {
    ss << "nullptr";
  }
  ss << ",\n";
  indent(ss, indent_size) << "thread predicate = "
                          << threadPredicate().toString() << ",\n";
  indent(ss, indent_size) << "allreduce = "
                          << (isAllreduce() ? "true" : "false") << " )\n";
  return ss.str();
}

std::string GridReduction::toInlineString(int indent_size) const {
  TORCH_CHECK(false, "Tensor op can not be printed inline");
}

NVFUSER_DEFINE_CLONE_AND_CREATE(GridReduction)

GroupedGridReduction::GroupedGridReduction(
    IrBuilderPasskey passkey,
    std::vector<BinaryOpType> reduction_op_types,
    std::vector<Val*> init_vals,
    std::vector<Val*> outputs,
    std::vector<Val*> inputs,
    std::vector<Allocate*> reduction_buffers,
    Allocate* sync_buffer,
    Val* entrance_index,
    Val* entrances,
    Val* buffer_stride,
    bool is_allreduce)
    : GroupedReductionOp(
          passkey,
          std::move(reduction_op_types),
          std::move(init_vals),
          std::move(outputs),
          std::move(inputs),
          is_allreduce) {
  TORCH_INTERNAL_ASSERT(
      passkey.ir_container_->isA<kir::Kernel>(),
      "IR type only valid for Kernel container.");
  TORCH_INTERNAL_ASSERT(
      attributes().size() == numGroupedReductionOpAttr(),
      "The numGroupedReductionOpAttr() does not match the number of attributes GroupedReductionOp has."
      "If you changed GroupedReductionOp, please change numGroupedReductionOpAttr() accordingly.");
  addAttribute(sync_buffer);
  addAttribute(entrance_index);
  addAttribute(entrances);
  addAttribute(buffer_stride);
  addAttribute(
      IrBuilder::create<Attribute<ParallelTypeBitmap>>(passkey.ir_container_));
  for (auto buffer : reduction_buffers) {
    addAttribute(buffer);
  }
}

std::string GroupedGridReduction::toString(int indent_size) const {
  std::stringstream ss;
  indent(ss, indent_size) << "GroupedGridReduction(\n";
  ++indent_size;
  for (const auto i : c10::irange(numHorizontallyGroupedExprs())) {
    indent(ss, indent_size)
        << output(i)->toString() << " = reduction( " << input(i)->toString()
        << ", op = " << getReductionOpType(i)
        << ", initial value = " << initVal(i)->toString()
        << ", reduction buffer = "
        << reduction_buffers().at(i)->buffer()->toString() << " )\n";
  }
  indent(ss, indent_size) << "sync buffer = "
                          << sync_buffer()->buffer()->toString() << ",\n";
  indent(ss, indent_size) << "read predicate = ";
  if (predicate() != nullptr) {
    ss << predicate()->toString();
  } else {
    ss << "nullptr";
  }
  ss << ",\n";
  indent(ss, indent_size) << "write predicate = ";
  if (writePredicate() != nullptr) {
    ss << writePredicate()->toString();
  } else {
    ss << "nullptr";
  }
  ss << ",\n";
  indent(ss, indent_size) << "thread predicate = "
                          << threadPredicate().toString() << ",\n";
  indent(ss, indent_size) << "allreduce = "
                          << (isAllreduce() ? "true" : "false") << " )\n";
  --indent_size;
  return ss.str();
}

std::string GroupedGridReduction::toInlineString(int indent_size) const {
  TORCH_CHECK(false, "Tensor op can not be printed inline");
}

NVFUSER_DEFINE_CLONE_AND_CREATE(GroupedGridReduction)

GridBroadcast::GridBroadcast(
    IrBuilderPasskey passkey,
    BroadcastOp* broadcast_op,
    Allocate* broadcast_buffer,
    Allocate* sync_buffer)
    : Expr(passkey) {
  TORCH_INTERNAL_ASSERT(
      passkey.ir_container_->isA<kir::Kernel>(),
      "IR type only valid for Kernel container.");
  addAttribute(broadcast_op);
  addAttribute(broadcast_buffer);
  addAttribute(sync_buffer);
}

std::string GridBroadcast::toString(int indent_size) const {
  std::stringstream ss;
  const auto* broadcast_op = this->broadcast_op();
  indent(ss, indent_size) << broadcast_op->out()->toString() << " = "
                          << "GRID_BROADCAST(in="
                          << broadcast_op->in()->toString() << ")\n";
  indent(ss, indent_size) << kTab << ".broadcast_buffer="
                          << broadcast_buffer()->buffer()->toString() << "\n";
  indent(ss, indent_size) << kTab << ".sync_buffer="
                          << sync_buffer()->buffer()->toString() << "\n";
  return ss.str();
}

std::string GridBroadcast::toInlineString(int indent_size) const {
  TORCH_CHECK(false, "Tensor op can not be printed inline");
}

NVFUSER_DEFINE_CLONE_AND_CREATE(GridBroadcast)

GridWelford::GridWelford(
    IrBuilderPasskey passkey,
    WelfordOp* welford_op,
    Allocate* var_buffer,
    Allocate* avg_buffer,
    Allocate* n_buffer,
    Allocate* sync_buffer,
    Val* entrance_index,
    Val* entrances)
    : Expr(passkey) {
  TORCH_INTERNAL_ASSERT(
      passkey.ir_container_->isA<kir::Kernel>(),
      "IR type only valid for Kernel container.");
  addAttribute(welford_op);
  addAttribute(var_buffer);
  addAttribute(avg_buffer);
  addAttribute(n_buffer);
  addAttribute(sync_buffer);
  addAttribute(entrance_index);
  addAttribute(entrances);
  addAttribute(
      IrBuilder::create<Attribute<ParallelTypeBitmap>>(passkey.ir_container_));
}

std::string GridWelford::toString(int indent_size) const {
  std::stringstream ss;
  const auto* welford_op = this->welford_op();
  indent(ss, indent_size) << welford_op->outAvg()->toString() << " (Avg),\n";
  indent(ss, indent_size) << welford_op->outVar()->toString() << " (Var),\n";
  indent(ss, indent_size) << welford_op->outN()->toString() << " (Count)\n";
  indent(ss, indent_size) << " = Welford (\n";
  ++indent_size;
  indent(ss, indent_size) << welford_op->inAvg()->toString() << " (Avg),\n";
  indent(ss, indent_size) << welford_op->inVar()->toString() << " (Var),\n";
  indent(ss, indent_size) << welford_op->inN()->toString() << " (Count)\n";
  indent(ss, indent_size) << "initial value =\n";
  ++indent_size;
  indent(ss, indent_size) << welford_op->initAvg()->toString() << " (Avg),\n";
  indent(ss, indent_size) << welford_op->initVar()->toString() << " (Var),\n";
  indent(ss, indent_size) << welford_op->initN()->toString() << " (Count),\n";
  --indent_size;
  indent(ss, indent_size) << "reduction buffer =\n";
  ++indent_size;
  indent(ss, indent_size) << avg_buffer()->buffer()->toString() << " (Avg),\n";
  indent(ss, indent_size) << var_buffer()->buffer()->toString() << " (Var),\n";
  indent(ss, indent_size) << N_buffer()->buffer()->toString() << " (Count),\n";
  --indent_size;
  indent(ss, indent_size) << "sync buffer = "
                          << sync_buffer()->buffer()->toString() << ",\n";
  indent(ss, indent_size) << "read predicate = ";
  if (welford_op->predicate() != nullptr) {
    ss << welford_op->predicate();
  } else {
    ss << "nullptr";
  }
  ss << ",\n";
  indent(ss, indent_size) << "write predicate = ";
  if (welford_op->writePredicate() != nullptr) {
    ss << welford_op->writePredicate()->toString();
  } else {
    ss << "nullptr";
  }
  ss << ",\n";
  indent(ss, indent_size) << "grid read predicate = ";
  if (predicate() != nullptr) {
    ss << predicate()->toString();
  } else {
    ss << "nullptr";
  }
  ss << ",\n";
  indent(ss, indent_size) << "grid write predicate = ";
  if (writePredicate() != nullptr) {
    ss << writePredicate()->toString();
  } else {
    ss << "nullptr";
  }
  ss << ",\n";
  indent(ss, indent_size) << "thread predicate = "
                          << threadPredicate().toString() << ",\n";
  indent(ss, indent_size) << "allreduce = "
                          << (welford_op->isAllreduce() ? "true" : "false")
                          << " )\n";
  return ss.str();
}

std::string GridWelford::toInlineString(int indent_size) const {
  TORCH_CHECK(false, "Tensor op can not be printed inline");
}

NVFUSER_DEFINE_CLONE_AND_CREATE(GridWelford)

GroupedGridWelford::GroupedGridWelford(
    IrBuilderPasskey passkey,
    std::vector<WelfordTriplet> output_vals,
    std::vector<WelfordTriplet> input_vals,
    std::vector<WelfordTriplet> init_vals,
    std::array<std::vector<Allocate*>, 3> reduction_buffers,
    Allocate* sync_buffer,
    Val* entrance_index,
    Val* entrances,
    Val* buffer_stride,
    bool is_allreduce,
    bool use_outer_opt)
    : GroupedWelfordOp(
          passkey,
          std::move(output_vals),
          std::move(input_vals),
          std::move(init_vals),
          is_allreduce) {
  TORCH_INTERNAL_ASSERT(
      passkey.ir_container_->isA<kir::Kernel>(),
      "IR type only valid for Kernel container.");
  TORCH_INTERNAL_ASSERT(
      attributes().size() == numGroupedWelfordOpAttr(),
      "The numGroupedWelfordOpAttr() does not match the number of attributes GroupedWelfordOp has."
      "If you changed GroupedReductionOp, please change numGroupedWelfordOpAttr() accordingly.");
  addAttribute(sync_buffer);
  addAttribute(entrance_index);
  addAttribute(entrances);
  addAttribute(buffer_stride);
  addAttribute(
      IrBuilder::create<Attribute<ParallelTypeBitmap>>(passkey.ir_container_));
  TORCH_INTERNAL_ASSERT(
      reduction_buffers[0].size() == reduction_buffers[1].size());
  TORCH_INTERNAL_ASSERT(
      reduction_buffers[0].size() == reduction_buffers[2].size());
  for (auto i : c10::irange(reduction_buffers[0].size())) {
    addAttribute(reduction_buffers[0].at(i));
    addAttribute(reduction_buffers[1].at(i));
    addAttribute(reduction_buffers[2].at(i));
  }

  addAttribute(
      IrBuilder::create<Attribute<bool>>(passkey.ir_container_, use_outer_opt));
}

int GroupedGridWelford::getSmemBufferSize(int bdimx, int bdimy, int bdimz)
    const {
  auto out_tv = ir_utils::getTvOutput(this);
  auto kernel = dynamic_cast<kir::Kernel*>(container());
  TORCH_INTERNAL_ASSERT(kernel != nullptr);

  // By default, the required size is the same as the normal Welford reduction
  if (!useOuterOpt()) {
    return bdimx * bdimy * bdimz * dataTypeSize(out_tv->getDataType().value()) *
        2 +
        bdimx * bdimy * bdimz *
        dataTypeSize(DataType::Index, kernel->indexType());
  }

  // In the outer-reduction version, the size is blockDim.x * NumberOfWarps *
  // GroupCount

  int group_count = 1;
  for (auto axis : out_tv->domain()->domain()) {
    auto pt = axis->getParallelType();
    if (pt == ParallelType::Group) {
      auto extent_int = axis->extent()->getInt();
      TORCH_INTERNAL_ASSERT(extent_int.has_value());
      group_count *= (int)extent_int.value();
    }
  }

  TORCH_INTERNAL_ASSERT(group_count > 1);

  int num_warps = bdimx * bdimy / 32;
  TORCH_INTERNAL_ASSERT((bdimx * bdimy) % 32 == 0);

  int buf_size_for_avg_var = bdimx * num_warps * group_count *
      dataTypeSize(out_tv->getDataType().value());
  int buf_size_for_N =
      num_warps * dataTypeSize(DataType::Index, kernel->indexType());

  return buf_size_for_avg_var * 2 + buf_size_for_N;
}

std::string GroupedGridWelford::toString(int indent_size) const {
  std::stringstream ss;
  indent(ss, indent_size) << "GroupedGridWelford(\n";
  ++indent_size;
  for (const auto i : c10::irange(numHorizontallyGroupedExprs())) {
    indent(ss, indent_size) << outAvg(i)->toString() << " (Avg),\n";
    indent(ss, indent_size) << outVar(i)->toString() << " (Var),\n";
    indent(ss, indent_size) << outN(i)->toString() << " (Count)\n";
    indent(ss, indent_size) << " = Welford (\n";
    ++indent_size;
    indent(ss, indent_size) << inAvg(i)->toString() << " (Avg),\n";
    indent(ss, indent_size) << inVar(i)->toString() << " (Var),\n";
    indent(ss, indent_size) << inN(i)->toString() << " (Count)\n";
    indent(ss, indent_size) << "initial value =\n";
    ++indent_size;
    indent(ss, indent_size) << initAvg(i)->toString() << " (Avg),\n";
    indent(ss, indent_size) << initVar(i)->toString() << " (Var),\n";
    indent(ss, indent_size) << initN(i)->toString() << " (Count),\n";
    --indent_size;
    indent(ss, indent_size) << "reduction buffer =\n";
    ++indent_size;
    indent(ss, indent_size)
        << reduction_buffers()[0].at(i)->buffer()->toString() << " (Avg),\n";
    indent(ss, indent_size)
        << reduction_buffers()[1].at(i)->buffer()->toString() << " (Var),\n";
    indent(ss, indent_size)
        << reduction_buffers()[2].at(i)->buffer()->toString() << " (Count) )\n";
    indent_size -= 2;
  }
  indent(ss, indent_size) << "sync buffer = "
                          << sync_buffer()->buffer()->toString() << ",\n";
  indent(ss, indent_size) << "read predicate = ";
  if (predicate() != nullptr) {
    ss << predicate()->toString();
  } else {
    ss << "nullptr";
  }
  ss << ",\n";
  indent(ss, indent_size) << "write predicate = ";
  if (writePredicate() != nullptr) {
    ss << writePredicate()->toString();
  } else {
    ss << "nullptr";
  }
  ss << ",\n";
  indent(ss, indent_size) << "thread predicate = "
                          << threadPredicate().toString() << ",\n";
  indent(ss, indent_size) << "allreduce = "
                          << (isAllreduce() ? "true" : "false") << " )\n";
  return ss.str();
}

std::string GroupedGridWelford::toInlineString(int indent_size) const {
  TORCH_CHECK(false, "Tensor op can not be printed inline");
}

NVFUSER_DEFINE_CLONE_AND_CREATE(GroupedGridWelford)

VectorizedWelfordOp::VectorizedWelfordOp(
    IrBuilderPasskey passkey,
    const WelfordTriplet& output,
    const WelfordTriplet& input,
    const WelfordTriplet& init,
    Val* count,
    Val* reciprocal_of_count,
    Bool* hoisted_predicate)
    : WelfordOp(passkey, output, input, init, false) {
  addAttribute(count);
  addAttribute(reciprocal_of_count);
  addAttribute(hoisted_predicate);
}

NVFUSER_DEFINE_CLONE_AND_CREATE(VectorizedWelfordOp)

AllocateFusedReduction::AllocateFusedReduction(
    IrBuilderPasskey passkey,
    Expr* grid_expr)
    : Expr(passkey) {
  TORCH_INTERNAL_ASSERT(
      passkey.ir_container_->isA<kir::Kernel>(),
      "IR type only valid for Kernel container.");
  addAttribute(grid_expr);
}

std::string AllocateFusedReduction::toString(int indent_size) const {
  std::stringstream ss;
  indent(ss, indent_size) << "AllocateFusedReduction(reduction buffer="
                          << out()->toString() << ")\n";
  return ss.str();
}

std::string AllocateFusedReduction::toInlineString(int indent_size) const {
  TORCH_CHECK(false, "Tensor op can not be printed inline");
}

TensorIndex* AllocateFusedReduction::out() const {
  TORCH_INTERNAL_ASSERT(gridExpr() != nullptr);
  if (gridExpr()->isA<GridReduction>() ||
      gridExpr()->isA<GroupedGridReduction>()) {
    return gridExpr()->outputs().at(0)->as<kir::TensorIndex>();
  } else if (auto grid_welford = dynamic_cast<GridWelford*>(gridExpr())) {
    return grid_welford->welford_op()->out()->as<kir::TensorIndex>();
  } else if (
      auto grouped_grid_welford =
          dynamic_cast<GroupedGridWelford*>(gridExpr())) {
    return grouped_grid_welford->out(0)->as<kir::TensorIndex>();
  } else {
    TORCH_INTERNAL_ASSERT(
        false, "Invalid grid expression: ", gridExpr()->toString());
  }
}

const ParallelTypeBitmap& AllocateFusedReduction::threadPredicate() const {
  TORCH_INTERNAL_ASSERT(gridExpr() != nullptr);
  if (auto grid_reduction = dynamic_cast<GridReduction*>(gridExpr())) {
    return grid_reduction->threadPredicate();
  } else if (auto grid_welford = dynamic_cast<GridWelford*>(gridExpr())) {
    return grid_welford->threadPredicate();
  } else if (
      auto grouped_grid_reduction =
          dynamic_cast<GroupedGridReduction*>(gridExpr())) {
    return grouped_grid_reduction->threadPredicate();
  } else if (
      auto grouped_grid_welford =
          dynamic_cast<GroupedGridWelford*>(gridExpr())) {
    return grouped_grid_welford->threadPredicate();
  } else {
    TORCH_INTERNAL_ASSERT(
        false, "Invalid grid expression: ", gridExpr()->toString());
  }
}

NVFUSER_DEFINE_CLONE_AND_CREATE(AllocateFusedReduction)

} // namespace kir
} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
