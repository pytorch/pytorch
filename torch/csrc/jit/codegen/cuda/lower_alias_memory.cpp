#include <torch/csrc/jit/codegen/cuda/lower_alias_memory.h>

#include <torch/csrc/jit/codegen/cuda/expr_evaluator.h>
#include <torch/csrc/jit/codegen/cuda/instrumentation.h>
#include <torch/csrc/jit/codegen/cuda/ir_iostream.h>
#include <torch/csrc/jit/codegen/cuda/ir_utils.h>
#include <torch/csrc/jit/codegen/cuda/kernel_ir.h>
#include <torch/csrc/jit/codegen/cuda/kernel_ir_dispatch.h>
#include <torch/csrc/jit/codegen/cuda/lower2device.h>
#include <torch/csrc/jit/codegen/cuda/lower_utils.h>

#include <sstream>
#include <unordered_map>
#include <unordered_set>

// The goal of this pass is to change allocations to use other
// allocations when possible. To do so, there are 3 main stages and
// corresponding classes.
//
// - Analyze live ranges of tensors (class AllocationInfoMap)
// - Find allocations of tensors that can reuse other allocations
//   (class ReusableAllocationFinder)
// - Replace those allocation expressions with their alias fields
//   pointing to reused allocations (class AllocationReuseModifier)

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {

namespace {
// Alias used for std::transform
IterDomain* exactConcreteId(IterDomain* id) {
  return GpuLower::current()->caMap()->getConcreteMappedID(
      id, IdMappingMode::EXACT);
}

//! Checks that the current loop nest is realizing a serial
//! broadcast so that each index of producer buffer can be visited
//! multiple times, in which case the aggressive is not valid.
//!
//! Need to look at all the loops at each consumer expression of the
//! producer tensor rather than just the consumer IterDomains of the
//! expression. Here's an example case from
//! FusionIssue2163ReproInvalidAlias:
//!
//! T3_l[ iS31{( ceilDiv(16, 8) )}, iS32{8} ] ca_pos( 2 ) produce_pos( 1)
//!  = T2_l[ iS33{( ceilDiv(16, 8) )}, iS34{8} ] ca_pos( 1 );
//! T7_l[ iS25{( ceilDiv(16, 8) )}, bS11{1}, iS26{8} ] ca_pos( 3 ) produce_pos(
//! 3)
//!  = broadcast( T3_l[ iS31{( ceilDiv(16, 8) )}, iS32{8} ] ca_pos( 2 )
//!  produce_pos( 1) )
//!
//! When T2 is viewed just as the consumer of T3, it doesn't look like
//! there's a consumer IterDomain that could make T2 used multiple
//! times, but that's actually not the case. T3 is computed at
//! position 2 with its consumer T7, which adds a broadcast IterDomain
//! between the two domains, and that is eventually concretized, T2 is
//! indeed used multiple times. See also issue #2163.
bool isSerialBroadcastResolution(
    TensorView* producer,
    const std::vector<kir::ForLoop*>& for_loops) {
  //! Note: see issue #1785:
  //!  serial broadcast resolution doesn't only happen to
  //! immediate outputs of broadcast ops. We can also have
  //! example:
  //!  T1[I,B] = broadcast(T0[I]])
  //!  T3[I,I] = T1[I,B] + T2[I,I]
  //!  T4[I,I] = T3[I,I]
  //!  and generates the following loop:
  //! alloc T0[4]
  //! For i in 0..3
  //!   T0[...] =
  //!
  //! For j in 0...X:
  //!   alloc T3[4]
  //!   for k in 0..3:
  //!     alloc T1[1]
  //!     T1[0] = T0[k] // <- This is actually a broadcast resolution
  //!     T3[k] = T1[0] + T2[...]
  //!   T4[...] = T3[...]
  //!
  //! In this case we are actually visiting each pixel of T0 in each iteration
  //!  of the j loop while T1 was the broadcasted tensor causing this reuse.
  //!
  //! The current version of checking covers this scenario by checking the root
  //!  ids of the consumer concrete loop id's. Any time a local tensor like T0
  //!  appears in a re-use scenario like above, we should see a serial loop id
  //!  that was derived from some root id that doesn't concretely map to T0's
  //!  domain.

  // Serial concrete loop ids
  std::vector<Val*> serial_loop_concrete_ids;

  for (auto for_loop : for_loops) {
    // ForLoop::iter_domain() should be the concrete domain, but just
    // in case.
    auto concrete_loop_id = GpuLower::current()->caMap()->getConcreteMappedID(
        for_loop->iter_domain(), IdMappingMode::LOOP);

    // Check for any serial loop id with non-trivial extent. If the
    // concrete ID is a broadcast, it shouldn't materialize an actual
    // loop, so that can be ignored as well.
    if (!concrete_loop_id->isThread() &&
        !concrete_loop_id->extent()->isOneInt() &&
        !concrete_loop_id->isBroadcast()) {
      serial_loop_concrete_ids.push_back(concrete_loop_id);
    }
  }

  // Collect the root id's that the serial loop iterdomain
  //  are transformed from.
  // NOTE: This does not necessarily capture the actual root domains
  //  as the concrete domains may be post-view domains. We need to
  //  traverse across view boundaries as we do in indexing. This
  //  should not result in false aliasing but may miss safe aliasing
  //  opportunities.
  auto serial_loop_roots =
      InputsOf::outputs(FusionGuard::getCurFusion(), serial_loop_concrete_ids);

  // Collect exact concrete id's in producer's root domain
  std::unordered_set<IterDomain*> producer_exact_concrete_root_ids;
  auto producer_root =
      TensorDomain::noReductions(producer->getMaybeRFactorDomain());
  std::transform(
      producer_root.begin(),
      producer_root.end(),
      std::inserter(
          producer_exact_concrete_root_ids,
          producer_exact_concrete_root_ids.begin()),
      exactConcreteId);

  // Check if serial loop roots indexes any exact root id's that
  //  is not within the set of producer's root exact id's. These
  //  id's will imply that the same producer pixel is accessed
  //  in multiple iterations of the materialized serial loop.
  for (auto serial_loop_root :
       ir_utils::filterByType<IterDomain>(serial_loop_roots)) {
    if (!producer_exact_concrete_root_ids.count(
            GpuLower::current()->caMap()->getConcreteMappedID(
                serial_loop_root, IdMappingMode::EXACT))) {
      return true;
    }
  }

  return false;
}

//! Get string representation of Allocate size for symbolic comparison
//!
//!  TODO: Some expr simplifications could also be helpful
class SymbolicSizePrinter : private OptOutConstDispatch {
 public:
  static std::string printSize(const kir::Allocate* allocate) {
    SymbolicSizePrinter printer;
    printer.handle(allocate->size());
    return printer.os_.str();
  }

 private:
  using OptOutConstDispatch::handle;

  void handle(const Int* node) final {
    if (auto def = node->definition()) {
      OptOutConstDispatch::handle(def);
    } else if (node->isConst()) {
      os_ << *node->value();
    } else {
      os_ << "ki" << node->name();
    }
  }

  void handle(const NamedScalar* named_scalar) final {
    os_ << "@" << named_scalar->name();
  }

  void handle(const UnaryOp* unary_op) final {
    os_ << unary_op->getUnaryOpType() << "(";
    OptOutConstDispatch::handle(unary_op);
    os_ << ")";
  }

  void handle(const BinaryOp* binary_op) final {
    os_ << binary_op->getBinaryOpType() << "(";
    OptOutConstDispatch::handle(binary_op->lhs());
    os_ << ",";
    OptOutConstDispatch::handle(binary_op->rhs());
    os_ << ")";
  }

 private:
  std::stringstream os_;
};

class AllocationInfoMap;

//! A debug printer internal to this pass to support
//!  future expansion and inline annotation of pass info.
class BufferReuseDebugPrinter {
  enum class DebugLineType { EXPR, START_BLOCK, END_BLOCK };

  struct ExprInfo {
    int lineno = 0;
    DebugLineType line_type = DebugLineType::EXPR;
  };

  using DebugEntry = std::pair<ExprInfo, Expr*>;
  using DebugEntryPtr = std::unique_ptr<DebugEntry>;

 public:
  BufferReuseDebugPrinter() : ir_printer_(os_){};

  std::string dumpDebugInfo(const AllocationInfoMap* allocation_info_map) {
    allocation_info_map_ = allocation_info_map;
    os_.clear();
    for (auto& debug_entry : debug_info_) {
      switch (debug_entry->first.line_type) {
        case DebugLineType::START_BLOCK:
          startBlock();
          break;
        case DebugLineType::END_BLOCK:
          endBlock();
          break;
        case DebugLineType::EXPR:
          os_ << debug_entry->first.lineno;
          handle(debug_entry->second);
          break;
        default:
          TORCH_INTERNAL_ASSERT(false, "unreachable");
      }
    }
    os_ << "\n\n";
    return os_.str();
  }

  void pushBack(int lineno, Expr* expr) {
    makeExprEntry(lineno, expr);
  }

  void pushScope() {
    makeScopeEntry(DebugLineType::START_BLOCK);
  }

  void popScope() {
    makeScopeEntry(DebugLineType::END_BLOCK);
  }

 private:
  void makeExprEntry(int lineno, Expr* expr) {
    auto debug_entry_ptr = std::make_unique<DebugEntry>();
    debug_entry_ptr->first.lineno = lineno;
    debug_entry_ptr->second = expr;
    debug_info_.emplace_back(std::move(debug_entry_ptr));
  }

  void makeScopeEntry(DebugLineType line_type) {
    TORCH_INTERNAL_ASSERT(
        line_type == DebugLineType::END_BLOCK ||
        line_type == DebugLineType::START_BLOCK);
    auto debug_entry_ptr = std::make_unique<DebugEntry>();
    debug_entry_ptr->first.line_type = line_type;
    debug_entry_ptr->second = nullptr;
    debug_info_.emplace_back(std::move(debug_entry_ptr));
  }

  void handle(const Expr* node) {
    if (auto for_loop = dynamic_cast<const kir::ForLoop*>(node)) {
      handle(for_loop);
    } else if (auto ite = dynamic_cast<const kir::IfThenElse*>(node)) {
      handle(ite);
    } else {
      indent();
      ir_printer_.handle(node);
    }
    if (auto alloc = dynamic_cast<const kir::Allocate*>(node)) {
      printAllocInfo(alloc);
    }
  }

  void handle(const kir::ForLoop* node) {
    indent();
    os_ << "FOR ";
    ir_printer_.handle(node->index());
    os_ << " in ";
    ir_printer_.handle(node->iter_domain());
    os_ << ":\n";
  }

  void handle(const kir::IfThenElse* node) {
    // This pass doesn't yet need to handle
    //  ite but could fill in the blank here
    //  if this printer can be used for
    //  other passes or we have more
    //  complex ite pattern.
    TORCH_INTERNAL_ASSERT(false, "unsupported");
  }

  void printAllocInfo(const kir::Allocate* alloc);

  std::stringstream& indent() {
    for (const auto i : c10::irange(indent_level_)) {
      (void)i; // Suppress unused variable warning
      os_ << "  ";
    }
    return os_;
  }

  void startBlock() {
    indent_level_++;
  }

  void endBlock() {
    indent_level_--;
  }

 private:
  std::stringstream os_;
  IrPrinter ir_printer_;
  int indent_level_ = 0;

  std::vector<DebugEntryPtr> debug_info_;

  const AllocationInfoMap* allocation_info_map_ = nullptr;
};

//! Utility class for modeling the liveness interval.
//! The first write and last read
//! is based on the position on the linear order within
//! the Kernel IR.
//!  The interval is semi-open,
//!     i.e. [First_Write, Last_Read)
//!  So the buffer is NOT available at exactly First_Write
//!   position while it IS available at Last_Read.
class BufferLiveInterval {
 public:
  // Simple detection of intersection of two intervals
  bool intersect(BufferLiveInterval* other) {
    if (first_write_pos_ <= other->first_write_pos_) {
      return other->first_write_pos_ < last_read_pos_;
    } else {
      return first_write_pos_ < other->last_read_pos_;
    }
  }

  void markWrite(int pos) {
    if (first_write_pos_ == -1) {
      first_write_pos_ = pos;
    }
  }

  void markRead(int pos) {
    last_read_pos_ = pos;
    TORCH_INTERNAL_ASSERT(
        first_write_pos_ > 0,
        "lower_alias_memory: a read seen before any write");
    TORCH_INTERNAL_ASSERT(
        pos >= first_write_pos_,
        "lower_alias_memory: marking a read (",
        pos,
        ") before write (",
        first_write_pos_,
        ")");
    all_read_pos_.push_back(pos);
  }

  const auto& allReads() {
    return all_read_pos_;
  }

  auto firstWrite() const {
    return first_write_pos_;
  }

  auto lastRead() const {
    return last_read_pos_;
  }

  std::string toString() {
    std::stringstream ss;
    ss << "[ " << first_write_pos_ << " , " << last_read_pos_ << " )";
    return ss.str();
  }

 private:
  int first_write_pos_ = -1;
  int last_read_pos_ = -1;
  std::vector<int> all_read_pos_;
};

using BufferLiveIntervalPtrList = std::vector<BufferLiveInterval*>;

//! Thin struct to keep track of loops. The actual loop body is
//!  considered live in [start_pos, end_pos)
struct ScopeInfo {
  int start_pos = -1;
  int end_pos = -1;

  // nullptr means it's global scope
  kir::ForLoop* loop = nullptr;
};

//! Assign an integer position to each expression to help representing
//! scope ranges. The position starts from 1.
class ExprPosMap {
 public:
  //! Get the position of an expr
  int get(const Expr* expr) const {
    return expr_pos_map_.at(expr);
  }

  //! Get the current position
  int getCurrentPos() const {
    return current_pos_;
  }

  //! Advance the position counter
  void moveToNext() {
    ++current_pos_;
  }

  //! Record the current position as the position of an expr
  void setPosAtCurrent(const Expr* expr) {
    expr_pos_map_[expr] = current_pos_;
  }

 private:
  //! Position counter. The first expression is assigned position 1
  int current_pos_ = 0;

  //! Keep track of the positions of expressions
  std::unordered_map<const Expr*, int> expr_pos_map_;
};

// Create ScopeInfo for each loop
class ScopeMap : private kir::IrVisitor {
 public:
  ScopeMap(const std::vector<Expr*>& exprs) {
    global_scope_info_ = makeAndRegisterScopeInfo(nullptr);
    handle(exprs);
    global_scope_info_->end_pos = expr_pos_map_.getCurrentPos() + 1;

    // Make sure all loops have end_pos filled
    for (const auto& info : all_scope_info_) {
      TORCH_INTERNAL_ASSERT(info->end_pos != -1);
    }
  }

  using kir::IrVisitor::handle;

  void handle(Expr* expr) final {
    expr_pos_map_.moveToNext();
    expr_pos_map_.setPosAtCurrent(expr);
    kir::IrVisitor::handle(expr);
  }

  void handle(kir::ForLoop* for_loop) final {
    auto loop_info = makeAndRegisterScopeInfo(for_loop);
    kir::IrVisitor::handle(for_loop);
    loop_info->end_pos = expr_pos_map_.getCurrentPos() + 1;
  }

  void handle(kir::IfThenElse* ite) final {
    TORCH_INTERNAL_ASSERT(
        false, "lower_alias_memory: no support for IfThenElse at this phase.");
  }

  //! Factory function for internal loop information data
  ScopeInfo* makeAndRegisterScopeInfo(kir::ForLoop* loop) {
    auto loop_info_ptr = std::make_unique<ScopeInfo>();
    auto loop_info = loop_info_ptr.get();

    // When loop is null, it corresponds to the global scope
    loop_info->start_pos = loop == nullptr ? 0 : getExprPos(loop);
    loop_info->end_pos = -1; // This will be filled later
    loop_info->loop = loop;
    all_scope_info_.emplace_back(std::move(loop_info_ptr));

    if (loop != nullptr) {
      TORCH_INTERNAL_ASSERT(
          loop_to_scope_info_map_.emplace(loop, loop_info).second,
          "Duplicated scope info created for loop: ",
          loop->toString());
    }

    return loop_info;
  }

  ScopeInfo* getGlobalScopeInfo() const {
    return global_scope_info_;
  }

  std::vector<std::unique_ptr<ScopeInfo>>&& getAllScopeInfo() {
    return std::move(all_scope_info_);
  }

  ScopeInfo* getLoopScopeInfo(const kir::ForLoop* loop) const {
    auto it = loop_to_scope_info_map_.find(loop);
    TORCH_INTERNAL_ASSERT(
        it != loop_to_scope_info_map_.end(),
        "No scope info found for loop: ",
        loop->toString());
    return it->second;
  }

  int getExprPos(const Expr* expr) const {
    return expr_pos_map_.get(expr);
  }

 private:
  //! Owning list of collected scope info
  std::vector<std::unique_ptr<ScopeInfo>> all_scope_info_;

  //! Contains start and end position of the global scope
  ScopeInfo* global_scope_info_ = nullptr;

  //! map loop to scope info
  std::unordered_map<const kir::ForLoop*, ScopeInfo*> loop_to_scope_info_map_;

  ExprPosMap expr_pos_map_;
};

//! Utility class to record the read and write of each
//! allocated buffer.
//!
//! Note:
//!  this simplified interval analysis only works on pointwise ops and
//!  reductions and broadcast. With no non-trivial IfThenElse and no
//!  non-trivial re-computation.
//!
//!  Will probably at some point need dataflow and index analysis to precisely
//!  handle loop carried dependency.
struct AllocationInfo {
  kir::Allocate* alloc_expr = nullptr;
  const kir::Allocate* alias_to = nullptr;
  bool is_inner_alias = false;
  bool should_try_alias = true;
  MemoryType mem_type = MemoryType::Local;
  DataType data_type = DataType::Float;
  std::string size_expr;
  ScopeInfo* loop_info = nullptr;
  bool can_use_inner_alias = true;
  int alloc_pos = -1;
  std::unique_ptr<std::vector<AllocationInfo*>> inner_alias_list_ = nullptr;
  std::unique_ptr<BufferLiveInterval> inner_live_interval = nullptr;
  std::unique_ptr<BufferLiveIntervalPtrList> inner_subscribed_intevals =
      nullptr;
  std::unique_ptr<BufferLiveInterval> outer_live_interval = nullptr;
  std::unique_ptr<BufferLiveIntervalPtrList> outer_subscribed_intevals =
      nullptr;
};

//! Analysis pass to collect the liveness info of local and shared buffers:
//! The liveness info is illustrated as follows:
//!
//! For Idx0 ...
//!   Alloc(T1, register)
//!   Alloc(T2, register)
//!   Alloc(T3, register)
//!
//!   For Idx1 ...     <---------- Outer Live Interval of T1 begin
//!     For Idx2 ...
//!       T1 = ...            <--  Inner Live Interval of T1 begin
//!       T2 = ...
//!       T3 = T1 + ...    <-- Inner Live Interval of T1 end
//!       T5 = T3 + ...
//!     EndFor Idx2
//!   EndFor Idx1 <-------  Outer Live Interval of T1 end
//!
//!   Alloc(T4, register)
//!   For Idx3 ...
//!     T4 = ...
//!   EndFor Idx3
//! EndFor Idx0
//!
//!  Each buffer is associated with an `inner_live_interval` and an
//!  `outer_live_interval`,
//!   Inner interval marks the exprs that are the first write and last read of
//!   the buffer.
//!   Outer interval marks the beginning of the loop of first write and end of
//!   the loop of last read, both at the same loop level as the buffer
//!   allocation.
class AllocationInfoMap : private kir::IrVisitor {
 public:
  // Alias local memory if it exceeds this threshold
  static constexpr long kRegisterSizeThreshold = 1;

  AllocationInfoMap(const std::vector<Expr*>& exprs, bool debug_print)
      : scope_map_(exprs),
        debug_printer_(
            debug_print ? std::make_unique<BufferReuseDebugPrinter>()
                        : nullptr) {
    current_stack_.push_back(scope_map_.getGlobalScopeInfo());
    if (debug_printer_) {
      debug_printer_->pushScope();
    }
    handle(exprs);
    if (debug_printer_) {
      debug_printer_->popScope();
      std::cout << debug_printer_->dumpDebugInfo(this);
    }
    current_stack_.pop_back();
  }

  c10::optional<AllocationInfo*> getMaybeAllocationInfo(
      const kir::Allocate* alloc) const {
    auto it = allocation_info_map_.find(alloc);
    if (it == allocation_info_map_.end()) {
      return c10::nullopt;
    }
    return it->second;
  }

  const std::unordered_map<const kir::Allocate*, AllocationInfo*>&
  getAllocationInfoMap() const {
    return allocation_info_map_;
  }

  //! Mark the tensor of "from" be an alias of the tensor of "to"
  //! through inner alias analysis and keep track of the re-use.
  void useInnerAlias(AllocationInfo* from, AllocationInfo* to) {
    to->inner_alias_list_->push_back(from);
    to->inner_subscribed_intevals->push_back(from->inner_live_interval.get());
    setAlias(from, to);
    from->is_inner_alias = true;
  }

  //! Mark the tensor of "from" be an alias of the tensor of "to"
  //! through outer alias analysis and keep track of the re-use.
  void useOuterAlias(AllocationInfo* from, AllocationInfo* to) {
    to->outer_subscribed_intevals->push_back(from->outer_live_interval.get());
    setAlias(from, to);
  }

  //! To run before performing in-place sharing analysis.
  //!   Initializes the inner live intervals with each
  //!   allocation's inner live interval.
  void prepareInnerSharingAnalysis() {
    for (auto it : getAllocationInfoMap()) {
      auto alloc_info = it.second;
      // At beginning only use interval for each
      //  allocate is their corresponding live interval
      alloc_info->inner_subscribed_intevals->push_back(
          alloc_info->inner_live_interval.get());
    }
  }

  //! To run before performing outer interval based sharing analysis.
  //!   Initializes the outer live intervals with the outer live interval
  //!   of each allocation and copy inner sharing information.
  void prepareOuterSharingAnalysis() {
    for (auto it : getAllocationInfoMap()) {
      auto alloc_info = it.second;
      if (!alias_map_.count(alloc_info)) {
        alloc_info->outer_subscribed_intevals->push_back(
            alloc_info->outer_live_interval.get());
        // Update only if this buffer isn't an alias
        for (auto inner_alias : *(alloc_info->inner_alias_list_)) {
          alloc_info->outer_subscribed_intevals->push_back(
              inner_alias->outer_live_interval.get());
        }
      }
    }
  }

  const std::unordered_map<AllocationInfo*, AllocationInfo*>& getAliasMap()
      const {
    return alias_map_;
  }

 private:
  using kir::IrVisitor::handle;

  void handle(Expr* expr) final {
    if (debug_printer_) {
      debug_printer_->pushBack(scope_map_.getExprPos(expr), expr);
    }
    kir::IrVisitor::handle(expr);
    if (ir_utils::isTvOp(expr)) {
      collectLivenessInfoOfExpr(expr);
    }
  }

  void handle(kir::ForLoop* for_loop) final {
    auto loop_info = scope_map_.getLoopScopeInfo(for_loop);
    current_stack_.push_back(loop_info);
    if (debug_printer_) {
      debug_printer_->pushScope();
    }
    kir::IrVisitor::handle(for_loop);
    if (debug_printer_) {
      debug_printer_->popScope();
    }
    current_stack_.pop_back();
  }

  void handle(kir::IfThenElse* ite) final {
    TORCH_INTERNAL_ASSERT(
        false, "lower_alias_memory: no support for IfThenElse at this phase.");
  }

  // Generate allocation info for allocation after some pre-filtering
  //  conditions.
  void handle(kir::Allocate* alloc) final {
    if (alloc->alias()) {
      // We shouldn't really see a case like this in general, but
      //  some Fusion outputs could have been aliased to inputs.
      // It should be safe to ignore these in the use-def analysis.
      return;
    }

    auto tv = dynamic_cast<TensorView*>(alloc->buffer());
    if (!tv) {
      return;
    }

    // Collect the allocate info data

    // Collect memory type, skip global buffers
    auto mem_type = tv->getMemoryType();
    if (mem_type != MemoryType::Local && mem_type != MemoryType::Shared) {
      return;
    }

    // Skip smaller register sizes
    bool should_try_alias = true;
    if (mem_type == MemoryType::Local) {
      if (!alloc->size()->isConstInt()) {
        TORCH_WARN_ONCE(
            "Lower_alias_memory : dynamic sized register allocation");
        return;
      }
      if (alloc->size()->evaluateInt() <= kRegisterSizeThreshold) {
        should_try_alias = false;
      }
    }

    auto data_type = tv->dtype();
    auto size_print = SymbolicSizePrinter::printSize(alloc);

    // Make sure we don't have conflicting information on record
    TORCH_INTERNAL_ASSERT(!allocation_info_map_.count(alloc));
    TORCH_INTERNAL_ASSERT(!tv_to_allocation_map_.count(tv->name()));

    // make AllocationUseDefInfo:
    auto alloc_info = makeAllocationInfo();
    alloc_info->alloc_pos = scope_map_.getExprPos(alloc);
    alloc_info->alloc_expr = alloc;
    alloc_info->mem_type = mem_type;
    alloc_info->data_type = data_type;
    alloc_info->size_expr = size_print;
    alloc_info->loop_info = current_stack_.back();
    alloc_info->should_try_alias = should_try_alias;

    // record short cuts
    allocation_info_map_[alloc] = alloc_info;
    tv_to_allocation_map_[tv->name()] = alloc_info;
  }

  //! Factory function for internal use-def information data
  AllocationInfo* makeAllocationInfo() {
    auto alloc_info_ptr = std::make_unique<AllocationInfo>();
    auto alloc_info = alloc_info_ptr.get();

    alloc_info->inner_alias_list_ =
        std::make_unique<std::vector<AllocationInfo*>>();
    alloc_info->inner_live_interval = std::make_unique<BufferLiveInterval>();
    alloc_info->inner_subscribed_intevals =
        std::make_unique<BufferLiveIntervalPtrList>();
    alloc_info->outer_live_interval = std::make_unique<BufferLiveInterval>();
    alloc_info->outer_subscribed_intevals =
        std::make_unique<BufferLiveIntervalPtrList>();
    all_allocations_.emplace_back(std::move(alloc_info_ptr));
    return alloc_info;
  }

  // Iterate over the inputs and outputs of exprs and update
  //  the liveness info of local buffers if applicaable.
  void collectLivenessInfoOfExpr(Expr* expr) {
    if (!ir_utils::isTvOp(expr)) {
      return;
    }

    const auto expr_pos = scope_map_.getExprPos(expr);

    // Collect all tv's that resolves broadcast in this
    //  expr. The current analysis isn't enough to capture
    //  their liveness range.
    for (auto input_tv : ir_utils::filterByType<TensorView>(expr->inputs())) {
      auto maybe_alloc_info = getMaybeAllocInfoFromTV(input_tv);
      if (maybe_alloc_info.has_value()) {
        if (!isSerialBroadcastResolution(input_tv, for_loops_)) {
          maybe_alloc_info.value()->inner_live_interval->markRead(expr_pos);
        } else {
          // Disable inner alias info for this buffer, since line number based
          //  analysis is no longer precise enough for inplace sharing
          //  if a serial broadcast is realized.
          maybe_alloc_info.value()->can_use_inner_alias = false;
        }

        auto outer_loop_info =
            ascendLoopNestToSameLevelAs(maybe_alloc_info.value());

        if (outer_loop_info) {
          maybe_alloc_info.value()->outer_live_interval->markRead(
              outer_loop_info->end_pos);
        } else {
          // Allocate is inlined in the innermost loop,
          //  so outer live interval is the same as inner.
          maybe_alloc_info.value()->outer_live_interval->markRead(expr_pos);
        }
      }
    }
    for (auto output_tv : ir_utils::filterByType<TensorView>(expr->outputs())) {
      auto maybe_alloc_info = getMaybeAllocInfoFromTV(output_tv);
      if (maybe_alloc_info.has_value()) {
        // Reductions use outputs as read-write parameters, so their
        // outputs need to be marked as read as well
        const bool is_read_write = ir_utils::isReductionOp(expr);
        maybe_alloc_info.value()->inner_live_interval->markWrite(expr_pos);
        if (is_read_write) {
          maybe_alloc_info.value()->inner_live_interval->markRead(expr_pos);
        }
        auto outer_loop_info =
            ascendLoopNestToSameLevelAs(maybe_alloc_info.value());
        auto write_pos =
            outer_loop_info ? outer_loop_info->start_pos : expr_pos;
        maybe_alloc_info.value()->outer_live_interval->markWrite(write_pos);
        if (is_read_write) {
          auto read_pos = outer_loop_info ? outer_loop_info->end_pos : expr_pos;
          maybe_alloc_info.value()->outer_live_interval->markRead(read_pos);
        }
      }
    }
  }

  c10::optional<AllocationInfo*> getMaybeAllocInfoFromTV(TensorView* tv) const {
    auto alloc_it = tv_to_allocation_map_.find(tv->name());
    if (alloc_it == tv_to_allocation_map_.end()) {
      return c10::nullopt;
    }
    return alloc_it->second;
  }

  //! Find the loop level of expr that apears in the same scope as
  //!  the reference allocate. Eg.
  //!
  //!  For ...
  //!    For ...
  //!      Allocate    <---- reference arg
  //!      For ..
  //!          For ...
  //!      For ... <---- this function returns `ScopeInfo` for this loop
  //!          For ...
  //!             expr  <---- current expr (implied in current_stack_ and
  //!             current_pos_ )
  //! Assumes that expr either writes to or reads from the reference allocate.
  ScopeInfo* ascendLoopNestToSameLevelAs(AllocationInfo* reference) {
    auto allocate_loop_info = reference->loop_info;
    if (allocate_loop_info->loop == nullptr) {
      if (current_stack_.size() > 1) {
        return current_stack_[1];
      }
      return nullptr;
    }

    for (const auto idx : c10::irange(current_stack_.size() - 1)) {
      if (current_stack_[idx] == allocate_loop_info) {
        return current_stack_[idx + 1];
      }
    }

    TORCH_INTERNAL_ASSERT(
        current_stack_.back() == allocate_loop_info,
        "lower_alias_memory : expr outer loop inconsistent with allocate");

    // Returning a nullptr means the allocate is in the current stack frame.
    return nullptr;
  }

  //! Mark the tensor of "from" be an alias of the tensor of "to".
  void setAlias(AllocationInfo* from, AllocationInfo* to) {
    alias_map_[from] = to;
    from->alias_to = to->alloc_expr;
  }

 private:
  friend BufferReuseDebugPrinter;

  const ScopeMap scope_map_;

  //! Map TensorView name to Allocate node.
  //!  Note: this assumes that each tensor view is only allocated once.
  std::unordered_map<StmtNameType, AllocationInfo*> tv_to_allocation_map_;

  //! Allocation sites that will participate in this analysis
  std::unordered_map<const kir::Allocate*, AllocationInfo*>
      allocation_info_map_;

  //! Owning list of collected allocation info
  std::vector<std::unique_ptr<AllocationInfo>> all_allocations_;

  //! Keep track of stack
  std::vector<ScopeInfo*> current_stack_;

  //! Keeps track of all the allocations that have been set to alias
  std::unordered_map<AllocationInfo*, AllocationInfo*> alias_map_;

  //! Debug info:
  std::unique_ptr<BufferReuseDebugPrinter> debug_printer_ = nullptr;
};

void BufferReuseDebugPrinter::printAllocInfo(const kir::Allocate* alloc) {
  TORCH_INTERNAL_ASSERT(allocation_info_map_ != nullptr);
  std::string message_header(" \033[1;32m^^^^^ ---Buffer Reuse Info---  ");
  std::string message_end("  \033[0m\n");
  if (!allocation_info_map_->getMaybeAllocationInfo(alloc).has_value()) {
    // This buffer is not considered for any sharing, either
    //  because of un-supported op or size below threshold.
    return;
  }

  auto alloc_info = allocation_info_map_->getMaybeAllocationInfo(alloc).value();

  indent() << message_header;
  if (alloc_info->alias_to) {
    if (alloc_info->is_inner_alias) {
      os_ << "(inner) ";
    } else {
      os_ << "(outer) ";
    }
    os_ << " alias to alloc at pos "
        << allocation_info_map_->getMaybeAllocationInfo(alloc_info->alias_to)
               .value()
               ->alloc_pos
        << " ";
  } else {
    os_ << " not aliased ";
  }

  os_ << " , ";

  if (alloc_info->can_use_inner_alias) {
    os_ << "inner live interval: ";
    os_ << alloc_info->inner_live_interval->toString() << " , ";
  } else {
    os_ << "cannot use inner alias, ";
  }
  os_ << "size expr : " << alloc_info->size_expr << " , "
      << "outer live interval: " << alloc_info->outer_live_interval->toString();
  indent() << message_end;
}

//! Reuse Allocation nodes via pointer aliasing
class ReusableAllocationFinder : private kir::IrVisitor {
 public:
  static void find(
      const std::vector<Expr*>& exprs,
      AllocationInfoMap& allocation_info_map) {
    // Perform in-place sharing first and then outer liveness
    //  based sharing. Since outer liveness info can still
    //  be used with some buffers already aliasing through
    //  in-place re-use but wouldn't be the case if we did
    //  outer liveness based sharing first.
    ReusableAllocationFinder finder_inner_alias(
        exprs, allocation_info_map, true);
    ReusableAllocationFinder finder_outer_alias(
        exprs, allocation_info_map, false);
    return;
  }

 private:
  ReusableAllocationFinder(
      const std::vector<Expr*>& exprs,
      AllocationInfoMap& allocation_info_map,
      bool inner_aliasing_pass)
      : allocation_info_map_(allocation_info_map),
        inner_aliasing_pass_(inner_aliasing_pass) {
    if (inner_aliasing_pass_) {
      allocation_info_map_.prepareInnerSharingAnalysis();
    } else {
      allocation_info_map_.prepareOuterSharingAnalysis();
    }

    current_visible_buffer_stack_.emplace_back(
        std::make_unique<std::vector<AllocationInfo*>>());

    handle(exprs);

    current_visible_buffer_stack_.pop_back();
  }

  using kir::IrVisitor::handle;

  void handle(kir::Allocate* allocate) final {
    // Check that if this allocation site is one that
    //  we want to re-use or replace with an alias

    auto maybe_alloc_info =
        allocation_info_map_.getMaybeAllocationInfo(allocate);
    if (maybe_alloc_info.has_value() &&
        maybe_alloc_info.value()->alias_to == nullptr) {
      // Try to re-use existing allocates
      if (!tryReuseOtherAllocate(maybe_alloc_info.value())) {
        // If didn't re-use, should register this
        // allocate so that future allocates
        // can re-use this one.
        current_visible_buffer_stack_.back()->push_back(
            maybe_alloc_info.value());
      }
    }
  }

  bool tryReuseOtherAllocate(AllocationInfo* alloc_info) {
    if (!alloc_info->should_try_alias) {
      return false;
    }
    if (!alloc_info->inner_alias_list_->empty()) {
      // Avoid 2-hop aliasing for simplicity. Can support if really need  in
      // extreme cases.
      return false;
    }

    // Move backwards on list of re-usable allocates on the stack, prefer
    //  reusing nearest allocation
    for (auto reuse_stack_it = current_visible_buffer_stack_.rbegin();
         reuse_stack_it != current_visible_buffer_stack_.rend();
         reuse_stack_it++) {
      for (auto alloc_to_reuse_it = (*reuse_stack_it)->rbegin();
           alloc_to_reuse_it != (*reuse_stack_it)->rend();
           alloc_to_reuse_it++) {
        auto alloc_to_reuse = *alloc_to_reuse_it;

        // Check if this re-use candidate is an alias
        if (alloc_to_reuse->alias_to != nullptr) {
          continue;
        }

        // Check if this alloc has the same mem type
        if (alloc_info->mem_type != alloc_to_reuse->mem_type) {
          continue;
        }

        // Check if this alloc has the same size
        if (alloc_info->size_expr != alloc_to_reuse->size_expr) {
          continue;
        }

        // Check if this alloc has the same data type
        if (alloc_info->data_type != alloc_to_reuse->data_type) {
          continue;
        }

        // Check if live intervals have any overlap
        auto subscribed_intervals = inner_aliasing_pass_
            ? alloc_to_reuse->inner_subscribed_intevals.get()
            : alloc_to_reuse->outer_subscribed_intevals.get();

        auto alloc_live_interval = inner_aliasing_pass_
            ? alloc_info->inner_live_interval.get()
            : alloc_info->outer_live_interval.get();

        if (std::any_of(
                subscribed_intervals->begin(),
                subscribed_intervals->end(),
                [alloc_live_interval](auto subscribed_interval) {
                  return alloc_live_interval->intersect(subscribed_interval);
                })) {
          continue;
        }

        // Special checks for inner sharing pass
        if (inner_aliasing_pass_ &&
            !isValidInnerSharing(alloc_to_reuse, alloc_info)) {
          continue;
        }

        if (alloc_info->alloc_expr->buffer()->isA<TensorView>()) {
          if (!alloc_to_reuse->alloc_expr->buffer()->isA<TensorView>()) {
            continue;
          }
          auto this_tv = alloc_info->alloc_expr->buffer()->as<TensorView>();
          auto reuse_tv =
              alloc_to_reuse->alloc_expr->buffer()->as<TensorView>();
          // Check that either both tv's are vectorized acceses, or neither are.
          // Vectorized allocations require correct alignment so they can only
          // alias with other allocations with the right alignment
          const auto& va = GpuLower::current()->vectorizedAccesses();
          if ((va.find(this_tv) == va.end()) !=
              (va.find(reuse_tv) == va.end())) {
            return false;
          }

          // Shared memory is all aligned to 128 bits, local memory might not be
          if (this_tv->getMemoryType() == MemoryType::Local &&
              va.find(this_tv) != va.end()) {
            // Make sure alignment matches
            if (va.at(this_tv) != va.at(reuse_tv)) {
              return false;
            }
          }
        }

        // TODO:
        //  Outer interval based sharing supports arbitrary re-indexing into
        //    the same buffer and would require additional syncs if fully
        //    enabled.
        //  Need a few more checks to insert syncs if necessary before turning
        //    on this sharing.
        if (!inner_aliasing_pass_ &&
            alloc_info->mem_type == MemoryType::Shared) {
          continue;
        }

        // Now re-use the alloc here and be sure to update
        reuseAllocation(alloc_info, alloc_to_reuse);
        return true;
      }
    }
    return false;
  }

  void handle(kir::ForLoop* for_loop) final {
    current_visible_buffer_stack_.emplace_back(
        std::make_unique<std::vector<AllocationInfo*>>());
    kir::IrVisitor::handle(for_loop);
    current_visible_buffer_stack_.pop_back();
  }

  struct InPlaceSharingInfo {
    bool has_broadcast_between = false;
    bool has_unsupported_op = false;
  };

  //! Careful heavy check on inner sharing candidates,
  //!  current enforced conditions are:
  //!
  //! 1. The two buffers have producer-consumer relationship
  //! 2. No halo in the allocated iter domains
  //! 3. Require index equivalence when sharing across broadcast
  bool isValidInnerSharing(
      AllocationInfo* alloc_info,
      AllocationInfo* to_reuse) {
    // Disable if either of the buffers do not support inner sharing
    if (!alloc_info->can_use_inner_alias || !to_reuse->can_use_inner_alias) {
      return false;
    }
    // Assume inputs are TV allocations, which should have been checked
    //  before reaching this point.
    auto this_tv = alloc_info->alloc_expr->buffer()->as<TensorView>();
    auto reuse_tv = to_reuse->alloc_expr->buffer()->as<TensorView>();

    // Aggressively disable inner sharing for swizzled tvs since
    //  the indexing order is in general not tractable.
    // But outer sharing should still apply.
    if (this_tv->hasSwizzleOp() || reuse_tv->hasSwizzleOp()) {
      return false;
    }

    // Check the values in between the two buffers.
    auto vals_between_this_and_reuse =
        DependencyCheck::getAllValsBetween({this_tv}, {reuse_tv});
    if (vals_between_this_and_reuse.empty()) {
      vals_between_this_and_reuse =
          DependencyCheck::getAllValsBetween({reuse_tv}, {this_tv});
    }

    if (!vals_between_this_and_reuse.empty()) {
      // Temporarily disable sharing across difficult
      //  ops for inner sharing and can be relaxed gradually.
      auto topo_info = checkOpsInBetween(vals_between_this_and_reuse);

      // Avoid difficult and future introduced ops
      if (topo_info.has_unsupported_op) {
        return false;
      }

      // Get information on the allocated domains of the
      //  two buffers
      const auto& local_alloc_map =
          GpuLower::current()->localAllocationInfoMap();
      auto alloc_it = local_alloc_map.find(alloc_info->alloc_expr);
      auto to_reuse_it = local_alloc_map.find(to_reuse->alloc_expr);
      if (alloc_it == local_alloc_map.end() ||
          to_reuse_it == local_alloc_map.end()) {
        return false;
      }

      // Disable in-place reusing for halo ops, since halo
      //  can issue pointwise op multiple points at some points.
      if (alloc_it->second->has_halo || to_reuse_it->second->has_halo) {
        return false;
      }

      // Require matched iterdomains for sharing across broadcast
      if (topo_info.has_broadcast_between) {
        auto& alloc_domains = alloc_it->second->alloc_domains;
        auto& reuse_domains = to_reuse_it->second->alloc_domains;

        return allocationDomainsIndexMapped(alloc_domains, reuse_domains);
      }

      // If only pointwise and reduction ops in between and no broadcast
      //  should be ok to re-use in place.
      return true;
    }

    // this and reuse are not dependencies of each other,
    //  which means we cannot use inner sharing.
    return false;
  }

  InPlaceSharingInfo checkOpsInBetween(std::vector<Val*>& all_used_vals) {
    InPlaceSharingInfo info;
    std::unordered_set<Val*> all_used_val_set(
        all_used_vals.begin(), all_used_vals.end());

    for (auto val : all_used_vals) {
      if (auto tv = dynamic_cast<TensorView*>(val)) {
        auto tv_def = tv->definition();
        if (!tv_def) {
          continue;
        }
        if (!isPointwiseTvOp(tv_def) && !ir_utils::isReductionTvOp(tv_def)) {
          if (isBroadcastTvOp(tv_def)) {
            info.has_broadcast_between = true;
          } else {
            info.has_unsupported_op = true;
          }
        }
      }
    }
    return info;
  }

  bool allocationDomainsIndexMapped(
      std::vector<IterDomain*>& alloc_domains,
      std::vector<IterDomain*>& reuse_domains) {
    // Require that the allocated domains are exactly mapped.
    if (alloc_domains.size() != reuse_domains.size()) {
      return false;
    }

    // Check index map for the corresponding axes.
    for (const auto id_it : c10::irange(alloc_domains.size())) {
      if (!GpuLower::current()->caMap()->areMapped(
              alloc_domains[id_it],
              reuse_domains[id_it],
              IdMappingMode::EXACT)) {
        return false;
      }
    }
    return true;
  }

  void reuseAllocation(AllocationInfo* alloc_info, AllocationInfo* to_reuse) {
    // Update analysis result
    if (inner_aliasing_pass_) {
      allocation_info_map_.useInnerAlias(alloc_info, to_reuse);
    } else {
      allocation_info_map_.useOuterAlias(alloc_info, to_reuse);
    }
  }

  // Do we have a true pointwise op?
  // (ie. a TV op, excluding direct assignments and reductions)
  bool isPointwiseTvOp(const Expr* expr) {
    if (ir_utils::isTvOp(expr)) {
      return expr->isA<UnaryOp>() || expr->isA<BinaryOp>() ||
          expr->isA<TernaryOp>();
    }
    return false;
  }

  // Utility to capture broadcast ops
  bool isBroadcastTvOp(const Expr* expr) {
    if (!ir_utils::isTvOp(expr)) {
      return false;
    }
    return expr->isA<BroadcastOp>();
  }

 private:
  // Analysis result from the first pass collecting the use-defs
  AllocationInfoMap& allocation_info_map_;

  // Internal data keeping track of currently visible allocations as
  //  the pass iterate through the expr list, grouped by the stack
  //  layer of alloc ops.
  std::vector<std::unique_ptr<std::vector<AllocationInfo*>>>
      current_visible_buffer_stack_;

  // Marks state of current pass
  bool inner_aliasing_pass_ = true;
};

// Replace Allocate exprs as determined by the alias analysis
class AllocationReuseModifier : private kir::ExprMutator {
 public:
  static std::vector<Expr*> modify(
      const std::vector<Expr*>& exprs,
      const AllocationInfoMap& allocation_info_map) {
    AllocationReuseModifier modifier(exprs, allocation_info_map);
    return modifier.exprs_;
  }

 private:
  AllocationReuseModifier(
      const std::vector<Expr*>& exprs,
      const AllocationInfoMap& allocation_info_map)
      : allocation_info_map_(allocation_info_map) {
    traverseAndInsert(exprs);
  }

  using kir::ExprMutator::handle;

  //! Replace an kir::Allocate with a new aliased Allocate
  void handle(kir::Allocate* allocate) final {
    auto maybe_alloc_info =
        allocation_info_map_.getMaybeAllocationInfo(allocate);
    if (!maybe_alloc_info.has_value()) {
      return;
    }

    AllocationInfo* alloc_info_from = maybe_alloc_info.value();

    auto alias_it = allocation_info_map_.getAliasMap().find(alloc_info_from);
    if (alias_it == allocation_info_map_.getAliasMap().end()) {
      return;
    }

    kir::Allocate* alloc_expr_to = alias_it->second->alloc_expr;

    // Currently, we don't allow 2-hop alias, ie., aliasing of an
    // aliased tensor, so alloc_expr_to should be still the allocation
    // expression of the aliased allocation. This assertion should be
    // removed if 2-hop aliasing is enabled.
    TORCH_INTERNAL_ASSERT(
        alloc_expr_to == getMaybeNewAllocate(alloc_expr_to),
        "Invalid updated allocation found. Original: ",
        alloc_expr_to->toString(),
        ". Updated: ",
        getMaybeNewAllocate(alloc_expr_to)->toString());

    kir::Allocate* old_alloc = alloc_info_from->alloc_expr;
    kir::Allocate* new_alloc = IrBuilder::create<kir::Allocate>(
        old_alloc->buffer(),
        old_alloc->memoryType(),
        old_alloc->shape(),
        old_alloc->zeroInit(),
        alloc_expr_to);

    registerReplace(old_alloc, new_alloc);

    TORCH_INTERNAL_ASSERT(old2new_.emplace(old_alloc, new_alloc).second);

    // TODO: Consider more robust way to keep the information map up-to-date
    GpuLower::current()->propagateExprInfo(old_alloc, new_alloc);
  }

  kir::Allocate* getMaybeNewAllocate(kir::Allocate* allocate) const {
    auto it = old2new_.find(allocate);
    if (it == old2new_.end()) {
      return allocate;
    } else {
      return it->second;
    }
  }

 private:
  const AllocationInfoMap& allocation_info_map_;

  //! Keep track of new Allocate exprs
  std::unordered_map<kir::Allocate*, kir::Allocate*> old2new_;
};

} // namespace

std::vector<Expr*> reuseMemoryAllocations(const std::vector<Expr*>& exprs) {
  FUSER_PERF_SCOPE("reuseMemoryAllocations");

  bool debug_print = isDebugDumpEnabled(DebugDumpOption::BufferReuseInfo);

  AllocationInfoMap allocation_info_map(exprs, debug_print);

  ReusableAllocationFinder::find(exprs, allocation_info_map);

  return AllocationReuseModifier::modify(exprs, allocation_info_map);
}

} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
