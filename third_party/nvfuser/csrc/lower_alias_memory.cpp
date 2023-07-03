#include <lower_alias_memory.h>

#include <instrumentation.h>
#include <ir_iostream.h>
#include <ir_utils.h>
#include <kernel_expr_evaluator.h>
#include <kernel_ir.h>
#include <lower2device.h>
#include <lower_utils.h>

#include <sstream>
#include <unordered_map>
#include <unordered_set>

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
//!  broadcast so that each index of producer buffer can be visited
//!  multiple times, in which case the aggressive is not valid.
bool isSerialBroadcastResolution(TensorView* producer, TensorView* consumer) {
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

  // Serial concrete loop id's that cover consumer's iter domain.
  std::vector<Val*> consumer_serial_loop_concrete_ids;

  for (auto consumer_leaf_id : consumer->domain()->domain()) {
    auto concrete_loop_id = GpuLower::current()->caMap()->getConcreteMappedID(
        consumer_leaf_id, IdMappingMode::LOOP);

    // Check for any serial loop id with non-trivial extent
    if (!concrete_loop_id->isThread() &&
        !concrete_loop_id->extent()->isOneInt()) {
      consumer_serial_loop_concrete_ids.push_back(concrete_loop_id);
    }
  }

  // Collect the root id's that the serial loop iterdomain
  //  are transformed from.
  auto serial_loop_roots = InputsOf::outputs(
      FusionGuard::getCurFusion(), consumer_serial_loop_concrete_ids);

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

class BufferUseDefInfo;
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

  std::string dumpDebugInfo() {
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

 private:
  friend class BufferUseDefInfo;

  void pushBack(int lineno, Expr* expr) {
    makeExprEntry(lineno, expr);
  }

  void pushScope() {
    makeScopeEntry(DebugLineType::START_BLOCK);
  }

  void popScope() {
    makeScopeEntry(DebugLineType::END_BLOCK);
  }

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
  BufferUseDefInfo* buffer_info_ = nullptr;
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
        "lower_alias_memory: a read seen before any write")
    TORCH_INTERNAL_ASSERT(
        pos > first_write_pos_,
        "lower_alias_memory: marking a read before write");
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

using ScopeInfoOwningPtr = std::unique_ptr<ScopeInfo>;
using ScopeInfoOwningPtrList = std::vector<ScopeInfoOwningPtr>;

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
struct AllocationUseDefInfo {
  kir::Allocate* alloc_expr = nullptr;
  kir::Allocate* alias_to = nullptr;
  bool is_inner_alias = false;
  bool should_try_alias = true;
  MemoryType mem_type = MemoryType::Local;
  DataType data_type = DataType::Float;
  std::string size_expr;
  ScopeInfo* loop_info = nullptr;
  bool can_use_inner_alias = true;
  int alloc_pos = -1;
  std::unique_ptr<std::vector<AllocationUseDefInfo*>> inner_alias_list_ =
      nullptr;
  std::unique_ptr<BufferLiveInterval> inner_live_interval = nullptr;
  std::unique_ptr<BufferLiveIntervalPtrList> inner_subscribed_intevals =
      nullptr;
  std::unique_ptr<BufferLiveInterval> outer_live_interval = nullptr;
  std::unique_ptr<BufferLiveIntervalPtrList> outer_subscribed_intevals =
      nullptr;
};

using AllocationInfoOwningPtr = std::unique_ptr<AllocationUseDefInfo>;
using AllocationInfoOwningList = std::vector<AllocationInfoOwningPtr>;
using AllocationInfoPtr = AllocationUseDefInfo*;
using AllocationInfoList = std::vector<AllocationInfoPtr>;

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
//!   Outer interval marks the begining of the loop of first write and end of
//!   the loop of last read, both at the same loop level as the buffer
//!   allocation.
class BufferUseDefInfo {
 public:
  // Alias local memory if it exceeds this threshold
  static constexpr long kRegisterSizeThreshold = 1;

  BufferUseDefInfo(
      const std::vector<Expr*>& exprs,
      BufferReuseDebugPrinter* debug_printer = nullptr)
      : debug_printer_(debug_printer) {
    if (debug_printer) {
      debug_printer->buffer_info_ = this;
    }
    collectScopeInfo(exprs);
    collectScopeUseDefInfo(exprs);
  }

  //! Returns live interval info of buffer if previously
  //!  computed.
  c10::optional<AllocationInfoPtr> getMaybeReuseInfoFor(
      kir::Allocate* allocate) const {
    auto alloc_it = map_allocate_to_info_.find(allocate);
    if (alloc_it == map_allocate_to_info_.end()) {
      return c10::nullopt;
    }
    auto alloc = alloc_it->second;
    return alloc;
  }

  //! Realize alias of two buffers through inner alias analysis and
  //!  keep track of the re-use.
  void useInnerAlias(AllocationInfoPtr from, AllocationInfoPtr to) {
    to->inner_alias_list_->push_back(from);
    to->inner_subscribed_intevals->push_back(from->inner_live_interval.get());
    setAlias(from, to);
    from->is_inner_alias = true;
  }

  //! Realize alias of two buffers through outer alias analysis and
  //!  keep track of the re-use.
  void useOuterAlias(AllocationInfoPtr from, AllocationInfoPtr to) {
    to->outer_subscribed_intevals->push_back(from->outer_live_interval.get());
    setAlias(from, to);
  }

  //! To run before performing in-place sharing analysis.
  //!   Initializes the inner live intervals with each
  //!   allocation's inner live interval.
  void prepareInnerSharingAnalysis() {
    for (auto it : map_allocate_to_info_) {
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
    for (auto it : map_allocate_to_info_) {
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

 private:
  void handle(Expr* expr) {
    current_pos_++;
    if (debug_printer_) {
      debug_printer_->pushBack(current_pos_, expr);
    }
    if (auto alloc = dynamic_cast<kir::Allocate*>(expr)) {
      handle(alloc);
    } else if (auto for_loop = dynamic_cast<kir::ForLoop*>(expr)) {
      handle(for_loop);
    } else if (auto ite = dynamic_cast<kir::IfThenElse*>(expr)) {
      handle(ite);
    } else {
      collectLivenessInfo(expr);
    }
  }

  void handleScope(const std::vector<Expr*>& exprs) {
    if (debug_printer_) {
      debug_printer_->pushScope();
    }
    for (auto expr : exprs) {
      handle(expr);
    }
    if (debug_printer_) {
      debug_printer_->popScope();
    }
  }

  void handle(kir::ForLoop* for_loop) {
    auto loop_info = map_loop_pos_to_loop_info_.at(current_pos_);
    current_stack_.push_back(loop_info);
    handleScope(for_loop->body().exprs());
    current_stack_.pop_back();
  }

  void handle(kir::IfThenElse* ite) {
    TORCH_INTERNAL_ASSERT(
        false, "lower_alias_memory: no support for IfThenElse at this phase.");
  }

  // Generate allocation info for allocation after some pre-filtering
  //  conditions.
  void handle(kir::Allocate* alloc) {
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
      const auto register_size = expr_evaluator_.evaluate(alloc->size());
      if (!register_size.has_value()) {
        TORCH_WARN_ONCE(
            "Lower_alias_memory : dynamic sized register allocation");
        return;
      }
      if (register_size->as<int64_t>() <= kRegisterSizeThreshold) {
        should_try_alias = false;
      }
    }

    auto data_type = tv->dtype();
    auto size_print = SymbolicSizePrinter::printSize(alloc);

    // Make sure we don't have conflicting information on record
    TORCH_INTERNAL_ASSERT(!map_allocate_to_info_.count(alloc));
    TORCH_INTERNAL_ASSERT(!map_tv_to_allocations_.count(tv->name()));

    // make AllocationUseDefInfo:
    auto alloc_info = makeUseDefInfo();
    alloc_info->alloc_expr = alloc;
    alloc_info->mem_type = mem_type;
    alloc_info->data_type = data_type;
    alloc_info->size_expr = size_print;
    alloc_info->loop_info = current_stack_.back();
    alloc_info->should_try_alias = should_try_alias;

    // record short cuts
    map_allocate_to_info_[alloc] = alloc_info;
    map_tv_to_allocations_[tv->name()] = alloc_info;
  }

  void collectScopeUseDefInfo(const std::vector<Expr*>& exprs) {
    // Reset position pointer
    resetExprCounter();
    TORCH_INTERNAL_ASSERT(global_scope_info_ != nullptr);
    current_stack_.push_back(global_scope_info_);
    handleScope(exprs);
  }

  void collectScopeInfo(const std::vector<Expr*>& exprs) {
    // Reset position pointer
    resetExprCounter();
    collectScopeInfoWithinLoop(exprs, nullptr);
  }

  void collectScopeInfoWithinLoop(
      const std::vector<Expr*>& exprs,
      kir::ForLoop* current_loop) {
    auto loop_info = makeScopeInfo(current_loop);
    for (auto expr : exprs) {
      current_pos_++;
      if (auto for_loop = dynamic_cast<kir::ForLoop*>(expr)) {
        collectScopeInfoWithinLoop(for_loop->body().exprs(), for_loop);
      }
    }
    loop_info->end_pos = current_pos_ + 1;
  }

  void resetExprCounter() {
    current_pos_ = -1;
  }

  // Iterate over the inputs and outputs of exprs and update
  //  the liveness info of local buffers if applicaable.
  void collectLivenessInfo(const Expr* expr) {
    if (!ir_utils::isTvOp(expr)) {
      return;
    }

    auto out_tv = expr->outputs()[0]->as<TensorView>();

    // Collect all tv's that resolves broadcast in this
    //  expr. The current analysis isn't enough to capture
    //  their liveness range.
    for (auto input_tv : ir_utils::filterByType<TensorView>(expr->inputs())) {
      auto maybe_alloc_info = getMaybeAllocInfoFromTV(input_tv);
      if (maybe_alloc_info.has_value()) {
        if (!isSerialBroadcastResolution(input_tv, out_tv)) {
          maybe_alloc_info.value()->inner_live_interval->markRead(current_pos_);
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
          maybe_alloc_info.value()->outer_live_interval->markRead(current_pos_);
        }
      }
    }
    for (auto output_tv : ir_utils::filterByType<TensorView>(expr->outputs())) {
      auto maybe_alloc_info = getMaybeAllocInfoFromTV(output_tv);
      if (maybe_alloc_info.has_value()) {
        maybe_alloc_info.value()->inner_live_interval->markWrite(current_pos_);
        auto outer_loop_info =
            ascendLoopNestToSameLevelAs(maybe_alloc_info.value());
        if (outer_loop_info) {
          maybe_alloc_info.value()->outer_live_interval->markWrite(
              outer_loop_info->start_pos);
        } else {
          maybe_alloc_info.value()->outer_live_interval->markWrite(
              current_pos_);
        }
      }
    }
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
  ScopeInfo* ascendLoopNestToSameLevelAs(AllocationUseDefInfo* reference) {
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

  c10::optional<AllocationInfoPtr> getMaybeAllocInfoFromTV(TensorView* tv) {
    auto alloc_it = map_tv_to_allocations_.find(tv->name());
    if (alloc_it == map_tv_to_allocations_.end()) {
      return c10::nullopt;
    }
    return alloc_it->second;
  }

  //! Factory function for internal loop information data
  ScopeInfo* makeScopeInfo(kir::ForLoop* loop) {
    auto loop_info_ptr = std::make_unique<ScopeInfo>();
    auto loop_info = loop_info_ptr.get();
    loop_info->start_pos = current_pos_;
    loop_info->end_pos = -1;
    loop_info->loop = loop;
    all_loop_infos_.emplace_back(std::move(loop_info_ptr));

    if (loop == nullptr) {
      TORCH_INTERNAL_ASSERT(
          !global_scope_info_, "Should only create global scope info once!");
      global_scope_info_ = loop_info;
    } else {
      map_loop_pos_to_loop_info_[current_pos_] = loop_info;
    }
    return loop_info;
  }

  //! Factory function for internal use-def information data
  AllocationUseDefInfo* makeUseDefInfo() {
    auto alloc_info_ptr = std::make_unique<AllocationUseDefInfo>();
    auto alloc_info = alloc_info_ptr.get();

    alloc_info->alloc_pos = current_pos_;
    alloc_info->inner_alias_list_ =
        std::make_unique<std::vector<AllocationUseDefInfo*>>();
    alloc_info->inner_live_interval = std::make_unique<BufferLiveInterval>();
    alloc_info->inner_subscribed_intevals =
        std::make_unique<BufferLiveIntervalPtrList>();
    alloc_info->outer_live_interval = std::make_unique<BufferLiveInterval>();
    alloc_info->outer_subscribed_intevals =
        std::make_unique<BufferLiveIntervalPtrList>();
    all_allocations_.emplace_back(std::move(alloc_info_ptr));
    return alloc_info;
  }

  // Realize buffer alias and keep track of the alias info.
  void setAlias(AllocationInfoPtr from, AllocationInfoPtr to) {
    alias_map_[from] = to;
    from->alloc_expr->setAlias(to->alloc_expr);
    from->alias_to = to->alloc_expr;
  }

 private:
  friend BufferReuseDebugPrinter;
  friend class SerialBroadcastIntervalExpansion;

  //! Allocation sites that will participate in this analysis
  std::unordered_map<const kir::Allocate*, AllocationInfoPtr>
      map_allocate_to_info_;

  //! Map TensorView name to Allocate node.
  //!  Note: this assumes that each tensor view is only allocated once.
  std::unordered_map<StmtNameType, AllocationInfoPtr> map_tv_to_allocations_;

  //! Keeps track of all the allocations that have been set to alias
  std::unordered_map<AllocationInfoPtr, AllocationInfoPtr> alias_map_;

  //! Keep track of stack:
  std::vector<ScopeInfo*> current_stack_;

  //! Contains start and end position of the global scope
  ScopeInfo* global_scope_info_ = nullptr;

  //! map loop start position to loop info
  std::unordered_map<int, ScopeInfo*> map_loop_pos_to_loop_info_;

  //! Owning list of collected allocation info
  AllocationInfoOwningList all_allocations_;

  //! Owning list of collected allocation info
  ScopeInfoOwningPtrList all_loop_infos_;

  //! Expression Evaluator to infer size of register allocation
  kir::ExpressionEvaluator expr_evaluator_;

  //! Position counter when iterating through the exprs list
  int current_pos_ = -1;

  //! Debug info:
  BufferReuseDebugPrinter* debug_printer_ = nullptr;
};

void BufferReuseDebugPrinter::printAllocInfo(const kir::Allocate* alloc) {
  TORCH_INTERNAL_ASSERT(buffer_info_ != nullptr);
  std::string message_header(" \033[1;32m^^^^^ ---Buffer Reuse Info---  ");
  std::string message_end("  \033[0m\n");
  if (!buffer_info_->map_allocate_to_info_.count(alloc)) {
    // This buffer is not considered for any sharing, either
    //  because of un-supported op or size below threshold.
    return;
  }

  auto alloc_info = buffer_info_->map_allocate_to_info_.at(alloc);

  indent() << message_header;
  if (alloc_info->alias_to) {
    if (alloc_info->is_inner_alias) {
      os_ << "(inner) ";
    } else {
      os_ << "(outer) ";
    }
    os_ << " alias to alloc at pos "
        << buffer_info_->getMaybeReuseInfoFor(alloc_info->alias_to)
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
  }
  os_ << "size expr : " << alloc_info->size_expr << " , "
      << "outer live interval: " << alloc_info->outer_live_interval->toString();
  indent() << message_end;
}

//! Reuse Allocation nodes via pointer aliasing
class AllocateReuseModifier {
 public:
  static void modify(const std::vector<Expr*>& exprs) {
    AllocateReuseModifier modifier(exprs);
  }

  static void debugPrint(const std::vector<Expr*>& exprs) {
    BufferReuseDebugPrinter debug_printer;
    AllocateReuseModifier modifier(exprs, &debug_printer);
    std::cout << debug_printer.dumpDebugInfo();
  }

 private:
  AllocateReuseModifier(
      const std::vector<Expr*>& exprs,
      BufferReuseDebugPrinter* debug_printer_ = nullptr)
      : buffer_info_(exprs, debug_printer_) {
    // Perform in-place sharing first and then outer liveness
    //  based sharing. Since outer liveness info can still
    //  be used with some buffers already aliasing through
    //  in-place re-use but wouldn't be the case if we did
    //  outer liveness based sharing first.
    buffer_info_.prepareInnerSharingAnalysis();
    handleScope(exprs);

    inner_aliasing_pass_ = false;

    buffer_info_.prepareOuterSharingAnalysis();
    handleScope(exprs);
  }

  // Second visit of an allocate op
  void handle(kir::Allocate* allocate) {
    // Check that if this allocation site is one that
    //  we want to re-use or replace with an alias

    auto maybe_alloc_info = buffer_info_.getMaybeReuseInfoFor(allocate);
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

  bool tryReuseOtherAllocate(AllocationInfoPtr alloc_info) {
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
          if (!alloc_info->alloc_expr->buffer()->isA<TensorView>()) {
            continue;
          }
          auto this_tv = alloc_info->alloc_expr->buffer()->as<TensorView>();
          auto reuse_tv = alloc_info->alloc_expr->buffer()->as<TensorView>();
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
        reUseAllocation(alloc_info, alloc_to_reuse);
        return true;
      }
    }
    return false;
  }

  void handle(Expr* expr) {
    if (auto ite = dynamic_cast<kir::IfThenElse*>(expr)) {
      handle(ite);
    } else if (auto for_loop = dynamic_cast<kir::ForLoop*>(expr)) {
      handle(for_loop);
    } else if (auto allocate = dynamic_cast<kir::Allocate*>(expr)) {
      handle(allocate);
    }
  }

  void handle(const kir::ForLoop* for_loop) {
    handleScope(for_loop->body().exprs());
  }

  void handle(const kir::IfThenElse* for_loop) {
    TORCH_INTERNAL_ASSERT(
        false,
        "lower_alias_memory: IfThenElse before unrolling is not yet supported");
  }

  void handleScope(const std::vector<Expr*>& exprs) {
    current_visible_buffer_stack_.emplace_back(
        std::make_unique<AllocationInfoList>());
    for (auto expr : exprs) {
      handle(expr);
    }
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
      AllocationUseDefInfo* alloc_info,
      AllocationUseDefInfo* to_reuse) {
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
      auto& local_alloc_map = GpuLower::current()->localAllocationInfoMap();
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

  void reUseAllocation(
      AllocationUseDefInfo* alloc_info,
      AllocationUseDefInfo* to_reuse) {
    // Update analysis result
    if (inner_aliasing_pass_) {
      buffer_info_.useInnerAlias(alloc_info, to_reuse);
    } else {
      buffer_info_.useOuterAlias(alloc_info, to_reuse);
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

  // Utility to capture reduction ops
  bool isBroadcastTvOp(const Expr* expr) {
    if (!ir_utils::isTvOp(expr)) {
      return false;
    }
    return expr->isA<BroadcastOp>();
  }

 private:
  // Analysis result from the first pass collecting the use-defs
  BufferUseDefInfo buffer_info_;

  // Internal data keeping track of currently visible allocations as
  //  the pass iterate through the expr list, grouped by the stack
  //  layer of alloc ops.
  std::vector<std::unique_ptr<AllocationInfoList>>
      current_visible_buffer_stack_;

  // Marks state of current pass
  bool inner_aliasing_pass_ = true;
};

} // namespace

std::vector<Expr*> reuseMemoryAllocations(const std::vector<Expr*>& exprs) {
  FUSER_PERF_SCOPE("reuseMemoryAllocations");
  bool debug_print = isDebugDumpEnabled(DebugDumpOption::BufferReuseInfo);
  if (debug_print) {
    AllocateReuseModifier::debugPrint(exprs);
  }
  AllocateReuseModifier::modify(exprs);
  return exprs;
}

} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
