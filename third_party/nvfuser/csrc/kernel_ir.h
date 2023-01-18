#pragma once

#include <ir_all_nodes.h>
#include <ir_base_nodes.h>
#include <parallel_type_bitmap.h>
#include <type.h>
#include <utils.h>

#include <c10/macros/Export.h>
#include <c10/util/Optional.h>

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {

class IrBuilderPasskey;

namespace kir {
class Kernel;

// Values
class Predicate;
class TensorIndex;

// Expressions
class Allocate;
class BlockSync;
class GridSync;
class CpAsyncWait;
class CpAsyncCommit;
class InitMagicZero;
class UpdateMagicZero;
class ForLoop;
class IfThenElse;
class GridReduction;
class GroupedGridReduction;
class GridBroadcast;
class GridWelford;
class GroupedGridWelford;
class AllocateFusedReduction;

// Expr container
class Scope;

class TORCH_CUDA_CU_API Predicate final : public Val {
 public:
  explicit Predicate(
      IrBuilderPasskey passkey,
      PredicateType ptype,
      const Expr* expr = nullptr,
      Bool* thread_pred = nullptr);

  explicit Predicate(IrBuilderPasskey passkey, ForLoop* unrolled_loop);

  explicit Predicate(IrBuilderPasskey passkey, Bool* value);

  std::string toString(int indent_size = 0) const override;

  std::string toInlineString(int indent_size = 0) const override;

  PredicateType predicate_type() const {
    return ptype_;
  }

  const Expr* expr() const {
    TORCH_INTERNAL_ASSERT(
        ptype_ != PredicateType::Unswitch &&
        ptype_ != PredicateType::Vectorize && ptype_ != PredicateType::Manual);
    return expr_;
  }

  Bool* thread_pred() const {
    TORCH_INTERNAL_ASSERT(
        ptype_ == PredicateType::Inline ||
        ptype_ == PredicateType::Misaligned || ptype_ == PredicateType::Shift ||
        ptype_ == PredicateType::Padding ||
        ptype_ == PredicateType::ReductionWrite);
    return thread_pred_;
  }

  ForLoop* unrolled_loop() const {
    TORCH_INTERNAL_ASSERT(ptype_ == PredicateType::Unswitch);
    return unrolled_loop_;
  }

  bool hasValue() const {
    return value_ != nullptr;
  }

  Bool* value() const {
    TORCH_INTERNAL_ASSERT(
        value_ != nullptr,
        "The conditional expression for this Predicate is invalid.");
    return value_;
  }

  void setValue(Bool* value) {
    TORCH_INTERNAL_ASSERT(value != nullptr, "The Bool expression is invalid.");
    value_ = value;
  }

  bool isConst() const final {
    return hasValue() && value_->isConst();
  }

 private:
  PredicateType ptype_ = PredicateType::Manual;

  // For PredicateCompute::getInlinePredicate,
  // ShiftPredicateInserter::getShiftPredicate and getPaddingPredicate
  const Expr* expr_ = nullptr;

  // For PredicateCompute::getInlinePredicate
  Bool* thread_pred_ = nullptr;

  // For ParallelType::Unswitch - UnswitchPredicate::get
  ForLoop* unrolled_loop_ = nullptr;

  // The Bool conditional value
  // The value is nullptr until lower_predicate pass
  Bool* value_ = nullptr;
};

class TORCH_CUDA_CU_API TensorIndex final : public Val {
 public:
  TensorIndex(IrBuilderPasskey, const TensorView* view, Val* index);

  Val* index() const {
    return index_;
  }

  TensorView* view() const {
    TORCH_INTERNAL_ASSERT(view_ != nullptr);
    return const_cast<TensorView*>(view_); // NOLINT
  }

  std::string toString(int indent_size = 0) const override;

  std::string toInlineString(int indent_size = 0) const override;

 private:
  const TensorView* view_ = nullptr;
  Val* index_ = nullptr;
};

//! Allocate is a lower level Node that describes a buffer of memory that
//! is required as an intermediate within a kernel. The extent is the expression
//! of the size of the buffer that is generated from the TensorView that
//! describes the output of an operation.
class TORCH_CUDA_CU_API Allocate final : public Expr {
 public:
  using Expr::Expr;

  //! Allocation of a multi-dimensional buffer
  //!
  //! param shape Size of each dimension
  explicit Allocate(
      IrBuilderPasskey passkey,
      Val* buffer,
      MemoryType memory_type,
      std::vector<Val*> shape = {},
      bool zero_init = false,
      Allocate* alias = nullptr);

  //! Allocation of a non-dimensional buffer
  //!
  //! param size Size of allocation
  explicit Allocate(
      IrBuilderPasskey passkey,
      Val* buffer,
      MemoryType memory_type,
      Val* size,
      bool zero_init = false);

  virtual const char* getOpString() const override {
    return "Allocate";
  }

  NVFUSER_DECLARE_CLONE_AND_CREATE

  std::string toString(int indent_size = 0) const override;
  std::string toInlineString(int indent_size = 0) const override;

  Val* buffer() const {
    return attributeVal(0);
  }

  MemoryType memoryType() const {
    return attribute(1)->as<Attribute<MemoryType>>()->value;
  }

  //! Total size
  Val* size() const {
    return input(0);
  }

  //! Size of each dimension
  std::vector<Val*> shape() const {
    std::vector<Val*> result;
    result.reserve(attributes().size() - 4);
    for (auto i = attributes().begin() + 4; i != attributes().end(); ++i) {
      result.emplace_back((*i)->as<Val>());
    }
    return result;
  }

  bool zeroInit() const {
    return attribute(2)->as<Attribute<bool>>()->value;
  }

  // This alias tracks the next Allocate node in a linked chain of aliases
  // If the alias is nullptr, then the Allocate node uses memory in the kernel
  const Allocate* alias() const {
    return dynamic_cast<const Allocate*>(attribute(3));
  }
};

// Sync represents __syncthreads barrier for block level coordination.
//
// TODO(kir): change name to SyncThreads as we could have other barriers.
//
class TORCH_CUDA_CU_API BlockSync final : public Expr {
 public:
  using Expr::Expr;

  explicit BlockSync(IrBuilderPasskey passkey, bool war_sync = false);

  virtual const char* getOpString() const override {
    return "BlockSync";
  }

  NVFUSER_DECLARE_CLONE_AND_CREATE

  std::string toString(int indent_size = 0) const override;
  std::string toInlineString(int indent_size = 0) const override;

  // TODO: war_sync_ is only used for testing/validation purposes.
  bool isWarHazardSync() const {
    return attribute(0)->as<Attribute<bool>>()->value;
  }
};

// Synchronize all blocks in device, implies cooperative group launch is
// required.
class TORCH_CUDA_CU_API GridSync final : public Expr {
 public:
  using Expr::Expr;

  explicit GridSync(
      IrBuilderPasskey passkey,
      ParallelTypeBitmap sync_dims,
      Val* sync_buffer);

  NVFUSER_DECLARE_CLONE_AND_CREATE

  virtual const char* getOpString() const override {
    return "GridSync";
  }

  std::string toString(int indent_size = 0) const override;
  std::string toInlineString(int indent_size = 0) const override;

  ParallelTypeBitmap syncDims() const {
    return attribute(0)->as<Attribute<ParallelTypeBitmap>>()->value;
  }

  Val* syncBuffer() const {
    return attributeVal(1);
  }
};

// CpAsyncWait represents wait intrinsics for cp.async
class TORCH_CUDA_CU_API CpAsyncWait final : public Expr {
 public:
  using Expr::Expr;

  explicit CpAsyncWait(IrBuilderPasskey passkey, unsigned int keep_stages = 0);

  NVFUSER_DECLARE_CLONE_AND_CREATE

  virtual const char* getOpString() const override {
    return "CpAsyncWait";
  }

  std::string toString(int indent_size = 0) const override;
  std::string toInlineString(int indent_size = 0) const override;

  //! Returns the remaining number of stages that are not synchronized
  //!  after this op.
  unsigned int keepStages() const {
    return attribute(0)->as<Attribute<unsigned int>>()->value;
  }
};

// CpAsyncCommit represents commit intrinsics for cp.async
//  A commit intrinsic communicates delimiter of transaction groups
// to the async load hardware. Example usage see [Cicular buffer].
class TORCH_CUDA_CU_API CpAsyncCommit final : public Expr {
 public:
  using Expr::Expr;

  explicit CpAsyncCommit(IrBuilderPasskey passkey);

  NVFUSER_DECLARE_CLONE_AND_CREATE

  virtual const char* getOpString() const override {
    return "CpAsyncCommit";
  }

  std::string toString(int indent_size = 0) const override;
  std::string toInlineString(int indent_size = 0) const override;
};

// Simply prints "DEFINE_MAGIC_ZERO" in the code in accordance with magic_zero
// in helpers.cu
class TORCH_CUDA_CU_API InitMagicZero final : public Expr {
 public:
  using Expr::Expr;

  explicit InitMagicZero(IrBuilderPasskey passkey);

  NVFUSER_DECLARE_CLONE_AND_CREATE

  virtual const char* getOpString() const override {
    return "InitMagicZero";
  }

  std::string toString(int indent_size = 0) const override;
  std::string toInlineString(int indent_size = 0) const override;
};

// Simply prints "UPDATE_MAGIC_ZERO" in the code in accordance with magic_zero
// in helpers.cu
class TORCH_CUDA_CU_API UpdateMagicZero final : public Expr {
 public:
  using Expr::Expr;

  explicit UpdateMagicZero(IrBuilderPasskey passkey);

  NVFUSER_DECLARE_CLONE_AND_CREATE

  virtual const char* getOpString() const override {
    return "UpdateMagicZero";
  }

  std::string toString(int indent_size = 0) const override;
  std::string toInlineString(int indent_size = 0) const override;
};

class TORCH_CUDA_CU_API SMemAddress final : public Expr {
 public:
  using Expr::Expr;

  explicit SMemAddress(IrBuilderPasskey passkey, Val* out, TensorView* smem_tv);

  NVFUSER_DECLARE_CLONE_AND_CREATE

  virtual const char* getOpString() const override {
    return "SMemAddress";
  }

  TensorView* smemTv() const {
    return input(0)->as<TensorView>();
  }

  std::string toString(int indent_size = 0) const override;
  std::string toInlineString(int indent_size = 0) const override;
};

// TODO(kir): promote to IR node
class TORCH_CUDA_CU_API Scope {
 public:
  explicit Scope(Expr* owner) : owner_(owner) {}

  std::string toString(int indent_size = 0) const;

  const std::vector<Expr*>& exprs() const {
    return exprs_;
  }

  bool empty() const {
    return exprs_.empty();
  }

  auto size() const {
    return exprs_.size();
  }

  auto& at(size_t i) {
    return exprs_.at(i);
  }

  auto& at(size_t i) const {
    return exprs_.at(i);
  }

  auto& operator[](size_t i) {
    return at(i);
  }

  auto& operator[](size_t i) const {
    return at(i);
  }

  // Insert expr before expression at pos
  void insert(size_t pos, Expr* expr);

  // Insert expr before ref
  void insert_before(Expr* ref, Expr* expr);

  // Insert expr after ref
  void insert_after(Expr* ref, Expr* expr);

  void push_back(Expr* e) {
    exprs_.push_back(e);
  }

  // Erase expr at pos
  void erase(size_t pos);

  // Erase expr ref
  void erase(Expr* ref);

  bool contains(Expr* expr) const;

  void clear();

  Expr* owner() const {
    return owner_;
  }

  bool operator==(const Scope&) const {
    TORCH_INTERNAL_ASSERT(false, "Should not reach here");
  }

 private:
  // Insert expr before pos
  void insert(std::vector<Expr*>::const_iterator pos, Expr* expr);

  // Erase expr at pos
  void erase(std::vector<Expr*>::const_iterator pos);

 private:
  std::vector<Expr*> exprs_;

  //! Owner exprssion of this scope, e.g., IfThenElse
  Expr* owner_ = nullptr;
};

//! ForLoop provides scoping around an int iterator from 0 to range. Exprs
//! placed in its body are considered inside the scope of the for loop. In the
//! future the implementation should look quite different so that we can do
//! proper dependency annalysis like in Fusion.
//!
//! TODO(kir): this is not a real expression
//!
//! ForLoop may represent a part of an iteration domain representend
//! by iter_domain_. In that case, the loop extent field, extent_, may
//! be smaller than the extent of iter_domain_.
class TORCH_CUDA_CU_API ForLoop final : public Expr {
 public:
  using Expr::Expr;

  //! By default, start and stop are the same as those of iter_domain.
  //! Step is one by default.
  //!
  //! TODO: cleaner way to set options?
  ForLoop(
      IrBuilderPasskey passkey,
      IterDomain* iter_domain,
      Val* index,
      Val* start,
      Val* stop,
      Val* step,
      bool vectorize,
      Val* vectorize_shift,
      bool unroll_required,
      DoubleBufferLoopStage double_buffer_loop_stage);

  ForLoop(IrBuilderPasskey passkey, IterDomain* iter_domain);

  ForLoop(IrBuilderPasskey passkey, const ForLoop* other);

  NVFUSER_DECLARE_CLONE_AND_CREATE

  virtual const char* getOpString() const override {
    return "ForLoop";
  }

  std::string toString(int indent_size = 0) const override;
  std::string toInlineString(int indent_size = 0) const override;

  Val* index() const {
    return input(0);
  }

  Val* start() const;

  Val* stop() const;

  Val* step() const;

  Val* simplifiedStop() const;

  // [pre | vectorize | post] <= inner-most, merged root domain
  // shift_ is applied to vectorize and post sections.
  Val* vectorize_shift() const {
    return attributeVal(4);
  }

  IterDomain* iter_domain() const {
    return input(1)->as<IterDomain>();
  }

  // TODO: Return pointer instead of reference to be more consistent
  Scope& body() {
    return attribute(7)->as<Attribute<Scope>>()->value;
  }

  const Scope& body() const {
    return attribute(7)->as<Attribute<Scope>>()->value;
  }

  // vectorize is true when the for-loop contains a vectorize set
  // the flag is used to omit the for-loop from the kernel
  bool vectorize() const {
    return attribute(3)->as<Attribute<bool>>()->value;
  }

  //! True if unrolled (i.e., "#pragma unroll" is attached)
  bool isUnrolled() const;

  //! True if unroll is required for avoiding stack allocation
  bool isUnrollRequired() const {
    return attribute(5)->as<Attribute<bool>>()->value;
  }

  //! Set unrolling required
  void requireUnroll() {
    attribute(5)->as<Attribute<bool>>()->value = true;
  }

  //! True if no actual for-loop is materialized
  bool isTrivial() const;

  //! True if loop is grouped reduction/welford
  bool isGroup() const;

  //! Returns the stage of a double buffered iterdomain
  //!  that this for loop materializes.
  auto doubleBufferLoopStage() const {
    return attribute(6)->as<Attribute<DoubleBufferLoopStage>>()->value;
  }

 private:
  //! Returns if a loop could be unrolled.
  bool isUnrollable() const;
};

//! IfThenElse provides scoping for an boolean operator. Exprs placed in its
//! body are considered inside the scope of the if statement. In the future the
//! implementation should look quite different so that we can do proper
//! dependency annalysis like in Fusion.
//!
//! TODO(kir): this is not a real expression
//!
class TORCH_CUDA_CU_API IfThenElse final : public Expr {
 public:
  using Expr::Expr;

  explicit IfThenElse(IrBuilderPasskey passkey, Predicate* cond);

  NVFUSER_DECLARE_CLONE_AND_CREATE

  virtual const char* getOpString() const override {
    return "IfThenElse";
  }

  std::string toString(int indent_size = 0) const override;
  std::string toInlineString(int indent_size = 0) const override;

  Scope& thenBody() {
    return attribute(0)->as<Attribute<Scope>>()->value;
  }
  const Scope& thenBody() const {
    return attribute(0)->as<Attribute<Scope>>()->value;
  }

  Scope& elseBody() {
    return attribute(1)->as<Attribute<Scope>>()->value;
  }

  const Scope& elseBody() const {
    return attribute(1)->as<Attribute<Scope>>()->value;
  }

  bool hasElse() const {
    return !elseBody().empty();
  }
};

//! Grid reduction operation
//!
//! This node is used only after lowering a fusion to explicitly mark a grid
//! reduction and the buffer allocation needed to do it.
//!
//! This node provides FusionExecutor the information it needs to allocate the
//! reduction and sync buffers.
class TORCH_CUDA_CU_API GridReduction final : public ReductionOp {
  static constexpr int num_reduction_op_attr = 3;

 public:
  using ReductionOp::ReductionOp;

  GridReduction(
      IrBuilderPasskey passkey,
      BinaryOpType reduction_op_type,
      Val* init,
      Val* out,
      Val* in,
      Allocate* reduction_buffer,
      Allocate* sync_buffer,
      Val* entrance_index,
      Val* entrances,
      bool is_allreduce = false);

  NVFUSER_DECLARE_CLONE_AND_CREATE

  virtual const char* getOpString() const override {
    return "GridReduction";
  }

  std::string toString(int indent_size = 0) const override;
  std::string toInlineString(int indent_size = 0) const override;

  Allocate* reduction_buffer() const {
    return attribute(num_reduction_op_attr)->as<Allocate>();
  }

  Allocate* sync_buffer() const {
    return attribute(num_reduction_op_attr + 1)->as<Allocate>();
  }

  // Which instance of entering this grid reduction is this iteration?
  Val* entrance_index() const {
    return attributeVal(num_reduction_op_attr + 2);
  }

  // How many times will this grid reduction be entered
  Val* entrances() const {
    return attributeVal(num_reduction_op_attr + 3);
  }

  // gridReduce has template flags for thread predicates. In order to
  // use them, the thread predicate is held here separately from
  // Expr::predicate_.
  const ParallelTypeBitmap& threadPredicate() const {
    return attribute(num_reduction_op_attr + 4)
        ->as<Attribute<ParallelTypeBitmap>>()
        ->value;
  }

  ParallelTypeBitmap& threadPredicate() {
    return attribute(num_reduction_op_attr + 4)
        ->as<Attribute<ParallelTypeBitmap>>()
        ->value;
  }

  GridReduction* withThreadPredicate(
      const ParallelTypeBitmap& thread_predicate) {
    auto result = shallowCopy()->as<GridReduction>();
    result->threadPredicate() = thread_predicate;
    return result;
  }
};

class TORCH_CUDA_CU_API GroupedGridReduction final : public GroupedReductionOp {
 public:
  using GroupedReductionOp::GroupedReductionOp;

  GroupedGridReduction(
      IrBuilderPasskey passkey,
      std::vector<BinaryOpType> reduction_op_type,
      std::vector<Val*> init,
      std::vector<Val*> out,
      std::vector<Val*> in,
      std::vector<Allocate*> reduction_buffers,
      Allocate* sync_buffer,
      Val* entrance_index,
      Val* entrances,
      Val* buffer_stride,
      bool is_allreduce = false);

  NVFUSER_DECLARE_CLONE_AND_CREATE

  // number of attributes in the parent class
  int numGroupedReductionOpAttr() const {
    return 2 + outputs().size();
  }

  virtual const char* getOpString() const override {
    return "GroupedGridReduction";
  }

  std::string toString(int indent_size = 0) const override;
  std::string toInlineString(int indent_size = 0) const override;

  std::vector<Allocate*> reduction_buffers() const {
    auto offset = numGroupedReductionOpAttr() + 5;
    auto size = outputs().size();
    std::vector<Allocate*> result;
    result.reserve(size);
    for (auto i : c10::irange(offset, offset + size)) {
      result.emplace_back(attribute(i)->as<Allocate>());
    }
    return result;
  }

  Allocate* reduction_buffer(size_t i) const {
    return reduction_buffers().at(i);
  }

  Allocate* sync_buffer() const {
    return attribute(numGroupedReductionOpAttr())->as<Allocate>();
  }

  // Which instance of entering this grid reduction is this iteration?
  Val* entrance_index() const {
    return attributeVal(numGroupedReductionOpAttr() + 1);
  }

  // How many times will this grid reduction be entered
  Val* entrances() const {
    return attributeVal(numGroupedReductionOpAttr() + 2);
  }

  // Stride of reduction buffers
  Val* buffer_stride() const {
    return attributeVal(numGroupedReductionOpAttr() + 3);
  }

  // gridReduce has template flags for thread predicates. In order to
  // use them, the thread predicate is held here separately from
  // Expr::predicate_.
  const ParallelTypeBitmap& threadPredicate() const {
    return attribute(numGroupedReductionOpAttr() + 4)
        ->as<Attribute<ParallelTypeBitmap>>()
        ->value;
  }

  ParallelTypeBitmap& threadPredicate() {
    return attribute(numGroupedReductionOpAttr() + 4)
        ->as<Attribute<ParallelTypeBitmap>>()
        ->value;
  }

  GroupedGridReduction* withThreadPredicate(
      const ParallelTypeBitmap& thread_predicate) {
    auto result = shallowCopy()->as<GroupedGridReduction>();
    result->threadPredicate() = thread_predicate;
    return result;
  }
};

//! Grid broadcast operation
//!
//! This node is used only after lowering a fusion to explicitly mark a grid
//! broadcast and the buffer allocation needed to do it.
//!
//! This node provides FusionExecutor the information it needs to allocate the
//! broadcast and sync buffers.
class TORCH_CUDA_CU_API GridBroadcast final : public Expr {
 public:
  using Expr::Expr;

  GridBroadcast(
      IrBuilderPasskey passkey,
      BroadcastOp* broadcast_op,
      Allocate* broadcast_buffer,
      Allocate* sync_buffer);

  NVFUSER_DECLARE_CLONE_AND_CREATE

  virtual const char* getOpString() const override {
    return "GridBroadcast";
  }

  std::string toString(int indent_size = 0) const override;
  std::string toInlineString(int indent_size = 0) const override;

  BroadcastOp* broadcast_op() const {
    return attribute(0)->as<BroadcastOp>();
  }

  Allocate* broadcast_buffer() const {
    return attribute(1)->as<Allocate>();
  }

  Allocate* sync_buffer() const {
    return attribute(2)->as<Allocate>();
  }
};

//! Grid welford operation
//!
//! This node is used only after lowering a fusion to explicitly mark a grid
//! reduction and the buffer allocation needed to do it.
//!
//! This node provides FusionExecutor the information it needs to allocate the
//! reduction and sync buffers.
//!
//! TODO: Make this a subclass of WelfordOp
class TORCH_CUDA_CU_API GridWelford final : public Expr {
 public:
  using Expr::Expr;

  GridWelford(
      IrBuilderPasskey passkey,
      WelfordOp* welford_op,
      Allocate* var_buffer,
      Allocate* avg_buffer,
      Allocate* n_buffer,
      Allocate* sync_buffer,
      Val* entrance_index,
      Val* entrances);

  NVFUSER_DECLARE_CLONE_AND_CREATE

  virtual const char* getOpString() const override {
    return "GridWelford";
  }

  std::string toString(int indent_size = 0) const override;
  std::string toInlineString(int indent_size = 0) const override;

  WelfordOp* welford_op() const {
    return attribute(0)->as<WelfordOp>();
  }

  Allocate* var_buffer() const {
    return attribute(1)->as<Allocate>();
  }

  Allocate* avg_buffer() const {
    return attribute(2)->as<Allocate>();
  }

  Allocate* N_buffer() const {
    return attribute(3)->as<Allocate>();
  }

  Allocate* sync_buffer() const {
    return attribute(4)->as<Allocate>();
  }

  // Which instance of entering this grid reduction is this iteration?
  Val* entrance_index() const {
    return attributeVal(5);
  }

  // How many times will this grid reduction be entered
  Val* entrances() const {
    return attributeVal(6);
  }

  // gridReduce has template flags for thread predicates. In order to
  // use them, the thread predicate is held here separately from
  // Expr::predicate_.
  const ParallelTypeBitmap& threadPredicate() const {
    return attribute(7)->as<Attribute<ParallelTypeBitmap>>()->value;
  }
  ParallelTypeBitmap& threadPredicate() {
    return attribute(7)->as<Attribute<ParallelTypeBitmap>>()->value;
  }

  GridWelford* withThreadPredicate(const ParallelTypeBitmap& thread_predicate) {
    auto result = shallowCopy()->as<GridWelford>();
    result->threadPredicate() = thread_predicate;
    return result;
  }
};

class TORCH_CUDA_CU_API GroupedGridWelford final : public GroupedWelfordOp {
 public:
  using GroupedWelfordOp::GroupedWelfordOp;

  // input, output and init vals are vectors of triplets
  GroupedGridWelford(
      IrBuilderPasskey passkey,
      std::vector<WelfordTriplet> output_vals,
      std::vector<WelfordTriplet> input_vals,
      std::vector<WelfordTriplet> init_vals,
      std::array<std::vector<Allocate*>, 3> reduction_buffers,
      Allocate* sync_buffer,
      Val* entrance_index,
      Val* entrances,
      Val* buffer_stride,
      bool is_allreduce = false,
      bool use_outer_opt = false);

  NVFUSER_DECLARE_CLONE_AND_CREATE

  int numGroupedWelfordOpAttr() const {
    return 1 + outputs().size();
  }

  virtual const char* getOpString() const override {
    return "GroupedGridWelford";
  }

  std::string toString(int indent_size = 0) const override;
  std::string toInlineString(int indent_size = 0) const override;

  std::array<std::vector<Allocate*>, 3> reduction_buffers() const {
    auto offset = numGroupedWelfordOpAttr() + 5;
    auto size = outputs().size() / 3;
    std::array<std::vector<Allocate*>, 3> result;
    result[0].reserve(size);
    result[1].reserve(size);
    result[2].reserve(size);
    for (auto i : c10::irange(size)) {
      result[0].emplace_back(attribute(offset + i * 3)->as<Allocate>());
      result[1].emplace_back(attribute(offset + i * 3 + 1)->as<Allocate>());
      result[2].emplace_back(attribute(offset + i * 3 + 2)->as<Allocate>());
    }
    return result;
  }

  Allocate* sync_buffer() const {
    return attribute(numGroupedWelfordOpAttr())->as<Allocate>();
  }

  // Which instance of entering this grid reduction is this iteration?
  Val* entrance_index() const {
    return attributeVal(numGroupedWelfordOpAttr() + 1);
  }

  // How many times will this grid reduction be entered
  Val* entrances() const {
    return attributeVal(numGroupedWelfordOpAttr() + 2);
  }

  // Stride of reduction buffers
  Val* buffer_stride() const {
    return attributeVal(numGroupedWelfordOpAttr() + 3);
  }

  // gridReduce has template flags for thread predicates. In order to
  // use them, the thread predicate is held here separately from
  // Expr::predicate_.
  const ParallelTypeBitmap& threadPredicate() const {
    return attribute(numGroupedWelfordOpAttr() + 4)
        ->as<Attribute<ParallelTypeBitmap>>()
        ->value;
  }
  ParallelTypeBitmap& threadPredicate() {
    return attribute(numGroupedWelfordOpAttr() + 4)
        ->as<Attribute<ParallelTypeBitmap>>()
        ->value;
  }

  GroupedGridWelford* withThreadPredicate(
      const ParallelTypeBitmap& thread_predicate) {
    auto result = shallowCopy()->as<GroupedGridWelford>();
    result->threadPredicate() = thread_predicate;
    return result;
  }

  // True if the outer-optimized kernel should be used
  bool useOuterOpt() const {
    auto offset = numGroupedWelfordOpAttr() + 5 + outputs().size();
    return attribute(offset)->as<Attribute<bool>>()->value;
  }

  //! Return the required smem buffer size
  int getSmemBufferSize(int bdimx, int bdimy, int bdimz) const;
};

//! Represents a WelfordOp with the division by count is hoisted out
//! of an innermost loop
class TORCH_CUDA_CU_API VectorizedWelfordOp final : public WelfordOp {
 public:
  using WelfordOp::WelfordOp;

  VectorizedWelfordOp(
      IrBuilderPasskey,
      const WelfordTriplet& output,
      const WelfordTriplet& input,
      const WelfordTriplet& init,
      Val* count,
      Val* reciprocal_of_count,
      Bool* hoisted_predicate);

  NVFUSER_DECLARE_CLONE_AND_CREATE

  virtual const char* getOpString() const override {
    return "VectorizedWelfordOp";
  }

  //! New count that should be set to outN
  Val* count() const {
    return attributeVal(WelfordOp::kNumAttrs);
  }

  //! Reciprocal of count
  Val* reciprocalOfCount() const {
    return attributeVal(WelfordOp::kNumAttrs + 1);
  }

  //! Predicate of this expression hoisted out of an innermost loop
  Bool* hoistedPredicate() const {
    return attributeVal(WelfordOp::kNumAttrs + 2)->as<Bool>();
  }
};

// Allocate an instance of the fused reduction class.
class TORCH_CUDA_CU_API AllocateFusedReduction final : public Expr {
  explicit AllocateFusedReduction(IrBuilderPasskey passkey, Expr* grid_expr);

 public:
  using Expr::Expr;

  explicit AllocateFusedReduction(
      IrBuilderPasskey passkey,
      GridReduction* grid_reduction)
      : AllocateFusedReduction(passkey, dynamic_cast<Expr*>(grid_reduction)) {}

  explicit AllocateFusedReduction(
      IrBuilderPasskey passkey,
      GridWelford* grid_welford)
      : AllocateFusedReduction(passkey, dynamic_cast<Expr*>(grid_welford)) {}

  explicit AllocateFusedReduction(
      IrBuilderPasskey passkey,
      GroupedGridReduction* grouped_grid_reduction)
      : AllocateFusedReduction(
            passkey,
            dynamic_cast<Expr*>(grouped_grid_reduction)) {}

  explicit AllocateFusedReduction(
      IrBuilderPasskey passkey,
      GroupedGridWelford* grouped_grid_welford)
      : AllocateFusedReduction(
            passkey,
            dynamic_cast<Expr*>(grouped_grid_welford)) {}

  NVFUSER_DECLARE_CLONE_AND_CREATE

  virtual const char* getOpString() const override {
    return "AllocateFusedReduction";
  }

  std::string toString(int indent_size = 0) const override;
  std::string toInlineString(int indent_size = 0) const override;

  //! GridReduction, GridWelford, GroupedGridReduction or GroupedGridWelford
  Expr* gridExpr() const {
    return attribute(0)->asExpr();
  }

  TensorIndex* out() const;

  const ParallelTypeBitmap& threadPredicate() const;
};

} // namespace kir
} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
