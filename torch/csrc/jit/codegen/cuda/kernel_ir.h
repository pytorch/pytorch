#pragma once

#include <torch/csrc/jit/codegen/cuda/ir_all_nodes.h>
#include <torch/csrc/jit/codegen/cuda/ir_base_nodes.h>
#include <torch/csrc/jit/codegen/cuda/parallel_type_bitmap.h>
#include <torch/csrc/jit/codegen/cuda/type.h>
#include <torch/csrc/jit/codegen/cuda/utils.h>

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

// Abstract nodes
class Val;
class Expr;

// Values
class Bool;
class Double;
class Int;
class NamedScalar;

class IterDomain;
class TensorDomain;
class TensorView;

// Expressions
class UnaryOp;
class BinaryOp;
class TernaryOp;
class RNGOp;
class ReductionOp;
class WelfordOp;
class BroadcastOp;

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

  PredicateType predicate_type() const {
    return ptype_;
  }

  const Expr* expr() const {
    TORCH_INTERNAL_ASSERT(
        ptype_ != PredicateType::Unswitch &&
        ptype_ != PredicateType::Vectorize && ptype_ != PredicateType::Manual);
    return expr_;
  }

  Bool* thread_pred() {
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
  TensorIndex(
      IrBuilderPasskey,
      const TensorView* view,
      std::vector<Val*> indices);

  std::vector<Val*>::size_type nDims() const {
    return indices_.size();
  }

  Val* index(int i) const;

  const std::vector<Val*>& indices() const {
    return indices_;
  }

  TensorView* view() const {
    TORCH_INTERNAL_ASSERT(view_ != nullptr);
    return const_cast<TensorView*>(view_); // NOLINT
  }

 private:
  const TensorView* view_ = nullptr;
  std::vector<Val*> indices_;
};

//! Allocate is a lower level Node that describes a buffer of memory that
//! is required as an intermediate within a kernel. The extent is the expression
//! of the size of the buffer that is generated from the TensorView that
//! describes the output of an operation.
class TORCH_CUDA_CU_API Allocate final : public Expr {
 public:
  //! Allocation of a multi-dimensional buffer
  //!
  //! param shape Size of each dimension
  explicit Allocate(
      IrBuilderPasskey passkey,
      Val* buffer,
      MemoryType memory_type,
      std::vector<Val*> shape = {},
      bool zero_init = false);

  //! Allocation of a non-dimensional buffer
  //!
  //! param size Size of allocation
  explicit Allocate(
      IrBuilderPasskey passkey,
      Val* buffer,
      MemoryType memory_type,
      Val* size,
      bool zero_init = false);

  Val* buffer() const {
    return buffer_;
  }

  MemoryType memoryType() const {
    return memory_type_;
  }

  Val* size() const {
    return size_;
  }

  const std::vector<Val*>& shape() const {
    return shape_;
  }

  bool zeroInit() const {
    return zero_init_;
  }

  const Allocate* alias() const {
    return alias_;
  }

  void setAlias(const Allocate* alias) {
    TORCH_INTERNAL_ASSERT(alias != this);
    TORCH_INTERNAL_ASSERT(alias->memoryType() == memory_type_);
    alias_ = alias;
  }

 private:
  Val* buffer_ = nullptr;
  MemoryType memory_type_ = MemoryType::Local;
  //! Size of each dimension
  std::vector<Val*> shape_;
  bool zero_init_ = false;
  //! Total size
  Val* size_ = nullptr;

  // This alias tracks the next Allocate node in a linked chain of aliases
  // If the alias is nullptr, then the Allocate node uses memory in the kernel
  const Allocate* alias_ = nullptr;
};

// Sync represents __syncthreads barrier for block level coordination.
//
// TODO(kir): change name to SyncThreads as we could have other barriers.
//
class TORCH_CUDA_CU_API BlockSync final : public Expr {
 public:
  explicit BlockSync(IrBuilderPasskey passkey, bool war_sync = false);

  bool isWarHazardSync() const {
    return war_sync_;
  }

 private:
  // TODO: war_sync_ is only used for testing/validation purposes.
  bool war_sync_ = false;
};

// CpAsyncWait represents wait intrinsics for cp.async
class TORCH_CUDA_CU_API CpAsyncWait final : public Expr {
 public:
  explicit CpAsyncWait(IrBuilderPasskey passkey, unsigned int keep_stages = 0);

  //! Returns the remaining number of stages that are not synchronized
  //!  after this op.
  unsigned int keepStages() const {
    return keep_stages_;
  }

 private:
  //! Number of stage to leave un-sync'ed by this op.
  unsigned int keep_stages_ = 0;
};

// CpAsyncCommit represents commit intrinsics for cp.async
//  A commit intrinsic communicates delimiter of transaction groups
// to the async load hardware. Example usage see [Cicular buffer].
class TORCH_CUDA_CU_API CpAsyncCommit final : public Expr {
 public:
  explicit CpAsyncCommit(IrBuilderPasskey passkey);
};

// Synchronize all blocks in device, implies cooperative group launch is
// required.
class TORCH_CUDA_CU_API GridSync final : public Expr {
 public:
  explicit GridSync(
      IrBuilderPasskey passkey,
      ParallelTypeBitmap sync_dims,
      Val* sync_buffer);

  ParallelTypeBitmap syncDims() const {
    return sync_dims_;
  }

  Val* syncBuffer() const {
    return sync_buffer_;
  }

 private:
  ParallelTypeBitmap sync_dims_;
  Val* sync_buffer_ = nullptr;
};

// Simply prints "DEFINE_MAGIC_ZERO" in the code in accordance with magic_zero
// in helpers.cu
class TORCH_CUDA_CU_API InitMagicZero final : public Expr {
 public:
  explicit InitMagicZero(IrBuilderPasskey passkey);
};

// Simply prints "UPDATE_MAGIC_ZERO" in the code in accordance with magic_zero
// in helpers.cu
class TORCH_CUDA_CU_API UpdateMagicZero final : public Expr {
 public:
  explicit UpdateMagicZero(IrBuilderPasskey passkey);
};

// TODO(kir): promote to IR node
class TORCH_CUDA_CU_API Scope {
 public:
  explicit Scope(Expr* owner) : owner_(owner) {}

  const std::vector<Expr*>& exprs() const {
    return exprs_;
  }

  bool empty() const {
    return exprs_.empty();
  }

  auto size() const {
    return exprs_.size();
  }

  auto& operator[](size_t i) {
    return exprs_[i];
  }

  auto& operator[](size_t i) const {
    return exprs_[i];
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

  Val* index() const {
    return index_;
  }

  Val* start() const;

  Val* stop() const;

  Val* step() const;

  Val* vectorize_shift() const {
    return vectorize_shift_;
  }

  IterDomain* iter_domain() const {
    return iter_domain_;
  }

  // TODO: Return pointer instead of reference to be more consistent
  Scope& body() {
    return body_;
  }

  const Scope& body() const {
    return body_;
  }

  bool vectorize() const {
    return vectorize_;
  }

  //! True if unrolled (i.e., "#pragma unroll" is attached)
  bool isUnrolled() const;

  //! True if unrolling is required
  bool isUnrollRequired() const {
    return unroll_required_;
  }

  //! Set unrolling required
  void requireUnroll() {
    unroll_required_ = true;
  }

  //! True if no actual for-loop is materialized
  bool isTrivial() const;

  //! Returns the stage of a double buffered iterdomain
  //!  that this for loop materializes.
  auto doubleBufferLoopStage() const {
    return double_buffer_loop_stage_;
  }

 private:
  //! Returns if a loop could be unrolled.
  bool isUnrollable() const;

 private:
  IterDomain* const iter_domain_ = nullptr;

  Val* index_ = nullptr;
  Val* start_ = nullptr;
  Val* stop_ = nullptr;
  Val* step_ = nullptr;

  // vectorize is true when the for-loop contains a vectorize set
  // the flag is used to omit the for-loop from the kernel
  bool vectorize_ = false;
  // [pre | vectorize | post] <= inner-most, merged root domain
  // shift_ is applied to vectorize and post sections.
  Val* vectorize_shift_ = nullptr;

  //! True if unroll is required for avoiding stack allocation
  bool unroll_required_ = false;

  Scope body_;

  //! Tracks if this for loop is implementing a stage of
  //!  a double buffered iterdomain.
  DoubleBufferLoopStage double_buffer_loop_stage_ =
      DoubleBufferLoopStage::NotApplicable;
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
  explicit IfThenElse(IrBuilderPasskey passkey, Predicate* cond);

  Scope& thenBody() {
    return then_body_;
  }
  const Scope& thenBody() const {
    return then_body_;
  }

  Scope& elseBody() {
    return else_body_;
  }

  const Scope& elseBody() const {
    return else_body_;
  }

  bool hasElse() const {
    return !else_body_.empty();
  }

 private:
  Scope then_body_;
  Scope else_body_;
};

//! Grid reduction operation
//!
//! This node is used only after lowering a fusion to explicitly mark a grid
//! reduction and the buffer allocation needed to do it.
//!
//! This node provides FusionExecutor the information it needs to allocate the
//! reduction and sync buffers.
class TORCH_CUDA_CU_API GridReduction final : public ReductionOp {
 public:
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

  Allocate* reduction_buffer() const {
    return reduction_buffer_;
  }

  Allocate* sync_buffer() const {
    return sync_buffer_;
  }

  // Which instance of entering this grid reduction is this iteration?
  Val* entrance_index() const {
    return entrance_index_;
  }

  // How many times will this grid reduction be entered
  Val* entrances() const {
    return entrances_;
  }

  const ParallelTypeBitmap& threadPredicate() const {
    return thread_predicate_;
  }

  void setThreadPredicate(const ParallelTypeBitmap& thread_predicate) {
    thread_predicate_ = thread_predicate;
  }

 private:
  Allocate* reduction_buffer_ = nullptr;
  Allocate* sync_buffer_ = nullptr;
  // gridReduce has template flags for thread predicates. In order to
  // use them, the thread predicate is held here separately from
  // Expr::predicate_.
  ParallelTypeBitmap thread_predicate_;
  Val* entrance_index_ = nullptr;
  Val* entrances_ = nullptr;
};

class TORCH_CUDA_CU_API GroupedGridReduction final : public GroupedReductionOp {
 public:
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

  const std::vector<Allocate*>& reduction_buffers() const {
    return reduction_buffers_;
  }

  Allocate* reduction_buffer(size_t i) const {
    return reduction_buffers_.at(i);
  }

  Allocate* sync_buffer() const {
    return sync_buffer_;
  }

  // Which instance of entering this grid reduction is this iteration?
  Val* entrance_index() const {
    return entrance_index_;
  }

  // How many times will this grid reduction be entered
  Val* entrances() const {
    return entrances_;
  }

  Val* buffer_stride() const {
    return buffer_stride_;
  }

  const ParallelTypeBitmap& threadPredicate() const {
    return thread_predicate_;
  }

  void setThreadPredicate(const ParallelTypeBitmap& thread_predicate) {
    thread_predicate_ = thread_predicate;
  }

 private:
  std::vector<Allocate*> reduction_buffers_;
  Allocate* sync_buffer_ = nullptr;
  // gridReduce has template flags for thread predicates. In order to
  // use them, the thread predicate is held here separately from
  // Expr::predicate_.
  ParallelTypeBitmap thread_predicate_;
  Val* entrance_index_ = nullptr;
  Val* entrances_ = nullptr;
  // Stride of reduction buffers
  Val* buffer_stride_ = nullptr;
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
  GridBroadcast(
      IrBuilderPasskey passkey,
      BroadcastOp* broadcast_op,
      Allocate* broadcast_buffer,
      Allocate* sync_buffer);

  BroadcastOp* broadcast_op() const {
    return broadcast_op_;
  }

  Allocate* broadcast_buffer() const {
    return broadcast_buffer_;
  }

  Allocate* sync_buffer() const {
    return sync_buffer_;
  }

 private:
  BroadcastOp* broadcast_op_ = nullptr;
  Allocate* broadcast_buffer_ = nullptr;
  Allocate* sync_buffer_ = nullptr;
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
  GridWelford(
      IrBuilderPasskey passkey,
      WelfordOp* welford_op,
      Allocate* var_buffer,
      Allocate* avg_buffer,
      Allocate* n_buffer,
      Allocate* sync_buffer,
      Val* entrance_index,
      Val* entrances);

  WelfordOp* welford_op() const {
    return welford_op_;
  }

  Allocate* var_buffer() const {
    return var_buffer_;
  }

  Allocate* avg_buffer() const {
    return avg_buffer_;
  }

  Allocate* N_buffer() const {
    return n_buffer_;
  }

  Allocate* sync_buffer() const {
    return sync_buffer_;
  }

  // Which instance of entering this grid reduction is this iteration?
  Val* entrance_index() const {
    return entrance_index_;
  }

  // How many times will this grid reduction be entered
  Val* entrances() const {
    return entrances_;
  }

  const ParallelTypeBitmap& threadPredicate() const {
    return thread_predicate_;
  }

  void setThreadPredicate(const ParallelTypeBitmap& thread_predicate) {
    thread_predicate_ = thread_predicate;
  }

 private:
  WelfordOp* welford_op_ = nullptr;
  Allocate* var_buffer_ = nullptr;
  Allocate* avg_buffer_ = nullptr;
  Allocate* n_buffer_ = nullptr;
  Allocate* sync_buffer_ = nullptr;
  Val* entrance_index_ = nullptr;
  Val* entrances_ = nullptr;
  // gridReduce has template flags for thread predicates. In order to
  // use them, the thread predicate is held here separately from
  // Expr::predicate_.
  ParallelTypeBitmap thread_predicate_;
};

class TORCH_CUDA_CU_API GroupedGridWelford final : public GroupedWelfordOp {
 public:
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
      bool is_allreduce = false);

  const std::array<std::vector<Allocate*>, 3>& reduction_buffers() const {
    return reduction_buffers_;
  }

  Allocate* sync_buffer() const {
    return sync_buffer_;
  }

  // Which instance of entering this grid reduction is this iteration?
  Val* entrance_index() const {
    return entrance_index_;
  }

  // How many times will this grid reduction be entered
  Val* entrances() const {
    return entrances_;
  }

  Val* buffer_stride() const {
    return buffer_stride_;
  }

  const ParallelTypeBitmap& threadPredicate() const {
    return thread_predicate_;
  }

  void setThreadPredicate(const ParallelTypeBitmap& thread_predicate) {
    thread_predicate_ = thread_predicate;
  }

 private:
  std::array<std::vector<Allocate*>, 3> reduction_buffers_;
  Allocate* sync_buffer_ = nullptr;
  // gridReduce has template flags for thread predicates. In order to
  // use them, the thread predicate is held here separately from
  // Expr::predicate_.
  ParallelTypeBitmap thread_predicate_;
  Val* entrance_index_ = nullptr;
  Val* entrances_ = nullptr;
  // Stride of reduction buffers
  Val* buffer_stride_ = nullptr;
};

// Allocate an instance of the fused reduction class.
class TORCH_CUDA_CU_API AllocateFusedReduction final : public Expr {
 public:
  explicit AllocateFusedReduction(
      IrBuilderPasskey passkey,
      GridReduction* grid_reduction);

  explicit AllocateFusedReduction(
      IrBuilderPasskey passkey,
      GridWelford* grid_welford);

  explicit AllocateFusedReduction(
      IrBuilderPasskey passkey,
      GroupedGridReduction* grouped_grid_reduction);

  explicit AllocateFusedReduction(
      IrBuilderPasskey passkey,
      GroupedGridWelford* grouped_grid_welford);

  Expr* gridExpr() const {
    return grid_expr_;
  }

  TensorIndex* out() const;

  const ParallelTypeBitmap& threadPredicate() const;

 private:
  //! GridReduction, GridWelford, GroupedGridReduction or GroupedGridWelford
  Expr* grid_expr_ = nullptr;
};

//! An IR node consisting of a pair of integers
//!  to facilitate definition of 2D swizzle operators.
//! All swizzle 2D ops takes two inputs and outputs
//!  an integer pair.
//! TODO:
//!  currently this IR node is only allowed as input
//!  to the new PairSelect node. In follow ups would
//!  possibly build out to support out of line
//!  definition of the pair alone.
class TORCH_CUDA_CU_API IntPair : public Val {
 public:
  IntPair(IrBuilderPasskey passkey);
};

//! An IR node marking selection of first or second
//!  value from a pair of integers, e.g.:
//! Pair(X,Y) -> X or Y.
//! This IR node is used to facilitate generation
//!  of inline 2D swizzle math.
class TORCH_CUDA_CU_API PairSelect : public Expr {
 public:
  //! Indicates which value from the input
  //!  integer pair to output.
  enum class Selection { X = 0, Y };

  PairSelect(IrBuilderPasskey, Val* out, IntPair* in, Selection selection);

  Val* out() const {
    return out_;
  }

  IntPair* in() const {
    return in_;
  }

  auto selection() const {
    return selection_;
  }

 private:
  Val* const out_ = nullptr;
  IntPair* const in_ = nullptr;
  Selection selection_;
};

//! An integer IR node that will be generated
//!  using custom integer swizzle functions
//!  from the cuda runtime functions.
//! Most supported swizzle functions require
//!  the sizes of each dimension defined so
//!  all operators will take the extents as inputs.
class TORCH_CUDA_CU_API Swizzle2DInt : public Expr {
 public:
  Swizzle2DInt(
      IrBuilderPasskey,
      IntPair* out,
      Val* in_x,
      Val* in_y,
      Val* extent_x,
      Val* extent_y,
      Swizzle2DType swizzle_type);

  IntPair* out() const {
    return out_;
  }

  Val* inX() const {
    return in_x_;
  }

  Val* inY() const {
    return in_y_;
  }

  Val* extentX() const {
    return extent_x_;
  }

  Val* extentY() const {
    return extent_y_;
  }

  const auto& swizzleType() const {
    return swizzle_type_;
  }

 private:
  IntPair* const out_ = nullptr;

  Val* const in_x_ = nullptr;
  Val* const in_y_ = nullptr;
  Val* const extent_x_ = nullptr;
  Val* const extent_y_ = nullptr;
  Swizzle2DType swizzle_type_;
};

} // namespace kir
} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
