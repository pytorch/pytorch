#include <torch/csrc/jit/codegen/cuda/lower_misaligned_vectorization.h>

#include <torch/csrc/jit/codegen/cuda/index_compute.h>
#include <torch/csrc/jit/codegen/cuda/instrumentation.h>
#include <torch/csrc/jit/codegen/cuda/ir_iostream.h>
#include <torch/csrc/jit/codegen/cuda/ir_utils.h>
#include <torch/csrc/jit/codegen/cuda/kernel_ir.h>
#include <torch/csrc/jit/codegen/cuda/kernel_ir_dispatch.h>
#include <torch/csrc/jit/codegen/cuda/lower2device.h>
#include <torch/csrc/jit/codegen/cuda/lower_utils.h>
#include <torch/csrc/jit/codegen/cuda/predicate_compute.h>

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {

namespace {

class MisalignedVectorizationModifier : public kir::ExprMutator {
 public:
  MisalignedVectorizationModifier() = delete;

  static std::vector<Expr*> processMisalignedVectorization(
      const std::vector<Expr*>& exprs) {
    FUSER_PERF_SCOPE("GpuLower::Lower::processMisalignedVectorization");
    MisalignedVectorizationModifier mvm(exprs);
    return mvm.exprs_;
  }

 private:
  MisalignedVectorizationModifier(const std::vector<Expr*>& exprs) {
    FUSER_PERF_SCOPE("GpuLower::Lower::MisalignedVectorizationModifier");
    // Run through loop nests
    // Find for-loops with misaligned vectorization domains
    kir::ExprMutator::traverseAndInsert(exprs);
  }

  void handle(kir::ForLoop* fl) final {
    kir::Scope* scope = scope_.empty() ? nullptr : scope_.back();
    if (containsAnyDirectChildMisalignedVectorize(fl)) {
      for_loops_.push_back(fl);
      auto new_fl = handleMisalignedVectorize(for_loops_, fl);
      for_loops_.pop_back();

      kir::ExprMutator::registerReplace(fl, new_fl, scope);
    } else {
      kir::ExprMutator::handle(fl);
    }
  }

  struct ReferenceTensors {
    // Input TensorView to Vectorize Set operation
    TensorView* in_tv = nullptr;
    // Output TensorView to Vectorize Set operation
    TensorView* out_tv = nullptr;
    // TensorView in global memory
    TensorView* global_tv = nullptr;
    // TensorView with vectorize IterDomain and not in global memory
    TensorView* vec_tv = nullptr;
  };

  ReferenceTensors getReferenceTensors(Expr* vectorized_expr) {
    TORCH_INTERNAL_ASSERT(vectorized_expr != nullptr);
    TORCH_INTERNAL_ASSERT(
        vectorized_expr->outputs().front()->isA<TensorView>());
    TORCH_INTERNAL_ASSERT(vectorized_expr->inputs().front()->isA<TensorView>());

    auto in_tv = vectorized_expr->inputs().front()->as<TensorView>();
    auto out_tv = vectorized_expr->outputs().front()->as<TensorView>();

    const bool global_vectorize_write_op =
        (out_tv->getMemoryType() == MemoryType::Global &&
         in_tv->getMemoryType() == MemoryType::Local);
    const bool global_vectorize_read_op =
        (out_tv->getMemoryType() == MemoryType::Local &&
         in_tv->getMemoryType() == MemoryType::Global);
    TORCH_INTERNAL_ASSERT(
        global_vectorize_write_op || global_vectorize_read_op,
        "Unsupported vectorize memory configuration detected.");

    // TensorView on global memory. This is the tensor that may have
    // a non-aligned base address.
    auto global_tv =
        (out_tv->getMemoryType() == MemoryType::Global) ? out_tv : in_tv;

    // TensorView with the misaligned vec iterDomain. It is the consumer
    // of vectorized load or the producer of vectorized store. It is
    // assumed that when the output TV is not on global memory, this
    // expression is a vectorized load, so the output TV is vec_tv.
    auto vec_tv =
        (out_tv->getMemoryType() != MemoryType::Global) ? out_tv : in_tv;

    return {in_tv, out_tv, global_tv, vec_tv};
  }

  struct VectorizeData {
    Val* vector_size = nullptr;
    Val* shift = nullptr;
    Val* extent = nullptr;
    Val* remainder = nullptr;
    Val* extent_minus_remainder = nullptr;
    Val* last_root_domain_index = nullptr;
    Val* last_root_domain_index_shift = nullptr;
  };

  // Create constants for handling misaligned addresses
  VectorizeData createVectorizeConstants(
      const std::vector<kir::ForLoop*>& for_loop_structure,
      const ReferenceTensors& tensors,
      kir::IfThenElse* parent_scope_ite) {
    // Generate vectorize index
    auto indices = (tensors.out_tv->getMemoryType() == MemoryType::Global)
        ? Index::getConsumerStridedIndices(tensors.out_tv, for_loop_structure)
        : Index::getProducerStridedIndices(
              tensors.in_tv, tensors.out_tv, for_loop_structure);

    // >>>>>>>>>>>>>
    // Number of elements in vectorize access
    auto vector_size =
        tensors.vec_tv->domain()->domain().back()->extent()->as<Int>();

    // Size of memory type for the elements
    Int* data_size_in_bytes =
        IrBuilder::create<Int>(dataTypeSize(tensors.vec_tv->dtype()));

    // The number of bytes in the vectorize access
    auto vector_size_in_bytes =
        IrBuilder::mulExpr(vector_size, data_size_in_bytes);

    auto index =
        IrBuilder::create<kir::TensorIndex>(tensors.global_tv, indices);
    auto address = createNamedScalarFromValue(
        parent_scope_ite->thenBody(), index, "address", true);

    // offset_size = (address % vector_size_bytes) / data_type_size_bytes
    // shift_init = vector_size - offset_size
    auto a = IrBuilder::modExpr(address, vector_size_in_bytes);
    auto b = IrBuilder::divExpr(a, data_size_in_bytes);
    auto c = IrBuilder::subExpr(vector_size, b);
    auto shift_init = createNamedScalarFromValue(
        parent_scope_ite->thenBody(), c, "shift_val");

    // shift = (shift_init == vector_size) ? 0 : shift_init
    // The number of elements until the first aligned address
    auto shift_pred = IrBuilder::eqExpr(shift_init, vector_size);
    auto shift_val = IrBuilder::whereExpr(
        shift_pred, GpuLower::current()->kernel()->zeroVal(), shift_init);

    // >>>>>>>>>>>>>
    auto shift = createNamedScalarFromValue(
        parent_scope_ite->thenBody(), shift_val, "shift");

    // >>>>>>>>>>>>>
    // Get full extent for the inner-most, merged root domain
    auto extent = getVectorizeExtent(tensors.in_tv, tensors.out_tv);

    // remainder = (extent - shift) % vector_size
    // The number of elements remaining not accessed by vectorized operations
    auto remaining_extent = IrBuilder::subExpr(extent, shift);
    auto remainder_val = IrBuilder::modExpr(remaining_extent, vector_size);
    auto remainder = createNamedScalarFromValue(
        parent_scope_ite->thenBody(), remainder_val, "remainder");

    // (extent - remainder) is the upper-bound for the vectorize section
    auto extent_remainder_val = IrBuilder::subExpr(extent, remainder);

    // >>>>>>>>>>>>>
    auto extent_minus_remainder = createNamedScalarFromValue(
        parent_scope_ite->thenBody(),
        extent_remainder_val,
        "extent_minus_remainder");

    // >>>>>>>>>>>>>
    auto last_root_domain_index = createNamedScalarFromValue(
        parent_scope_ite->thenBody(), indices.back(), "last_root_domain_index");

    // >>>>>>>>>>>>>
    auto last_root_domain_index_shift =
        IrBuilder::addExpr(last_root_domain_index, shift);

    return {
        vector_size,
        shift,
        extent,
        remainder,
        extent_minus_remainder,
        last_root_domain_index,
        last_root_domain_index_shift};
  }

  // Vectorized : [shift - (extent-remainder))
  // From the first to the last aligned address
  kir::IfThenElse* createVectorizeSection(
      const std::vector<kir::ForLoop*>& child_loops,
      const VectorizeData& params) {
    auto vectorized_child_loops = cloneForLoops(
        child_loops, params.vector_size, nullptr, true, params.shift);

    // Vectorize Range: [shift - (extent-remainder))
    // (last_root_domain_index + shift) < (extent - remainder)
    Val* vectorize_cond = IrBuilder::ltExpr(
        params.last_root_domain_index_shift, params.extent_minus_remainder);

    kir::Predicate* vectorize_pred =
        IrBuilder::create<kir::Predicate>(vectorize_cond->as<Bool>());
    kir::IfThenElse* vectorize_ite =
        IrBuilder::create<kir::IfThenElse>(vectorize_pred);

    for (auto cloned_loop : vectorized_child_loops) {
      vectorize_ite->thenBody().push_back(cloned_loop);
    }

    return vectorize_ite;
  }

  // Initial : [0 - shift)
  // From the initial address until the first aligned address
  kir::IfThenElse* createInitialSection(
      const std::vector<kir::ForLoop*>& child_loops,
      const VectorizeData& params) {
    auto pre_child_loops = cloneForLoops(
        child_loops, params.vector_size, params.shift, false, nullptr);

    // Initial Range: [0 - shift)
    // last_root_domain_index == 0
    Val* initial_cond = IrBuilder::eqExpr(
        params.last_root_domain_index,
        GpuLower::current()->kernel()->zeroVal());

    kir::Predicate* initial_pred =
        IrBuilder::create<kir::Predicate>(initial_cond->as<Bool>());
    kir::IfThenElse* initial_ite =
        IrBuilder::create<kir::IfThenElse>(initial_pred);

    for (auto cloned_loop : pre_child_loops) {
      initial_ite->thenBody().push_back(cloned_loop);
    }

    return initial_ite;
  }

  // Remainder : [(extent-remainder) - extent)
  // From the last aligned address until the end of the extent
  kir::IfThenElse* createRemainderSection(
      const std::vector<kir::ForLoop*>& child_loops,
      const VectorizeData& params) {
    auto post_child_loops = cloneForLoops(
        child_loops, params.vector_size, params.remainder, false, params.shift);

    // Remainder Range: [(extent-remainder) - extent)
    // (extent - remainder) <= last_root_domain_index + shift < extent
    Val* lower_bound = IrBuilder::geExpr(
        params.last_root_domain_index_shift, params.extent_minus_remainder);
    Val* upper_bound =
        IrBuilder::ltExpr(params.last_root_domain_index_shift, params.extent);
    Val* remainder_cond = IrBuilder::andExpr(lower_bound, upper_bound);

    kir::Predicate* remainder_pred =
        IrBuilder::create<kir::Predicate>(remainder_cond->as<Bool>());
    kir::IfThenElse* remainder_ite =
        IrBuilder::create<kir::IfThenElse>(remainder_pred);

    for (auto cloned_loop : post_child_loops) {
      remainder_ite->thenBody().push_back(cloned_loop);
    }

    return remainder_ite;
  }

  kir::ForLoop* handleMisalignedVectorize(
      std::vector<kir::ForLoop*> for_loop_structure,
      const kir::ForLoop* parent_for_loop) {
    auto child_loops = findChildForLoops(parent_for_loop);

    // Assumption: All vectorize operations have the same shift
    auto vectorized_expr =
        findFirstVectorizedSetOp(for_loop_structure, child_loops);
    TORCH_INTERNAL_ASSERT(vectorized_expr != nullptr);

    auto reference_tensors = getReferenceTensors(vectorized_expr);

    // The parent_for_loop contains allocate, read, compute, write operations
    const auto new_parent_for_loop =
        IrBuilder::create<kir::ForLoop>(parent_for_loop);

    // Transfer all expressions except for-loops to new parent for-loop
    // All expressions are placed at the beginning of the new for-loop
    copyExprsExceptForLoops(parent_for_loop, new_parent_for_loop);

    // Get the predicate for all but the last root domain
    auto pred_except_last_root_domain = IrBuilder::create<kir::Predicate>(
        PredicateType::Misaligned,
        vectorized_expr,
        GpuLower::current()->kernel()->trueVal());
    kir::IfThenElse* pred_ite =
        IrBuilder::create<kir::IfThenElse>(pred_except_last_root_domain);
    new_parent_for_loop->body().push_back(pred_ite);

    auto constants = createVectorizeConstants(
        for_loop_structure, reference_tensors, pred_ite);

    // The last root domain is divided into three sections.
    // | Initial - N/A Shift | Vectorize - Shift | Remainder - Shift |

    // Vectorized set operation with vectorize shift
    auto vectorize_ite = createVectorizeSection(child_loops, constants);
    pred_ite->thenBody().push_back(vectorize_ite);

    // Standard set operation without vectorize shift
    auto initial_ite = createInitialSection(child_loops, constants);
    pred_ite->thenBody().push_back(initial_ite);

    // Standard set operation with vectorize shift
    auto remainder_ite = createRemainderSection(child_loops, constants);
    pred_ite->thenBody().push_back(remainder_ite);

    return new_parent_for_loop;
  }

  // Determine that the expression is UnaryOpType::Set AND
  // the output TensorView domain is vectorized
  bool isVectorizeSetOp(kir::ForLoop* fl, Expr* expr) {
    if (fl->iter_domain()->getParallelType() !=
        ParallelType::MisalignedVectorize) {
      return false;
    }

    if (expr->isA<UnaryOp>()) {
      auto unaryOp = expr->as<UnaryOp>();
      if (unaryOp->out()->isA<TensorView>()) {
        auto out_tv = unaryOp->out()->as<TensorView>();
        return unaryOp->getUnaryOpType() == UnaryOpType::Set &&
            out_tv->domain()->hasVectorize();
      }
    }
    return false;
  }

  // Clone each for loop
  // loop_stop value - for (index = start; index < stop; index += step)
  // pred_stop value - Predicate loop body as (index < pred_stop) if non null
  // vectorize flag - Do not generate for loop header
  // shift value - Add shift to global indices generated within for loop
  std::vector<kir::ForLoop*> cloneForLoops(
      const std::vector<kir::ForLoop*>& for_loops_,
      Val* loop_stop,
      Val* pred_stop,
      bool vectorize,
      Val* vectorize_shift) {
    std::vector<kir::ForLoop*> cloned_for_loops;

    for (auto fl : for_loops_) {
      auto first_expr = fl->body().exprs().front();
      bool has_vectorize_op = isVectorizeSetOp(fl, first_expr);

      // If the for loop contains a vectorize Set operation, then
      // it should only contain a single expression
      TORCH_INTERNAL_ASSERT(
          !has_vectorize_op || fl->body().exprs().size() == 1);

      const auto new_loop = IrBuilder::create<kir::ForLoop>(
          fl->iter_domain(),
          fl->index(),
          GpuLower::current()->kernel()->zeroVal(),
          loop_stop,
          GpuLower::current()->kernel()->oneVal(),
          vectorize && has_vectorize_op,
          vectorize_shift,
          fl->isUnrollRequired(),
          fl->doubleBufferLoopStage());

      auto body = &new_loop->body();

      // Predicate the loop body if pred_stop is not null. This is to
      // make sure the loop itself is completely unrollable.
      if (pred_stop != nullptr) {
        auto body_pred = IrBuilder::create<kir::Predicate>(
            IrBuilder::ltExpr(new_loop->index(), pred_stop)->as<Bool>());
        auto body_ite = IrBuilder::create<kir::IfThenElse>(body_pred);
        body->push_back(body_ite);
        body = &body_ite->thenBody();
      }

      for (auto expr : fl->body().exprs()) {
        body->push_back(expr);
      }

      cloned_for_loops.push_back(new_loop);
    }
    return cloned_for_loops;
  }

  // Add all expressions except for loops to new parent for loop
  void copyExprsExceptForLoops(
      const kir::ForLoop* for_loop,
      kir::ForLoop* new_loop) {
    std::vector<kir::ForLoop*> loops;
    for (auto expr : for_loop->body().exprs()) {
      if (!expr->isA<kir::ForLoop>()) {
        new_loop->body().push_back(expr);
      }
    }
  }

  // Find any child for loops inside parent for loop
  std::vector<kir::ForLoop*> findChildForLoops(const kir::ForLoop* for_loop) {
    std::vector<kir::ForLoop*> loops;
    for (auto expr : for_loop->body().exprs()) {
      if (auto nested_for_loop = dynamic_cast<kir::ForLoop*>(expr)) {
        loops.push_back(nested_for_loop);
      }
    }
    return loops;
  }

  // Find the first vectorize set - either read or write
  // Add child For-Loop to for_loop_structure
  // Enable vectorize flag in child For-Loop
  Expr* findFirstVectorizedSetOp(
      std::vector<kir::ForLoop*>& for_loop_structure,
      const std::vector<kir::ForLoop*>& for_loops_) {
    for (auto fl : for_loops_) {
      auto first_expr = fl->body().exprs().front();
      bool has_vectorize_op = isVectorizeSetOp(fl, first_expr);
      if (has_vectorize_op) {
        for_loop_structure.push_back(fl);
        return first_expr;
      }
    }
    return nullptr;
  }

  // Get full extent for the inner-most, merged root domain
  Val* getVectorizeExtent(TensorView* producer_tv, TensorView* consumer_tv) {
    auto p2c = PairwiseRootDomainMap(producer_tv, consumer_tv)
                   .mapProducerToConsumer(
                       producer_tv->domain(), consumer_tv->domain());

    auto consumer_root_right_of_ca_domains = IterVisitor::getInputsTo(
        {consumer_tv->domain()->domain().begin() +
             consumer_tv->getComputeAtPosition(),
         consumer_tv->domain()->domain().end()});
    auto producer_root_right_of_ca_domains = IterVisitor::getInputsTo(
        {producer_tv->domain()->domain().begin() +
             producer_tv->getComputeAtPosition(),
         producer_tv->domain()->domain().end()});

    const auto& consumer_contig = consumer_tv->domain()->contiguity();
    const auto& producer_contig = producer_tv->domain()->contiguity();

    auto producer_root_domain = producer_tv->getMaybeRFactorDomain();

    // Calculate extent of merged root domains
    Val* extent = nullptr;
    auto consumer_root_idx =
        int(consumer_tv->getMaybeRFactorDomain().size()) - 1;
    for (int i = int(producer_root_domain.size()) - 1; i >= 0; --i) {
      auto producer_root_id = producer_root_domain.at(i);

      // If the producer ID is reduction or broadcast, it should be safe
      // to ignore.
      if (producer_root_id->isReduction()) {
        continue;
      } else if (producer_root_id->isBroadcast()) {
        --consumer_root_idx;
        continue;
      }

      // There must be a matching consumer root ID as the producer ID is
      // not reduction and the expression between them is UnaryOpType::Set.
      auto it = p2c.find(producer_root_id);
      TORCH_INTERNAL_ASSERT(
          it != p2c.end(), "No matching consumer root ID found");
      auto consumer_root_id = it->second;

      // Don't extend the vectorization domain beyond the CA position
      if (std::find(
              consumer_root_right_of_ca_domains.begin(),
              consumer_root_right_of_ca_domains.end(),
              consumer_root_id) == consumer_root_right_of_ca_domains.end() ||
          std::find(
              producer_root_right_of_ca_domains.begin(),
              producer_root_right_of_ca_domains.end(),
              producer_root_id) == producer_root_right_of_ca_domains.end()) {
        break;
      }

      // We now know it's safe to extend the vectorization domain to these
      // axes. It shouldn't matter whether producer or consumer is used.
      if (extent == nullptr) {
        extent = consumer_root_id->extent();
      } else {
        extent = IrBuilder::mulExpr(extent, consumer_root_id->extent());
      }

      // If it's not contiguous, extending the vectorization domain
      // further is not possible
      if (!(producer_contig.at(i) && consumer_contig.at(consumer_root_idx))) {
        break;
      }

      --consumer_root_idx;
    }

    TORCH_INTERNAL_ASSERT(extent != nullptr);

    return extent;
  }

  Val* createNamedScalarFromValue(
      kir::Scope& body,
      Val* val,
      const std::string& name,
      bool address = false) {
    auto namedScalar = (address) ? IrBuilder::addressExprNamedScalar(name, val)
                                 : IrBuilder::setExprNamedScalar(name, val);
    TORCH_INTERNAL_ASSERT(namedScalar->definition() != nullptr);

    auto alloc = IrBuilder::create<kir::Allocate>(
        namedScalar,
        MemoryType::Local,
        GpuLower::current()->kernel()->oneVal());
    body.push_back(alloc);
    body.push_back(namedScalar->definition());
    return namedScalar;
  }
};

} // namespace

std::vector<Expr*> processMisalignedVectorization(
    const std::vector<Expr*>& exprs) {
  return MisalignedVectorizationModifier::processMisalignedVectorization(exprs);
}

bool containsAnyDirectChildMisalignedVectorize(const kir::ForLoop* fl) {
  for (auto expr : fl->body().exprs()) {
    if (expr->isA<kir::ForLoop>()) {
      auto child_fl = expr->as<kir::ForLoop>();
      if (child_fl->iter_domain()->getParallelType() ==
          ParallelType::MisalignedVectorize) {
        return true;
      }
    }
  }
  return false;
}

} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
