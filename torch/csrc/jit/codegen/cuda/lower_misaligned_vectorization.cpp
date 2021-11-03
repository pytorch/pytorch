#include <torch/csrc/jit/codegen/cuda/lower_misaligned_vectorization.h>

#include <torch/csrc/jit/codegen/cuda/index_compute.h>
#include <torch/csrc/jit/codegen/cuda/instrumentation.h>
#include <torch/csrc/jit/codegen/cuda/ir_iostream.h>
#include <torch/csrc/jit/codegen/cuda/ir_utils.h>
#include <torch/csrc/jit/codegen/cuda/kernel_ir.h>
#include <torch/csrc/jit/codegen/cuda/kernel_ir_builder.h>
#include <torch/csrc/jit/codegen/cuda/kernel_ir_printer.h>
#include <torch/csrc/jit/codegen/cuda/lower2device.h>
#include <torch/csrc/jit/codegen/cuda/lower_utils.h>
#include <torch/csrc/jit/codegen/cuda/predicate_compute.h>

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {

namespace {

class MisalignedVectorizationModifier {
 public:
  void process(const std::vector<kir::Expr*>& exprs) {
    FUSER_PERF_SCOPE(
        "GpuLower::Lower::MisalignedVectorizationModifier::process");
    // Run through loop nests
    // Find for-loops with misaligned vectorization domains
    for (auto* expr : exprs) {
      handle(expr);
    }
  }

  const std::unordered_map<kir::Expr*, kir::Expr*>& replacementMap() const {
    return expr_replacement_map_;
  }

 private:
  void handle(kir::Expr* expr) {
    if (auto for_loop = dynamic_cast<kir::ForLoop*>(expr)) {
      handle(for_loop);
    } else if (auto ite = dynamic_cast<kir::IfThenElse*>(expr)) {
      handle(ite);
    }
  }

  void handle(kir::ForLoop* fl) {
    for_loops_structure_.push_back(fl);

    // Make copy of exprs because we replace them inplace in fl
    const auto exprs_copy = fl->body().exprs();

    if (containsAnyDirectChildMisalignedVectorize(fl)) {
      auto new_fl = handleMisalignedVectorize(for_loops_structure_, fl);
      expr_replacement_map_.insert({fl, new_fl});
    } else {
      for (auto expr : exprs_copy) {
        handle(expr);
      }
    }

    for_loops_structure_.pop_back();
  }

  void handle(kir::IfThenElse* ite) {
    for (auto expr : ite->thenBody().exprs()) {
      handle(expr);
    }
    for (auto expr : ite->elseBody().exprs()) {
      handle(expr);
    }
  }

  struct ReferenceTensors {
    // Input TensorView to Vectorize Set operation
    kir::TensorView* in_tv = nullptr;
    // Output TensorView to Vectorize Set operation
    kir::TensorView* out_tv = nullptr;
    // TensorView in global memory
    kir::TensorView* global_tv = nullptr;
    // TensorView with vectorize IterDomain and not in global memory
    kir::TensorView* vec_tv = nullptr;
  };

  ReferenceTensors getReferenceTensors(kir::Expr* vectorized_expr) {
    TORCH_INTERNAL_ASSERT(vectorized_expr != nullptr);
    TORCH_INTERNAL_ASSERT(
        vectorized_expr->outputs().front()->isA<kir::TensorView>());
    TORCH_INTERNAL_ASSERT(
        vectorized_expr->inputs().front()->isA<kir::TensorView>());

    auto in_tv = vectorized_expr->inputs().front()->as<kir::TensorView>();
    auto out_tv = vectorized_expr->outputs().front()->as<kir::TensorView>();

    const bool global_vectorize_write_op =
        (out_tv->memoryType() == MemoryType::Global &&
         in_tv->memoryType() == MemoryType::Local);
    const bool global_vectorize_read_op =
        (out_tv->memoryType() == MemoryType::Local &&
         in_tv->memoryType() == MemoryType::Global);
    TORCH_INTERNAL_ASSERT(
        global_vectorize_write_op || global_vectorize_read_op,
        "Unsupported vectorize memory configuration detected.");

    // TensorView on global memory. This is the tensor that may have
    // a non-aligned base address.
    auto global_tv =
        (out_tv->memoryType() == MemoryType::Global) ? out_tv : in_tv;

    // TensorView with the misaligned vec iterDomain. It is the consumer
    // of vectorized load or the producer of vectorized store. It is
    // assumed that when the output TV is not on global memory, this
    // expression is a vectorized load, so the output TV is vec_tv.
    auto vec_tv = (out_tv->memoryType() != MemoryType::Global) ? out_tv : in_tv;

    return {in_tv, out_tv, global_tv, vec_tv};
  }

  struct VectorizeData {
    kir::Val* vector_size = nullptr;
    kir::Val* shift = nullptr;
    kir::Val* extent = nullptr;
    kir::Val* remainder = nullptr;
    kir::Val* extent_minus_remainder = nullptr;
    kir::Val* last_root_domain_index = nullptr;
    kir::Val* last_root_domain_index_shift = nullptr;
  };

  // Create constants for handling misaligned addresses
  VectorizeData createVectorizeConstants(
      const std::vector<kir::ForLoop*>& for_loop_structure,
      const ReferenceTensors& tensors,
      kir::IfThenElse* parent_scope_ite) {
    kir::IrBuilder ir_builder(GpuLower::current()->kernel());

    // Generate vectorize index
    auto indices = (tensors.out_tv->memoryType() == MemoryType::Global)
        ? Index::getConsumerStridedIndices(
              tensors.out_tv->fuserTv(), for_loop_structure)
        : Index::getProducerStridedIndices(
              tensors.in_tv->fuserTv(),
              tensors.out_tv->fuserTv(),
              for_loop_structure);

    // >>>>>>>>>>>>>
    // Number of elements in vectorize access
    auto vector_size =
        tensors.vec_tv->domain()->domain().back()->extent()->as<kir::Int>();

    // Size of memory type for the elements
    kir::Int* data_size_in_bytes =
        ir_builder.create<kir::Int>(dataTypeSize(tensors.vec_tv->dtype()));

    // The number of bytes in the vectorize access
    auto vector_size_in_bytes =
        ir_builder.mulExpr(vector_size, data_size_in_bytes);

    auto index = ir_builder.create<kir::TensorIndex>(
        tensors.global_tv->fuserTv(), indices);
    auto address = createNamedScalarFromValue(
        parent_scope_ite->thenBody(), index, "address", true);

    // offset_size = (address % vector_size_bytes) / data_type_size_bytes
    // shift_init = vector_size - offset_size
    auto a = ir_builder.modExpr(address, vector_size_in_bytes);
    auto b = ir_builder.divExpr(a, data_size_in_bytes);
    auto c = ir_builder.subExpr(vector_size, b);
    auto shift_init = createNamedScalarFromValue(
        parent_scope_ite->thenBody(), c, "shift_val");

    // shift = (shift_init == vector_size) ? 0 : shift_init
    // The number of elements until the first aligned address
    auto shift_pred = ir_builder.eqExpr(shift_init, vector_size);
    auto shift_val =
        ir_builder.whereExpr(shift_pred, ir_builder.zeroVal(), shift_init);

    // >>>>>>>>>>>>>
    auto shift = createNamedScalarFromValue(
        parent_scope_ite->thenBody(), shift_val, "shift");

    // >>>>>>>>>>>>>
    // Get full extent for the inner-most, merged root domain
    auto extent = getVectorizeExtent(tensors.in_tv, tensors.out_tv);

    // remainder = (extent - shift) % vector_size
    // The number of elements remaining not accessed by vectorized operations
    auto remaining_extent = ir_builder.subExpr(extent, shift);
    auto remainder_val = ir_builder.modExpr(remaining_extent, vector_size);
    auto remainder = createNamedScalarFromValue(
        parent_scope_ite->thenBody(), remainder_val, "remainder");

    // (extent - remainder) is the upper-bound for the vectorize section
    auto extent_remainder_val = ir_builder.subExpr(extent, remainder);

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
        ir_builder.addExpr(last_root_domain_index, shift);

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
    kir::IrBuilder ir_builder(GpuLower::current()->kernel());

    auto vectorized_child_loops =
        cloneForLoops(child_loops, params.vector_size, true, params.shift);

    // Vectorize Range: [shift - (extent-remainder))
    // (last_root_domain_index + shift) < (extent - remainder)
    kir::Val* vectorize_cond = ir_builder.ltExpr(
        params.last_root_domain_index_shift, params.extent_minus_remainder);

    kir::Predicate* vectorize_pred =
        ir_builder.create<kir::Predicate>(vectorize_cond->as<kir::Bool>());
    kir::IfThenElse* vectorize_ite =
        ir_builder.create<kir::IfThenElse>(vectorize_pred);

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
    kir::IrBuilder ir_builder(GpuLower::current()->kernel());

    auto pre_child_loops =
        cloneForLoops(child_loops, params.shift, false, nullptr);

    // Initial Range: [0 - shift)
    // last_root_domain_index == 0
    kir::Val* initial_cond =
        ir_builder.eqExpr(params.last_root_domain_index, ir_builder.zeroVal());

    kir::Predicate* initial_pred =
        ir_builder.create<kir::Predicate>(initial_cond->as<kir::Bool>());
    kir::IfThenElse* initial_ite =
        ir_builder.create<kir::IfThenElse>(initial_pred);

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
    kir::IrBuilder ir_builder(GpuLower::current()->kernel());

    auto post_child_loops =
        cloneForLoops(child_loops, params.remainder, false, params.shift);

    // Remainder Range: [(extent-remainder) - extent)
    // (extent - remainder) <= last_root_domain_index + shift < extent
    kir::Val* lower_bound = ir_builder.geExpr(
        params.last_root_domain_index_shift, params.extent_minus_remainder);
    kir::Val* upper_bound =
        ir_builder.ltExpr(params.last_root_domain_index_shift, params.extent);
    kir::Val* remainder_cond = ir_builder.andExpr(lower_bound, upper_bound);

    kir::Predicate* remainder_pred =
        ir_builder.create<kir::Predicate>(remainder_cond->as<kir::Bool>());
    kir::IfThenElse* remainder_ite =
        ir_builder.create<kir::IfThenElse>(remainder_pred);

    for (auto cloned_loop : post_child_loops) {
      remainder_ite->thenBody().push_back(cloned_loop);
    }

    return remainder_ite;
  }

  kir::ForLoop* handleMisalignedVectorize(
      std::vector<kir::ForLoop*> for_loop_structure,
      const kir::ForLoop* parent_for_loop) {
    kir::IrBuilder ir_builder(GpuLower::current()->kernel());

    auto child_loops = findChildForLoops(parent_for_loop);

    // Assumption: All vectorize operations have the same shift
    auto vectorized_expr =
        findFirstVectorizedSetOp(for_loop_structure, child_loops);
    TORCH_INTERNAL_ASSERT(vectorized_expr != nullptr);

    auto reference_tensors = getReferenceTensors(vectorized_expr);

    // The parent_for_loop contains allocate, read, compute, write operations
    const auto new_parent_for_loop =
        ir_builder.create<kir::ForLoop>(parent_for_loop);

    // Transfer all expressions except for-loops to new parent for-loop
    // All expressions are placed at the beginning of the new for-loop
    moveExprsExceptForLoops(parent_for_loop, new_parent_for_loop);

    // Get the predicate for all but the last root domain
    auto pred_except_last_root_domain = ir_builder.create<kir::Predicate>(
        PredicateType::Misaligned, vectorized_expr, ir_builder.trueVal());
    kir::IfThenElse* pred_ite =
        ir_builder.create<kir::IfThenElse>(pred_except_last_root_domain);
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
  bool isVectorizeSetOp(kir::ForLoop* fl, kir::Expr* expr) {
    if (fl->iter_domain()->parallelType() !=
        ParallelType::MisalignedVectorize) {
      return false;
    }

    if (expr->isA<kir::UnaryOp>()) {
      auto unaryOp = expr->as<kir::UnaryOp>();
      if (unaryOp->out()->isA<kir::TensorView>()) {
        auto out_tv = unaryOp->out()->as<kir::TensorView>();
        return unaryOp->operation() == UnaryOpType::Set &&
            out_tv->domain()->hasVectorize();
      }
    }
    return false;
  }

  // Clone each for loop
  // stop value - for (index = start; index < stop; index += step)
  // vectorize flag - Do not generate for loop header
  // shift value - Add shift to global indices generated within for loop
  std::vector<kir::ForLoop*> cloneForLoops(
      const std::vector<kir::ForLoop*>& for_loops,
      kir::Val* stop,
      bool vectorize,
      kir::Val* vectorize_shift) {
    kir::IrBuilder ir_builder(GpuLower::current()->kernel());
    std::vector<kir::ForLoop*> cloned_for_loops;

    for (auto fl : for_loops) {
      auto first_expr = fl->body().exprs().front();
      bool has_vectorize_op = isVectorizeSetOp(fl, first_expr);

      // If the for loop contains a vectorize Set operation, then
      // it should only contain a single expression
      TORCH_INTERNAL_ASSERT(
          !has_vectorize_op || fl->body().exprs().size() == 1);

      const auto new_loop = ir_builder.create<kir::ForLoop>(
          fl->iter_domain(),
          fl->index(),
          ir_builder.zeroVal(),
          stop,
          ir_builder.oneVal(),
          vectorize && has_vectorize_op,
          vectorize_shift);

      for (auto expr : fl->body().exprs()) {
        new_loop->body().push_back(expr);
      }

      cloned_for_loops.push_back(new_loop);
    }
    return cloned_for_loops;
  }

  // Add all expressions except for loops to new parent for loop
  void moveExprsExceptForLoops(
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
  kir::Expr* findFirstVectorizedSetOp(
      std::vector<kir::ForLoop*>& for_loop_structure,
      const std::vector<kir::ForLoop*>& for_loops) {
    for (auto fl : for_loops) {
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
  kir::Val* getVectorizeExtent(
      kir::TensorView* producer_tv,
      kir::TensorView* consumer_tv) {
    const auto gpu_lower = GpuLower::current();
    kir::IrBuilder ir_builder(gpu_lower->kernel());

    auto consumer_fuser_tv = consumer_tv->fuserTv();
    auto producer_fuser_tv = producer_tv->fuserTv();

    auto p2c =
        PairwiseRootDomainMap(producer_fuser_tv, consumer_fuser_tv)
            .mapProducerToConsumer(
                producer_fuser_tv->domain(), consumer_fuser_tv->domain());

    auto consumer_root_right_of_ca_domains = IterVisitor::getInputsTo(
        {consumer_fuser_tv->domain()->domain().begin() +
             consumer_fuser_tv->getComputeAtPosition(),
         consumer_fuser_tv->domain()->domain().end()});
    auto producer_root_right_of_ca_domains = IterVisitor::getInputsTo(
        {producer_fuser_tv->domain()->domain().begin() +
             producer_fuser_tv->getComputeAtPosition(),
         producer_fuser_tv->domain()->domain().end()});

    const auto& consumer_contig = consumer_fuser_tv->domain()->contiguity();
    const auto& producer_contig = producer_fuser_tv->domain()->contiguity();

    // No rfactor should exist in the producer TVs
    TORCH_INTERNAL_ASSERT(
        !producer_tv->domain()->hasRFactor(),
        "Invalid producer tensor: ",
        producer_fuser_tv);
    auto producer_root_domain = producer_fuser_tv->getRootDomain();

    // Calculate extent of merged root domains
    kir::Val* extent = nullptr;
    auto consumer_root_idx = int(consumer_fuser_tv->getRootDomain().size()) - 1;
    for (int i = int(producer_root_domain.size()) - 1; i >= 0; --i) {
      auto producer_root_id = producer_root_domain.at(i);

      TORCH_INTERNAL_ASSERT(
          !gpu_lower->trivialReductionInfo().isDerived(producer_root_id),
          "No trivial reduciton axis should exist: ",
          producer_root_id);

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
      auto consumer_extent = gpu_lower->lowerValue(consumer_root_id->extent());
      if (extent == nullptr) {
        extent = consumer_extent;
      } else {
        extent = ir_builder.mulExpr(extent, consumer_extent);
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

  kir::Val* createNamedScalarFromValue(
      kir::Scope& body,
      kir::Val* val,
      const std::string& name,
      bool address = false) {
    kir::IrBuilder ir_builder(GpuLower::current()->kernel());
    auto namedScalar = (address) ? ir_builder.addressExprNamedScalar(name, val)
                                 : ir_builder.setExprNamedScalar(name, val);
    TORCH_INTERNAL_ASSERT(namedScalar->definition() != nullptr);

    auto alloc = ir_builder.create<kir::Allocate>(
        namedScalar, MemoryType::Local, ir_builder.oneVal());
    body.push_back(alloc);
    body.push_back(namedScalar->definition());
    return namedScalar;
  }

 private:
  // We will track which loops in the incoming IR will be replaced and by what
  std::unordered_map<kir::Expr*, kir::Expr*> expr_replacement_map_;

  // A depth-first ordering of nested for loops
  // It is used for indexing and predicate generation
  std::vector<kir::ForLoop*> for_loops_structure_;
};

} // namespace

std::vector<kir::Expr*> processMisalignedVectorization(
    Fusion* fusion,
    const std::vector<kir::Expr*>& exprs) {
  FUSER_PERF_SCOPE("GpuLower::Lower::processMisalignedVectorization");

  MisalignedVectorizationModifier mvm;
  mvm.process(exprs);

  std::vector<kir::Expr*> mutated_exprs;
  mutated_exprs.reserve(exprs.size());
  for (auto expr : exprs) {
    mutated_exprs.push_back(
        ir_utils::applyReplacements(mvm.replacementMap(), expr));
  }

  return mutated_exprs;
}

bool containsAnyDirectChildMisalignedVectorize(const kir::ForLoop* fl) {
  for (auto expr : fl->body().exprs()) {
    if (expr->isA<kir::ForLoop>()) {
      auto child_fl = expr->as<kir::ForLoop>();
      if (child_fl->iter_domain()->parallelType() ==
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
