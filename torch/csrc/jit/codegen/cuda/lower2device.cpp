#include <torch/csrc/jit/codegen/cuda/lower2device.h>

#include <ATen/cuda/CUDAContext.h>
#include <torch/csrc/jit/codegen/cuda/expr_evaluator.h>
#include <torch/csrc/jit/codegen/cuda/fusion.h>
#include <torch/csrc/jit/codegen/cuda/instrumentation.h>
#include <torch/csrc/jit/codegen/cuda/ir_iostream.h>
#include <torch/csrc/jit/codegen/cuda/ir_utils.h>
#include <torch/csrc/jit/codegen/cuda/kernel_ir_printer.h>
#include <torch/csrc/jit/codegen/cuda/lower_alias_memory.h>
#include <torch/csrc/jit/codegen/cuda/lower_allocation.h>
#include <torch/csrc/jit/codegen/cuda/lower_expr_sort.h>
#include <torch/csrc/jit/codegen/cuda/lower_index.h>
#include <torch/csrc/jit/codegen/cuda/lower_insert_syncs.h>
#include <torch/csrc/jit/codegen/cuda/lower_loops.h>
#include <torch/csrc/jit/codegen/cuda/lower_magic_zero.h>
#include <torch/csrc/jit/codegen/cuda/lower_misaligned_vectorization.h>
#include <torch/csrc/jit/codegen/cuda/lower_predicate.h>
#include <torch/csrc/jit/codegen/cuda/lower_shift.h>
#include <torch/csrc/jit/codegen/cuda/lower_thread_predicate.h>
#include <torch/csrc/jit/codegen/cuda/lower_trivial_reductions.h>
#include <torch/csrc/jit/codegen/cuda/lower_unroll.h>
#include <torch/csrc/jit/codegen/cuda/lower_utils.h>
#include <torch/csrc/jit/codegen/cuda/lower_validation.h>
#include <torch/csrc/jit/codegen/cuda/lower_warp_reduce.h>

#include <list>
#include <unordered_map>
#include <unordered_set>

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {

// TODO(kir): revisit this
thread_local GpuLower* active_gpu_lower = nullptr; // NOLINT
namespace {

// Going to generate a map of tensor view root domain extents to reduce the
// number used during lowering. For example if we have:
//
// T2[i0, i1] = T1[i0, i1] + T2[i2, i3]
//
// We know it would be safe to use:
//
// T2[i0, i1] = T1[i0, i1] + T2[i0, i1]
//
// And that way we don't generate T2.size[0] and T2.size[1], instead we will
// reuse T1.size[0] and T1.size[1]
// This is important when doing CSE as T2 and T1 would otherwise look like
// they're using different values, even though we know they're the same
//
// There's some duplicate logic here that's in computeAt map, but it's not so
// concice there to pull out. May want to consider making this mapping its own
// class especially as it may be useful during scheduling.
std::unordered_map<Val*, Val*> getSimplificationMap(Fusion* fusion) {
  std::list<std::unordered_set<IterDomain*>> disjoint_root_sets;
  std::unordered_map<IterDomain*, std::unordered_set<IterDomain*>*>
      id_to_disjoint_root_set;

  auto map_root_ids = [&disjoint_root_sets, &id_to_disjoint_root_set](
                          IterDomain* id0, IterDomain* id1) {
    if (id0->isBroadcast() || id1->isBroadcast()) {
      return;
    }

    auto disjoint_set_0_it = id_to_disjoint_root_set.find(id0);
    auto disjoint_set_1_it = id_to_disjoint_root_set.find(id1);
    bool set_0_found = disjoint_set_0_it != id_to_disjoint_root_set.end();
    bool set_1_found = disjoint_set_1_it != id_to_disjoint_root_set.end();

    if (set_0_found && set_1_found) {
      if (disjoint_set_0_it->second == disjoint_set_1_it->second) {
        return;
      }
      // merge second disjoint set into first
      auto* set_0 = disjoint_set_0_it->second;
      auto* set_1 = disjoint_set_1_it->second;
      for (auto id : *set_1) {
        set_0->emplace(id);
        id_to_disjoint_root_set[id] = set_0;
      }
      // remove second set from disjoint_root_sets
      disjoint_root_sets.erase(std::find(
          disjoint_root_sets.begin(), disjoint_root_sets.end(), *set_1));
    } else if (set_0_found || set_1_found) {
      auto existing_set =
          set_0_found ? disjoint_set_0_it->second : disjoint_set_1_it->second;
      auto to_add_id = set_0_found ? id1 : id0;
      existing_set->emplace(to_add_id);
      id_to_disjoint_root_set[to_add_id] = existing_set;
      // add entry into existing set
    } else {
      // create new set entry
      disjoint_root_sets.emplace_back(std::unordered_set<IterDomain*>());
      auto* new_set = &disjoint_root_sets.back();
      new_set->emplace(id0);
      new_set->emplace(id1);
      id_to_disjoint_root_set[id0] = new_set;
      id_to_disjoint_root_set[id1] = new_set;
    }
  };

  auto fusion_vals = fusion->usedMathVals();
  for (auto producer_tv : ir_utils::filterByType<TensorView>(fusion_vals)) {
    auto consumer_tvs = ir_utils::consumerTvsOf(producer_tv);
    for (auto consumer_tv : consumer_tvs) {
      auto pairwise_map = PairwiseRootDomainMap(producer_tv, consumer_tv);
      auto c2p_root_map = pairwise_map.mapConsumerToProducer(
          consumer_tv->domain(), producer_tv->domain());
      for (auto entry : c2p_root_map) {
        auto c_id = entry.first;
        auto p_id = entry.second;
        map_root_ids(p_id, c_id);
      }
    }
  }

  // Map each set to an input ID (if it exists) that has the smallest ->name()
  // entry value
  std::unordered_map<std::unordered_set<IterDomain*>*, IterDomain*>
      set_to_input_id;

  // Loop over the root domains, of the inputs to the fusion. Pick an input ID
  // to use as the representative ID of the collected sets. Only consider inputs
  // as those are the ones that map to values like "T0.size[1]". They are he
  // ID's that propagated their extents into the problem. We could also check
  // the outputs as we do have C++ examples of using output dimensions for the
  // problem size instead of inputs. However, we don't do anything where we can
  // translate to those kinds of kernels integrated into PyTorch.
  for (auto input_tv : ir_utils::filterByType<TensorView>(fusion->inputs())) {
    for (auto id :
         TensorDomain::noReductions(input_tv->getMaybeRFactorDomain())) {
      auto id_set_it = id_to_disjoint_root_set.find(id);
      if (id_set_it == id_to_disjoint_root_set.end()) {
        continue;
      }
      auto* id_set = id_set_it->second;
      if (set_to_input_id.find(id_set) == set_to_input_id.end()) {
        set_to_input_id[id_set] = id;
      } else {
        auto input_id_of_set = set_to_input_id.at(id_set);
        // Swap id's if new name is less than previously set
        bool swap_ids = id->name() < input_id_of_set->name();
        // If new id is a const scalar but previously was'nt use the const
        // scalar
        swap_ids = swap_ids ||
            (id->extent()->isConstScalar() &&
             !input_id_of_set->extent()->isConstScalar());
        // If previous scalar was const and new isn't, don't swap
        swap_ids = swap_ids &&
            !(input_id_of_set->extent()->isConstScalar() &&
              !id->extent()->isConstScalar());

        if (swap_ids) {
          set_to_input_id[id_set] = id;
        }
      }
    }
  }

  // Finally make map from ID extents to the representitive ID extent.
  std::unordered_map<Val*, Val*> extent_to_min_input_id_extent;
  for (auto entry : set_to_input_id) {
    auto* set = entry.first;
    auto input_id = entry.second;
    for (auto id : *set) {
      extent_to_min_input_id_extent[id->extent()] = input_id->extent();
    }
  }
  return extent_to_min_input_id_extent;
}

class KIRCleaner : public kir::MutableIrVisitor {
 public:
  //! Remove nop IR nodes
  static std::vector<kir::Expr*> cleanUp(
      const std::vector<kir::Expr*>& loop_nests) {
    KIRCleaner cleaner;
    std::vector<kir::Expr*> out_loop_nests;
    for (auto loop_nest : loop_nests) {
      cleaner.handle(loop_nest);
      // No need to keep the loop nest if it's determined to be nop
      if (!cleaner.is_nop_) {
        out_loop_nests.push_back(loop_nest);
      }
    }
    return out_loop_nests;
  }

 private:
  void handle(kir::Expr* expr) {
    if (expr->isA<kir::ForLoop>() || expr->isA<kir::IfThenElse>()) {
      expr->accept(this);
    } else {
      // Any non-scoping expr is not considered nop
      is_nop_ = false;
    }
  }

  void visit(kir::ForLoop* fl) final {
    auto exprs = fl->body().exprs();
    fl->body().clear();
    for (auto expr : exprs) {
      handle(expr);
      // Add the expr to the loop body only when the expr is not nop
      if (!is_nop_) {
        fl->body().push_back(expr);
      }
    }
    // The loop is nop when no expr exists in the body
    is_nop_ = fl->body().empty();
  }

  void visit(kir::IfThenElse* ite) final {
    const auto conditional = ite->predicate()->value();

    // Visit the then block
    auto then_exprs = ite->thenBody().exprs();
    ite->thenBody().clear();
    if (!conditional->isConst() || conditional->value().value()) {
      for (auto expr : then_exprs) {
        handle(expr);
        if (!is_nop_) {
          ite->thenBody().push_back(expr);
        }
      }
    }

    const bool then_nop = ite->thenBody().empty();

    // Visit the else block
    auto else_exprs = ite->elseBody().exprs();
    ite->elseBody().clear();
    if (!conditional->isConst() || !conditional->value().value()) {
      for (auto expr : else_exprs) {
        handle(expr);
        if (!is_nop_) {
          ite->elseBody().push_back(expr);
        }
      }
    }

    const bool else_nop = ite->elseBody().empty();

    // If the then block is nop but the else is not, invert the
    // conditional and move the exprs in the else block to the then
    // block.
    if (then_nop && !else_nop) {
      kir::SimplifyingIrBuilder ir_builder(GpuLower::current()->kernel());
      kir::Bool* pred = ite->predicate()->value();
      kir::Bool* not_pred = ir_builder.notExpr(pred)->as<kir::Bool>();
      ite->predicate()->setValue(not_pred);
      for (auto expr : ite->elseBody().exprs()) {
        ite->thenBody().push_back(expr);
      }
      ite->elseBody().clear();
    }

    // This IfThenElse is nop if both the then and else blocks are nop
    is_nop_ = then_nop && else_nop;
  }

 private:
  //! True if the last visited expr is nop
  bool is_nop_ = false;
};

} // namespace

void GpuLower::replaceSymbolicSizes() {
  FUSER_PERF_SCOPE("GpuLower::Lower::replaceSymbolicSizes");

  kir::IrBuilder ir_builder(kernel());

  // Grab inputs and outputs
  std::vector<TensorView*> inputs_and_outputs;
  for (auto val : fusion_->inputs()) {
    if (ir_utils::isTV(val)) {
      inputs_and_outputs.push_back(val->as<TensorView>());
    }
  }
  // Symbolic size is necessary for outputs if there are no inputs.
  // Otherwise infer output sizes from the inputs via expression evaluation.
  if (fusion_->inputs().empty()) {
    for (auto val : fusion_->outputs()) {
      if (ir_utils::isTV(val)) {
        inputs_and_outputs.push_back(val->as<TensorView>());
      }
    }
  }

  // Generate map for all tensorview root domain values to map them to symbolic
  // values. i.e. T0->getRootDomain()[0] would map to a named scalar
  // "T0.size[0]". This map will be used when lowering fusion ir to kernel ir.
  for (TensorView* tv : inputs_and_outputs) {
    // Replace the domain with one based on Ti.size[j]
    const std::vector<IterDomain*>& root_td = tv->getRootDomain();

    size_t dim = 0;
    for (auto id : root_td) {
      const Val* orig_size = id->extent();

      // Output sizes could have reduction axes, which isn't what gets output.
      // NOLINTNEXTLINE(bugprone-branch-clone)
      if (id->isReduction() ||
          (id->getIterType() == IterType::BroadcastWithoutStride)) {
        continue;
      } else if (
          id->isRFactorProduct() ||
          // NOLINTNEXTLINE(bugprone-branch-clone)
          (id->getIterType() == IterType::BroadcastWithStride) ||
          orig_size->isConstScalar()) {
        dim++;
        continue;
      }

      // TODO(kir): consider a different implementation which doesn't
      //  hijack the kir_val_map_
      // Currently turn off this part for inputs of segmented fusion,
      //  since FusionKernelRuntime will provide these as integer inputs
      if (kir_val_map_.find(orig_size) == kir_val_map_.end() &&
          !orig_size->isFusionInput() && !orig_size->isConstScalar()) {
        std::stringstream ss;
        ss << "T" << tv->name() << ".size[" << dim++ << "]";
        kir_val_map_[orig_size] = ir_builder.create<kir::NamedScalar>(
            ss.str(), orig_size->getDataType().value());
      } else {
        dim++;
      }
    }
  }

  // Use a minimal number of sizes from provided tensors.
  auto extent_simplification_map = getSimplificationMap(fusion_);
  for (auto extent_entry : extent_simplification_map) {
    auto orig_extent = extent_entry.first;
    auto simplified_extent = extent_entry.second;
    if (kir_val_map_.count(orig_extent)) {
      if (kir_val_map_.count(simplified_extent)) {
        kir_val_map_[orig_extent] = kir_val_map_[simplified_extent];
      } else {
        kir_val_map_[orig_extent] = lowerValue(simplified_extent);
      }
    }
  }
}

void GpuLower::collectPaddedParallelDims() {
  ExpressionEvaluator ee(fusion_);
  bool can_be_single_warp = true;

  auto warp_size = at::cuda::warp_size();

  auto used_vals = fusion_->usedMathVals();
  for (auto tv : ir_utils::filterByType<TensorView>(used_vals)) {
    for (auto id : tv->domain()->domain()) {
      if (tv->definition()) {
        if (auto reduction = dynamic_cast<ReductionOp*>(tv->definition())) {
          if (ir_utils::getMaybeWarpReductionDim(reduction).has_value()) {
            warp_pad_info_.has_warp_reduction = true;
          }
        }
      }

      // Check ifi TIDx is padded in this kernel
      if (id->hasPaddingToMultipleOfWarp()) {
        TORCH_INTERNAL_ASSERT(
            id->getParallelType() == ParallelType::TIDx,
            "Padded types supported only on TIDx");
        warp_pad_info_.is_tidx_padded = true;
      }

      // Check all possible bindings of TIDx to see
      //  if TIDx will eventually be bound to a single warp.
      if (id->getParallelType() == ParallelType::TIDx) {
        auto eval_dim = ee.evaluate(id->extent());
        auto size_after_padding = id->getMaybeSizeAfterPadding();
        bool padding_to_single_warp = size_after_padding.has_value() &&
            size_after_padding.value() == warp_size;

        if ((!eval_dim.has_value() || eval_dim.value() > warp_size) &&
            !padding_to_single_warp) {
          // If we see any other TIDx binding that's larger than
          //  a warp or unknown, we shouldn't lower warp reduce
          //  to a single warp type.
          can_be_single_warp = false;
          warp_pad_info_.is_tidx_single_warp = false;
        } else if (can_be_single_warp) {
          if (padding_to_single_warp ||
              (eval_dim.has_value() && eval_dim.value() == warp_size)) {
            warp_pad_info_.is_tidx_single_warp = true;
          }
        }
      }
    }
  }
}

void GpuLower::lower() {
  FUSER_PERF_SCOPE("GpuLower::lower");

  TORCH_INTERNAL_ASSERT(fusion_ != nullptr);
  TORCH_INTERNAL_ASSERT(
      active_gpu_lower == nullptr, "Nested lowering passes are not supported");

  // TODO(kir): revisit this
  struct LowerGuard {
    LowerGuard(GpuLower* gpu_lower) {
      active_gpu_lower = gpu_lower;
    }
    ~LowerGuard() {
      active_gpu_lower = nullptr;
    }
  } lower_guard(this);

  FusionGuard fg(fusion_);

  // Start with a fresh kernel
  kernel_ = std::make_unique<kir::Kernel>();

  // prepare for lowering
  validateIr(fusion_);
  replaceSymbolicSizes();
  collectPaddedParallelDims();
  trivial_reduction_info_.build(fusion_, this);

  // In the future we may directly use this map, but for now it will propagate
  // and validate (to some extent) the parallelization strategy.
  // This is the first time nodes will be lowered to kir nodes. Since for now we
  // propagate the parallel strategy in some instances, we need to do it before
  // lowering.
  ca_parallel_map_ = ComputeAtMap(ComputeAtMap::MappingMode::PARALLEL);
  ca_parallel_map_.build(fusion_, current());

  // Want to run this after parallel map is created
  validateVectorize(fusion_);

  // Generate mappings to generate indices
  ca_index_map_ = ComputeAtMap(ComputeAtMap::MappingMode::INDEX);
  ca_index_map_.build(fusion_, current());

  // Generate mappings to generate and map to loop nests
  ca_loop_map_ = ComputeAtMap(ComputeAtMap::MappingMode::LOOP);
  ca_loop_map_.build(fusion_, current());

  parallelDimensionMap().build(fusion_);
  if (isDebugDumpEnabled(DebugDumpOption::ParallelDimensions)) {
    std::cout << parallelDimensionMap().toString();
  }

  // Compute thread predicates. Depends on parallel_dimension_map_
  thread_pred_map_.build(fusion_);

  // Depends on thread_pred_map_
  validateParallelize(fusion_);

  // Scan the whole fusion and build mappings about halo extensions of
  // all IterDomains
  haloInfo().build(fusion_);

  partialSplitMap().build(fusion_);

  validatePartialSplit(fusion_);

  // Detects all exprssions that don't need predicates
  predicateElimination().build(fusion_);

  nonDivisibleSplitInfo().build(fusion_);

  // Set the kernel inputs & outputs
  for (auto input : fusion_->inputs()) {
    kernel_->addInput(GpuLower::lowerValue(input));
  }

  for (auto output : fusion_->outputs()) {
    kernel_->addOutput(GpuLower::lowerValue(output));
  }

  // Run our passes keeping the lowered expressions and forwarding
  // them

  // Reorder expressions for loop-nest generation respecting computeAt
  // relationships
  auto sorted_exprs = reorderExprsForComputeAt();

  // Generate loop-nests and place each expression at its
  // corresponding loop
  const auto lowered_exprs = LoopNestGenerator::loweredExprs(sorted_exprs);

  // Insert allocations
  const auto alloced_exprs = insertAllocations(lowered_exprs);

  // Insert read after write smem syncs
  const auto raw_sync_exprs = insertRawThreadSynchronization(alloced_exprs);

  // Reuse memory locations
  const auto reuse_mem_exprs = reuseMemoryAllocations(raw_sync_exprs);

  // Inserts predicates after this, need to be careful in later passes when
  // inserting in loop nest structure as insertions could be on if then else
  // instead of directly on a for loop
  const auto unrolled_loops = UnrollPass::runPass(fusion_, reuse_mem_exprs);

  const auto unrolled_mv_loops =
      processMisalignedVectorization(fusion_, unrolled_loops);

  // Insert SyncThreads at end of for-loop to avoid WAR race condition
  const auto war_sync_exprs = insertWarThreadSynchronization(unrolled_mv_loops);

  const auto indexed_loops = IndexLowering::getIndexedExprs(war_sync_exprs);

  const auto exprs_with_fused_broadcast = fuseWarpReduce(indexed_loops);

  const auto conditional_loops =
      generateConditionalFromPredicate(fusion_, exprs_with_fused_broadcast);

  // Insert fake zero updates to make sure nvrtc doesn't blow out register use
  // on index and predicate reuse
  const auto register_adjusted = insertMagicZero(conditional_loops);

  const auto cleaned_up_loops = KIRCleaner::cleanUp(register_adjusted);

  // We now have the lowered expressions, finalize the kernel IR
  kernel_->finalize(cleaned_up_loops);
}

kir::Kernel* GpuLower::kernel() const {
  TORCH_CHECK(kernel_);
  return kernel_.get();
}

// Maps Fusion IR nodes to the Kernel IR counterparts
class GpuLower::KernelIrMapper : private OptInConstDispatch {
 public:
  explicit KernelIrMapper(GpuLower* gpu_lower)
      : gpu_lower_(gpu_lower), ir_builder_(gpu_lower->kernel()) {}

  kir::Val* lowerValue(const Val* value) {
    const auto it = gpu_lower_->kir_val_map_.find(value);
    if (it != gpu_lower_->kir_val_map_.end()) {
      return it->second;
    } else {
      handle(value);
      const auto kir_value = gpu_lower_->kir_val_map_[value];
      TORCH_CHECK(kir_value != nullptr);

      // Lower the value definition, if any
      if (value->isScalar()) {
        if (auto def = value->definition()) {
          const auto kir_def = lowerExpr(def);
          TORCH_INTERNAL_ASSERT(kir_value->definition() == kir_def);
        }
      }

      return kir_value;
    }
  }

  kir::Expr* lowerExpr(const Expr* expr) {
    const auto it = gpu_lower_->kir_expr_map_.find(expr);
    if (it != gpu_lower_->kir_expr_map_.end()) {
      return it->second;
    } else {
      handle(expr);
      const auto lowered_node = gpu_lower_->kir_expr_map_[expr];
      TORCH_CHECK(lowered_node != nullptr);
      return lowered_node;
    }
    // NOLINTNEXTLINE(clang-analyzer-cplusplus.NewDeleteLeaks)
  }

 private:
  void handle(const Statement* node) final {
    OptInConstDispatch::handle(node);
  }

  void handle(const Val* node) final {
    OptInConstDispatch::handle(node);
  }

  void handle(const Expr* node) final {
    OptInConstDispatch::handle(node);
  }

  void handle(const TensorDomain* node) final {
    const auto lowered_node = ir_builder_.create<kir::TensorDomain>(node);
    TORCH_CHECK(gpu_lower_->kir_val_map_.insert({node, lowered_node}).second);
  }

  void handle(const IterDomain* node) final {
    const auto lowered_node = ir_builder_.create<kir::IterDomain>(node);
    TORCH_CHECK(gpu_lower_->kir_val_map_.insert({node, lowered_node}).second);
  }

  void handle(const TensorView* node) final {
    const auto lowered_node = ir_builder_.create<kir::TensorView>(node);
    TORCH_CHECK(gpu_lower_->kir_val_map_.insert({node, lowered_node}).second);
  }

  void handle(const Bool* node) final {
    const auto lowered_node = ir_builder_.create<kir::Bool>(node);
    TORCH_CHECK(gpu_lower_->kir_val_map_.insert({node, lowered_node}).second);
  }

  void handle(const Double* node) final {
    const auto lowered_node = ir_builder_.create<kir::Double>(node);
    TORCH_CHECK(gpu_lower_->kir_val_map_.insert({node, lowered_node}).second);
  }

  void handle(const Int* node) final {
    const auto lowered_node = ir_builder_.create<kir::Int>(node);
    TORCH_CHECK(gpu_lower_->kir_val_map_.insert({node, lowered_node}).second);
  }

  void handle(const NamedScalar* node) final {
    const auto lowered_node = ir_builder_.create<kir::NamedScalar>(
        node->name(), node->getDataType().value());
    TORCH_CHECK(gpu_lower_->kir_val_map_.insert({node, lowered_node}).second);
  }

  void handle(const UnaryOp* node) final {
    const auto lowered_node = ir_builder_.create<kir::UnaryOp>(
        node->getUnaryOpType(),
        lowerValue(node->out()),
        lowerValue(node->in()));
    TORCH_CHECK(gpu_lower_->kir_expr_map_.insert({node, lowered_node}).second);
  }

  void handle(const BinaryOp* node) final {
    const auto lowered_node = ir_builder_.create<kir::BinaryOp>(
        node->getBinaryOpType(),
        lowerValue(node->out()),
        lowerValue(node->lhs()),
        lowerValue(node->rhs()));
    TORCH_CHECK(gpu_lower_->kir_expr_map_.insert({node, lowered_node}).second);
  }

  void handle(const TernaryOp* node) final {
    const auto lowered_node = ir_builder_.create<kir::TernaryOp>(
        node->getTernaryOpType(),
        lowerValue(node->out()),
        lowerValue(node->in1()),
        lowerValue(node->in2()),
        lowerValue(node->in3()));
    TORCH_CHECK(gpu_lower_->kir_expr_map_.insert({node, lowered_node}).second);
  }

  void handle(const ReductionOp* node) final {
    auto out_tv = node->out()->as<TensorView>();
    // If trivial reduction operation lower to set operation.
    if (std::all_of(
            out_tv->domain()->domain().begin(),
            out_tv->domain()->domain().end(),
            [&](IterDomain* id) {
              // If id is a reduction axis, is it a trivial reduction?
              if (id->isReduction()) {
                return gpu_lower_->trivialReductionInfo().isDerived(id);
              } else {
                return true;
              }
            })) {
      const auto lowered_node = ir_builder_.create<kir::UnaryOp>(
          UnaryOpType::Set, lowerValue(node->out()), lowerValue(node->in()));
      TORCH_CHECK(
          gpu_lower_->kir_expr_map_.insert({node, lowered_node}).second);
      return;
    }

    const auto lowered_node = ir_builder_.create<kir::ReductionOp>(
        node->getReductionOpType(),
        lowerValue(node->init()),
        lowerValue(node->out()),
        lowerValue(node->in()));
    TORCH_CHECK(gpu_lower_->kir_expr_map_.insert({node, lowered_node}).second);
  }

  void handle(const WelfordOp* node) final {
    auto lowerOptional = [&](Val* v) { return v ? lowerValue(v) : nullptr; };
    const auto lowered_node = ir_builder_.create<kir::WelfordOp>(
        lowerValue(node->outVar()),
        lowerValue(node->outAvg()),
        lowerValue(node->outN()),
        lowerValue(node->initVar()),
        lowerValue(node->initAvg()),
        lowerValue(node->initN()),
        lowerOptional(node->inVar()),
        lowerValue(node->inAvg()),
        lowerValue(node->inN()));

    TORCH_CHECK(gpu_lower_->kir_expr_map_.insert({node, lowered_node}).second);
  }

  void handle(const BroadcastOp* node) final {
    const auto lowered_node = ir_builder_.create<kir::BroadcastOp>(
        lowerValue(node->out()), lowerValue(node->in()));
    TORCH_CHECK(gpu_lower_->kir_expr_map_.insert({node, lowered_node}).second);
  }

  void handle(const TransposeOp* node) final {
    const auto lowered_node = ir_builder_.create<kir::UnaryOp>(
        UnaryOpType::Set, lowerValue(node->out()), lowerValue(node->in()));
    TORCH_CHECK(gpu_lower_->kir_expr_map_.insert({node, lowered_node}).second);
  }

  void handle(const ShiftOp* node) final {
    const auto lowered_node = ir_builder_.create<kir::UnaryOp>(
        UnaryOpType::Set, lowerValue(node->out()), lowerValue(node->in()));
    TORCH_CHECK(gpu_lower_->kir_expr_map_.insert({node, lowered_node}).second);
  }

  void handle(const GatherOp* node) final {
    const auto lowered_node = ir_builder_.create<kir::UnaryOp>(
        UnaryOpType::Set, lowerValue(node->out()), lowerValue(node->in()));
    TORCH_CHECK(gpu_lower_->kir_expr_map_.insert({node, lowered_node}).second);
  }

  void handle(const ViewOp* node) final {
    const auto lowered_node = ir_builder_.create<kir::UnaryOp>(
        UnaryOpType::Set, lowerValue(node->out()), lowerValue(node->in()));
    TORCH_CHECK(gpu_lower_->kir_expr_map_.insert({node, lowered_node}).second);
  }

 private:
  GpuLower* gpu_lower_ = nullptr;
  kir::IrBuilder ir_builder_;
};

kir::Val* GpuLower::lowerValue(const Val* val) {
  KernelIrMapper kir_mapper(this);
  return kir_mapper.lowerValue(val);
}

kir::Expr* GpuLower::lowerExpr(const Expr* expr) {
  KernelIrMapper kir_mapper(this);
  return kir_mapper.lowerExpr(expr);
}

GpuLower* GpuLower::current() {
  return active_gpu_lower;
}

} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
