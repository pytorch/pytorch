#include <torch/csrc/jit/codegen/cuda/lower_magic_zero.h>

#include <torch/csrc/jit/codegen/cuda/dispatch.h>
#include <torch/csrc/jit/codegen/cuda/instrumentation.h>
#include <torch/csrc/jit/codegen/cuda/ir_utils.h>
#include <torch/csrc/jit/codegen/cuda/kernel_ir_dispatch.h>
#include <torch/csrc/jit/codegen/cuda/lower2device.h>
#include <torch/csrc/jit/codegen/cuda/lower_index_compute.h>

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {

namespace {

class MagicZeroInserter : public kir::ExprMutator {
 public:
  static std::vector<Expr*> insert(const std::vector<Expr*>& exprs) {
    MagicZeroInserter inserter(exprs);
    return inserter.exprs_;
  }

 private:
  struct InsertionInfo {
    kir::Scope* scope = nullptr;
    kir::ForLoop* fl = nullptr;
  };

  MagicZeroInserter(const std::vector<Expr*>& exprs) {
    TORCH_INTERNAL_ASSERT(exprs.size());
    kir::ExprMutator::registerInsertBefore(
        exprs.front(), IrBuilder::create<kir::InitMagicZero>(), nullptr);
    kir::ExprMutator::traverseAndInsert(exprs);
  }

  void handle(kir::ForLoop* fl) final {
    if (fl->isUnrolled()) {
      if (scope_.empty()) {
        kir::ExprMutator::registerInsertAfter(
            fl, IrBuilder::create<kir::UpdateMagicZero>());
      } else {
        TORCH_INTERNAL_ASSERT(
            scope_.back()->exprs().size(), "Not expecting an empty loop.");
        kir::ExprMutator::registerInsertAfter(
            fl, IrBuilder::create<kir::UpdateMagicZero>(), scope_.back());
      }
    } else {
      kir::ExprMutator::handle(fl);
    }
  }

  std::vector<InsertionInfo> insertion_list_;
};

} // namespace

std::vector<Expr*> insertMagicZero(const std::vector<Expr*>& exprs) {
  FUSER_PERF_SCOPE("GpuLower::Lower::insertMagicZero");
  // Check if magic zero was even used, if not we don't have to define it or
  // update it.
  const auto gpu_lower = GpuLower::current();
  auto kernel = gpu_lower->kernel();
  const bool has_magic_zero =
      std::any_of(kernel->vals().begin(), kernel->vals().end(), [](Val* val) {
        return isMagicZero(val);
      });

  if (!has_magic_zero) {
    return exprs;
  }

  return MagicZeroInserter::insert(exprs);
}

bool isMagicZero(const Val* val) {
  if (!val->isA<NamedScalar>()) {
    return false;
  }
  auto ns = val->as<NamedScalar>();
  return ns->dtype() == DataType::Int &&
      ns->name() == std::string(kMagicZeroName);
}

bool isProtectedWithMagicZero(const Val* val) {
  if (val->definition() == nullptr || !val->definition()->isA<BinaryOp>()) {
    return false;
  }
  auto bop = val->definition()->as<BinaryOp>();
  return bop->getBinaryOpType() == BinaryOpType::Add && isMagicZero(bop->rhs());
}

bool needsMagicZero(
    kir::ForLoop* loop,
    IterDomain* reference_domain,
    Val* ind) {
  if (ind->isConstScalar()) {
    return false;
  }

  bool ref_dom_simple =
      reference_domain == nullptr || reference_domain->definition() != nullptr;
  bool ind_simple =
      ind == nullptr || (ind->definition() != nullptr && !ind->isZeroInt());
  return loop->isUnrolled() && (!ref_dom_simple || !ind_simple);
}

void protectNonPredicateIndexWithMagicZero(
    const std::vector<kir::ForLoop*>& loops,
    const std::vector<IterDomain*>& loop_domains,
    std::unordered_map<IterDomain*, Val*>& concrete_loop_idx_map) {
  // Find magic zero insertion point
  IterDomain* magic_zero_loop = nullptr;

  // Search for proper magic zero insertion point,
  //  prefer innermost.
  for (auto idx : c10::irange(loops.size())) {
    auto loop = loops[idx];
    auto concrete_loop_id = GpuLower::current()->caMap()->getConcreteMappedID(
        loop_domains[idx], IdMappingMode::EXACT);
    auto loop_ind = concrete_loop_idx_map.at(concrete_loop_id);

    // Save the concrete id if this loop id is decided to
    //  be the insertion point by the magic zero util.
    if (needsMagicZero(loop, concrete_loop_id, loop_ind)) {
      magic_zero_loop = concrete_loop_id;
    }
  }

  // Insert magic zero if insertion point found
  if (magic_zero_loop != nullptr &&
      concrete_loop_idx_map.count(magic_zero_loop)) {
    auto& ind = concrete_loop_idx_map.at(magic_zero_loop);
    ind = SimplifyingIrBuilder::addExpr(
        ind, GpuLower::current()->kernel()->magicZeroVal());
  }
}

namespace {

//! Protect loop_index_to_protect appearing in overall_index_val
IndexMagicZeroInfo protectIndexByReplacingLoopIndex(
    IterDomain* loop_id,
    Val* overall_index_val,
    Val* loop_index_to_protect) {
  auto protected_loop_index = SimplifyingIrBuilder::addExpr(
      loop_index_to_protect, GpuLower::current()->kernel()->magicZeroVal());

  std::unordered_map<Val*, Val*> replacement_map;
  replacement_map[loop_index_to_protect] = protected_loop_index;

  auto protected_index =
      ir_utils::replaceValInIndexVal(overall_index_val, replacement_map);

  IndexMagicZeroInfo info;
  info.index = protected_index;
  info.original_loop_index = loop_index_to_protect;
  info.protected_loop_index = protected_loop_index;
  info.loop_id = loop_id;
  return info;
}

} // namespace

IndexMagicZeroInfo protectPredicateIndexWithMagicZero(
    Val* index,
    const IndexFromIdGraph& id_graph,
    const std::vector<kir::ForLoop*>& loops) {
  // Gather the loop indices
  std::unordered_set<Val*> loop_indices;
  for (auto loop_id : id_graph.resolved_loop_domains) {
    auto concrete_loop_id = GpuLower::current()->caMap()->getConcreteMappedID(
        loop_id, IdMappingMode::EXACT);
    auto index_it = id_graph.initial_concrete_index_map.find(concrete_loop_id);
    TORCH_INTERNAL_ASSERT(
        index_it != id_graph.initial_concrete_index_map.end(),
        "Index not found for loop: ",
        concrete_loop_id->toString());
    auto loop_index = index_it->second;
    loop_indices.insert(loop_index);
  }

  // Figure out which loop indices are used in index
  const auto vals = DependencyCheck::getAllValsBetween(loop_indices, {index});

  // Traverser from the inner-most loop and apply the magic-zero
  // prorection if needed
  for (int i = static_cast<int>(loops.size()) - 1; i >= 0; --i) {
    auto loop = loops.at(i);
    auto loop_id = id_graph.resolved_loop_domains.at(i);
    TORCH_INTERNAL_ASSERT(GpuLower::current()->caMap()->areMapped(
        loop_id, loop->iter_domain(), IdMappingMode::PERMISSIVE));
    IterDomain* concrete_loop_id =
        GpuLower::current()->caMap()->getConcreteMappedID(
            loop_id, IdMappingMode::EXACT);
    auto index_it = id_graph.initial_concrete_index_map.find(concrete_loop_id);
    TORCH_INTERNAL_ASSERT(
        index_it != id_graph.initial_concrete_index_map.end());
    auto loop_index = index_it->second;

    const auto is_loop_index_used =
        std::find(vals.begin(), vals.end(), loop_index) != vals.end();

    if (!is_loop_index_used) {
      continue;
    }

    if (needsMagicZero(loop, concrete_loop_id, loop_index)) {
      return protectIndexByReplacingLoopIndex(loop_id, index, loop_index);
    }
  }

  // No loop is identified to require protection with magic zero. Just
  // return the index argument as is
  IndexMagicZeroInfo not_proteced;
  not_proteced.index = index;
  return not_proteced;
}

} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
