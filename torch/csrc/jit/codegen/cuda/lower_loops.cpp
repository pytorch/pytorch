#include <torch/csrc/jit/codegen/cuda/lower_loops.h>

#include <torch/csrc/jit/codegen/cuda/arith.h>
#include <torch/csrc/jit/codegen/cuda/ir_iostream.h>
#include <torch/csrc/jit/codegen/cuda/ir_utils.h>
#include <torch/csrc/jit/codegen/cuda/iter_visitor.h>
#include <torch/csrc/jit/codegen/cuda/kernel_expr_evaluator.h>
#include <torch/csrc/jit/codegen/cuda/lower2device.h>
#include <torch/csrc/jit/codegen/cuda/lower_utils.h>
#include <torch/csrc/jit/codegen/cuda/transform_replay.h>

#include <algorithm>
#include <deque>
#include <numeric>

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {

std::vector<Expr*> LoopNestGenerator::loweredExprs(
    const std::vector<Expr*>& exprs) {
  FUSER_PERF_SCOPE("GpuLower::Lower::LoopNestGenerator::loweredExprs");
  TORCH_INTERNAL_ASSERT(FusionGuard::getCurFusion() != nullptr);
  LoopNestGenerator generator(exprs);
  return generator.lowered_exprs_;
}

LoopNestGenerator::LoopNestGenerator(const std::vector<Expr*>& exprs) {
  generate(exprs);
}

namespace {

kir::ForLoop* openForHelper(kir::ForLoop* scope, IterDomain* id) {
  auto extent_with_halo = GpuLower::current()->haloInfo()->getExtent(id);
  kir::ForLoop* new_scope = nullptr;
  if (extent_with_halo) {
    // When an axis is extended with halo, unrolling and vectorization
    // are assumed to not be used for now.
    TORCH_INTERNAL_ASSERT(
        id->getParallelType() != ParallelType::Unroll &&
        !isParallelTypeVectorize(id->getParallelType()));
    // Use the extent that's extended by halo
    new_scope = IrBuilder::create<kir::ForLoop>(
        id,
        GpuLower::current()->caMap()->getIndexVariable(id),
        nullptr,
        extent_with_halo,
        nullptr,
        false,
        nullptr,
        false,
        DoubleBufferLoopStage::NotApplicable);
  } else {
    new_scope = IrBuilder::create<kir::ForLoop>(id);
  }
  if (scope != nullptr) {
    scope->body().insert(0, new_scope);
  }
  return new_scope;
}

} // namespace

void LoopNestGenerator::openFor(IterDomain* id) {
  if (for_loops_.size() > 0) {
    const auto new_scope = openForHelper(for_loops_.back(), id);
    // for_loop_allocations_.insert({new_scope, 0});
    for_loops_.push_back(new_scope);
  } else {
    for_loops_.push_back(openForHelper(nullptr, id));
    lowered_exprs_.insert(lowered_exprs_.begin(), for_loops_.back());
  }
}

void LoopNestGenerator::closeFor() {
  TORCH_INTERNAL_ASSERT(!for_loops_.empty());
  for_loops_.pop_back();
}

void LoopNestGenerator::pushFront(Expr* expr) {
  if (for_loops_.size() == 0) {
    lowered_exprs_.insert(lowered_exprs_.begin(), expr);
  } else {
    for_loops_.back()->body().insert(0, expr);
  }
}

void LoopNestGenerator::handle(Expr* expr) {
  // Check if it's a tensor view expression we need to place in the loop nest
  // structure
  if (!ir_utils::isTvOp(expr)) {
    // Close all the loops, scalar operations cannot be inside for loops based
    // on expr sorting.
    while (!for_loops_.empty()) {
      closeFor();
    }
    pushFront(expr);

    for (auto out : expr->outputs()) {
      TORCH_INTERNAL_ASSERT(
          out->getValType().value() == ValType::Scalar,
          "Unrecognized output type found in expr ",
          expr,
          " cannot lower ",
          out->getValType().value());

      pushFront(IrBuilder::create<kir::Allocate>(
          out, MemoryType::Local, GpuLower::current()->kernel()->oneVal()));
    }
    return;
  }

  TensorView* out_tv = expr->output(0)->as<TensorView>();

  // Grab the loop structure
  TORCH_INTERNAL_ASSERT(
      loop_structures_.find(out_tv) != loop_structures_.end(),
      "Could not find loop structure of ",
      out_tv);

  // Figure out what the entire loop structure should look like.
  std::vector<IterDomain*> loop_structure = loop_structures_.at(out_tv);

  // Ordering of loop_structure is global, so simply close loops we don't need,
  // and open the ones we do.

  while (!for_loops_.empty() &&
         std::find(
             loop_structure.begin(),
             loop_structure.end(),
             for_loops_.back()->iter_domain()) == loop_structure.end()) {
    closeFor();
  }

  for (auto loop : loop_structure) {
    auto find_it = std::find_if(
        for_loops_.begin(), for_loops_.end(), [loop](kir::ForLoop* fl) {
          return fl->iter_domain() == loop;
        });
    if (find_it == for_loops_.end()) {
      openFor(loop);
    }
  }

  pushFront(expr);
}

// Generate the loop nest structure and place it in lowered_exprs_
void LoopNestGenerator::generate(const std::vector<Expr*>& exprs) {
  TORCH_INTERNAL_ASSERT(lowered_exprs_.empty());

  // Figure out loop structure of each expression. This can be a bit convoluted,
  // for an example why see FusionAdvancedLowering6

  // Grab iteration domain dependencies, similar to the logic in
  // lower_expr_sort, EXCEPT dependencies are in opposite order,
  // inner loops are dependant on outer loops.

  const auto& ca_map = GpuLower::current()->caMap();

  std::unordered_map<IterDomain*, std::unordered_set<IterDomain*>>
      concrete_id_dependencies;
  for (auto tv : ir_utils::allTvs(FusionGuard::getCurFusion())) {
    std::unordered_set<IterDomain*> dependencies;

    for (auto tv_id : tv->domain()->domain()) {
      auto concrete_id =
          ca_map->getConcreteMappedID(tv_id, IdMappingMode::LOOP);

      if (concrete_id_dependencies.find(concrete_id) ==
          concrete_id_dependencies.end()) {
        concrete_id_dependencies[concrete_id] = dependencies;
      } else {
        concrete_id_dependencies[concrete_id].insert(
            dependencies.begin(), dependencies.end());
      }

      // Loops after tv_id are dependent on tv_id
      dependencies.emplace(concrete_id);
    }
  }

  // Fill out dependencies as IDs will have local dependency information, but
  // it's still not guaranteed to be global.

  // If loop structure is something like:
  // T0 [I0]
  // T1 [I0, I1]
  // T2 [I1, I2]
  //
  // I0 will be marked as a dependency of I1
  // I1 will be marked as a dependency of I2
  //
  // However, I0 will not be marked as a dep of I2, so we need to fill out the
  // dependency analysis. This is done by iterating through IterDomains filling
  // out all the dependencies of dependencies recursively.

  std::deque<IterDomain*> to_visit;
  std::unordered_set<IterDomain*> visited;

  std::transform(
      concrete_id_dependencies.begin(),
      concrete_id_dependencies.end(),
      std::back_inserter(to_visit),
      [](const auto& concrete_dep_entry) { return concrete_dep_entry.first; });

  while (!to_visit.empty()) {
    auto id = to_visit.front();
    to_visit.pop_front();

    auto& dependencies = concrete_id_dependencies.at(id);
    bool ready = std::all_of(
        dependencies.begin(), dependencies.end(), [&visited](IterDomain* id) {
          return visited.count(id);
        });

    if (!ready) {
      to_visit.push_back(id);
      continue;
    }

    for (auto dependency : dependencies) {
      auto dep_of_dep = concrete_id_dependencies.at(dependency);
      dependencies.insert(dep_of_dep.begin(), dep_of_dep.end());
    }
    visited.emplace(id);
  }

  // Generate loop structure for each tensor view
  for (auto tv : ir_utils::allTvs(FusionGuard::getCurFusion())) {
    // Zero dim tensor support
    if (tv->nDims() == 0) {
      loop_structures_[tv] = std::vector<IterDomain*>();
      continue;
    }

    auto last_id_concrete = ca_map->getConcreteMappedID(
        tv->axis((int)(tv->nDims() - 1)), IdMappingMode::LOOP);
    auto all_loops_it = concrete_id_dependencies.find(last_id_concrete);
    TORCH_INTERNAL_ASSERT(
        all_loops_it != concrete_id_dependencies.end(),
        "Should have processed all id's in all tvs.");
    std::vector<IterDomain*> loop_structure(
        all_loops_it->second.begin(), all_loops_it->second.end());
    // Dependencies of last domain doesn't include last domain, include it
    // manually
    loop_structure.emplace_back(last_id_concrete);
    // reverse sort (rbegin & rend) since we want the reverse of the order
    // given by IterDomainDependencySorter
    std::sort(
        loop_structure.rbegin(),
        loop_structure.rend(),
        ir_utils::IterDomainDependencySorter(
            concrete_id_dependencies, GpuLower::current()->caMap()));
    loop_structures_[tv] = loop_structure;
  }

  // Process the carefully ordered expressions
  for (auto it = exprs.rbegin(); it != exprs.rend(); ++it) {
    handle(*it);
  }
}

} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
