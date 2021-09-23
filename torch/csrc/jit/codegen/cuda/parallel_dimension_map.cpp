#include <torch/csrc/jit/codegen/cuda/parallel_dimension_map.h>

#include <torch/csrc/jit/codegen/cuda/expr_evaluator.h>
#include <torch/csrc/jit/codegen/cuda/ir_utils.h>
#include <torch/csrc/jit/codegen/cuda/iter_visitor.h>
#include <torch/csrc/jit/codegen/cuda/kernel_expr_evaluator.h>
#include <torch/csrc/jit/codegen/cuda/kernel_ir_builder.h>
#include <torch/csrc/jit/codegen/cuda/kernel_ir_printer.h>
#include <torch/csrc/jit/codegen/cuda/lower2device.h>

#include <sstream>

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {

void ParallelDimensionMap::build(Fusion* fusion) {
  // Scan all TVs to build ParallelType maps
  auto all_vals = fusion->usedMathVals();
  for (auto tv : ir_utils::filterByType<TensorView>(all_vals)) {
    for (auto id : tv->domain()->domain()) {
      registerConstantExtent(id);
      if (!isParallelTypeThread(id->getParallelType())) {
        continue;
      }
      handleParallelDomain(id);
    }
  }

  // Populate the dimension map for each parallel type
  for (const auto& kv : concrete_dom_map_) {
    auto pt = kv.first;
    const auto& concrete_dom_set = kv.second;
    TORCH_INTERNAL_ASSERT(!concrete_dom_set.empty());
    if (concrete_dom_set.size() == 1) {
      populateDimensionMapWithSingleCASet(pt, concrete_dom_set);
    } else {
      populateDimensionMapWithMultipleCASet(pt, concrete_dom_set);
    }
  }
}

void ParallelDimensionMap::registerConstantExtent(IterDomain* id) {
  ExpressionEvaluator ee(id->fusion());
  auto extent_int = ee.evaluate(id->extent());
  if (!extent_int.has_value()) {
    // Nothing to do if not constant
    return;
  }

  auto const_extent = extent_int.value();

  // Ignore if this is derived from a size-1 domain as it is likely a
  // size-1 broadcast domain and that does not represent the actual
  // dimension even if it's constant. Being size-1 may not always mean
  // it's a broadcast domain, but it'd be safe to assume it is mostly
  // the case. If it is not a broadcast, ignoring this domain does not
  // impact the correctness.
  auto extent_inputs = InputsOf::output(id->fusion(), id->extent());
  if (std::any_of(extent_inputs.begin(), extent_inputs.end(), [](Val* input) {
        return input->isOneInt();
      })) {
    return;
  }

  auto concrete_id = getCAMappedConcreteDomain(id);

  auto existing_it = constant_extent_map_.find(id);

  // Adds the constant extent to the set for the concrete domain. If
  // multiple constants are found, this concrete domain has multiple
  // distinctive extents, which can happen with broadcast.
  if (existing_it == constant_extent_map_.end()) {
    constant_extent_map_.insert({concrete_id, {const_extent}});
  } else {
    existing_it->second.insert(const_extent);
  }
}

// Adds the conrecte domain of id to the mappsed set for its
// parallel type
void ParallelDimensionMap::handleParallelDomain(IterDomain* id) {
  auto pt = id->getParallelType();
  TORCH_INTERNAL_ASSERT(isParallelTypeThread(pt));
  auto concrete_id = getCAMappedConcreteDomain(id);

  auto it = concrete_dom_map_.find(pt);
  if (it == concrete_dom_map_.end()) {
    concrete_dom_map_.insert({pt, {concrete_id}});
  } else {
    it->second.insert(concrete_id);
  }
}

void ParallelDimensionMap::populateDimensionMapWithSingleCASet(
    ParallelType pt,
    const std::unordered_set<IterDomain*>& dom_set) {
  TORCH_INTERNAL_ASSERT(dom_set.size() == 1);

  const auto gpu_lower = GpuLower::current();
  kir::IrBuilder ir_builder(gpu_lower->kernel());

  // pt is used by only one concrete domain
  auto id = *dom_set.begin();
  auto it = constant_extent_map_.find(id);

  if (it != constant_extent_map_.end()) {
    if (it->second.size() == 1) {
      dim_map_.insert({pt, ir_builder.create<kir::Int>(*(it->second.begin()))});
      exact_types_.insert(pt);
    } else {
      // Multiple constant dimensions found; Use the corresponding
      // symbolic parallel dim
      dim_map_.insert({pt, kir::NamedScalar::getParallelDim(pt)});
    }
  } else {
    // Prefer to use blockDim/gridDim if not constant
    dim_map_.insert({pt, kir::NamedScalar::getParallelDim(pt)});
    exact_types_.insert(pt);
  }
}

void ParallelDimensionMap::populateDimensionMapWithMultipleCASet(
    ParallelType pt,
    const std::unordered_set<IterDomain*>& dom_set) {
  TORCH_INTERNAL_ASSERT(dom_set.size() > 1);

  const auto gpu_lower = GpuLower::current();
  kir::IrBuilder ir_builder(gpu_lower->kernel());

  bool all_equal = true;
  kir::Val* known_dimension =
      gpu_lower->lowerValue((*dom_set.begin())->extent());
  // Set it -1 to signal it's not initialied yet
  int64_t known_const = -1;

  // Check all of concrete domains to see if they match all together.
  for (auto concrete_id : dom_set) {
    // If this concrete domain has a constant extent, check if it
    // matches with the known constant extent.
    auto it = constant_extent_map_.find(concrete_id);
    if (it != constant_extent_map_.end()) {
      const auto& const_extent_set = it->second;
      // If multiple constants are detected, it's not exact.
      if (const_extent_set.size() > 1) {
        all_equal = false;
        break;
      }
      auto this_const = *(const_extent_set.begin());
      // known_const is initialized to -1
      if (known_const == -1) {
        known_const = this_const;
      } else if (known_const == this_const) {
        // Matched with previously known const. The extent of this
        // domain must be equal to that's previously known.
        continue;
      } else {
        // Unmatched. This dom_set extents may not be unique.
        all_equal = false;
        break;
      }
    }

    // At this point, it still remains undetermined whether this id
    // matches with those previously looked at. Constant check failed,
    // but symbolic matching may succeed.
    if (!equalDim(
            known_dimension, gpu_lower->lowerValue(concrete_id->extent()))) {
      all_equal = false;
      break;
    }
  }

  // If all_equal is still true, the dimension of this paralel type
  // must be exact.
  if (all_equal) {
    exact_types_.insert(pt);
  }
  // Use the const value, if found, as its dimension
  if (all_equal && known_const != -1) {
    dim_map_.insert({pt, ir_builder.create<kir::Int>(known_const)});
  } else {
    dim_map_.insert({pt, kir::NamedScalar::getParallelDim(pt)});
  }
}

kir::Val* ParallelDimensionMap::get(ParallelType pt) const {
  TORCH_INTERNAL_ASSERT(isParallelTypeThread(pt), "Invalid ParallelType: ", pt);
  auto it = dim_map_.find(pt);
  if (it == dim_map_.end()) {
    return nullptr;
  } else {
    return it->second;
  }
}

bool ParallelDimensionMap::isExact(ParallelType pt) const {
  return exact_types_.find(pt) != exact_types_.end();
}

IterDomain* ParallelDimensionMap::getCAMappedConcreteDomain(IterDomain* id) {
  const auto gpu_lower = GpuLower::current();
  const auto& ca_map = gpu_lower->caIndexMap();
  return ca_map.getConcreteMappedID(id);
}

// Symbolically compares equality of two KIR vals. Comparison is done
// conservatively, so returning false does not guarantee non-equality.
bool ParallelDimensionMap::equalDim(kir::Val* dim1, kir::Val* dim2) {
  TORCH_INTERNAL_ASSERT(dim1 != nullptr && dim2 != nullptr);

  if (dim1 == dim2) {
    return true;
  }

  // When Both are Int, they are same if both have the same constant
  auto dim1_int = dynamic_cast<kir::Int*>(dim1);
  auto dim2_int = dynamic_cast<kir::Int*>(dim2);
  if (dim1_int && dim2_int) {
    if (dim1_int->isConst() && dim2_int->isConst()) {
      return dim1_int->value() == dim2_int->value();
    }
  }

  // When both are NamedScalar, they are same if Both have the same
  // name
  auto dim1_ns = dynamic_cast<kir::NamedScalar*>(dim1);
  auto dim2_ns = dynamic_cast<kir::NamedScalar*>(dim2);
  if (dim1_ns && dim2_ns) {
    return dim1_ns->name() == dim2_ns->name();
  }

  // Check recursively their definitions

  auto dim1_def = dim1->definition();
  auto dim2_def = dim2->definition();

  if (dim1_def == nullptr || dim2_def == nullptr) {
    return false;
  }

  // If both are BinaryOp or UnaryOp, check their inputs. Since these
  // Vals are IterDomain extents, UnaryOp should not occur, but
  // checking shouldn't be harmful.
  if ((dim1_def->isA<kir::BinaryOp>() && dim2_def->isA<kir::BinaryOp>() &&
       (dim1_def->as<kir::BinaryOp>()->operation() ==
        dim2_def->as<kir::BinaryOp>()->operation())) ||
      (dim1_def->isA<kir::UnaryOp>() && dim2_def->isA<kir::UnaryOp>() &&
       (dim1_def->as<kir::UnaryOp>()->operation() ==
        dim2_def->as<kir::UnaryOp>()->operation()))) {
    for (size_t i = 0; i < dim1_def->inputs().size(); ++i) {
      if (!equalDim(dim1_def->inputs()[0], dim2_def->inputs()[0])) {
        return false;
      }
    }
    return true;
  }

  return false;
}

std::string ParallelDimensionMap::toString() const {
  std::stringstream ss;

  const std::array<ParallelType, 6> ptypes{
      ParallelType::BIDx,
      ParallelType::BIDy,
      ParallelType::BIDz,
      ParallelType::TIDx,
      ParallelType::TIDy,
      ParallelType::TIDz};

  for (auto pt : ptypes) {
    ss << pt << ": ";
    auto dim = get(pt);
    if (dim != nullptr) {
      ss << kir::toString(dim);
      if (isExact(pt)) {
        ss << ", exact";
      } else {
        ss << ", non-exact";
      }
    } else {
      ss << "unused";
    }
    ss << "\n";
  }

  return ss.str();
}

} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
