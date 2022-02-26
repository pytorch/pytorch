#pragma once

#include <c10/macros/Export.h>

#include <torch/csrc/jit/codegen/cuda/ir_all_nodes.h>

#include <memory>
#include <vector>

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {

class ViewTransform;

//!
//! The goal of analyzeView is to find the minimum number of transformations
//! to convert from the original size to the new size. A naive view algorithm
//! would merge all axis together and then split according to the new sizes.
//!
//! This implementation will keep the original domains, if the domains are the
//! same size in the original and new shapes. If an original domain is not
//! evenly divisible by the new domain, we will merge the minimum number of
//! adjacent original domains.
//!
//! The view transformations are processed in the following order:
//! 1. Trivial Reductions - Removes size-1 broadcast dimensions
//! 2. Keep, Merge, Split - Used to create new rfactor domain
//! 3. Broadcast - Inserts size-1 dimensions
//!
//! Broadcast is handled last because size-1 dimension can be inserted anywhere
//! in the new shape.
//!

struct AnalyzeViewResult {
  bool has_broadcast = false;
  std::vector<bool> broadcast_axes;
  std::vector<int> trivial_reduction_axes;
  std::vector<std::shared_ptr<ViewTransform>> transforms;
};

struct AnalyzeViewConstraint {
  std::vector<int64_t> original_constraint;
  std::vector<int64_t> new_constraint;
};

// Find the transformations necessary to convert TensorView
// from original size to new size.
AnalyzeViewResult analyzeView(
    const TensorView* tv,
    const std::vector<int64_t>& original_sizes,
    const std::vector<int64_t>& new_sizes);

// Find the constraints derived from the view transformations
AnalyzeViewConstraint analyzeViewConstraint(
    const std::vector<int64_t>& original_sizes,
    const std::vector<int64_t>& new_sizes);

// Generate a new TensorDomain from the given view transformations.
// The original root domain is kept in the new TensorDomain,
// but a new rfactor domain is created from the view transformations.
TensorDomain* transformView(
    TensorDomain* original_domain,
    const std::vector<std::shared_ptr<ViewTransform>>& view_transforms);

} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
