#pragma once

#include <torch/csrc/jit/codegen/cuda/fusion.h>
#include <torch/csrc/jit/codegen/cuda/ir_all_nodes.h>
#include <torch/csrc/jit/codegen/cuda/iter_visitor.h>
#include <torch/csrc/jit/codegen/cuda/scheduler/registry.h>
#include <torch/csrc/jit/codegen/cuda/scheduler/utils.h>

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {

// TODO: Put implementations in a vectorize_helper.cpp
namespace scheduler_utils {

// Moved the definition of these to
// torch/csrc/jit/codegen/cuda/scheduler/utils.cpp as making new CPP files is
// painful for multiple reasons.

// Grab all values and expressions used to make the merged_domain and remove
// them from the fusion
void cleanUpInnermostMergedDomains(
    const std::vector<IterDomain*>& root_domain,
    IterDomain* merged_domain);

// Merge innermost domains for finding the widest vectorizable
// size. Return the merged domain or nullptr if no merge is done.
IterDomain* mergeInnermostDomains(
    const std::vector<IterDomain*>& domain,
    int num_merged_domains);

//! Attempt to expand vectorized domains to contig merged domains. Break point
//! identifies the point in which you can't propagate contiguous merges. For
//! example in pointwise this is the point where we want to split the
//! parallelization to take advantage of broadcast, and for reduction schedulers
//! it's the point where we switch from a reduction domain to an iter domain (or
//! vice versa).
size_t expandVectorizationToContigMergedDomains(
    Fusion* fusion,
    SchedulerRuntimeInfo& runtime_info,
    const std::vector<TensorView*> vectorizable_inputs_outputs,
    TensorView* reference_tv,
    int break_point,
    size_t default_word_size);

} // namespace scheduler_utils
} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
