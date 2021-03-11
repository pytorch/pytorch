#pragma once

#include <torch/csrc/WindowsTorchApiMacro.h>

#include <torch/csrc/jit/codegen/cuda/dispatch.h>
#include <torch/csrc/jit/codegen/cuda/ir_all_nodes.h>
#include <torch/csrc/jit/codegen/cuda/kernel_ir.h>

#include <unordered_set>

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {

//! Detect all IterDomains that are derived only from trivial
//! reductons, thus not necessary to appear in the final generated
//! kernel. The returned set includes all domains from root to
//! leaves. It also can include non-reduction, rfactor domains.
std::unordered_set<IterDomain*> detectTrivialReductionDerivedDomains(
    Fusion* fusion);

} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
