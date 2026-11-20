#pragma once

#include <rust/cxx.h>
#include <cstdint>

namespace torch::rust {

// Defined in Rust (exec/autograd.rs): what a CfgBackward node runs.
struct NodeState;

// Makes a CfgBackward node, with an edge to each of `inputs`, and gives it as
// history to `outputs`, as a custom autograd Function does. Both hold borrowed
// PyObject* tensors; `outputs` must have no history of their own. Fills
// `wrapped` with new references to the outputs to return. Returns false with
// a Python exception set on failure.
bool make_cfg_node(
    ::rust::Box<NodeState> state,
    ::rust::Slice<const std::uintptr_t> inputs,
    ::rust::Slice<const std::uintptr_t> outputs,
    ::rust::Vec<std::uintptr_t>& wrapped) noexcept;

} // namespace torch::rust
