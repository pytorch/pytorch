#pragma once

#include <cstdint>

namespace torch::rust {

// While enabled on a thread, every op reaching the PreDispatch key (that is,
// before autograd) is reported to Rust's on_torch_op and then continued. Ops
// that run inside a reported op are not reported. Enables nest; each needs a
// matching disable on the same thread.
void op_hook_enable();
void op_hook_disable();
// While enabled, stops (or restarts) running the hook.
void op_hook_pause(bool paused);

// The witness (src/exec/calls/witness.rs), apart from enable and pause above:
// while wanted on a thread, every op reaching the PreDispatch key at ANY depth
// is reported to Rust's on_witness_op, with no text made, and the key stays in
// the thread's include set below a reported op. The key is included iff the
// text above or the witness wants it. Returns op_hook_seeing().
bool op_hook_witness(bool wanted);
// Whether an op run now on this thread would reach the hook: false when
// something took the key out of the include set or put it into the exclude
// set (no_dispatch(), the end of a pre-dispatch mode, an op in flight that
// only the text path reported).
bool op_hook_seeing();
// How many reported ops are in flight on this thread.
std::uint32_t op_hook_depth();

// Defined by the cxx bridge (src/bindings/ffi.rs).
struct TensorIdent;
// The identity of the tensor behind a Python object while that object lives:
// its TensorImpl and its storage (0: none). Zeros for what is no tensor, told
// by the object's type alone (never runs Python code).
void tensor_ident(std::uintptr_t py_obj, TensorIdent& out) noexcept;

} // namespace torch::rust
