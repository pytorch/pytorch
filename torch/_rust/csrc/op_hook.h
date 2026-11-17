#pragma once

namespace torch::rust {

// While enabled on a thread, every op reaching the PreDispatch key (that is,
// before autograd) is reported to Rust's on_torch_op and then continued. Ops
// that run inside a reported op are not reported. Enables nest; each needs a
// matching disable on the same thread.
void op_hook_enable();
void op_hook_disable();
// While enabled, stops (or restarts) running the hook.
void op_hook_pause(bool paused);

} // namespace torch::rust
