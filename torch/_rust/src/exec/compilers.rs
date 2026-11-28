// A compiler's code runs on CPython.
//
// torch.compile and torch.export are Python programs of their own: Dynamo
// evaluates the frames of what it compiles, Inductor lowers and schedules
// graphs, AOT autograd traces them. A function the executor runs
// can call them (a module the program compiled itself, a `torch.func`
// transform). Run as the program's code, each of their code objects would be
// converted and observed, and a compiled function would be evaluated by two
// engines at once.
//
// What it is given
//   - a code object the frame hook is about to look up, or a call site's
//     clone is about to be made for (mod.rs's compiled and new_clones).
// What it guarantees
//   - is(): whether the code is a compiler's, from the file it was written
//     in, decided once per code. mark() is what the code's extra slot holds
//     from then on in place of a Compiled: nothing is converted for it;
//   - the frame hook runs such a frame, and every frame it enters, on
//     CPython (mod.rs's eval_frame). The instruction that entered it is
//     seen to have entered a frame, and the optimizer finds no verdict for
//     that code: the call is not movable, and a traced body that comes to
//     one all the same is refused.
// What it may not assume
//   - that a compiler's code runs no code of the program: it does (the
//     function Dynamo compiles, the function a transform is applied to).
//     Those frames are CPython's too, by this rule: under a compiler the
//     program runs as the compiler arranged it.
// Which
//   - torch._dynamo, torch._inductor, torch._functorch: the three parts of
//     torch.compile, and with torch._functorch the `torch.func` transforms,
//     which interpret the function they are given through layers of their
//     own;
//   - torch.export and torch._export: tracing front ends over the above.
// Which not
//   - the wrapper by which a function runs with Dynamo off
//     (`torch._dynamo.disable`, which `torch.utils.checkpoint.checkpoint`
//     and torch.cond's eager path are decorated with): it is no compiler
//     at work, it calls the program's function, which stays the
//     executor's;
//   - torch.jit: `torch.jit.is_tracing()` is a question ordinary tensor
//     code asks (torchvision's operators before every call), and as a
//     compiler's code each such call would end a region; what torch.jit
//     compiles runs outside Python;
//   - torch.fx: its tracer is the region tracer's own machinery while a
//     body is traced (see tracer_code), and outside a trace the file names
//     of torch.fx do not tell its tracer from what runs the modules it made
//     (GraphModule's call, Interpreter), which is the program's tensor code.

use std::ffi::c_void;

use pyo3::prelude::*;

// The extra-slot value of a compiler's code.
static MARK: u8 = 0;

pub fn mark() -> *mut c_void {
    &MARK as *const u8 as *mut c_void
}

// The packages of torch whose code is a compiler's, as paths under torch/.
const PACKAGES: &[&str] = &["_dynamo/", "_inductor/", "_functorch/", "export/", "_export/"];
// The wrapper `torch._dynamo.disable` makes: (file under torch/, qualname).
const DISABLED: (&str, &str) = ("_dynamo/eval_frame.py", "DisableContext.__call__.<locals>._fn");

// Whether `code` (a code object) was written in one of PACKAGES and is not
// the wrapper DISABLED.
pub fn is(code: &Bound<'_, PyAny>) -> bool {
    let Ok(file) = code.getattr("co_filename").and_then(|f| f.extract::<String>()) else {
        return false;
    };
    let Some(at) = file.rfind("/torch/") else { return false };
    let rest = &file[at + 7..];
    PACKAGES.iter().any(|p| rest.starts_with(p)) && !(rest == DISABLED.0 && code.getattr("co_qualname").is_ok_and(|q| q.to_string() == DISABLED.1))
}
