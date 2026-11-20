#pragma once

// C++ side of torch.vm.
//
// The compiler and its VM live in a Rust extension module
// (torch/csrc/cfg/rust). Everything that has to touch the dispatcher lives
// here and is handed to Rust as a table of C function pointers (TcfgApi)
// through a capsule, so the Rust crate does not link against libtorch.
//
// The table covers three things:
//   - capture: run one Python-level operation with a DispatchCaptureSink
//     installed and get back the tree of kernel invocations it made;
//   - sites: turn such a trace into a prepared call that feeds the operands of
//     the Python-level operation straight into the post-dispatch kernel,
//     guarded on the dispatch keys of its tensor operands;
//   - tensor inspection for shape guards;
//   - unboxed tensors: a kernel result does not need a Python object until
//     Python code looks at it. The VM can ask for results as tagged pointers,
//     TensorImpl* | 1, each an owned reference, pass them back as operands,
//     and box them when it has to.

#include <torch/csrc/python_headers.h>

#include <cstddef>
#include <cstdint>

extern "C" {

struct TcfgTrace;
struct TcfgSite;

enum TcfgStatus : int32_t {
  TCFG_OK = 0,
  // A guard of the site did not hold. No Python error is set; the caller runs
  // the generic operation instead.
  TCFG_MISS = 1,
  // The kernel raised. A Python error is set.
  TCFG_ERROR = 2,
};

enum TcfgPyOpKind : int32_t {
  // operands = [callee, positional..., keyword values...]
  TCFG_CALL = 0,
  // operands = [receiver, positional..., keyword values...]; callee is the
  // bound method that was called.
  TCFG_METHOD = 1,
  // operands = [lhs, rhs]
  TCFG_BINARY = 2,
  // operands = [self]: unary operators, truthiness, attribute getters
  TCFG_UNARY = 3,
};

enum TcfgCallFlags : uint32_t {
  // Return a tensor result unboxed (see above) when it has no Python object.
  TCFG_UNBOXED_RESULT = 1,
};

enum TcfgPyOpFlags : uint32_t {
  // Enter below the Autograd kernel when the captured call did not need it.
  TCFG_ALLOW_SKIP_AUTOGRAD = 1,
};

// One Python-level operation, as the VM executed it.
struct TcfgPyOp {
  int32_t kind;
  int32_t n_operands;
  PyObject* const* operands;
  // Per operand: the value is a compile-time constant of the function, so it
  // can be baked into the site without a guard. May be null.
  const uint8_t* is_const;
  // Tuple of keyword names; they name the trailing operands. May be null.
  PyObject* kwnames;
  PyObject* callee;
  PyObject* result;
  uint32_t flags;
};

struct TcfgTensorInfo {
  int32_t ndim;
  int32_t requires_grad;
  const int64_t* sizes;
  uint64_t key_set;
  // What else a graph that was compiled for a tensor depends on: how it is
  // laid out in its storage (strides is null for a tensor that has none, a
  // sparse one say), and its element type and device (the device type in the
  // upper half, the index plus one in the lower).
  const int64_t* strides;
  int64_t storage_offset;
  int32_t dtype;
  int32_t device;
  // What says whether a tensor holds what it held: where its data is, and
  // how often it was written to in place. TCFG_NO_VERSION for a tensor that
  // does not count that (one made under inference mode).
  const void* data;
  uint64_t version;
};

#define TCFG_NO_VERSION UINT64_MAX

#define TCFG_API_VERSION 11

// What frame_hook registers. It is offered the frame the interpreter made
// for a call of func, with the arguments bound and nothing run. It returns 0
// to leave the frame to the interpreter, or 1 after it has run the call and
// disposed of the frame (frame_adopt and frame_pop, or frame_interpret), with
// *result a new reference, or null with a Python error set.
typedef int32_t (
    *TcfgTakeFrame)(PyObject* func, void* frame, PyObject** result);

struct TcfgApi {
  uint32_t version;

  // Starts recording the dispatcher calls of the current thread. Captures
  // nest; each capture_end returns the calls made since its capture_begin.
  void (*capture_begin)();
  // Returns null when no dispatcher call was made.
  TcfgTrace* (*capture_end)();
  void (*trace_free)(TcfgTrace*);
  // New reference: list of (op, dispatch key, depth, is_redispatch).
  PyObject* (*trace_records)(const TcfgTrace*);

  // Returns null and writes the reason when the operation is not a single
  // dispatcher call whose arguments and result line up with the operands.
  TcfgSite* (*site_from_trace)(
      const TcfgTrace*,
      const TcfgPyOp*,
      char* reason,
      size_t reason_len);
  void (*site_free)(TcfgSite*);
  // operands as in the TcfgPyOp the site was built from (borrowed); tensor
  // operands may be unboxed. Returns a new reference (unboxed if requested
  // with TCFG_UNBOXED_RESULT), or null with *status set.
  PyObject* (*site_call)(
      TcfgSite*,
      PyObject* const* operands,
      uint32_t flags,
      int32_t* status);
  // New reference: dict describing the site.
  PyObject* (*site_info)(const TcfgSite*);

  // Returns 0 if obj is not exactly a Tensor or a Parameter. obj may be an
  // unboxed tensor.
  int32_t (*tensor_info)(PyObject* obj, TcfgTensorInfo* out);

  // Gives an unboxed tensor its Python object. Consumes the reference and
  // returns a new one, or null with a Python error set.
  PyObject* (*tensor_box)(PyObject* unboxed);
  void (*tensor_retain)(PyObject* unboxed);
  void (*tensor_release)(PyObject* unboxed);

  // Interpreter frames, so that whatever looks at the Python stack while a
  // compiled function runs (a traceback, a warning, sys._getframe, the module
  // a new class is attributed to) finds the function there. Both are null
  // where the layout of CPython's frames is not known: anything but 3.12.
  //
  // Pushes a frame for func, a Python function, and makes it the current
  // frame of the thread. *instr gets the address of the frame's instruction
  // pointer and *code that of the first instruction of the function's code,
  // so the caller can keep the pointer on what it is executing. *locals gets
  // the frame's variables, one owned reference or null each, in the order of
  // the code's co_localsplusnames: the caller keeps them what the variables
  // of the function are bound to, the cells of the closure excepted, which
  // are there already. The frame counts as one Python call against the
  // recursion limit. *object gets the address of the frame's frame object,
  // which is null until something looks at the frame. Returns null with a
  // Python error set: RecursionError, or MemoryError.
  void* (*frame_push)(
      PyObject* func,
      uint16_t*** instr,
      uint16_t** code,
      PyObject*** locals,
      PyObject*** object);
  // Pops the frame, which must be the current one. A frame object that
  // something still refers to (a traceback) takes the frame's contents over,
  // its variables among them.
  void (*frame_pop)(void* frame);
  // Whether something other than the stack refers to the frame.
  int32_t (*frame_held)(void* frame);
  // Makes a frame that is on the stack of the thread the current one and
  // returns the one that was, for the same call to put back: what runs in
  // between finds itself called by the function of that frame, whatever
  // frames were pushed since.
  void* (*frame_swap)(void* frame);
  // Whether the thread has a trace function (sys.settrace): the interpreter
  // calls it for the frames it enters and the lines it runs in them.
  int32_t (*frame_traced)();

  // The frames the interpreter makes itself: for the calls that C code makes
  // (a constructor, a property, an operator, a callback), and for those that
  // a function makes which the interpreter runs. frame_hook says who is
  // offered them, once. They are offered while some thread wants them:
  // frame_want(1), and frame_want(-1) when it no longer does. The
  // interpreter's own calls are slower meanwhile, since it then starts every
  // frame through this.
  void (*frame_hook)(TcfgTakeFrame take);
  void (*frame_want)(int32_t delta);
  // Makes a frame that was offered the current one of the thread, as
  // frame_push does for one it makes; frame_pop disposes of it. Returns 0
  // with RecursionError set, and the frame gone.
  int32_t (*frame_adopt)(
      void* frame,
      uint16_t*** instr,
      uint16_t** code,
      PyObject*** locals,
      PyObject*** object);
  // Has the interpreter run a frame that was offered. Returns a new
  // reference, or null with a Python error set.
  PyObject* (*frame_interpret)(void* frame);
  // For a frame that is about to return result and has a frame object: if
  // a debugger put its trace function on the frame (it does so for every
  // frame of the stack it is entered under), tells it of the return, as the
  // interpreter does. Returns -1 with a Python error set if it raised.
  int32_t (*frame_returns)(void* frame, PyObject* result);

  // Whether torch.jit.trace is recording on this thread: one of the things
  // nn.Module.__call__ looks at before it calls forward.
  int32_t (*is_tracing)();

  // The memory a compiled function keeps between its calls (workspace.h),
  // for the time of a call: tensors made on this thread get their memory
  // from it. Enter returns what leave is to put back.
  void* (*workspace_enter)(void* workspace);
  void (*workspace_leave)(void* workspace, void* previous);
};

} // extern "C"

namespace torch::cfg {

void initCfgBindings(PyObject* module);

} // namespace torch::cfg
