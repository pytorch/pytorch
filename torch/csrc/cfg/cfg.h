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

#define TCFG_API_VERSION 18

// A node of the autograd graph, as the engine reads it when it takes the
// backward of a graph down (node_info).
struct TcfgNodeInfo {
  // Valid until the next node_info of the thread.
  const char* name;
  uint64_t sequence_nr;
  // How many gradients it is handed, and how many edges what it gives goes
  // along.
  uint32_t n_inputs;
  uint32_t n_edges;
  uint32_t flags;
};

enum : uint32_t {
  // Written in Python: the engine does not call it.
  TCFG_NODE_PYTHON = 1,
  // Accumulates into the .grad of a leaf.
  TCFG_NODE_LEAF = 2,
};

// How node_call, op_call and program_run run kernels. Without a flag
// op_call and program_run run them as the thread is, and node_call with
// gradients disabled, as the autograd engine calls a node.
enum : uint32_t {
  // As the autograd kernel of an operator runs what is below it: with the
  // autograd keys excluded, so that nothing is recorded on the tape. The
  // kernel that counts a write or makes a view runs for an operator that
  // has one; what the kernel of any other operator calls in turn counts no
  // write and makes no view. The grad mode is left as it is.
  TCFG_RUN_BELOW_AUTOGRAD = 1,
  // With gradients enabled: eager can differentiate what ran.
  TCFG_RUN_ON_TAPE = 2,
};

// What graph_node registers. The node of a graph calls it, with the GIL
// held, when the autograd engine hands it gradients: grads has one entry per
// root (borrowed; null where none came), outs room for one new reference per
// contribution (left null where there is none), and needed says per
// contribution whether the autograd engine asks for it: it does not where
// the gradients of some tensors only are asked for and the contribution
// goes to none of them. flags is TCFG_RUN_ON_TAPE when the backward is to
// be differentiated again (create_graph). Returns 0, or -1 with a Python
// error set, which the node throws.
typedef int32_t (*TcfgBackward)(
    void* state,
    PyObject* const* grads,
    uint32_t n_grads,
    PyObject** outs,
    uint32_t n_outs,
    const uint8_t* needed,
    uint32_t flags);
// Called with the GIL held when the autograd engine lets go of what a
// backward kept (dropped is 0: a backward that did not retain the graph has
// run; the node then raises eager's error by itself), and when the node
// dies (1).
typedef void (*TcfgRelease)(void* state, int32_t dropped);

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
  // where the layout of CPython's frames is not known: it is for 3.12, and
  // for the default build of 3.14 on 64-bit platforms other than Windows.
  //
  // Pushes a frame for func, a Python function, and makes it the current
  // frame of the thread. *instr gets the address of the frame's instruction
  // pointer and *code that of the first instruction of the function's code,
  // so the caller can keep the pointer on what it is executing. *locals gets
  // the frame's variables, one owned reference or null each (in 3.14 as the
  // stack reference that stands for it), in the order of the code's
  // co_localsplusnames: the caller keeps them what the variables
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
  // calls it for the frames it enters and the lines it runs in them. In 3.14
  // the same goes for a debugger or a coverage tool that asked sys.monitoring
  // for the functions that start, which is how pdb listens there.
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
  // interpreter does: through the thread's trace function, or in 3.14
  // through what the debugger registered with sys.monitoring for the returns
  // of the frame's code. Returns -1 with a Python error set if it raised.
  int32_t (*frame_returns)(void* frame, PyObject* result);

  // Whether torch.jit.trace is recording on this thread: one of the things
  // nn.Module.__call__ looks at before it calls forward.
  int32_t (*is_tracing)();

  // The memory a compiled function keeps between its calls (workspace.h),
  // for the time of a call: tensors made on this thread get their memory
  // from it. Enter returns what leave is to put back.
  void* (*workspace_enter)(void* workspace);
  void (*workspace_leave)(void* workspace, void* previous);

  // The modes of the thread that a graph depends on, as one number: whether
  // gradients are recorded (bit 0), inference mode (1), autocast for CUDA
  // (2) and for the CPU (3), the default dtype (bits 8 to 15) and, from bit
  // 16, how many torch function modes are active. Bits 4 to 6: what runs
  // Python code under the operations of the thread where gradients are
  // recorded: a saved-tensor hook is on (4), a node-creation hook is on (5),
  // anomaly mode is on (6).
  uint64_t (*call_modes)();

  // New reference: what a graph that has the call of a site as a node is
  // built from, a dict with the operator ("op", "overload"), per schema
  // argument where its value comes from ("args": tuples of the name, whether
  // it is keyword-only, and "operand" with the operand and its conversion,
  // "packed" with the operands of an int list, or "const" with the value as a
  // Python object), per operand what the site checks of it ("guards": "none",
  // "tensor", "exact", "is_none" or "identity"), what it returns ("returns":
  // "tensor", "bool", "int" or "float"), and whether the operator writes to an
  // argument ("writes") or draws random numbers ("random").
  PyObject* (*site_export)(const TcfgSite*);

  // How many dispatcher calls a capture saw at its top level, those of the
  // captures that began and ended inside it included.
  int32_t (*trace_calls)(const TcfgTrace*);

  // The dispatch keys the thread adds to and takes from every call: what a
  // site compares with the ones it was captured under before it runs.
  void (*dispatch_keys)(uint64_t* included, uint64_t* excluded);

  // How many frames the interpreter has started or resumed on this thread
  // while some thread wanted them offered (frame_want): calls of functions,
  // class bodies, and every resumption of a generator or coroutine. Null
  // where frame_push is.
  uint64_t (*frame_count)();

  // Whether a tensor carries a forward-mode gradient, which a graph that was
  // compiled for tensors without one would drop. obj may be an unboxed
  // tensor.
  int32_t (*tensor_forward_grad)(PyObject* obj);

  // Version 16: the backward of a graph.
  //
  // tape_begin starts taking down the kernels the current thread runs: of
  // every call of the dispatcher the first kernel below the autograd keys,
  // with what it took and what it gave. What a kernel calls in turn is not
  // taken down. Unlike capture_begin it keeps the tensors of every kernel
  // alive until tape_end. tape_end returns a new reference to a list of
  // (operator, name, arguments, result, written, random, view): the
  // operator as op_find gives it, the arguments in the order of the schema,
  // the result (a tensor, a tuple, or None), the indices of the arguments
  // the kernel writes to, whether it draws random numbers, and whether
  // what it gives is a view of its first argument. Null with an error set
  // when a kernel threw.
  void (*tape_begin)();
  PyObject* (*tape_end)();

  // The node the gradient of a tensor goes to (its grad_fn, or what
  // accumulates into its .grad), as a new reference to the Python object of
  // the node, and which of the node's inputs the tensor is. Null without an
  // error for a tensor that needs no gradient.
  PyObject* (*node_of)(PyObject* tensor, uint32_t* input_nr);
  // Returns 0 for what is no node.
  int32_t (*node_info)(PyObject* node, TcfgNodeInfo* out);
  // Where the edge of a node leads: a new reference, and which input of
  // that node. Null without an error for an edge that leads nowhere.
  PyObject* (*node_next)(PyObject* node, uint32_t edge, uint32_t* input_nr);
  // Calls a node as the autograd engine calls it: grads is a tuple with one
  // entry per input (None where no gradient came), and what the node gives
  // is brought to the sizes, dtype and device of where each edge leads
  // (validate_outputs). New reference to a tuple with one entry per edge,
  // None where the node gave nothing. The node runs on the stream its
  // forward ran on, as in the autograd engine. Where a backward for some
  // inputs only is running on the calling thread, the node is called on a
  // thread of the engine's own with the thread-local state of the caller,
  // and the nodes it makes are numbered on with the caller's: a node asks
  // the backward that runs on its thread which gradients are wanted. A
  // tape that is being taken down on the calling thread sees the kernels
  // either way.
  PyObject* (*node_call)(PyObject* node, PyObject* grads, uint32_t flags);

  // An operator by its name and overload name: a new reference to what
  // tape_end names operators with. Null with an error for none.
  PyObject* (*op_find)(const char* name, const char* overload);
  // Calls an operator with the arguments of its schema, in order.
  PyObject* (*op_call)(PyObject* op, PyObject* args, uint32_t flags);

  // A straight-line program of operators over numbered values. ops is a
  // list as the engine's joint graph describes one: per operation
  // (operator, arguments, keyword arguments, [(place, value)], single),
  // an argument being ("v", value), ("c", constant), ("l", [arguments])
  // or ("t", [arguments]). Null with the reason written when an operation
  // cannot be made one.
  void* (*program_new)(
      PyObject* ops,
      uint32_t n_values,
      char* reason,
      size_t reason_len);
  void (*program_free)(void* program);
  // Runs a program: inputs[i] is value input_values[i]; returns a new
  // reference to a tuple of the values output_values name.
  PyObject* (*program_run)(
      void* program,
      PyObject* const* inputs,
      const uint32_t* input_values,
      uint32_t n_inputs,
      const uint32_t* output_values,
      uint32_t n_outputs,
      uint32_t flags);

  // Gives the tensors of roots (a tuple of tensors that were made below the
  // autograd keys) one node, "GraphBackward", with one edge per entry of
  // targets (a tuple of tensors; the same tensor once per contribution it
  // gets), in that order. The node calls backward with state when it is
  // handed gradients. Returns 0, or -1 with an error set.
  int32_t (*graph_node)(
      PyObject* roots,
      PyObject* targets,
      void* state,
      TcfgBackward backward,
      TcfgRelease release);

  // Of the frames frame_count counts, those that a kernel enters by itself:
  // the router of an operator whose kernel is chosen in Python
  // (eager_router of torch/_native/registry.py) with what it calls, and
  // what shows a warning that a kernel gave (the warnings module). Such a
  // frame is not Python code that the caller of the operator enters, and
  // is not offered to frame_hook. Null where frame_push is.
  uint64_t (*frame_count_kernel)();

  // Whether gradients are enabled on this thread: bit 0 of call_modes, for
  // what asks for nothing else at every call.
  uint32_t (*grad_enabled)();
};

} // extern "C"

namespace torch::cfg {

void initCfgBindings(PyObject* module);

} // namespace torch::cfg
