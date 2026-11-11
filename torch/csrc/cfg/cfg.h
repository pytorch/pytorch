#pragma once

// C++ side of torch.cfg_compile.
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
//   - tensor inspection for shape guards.

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
};

#define TCFG_API_VERSION 1

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
  // operands as in the TcfgPyOp the site was built from (borrowed). Returns a
  // new reference, or null with *status set.
  PyObject* (*site_call)(TcfgSite*, PyObject* const* operands, int32_t* status);
  // New reference: dict describing the site.
  PyObject* (*site_info)(const TcfgSite*);

  // Returns 0 if obj is not exactly a Tensor or a Parameter.
  int32_t (*tensor_info)(PyObject* obj, TcfgTensorInfo* out);
};

} // extern "C"

namespace torch::cfg {

void initCfgBindings(PyObject* module);

} // namespace torch::cfg
