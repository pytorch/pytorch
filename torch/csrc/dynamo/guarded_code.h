#pragma once

#ifdef __cplusplus

#include <torch/csrc/dynamo/utils.h>
#include <torch/csrc/utils/pybind.h>

extern "C" {

#endif

typedef struct GuardedCode GuardedCode;

#ifdef __cplusplus

typedef struct VISIBILITY_HIDDEN GuardedCode {
  // modified user bytecode (protected by guard_manager's guards)
  py::object code;
  // check the guards: lambda: <locals of user function>: bool
  py::object guard_manager;
  // CompileId corresponding to this compilation
  py::object compile_id;
  // Reference to string representation of the CompileContext
  std::string trace_annotation{"Unknown"};
} GuardedCode;

} // extern "C"
#endif
