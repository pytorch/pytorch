#pragma once

#include <c10/util/Exception.h>

/// This file provides some simple utilities for detecting common deadlocks in
/// PyTorch.  For now, we focus exclusively on detecting Python GIL deadlocks,
/// as the GIL is a wide ranging lock that is taken out in many situations.
/// The basic strategy is before performing an operation that may block, you
/// can use TORCH_ASSERT_NO_GIL_WITHOUT_PYTHON_DEP() to assert that the GIL is
/// not held.  This macro is to be used in contexts where no static dependency
/// on Python is available (we will handle indirecting a virtual call for you).
///
/// If the GIL is held by a torchdeploy interpreter, we always report false.
/// If you are in a context where Python bindings are available, it's better
/// to directly assert on PyGILState_Check (as it avoids a vcall and also
/// works correctly with torchdeploy.)

namespace c10 {

#define TORCH_ASSERT_NO_GIL_WITHOUT_PYTHON_DEP() \
  TORCH_INTERNAL_ASSERT(                         \
      !c10::impl::check_python_gil(),            \
      "Holding GIL before a blocking operation!  Please release the GIL before blocking, or see https://github.com/pytorch/pytorch/issues/56297 for how to release the GIL for destructors of objects")

namespace impl {

C10_API bool check_python_gil();

struct C10_API PythonGILHooks {
  virtual ~PythonGILHooks() = default;
  // Returns true if we hold the GIL.  If not linked against Python we
  // always return false.
  virtual bool check_python_gil() const = 0;
};

C10_API void SetPythonGILHooks(PythonGILHooks* factory);

// DO NOT call this registerer from a torch deploy instance!  You will clobber
// other registrations
struct C10_API PythonGILHooksRegisterer {
  explicit PythonGILHooksRegisterer(PythonGILHooks* factory) {
    SetPythonGILHooks(factory);
  }
  ~PythonGILHooksRegisterer() {
    SetPythonGILHooks(nullptr);
  }
};

} // namespace impl
} // namespace c10
