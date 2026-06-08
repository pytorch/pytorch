// Safe Python.h include for MSVC Debug builds.
//
// On MSVC with _DEBUG defined (Debug configuration), Python.h emits
// #pragma comment(lib, "python3XX_d.lib") and exposes debug-only CPython APIs.
// When building against a release Python (no Py_DEBUG), this causes link errors.
// This header temporarily undefines _DEBUG around #include <Python.h>,
// matching the same approach used by pybind11.

#pragma once

#if defined(_MSC_VER) && defined(_DEBUG) && !defined(Py_DEBUG)
#  pragma push_macro("_DEBUG")
#  undef _DEBUG
#  include <Python.h>
#  pragma pop_macro("_DEBUG")
#else
#  include <Python.h>
#endif
