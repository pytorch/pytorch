#include <Python.h>

extern PyObject* initModule(void);

// Forward decl for the Rust crate's PyO3-generated module init. When the Rust
// crate is link_whole'd into _C (Buck fbcode/xplat/ovrsource builds), this
// resolves to the real symbol defined in the same module as the call site.
// When it isn't (OSS builds, or configs that ship _rust as a separate
// extension), it must resolve to a NULL-returning fallback so _C still links
// and we simply skip attaching the submodule.
//
// This has to live in stub.c (in _C.so) rather than Module.cpp (in _C_impl.so
// in split-shared builds), because _C_impl.so's symbol lookup doesn't see
// _C.so's symbols.
#ifdef _WIN32
// MSVC has no __attribute__((weak)). /alternatename redirects PyInit__rust to
// the fallback only when PyInit__rust is otherwise undefined at link time,
// which is the portable equivalent of a weak symbol. x64/arm64 use undecorated
// C symbol names; 32-bit x86 prepends a leading underscore.
extern PyObject* PyInit__rust(void);
PyObject* PyInit__rust_weak_fallback(void) {
  return NULL;
}
#if defined(_WIN64)
#pragma comment(linker, "/alternatename:PyInit__rust=PyInit__rust_weak_fallback")
#else
#pragma comment( \
    linker, "/alternatename:_PyInit__rust=_PyInit__rust_weak_fallback")
#endif
#else
__attribute__((weak)) PyObject* PyInit__rust(void);
#endif

#ifndef _WIN32
#ifdef __cplusplus
extern "C"
#endif
__attribute__((visibility("default"))) PyObject* PyInit__C(void);
#endif

PyMODINIT_FUNC PyInit__C(void)
{
  PyObject* mod = initModule();
  if (!mod) {
    return mod;
  }
#ifndef _WIN32
  // The weak symbol is NULL when the Rust crate isn't linked into _C; on
  // Windows the /alternatename fallback guarantees a callable symbol instead.
  if (!PyInit__rust) {
    return mod;
  }
#endif
  PyObject* rust = PyInit__rust();
  if (!rust) {
    // A NULL return with a Python exception set is a genuine init failure;
    // propagate it. The Windows /alternatename fallback returns NULL *without*
    // setting an exception to signal "rust not linked in" -- skip in that case.
    if (PyErr_Occurred()) {
      Py_DECREF(mod);
      return NULL;
    }
    return mod;
  }
  // PyModule_AddObjectRef does not steal `rust`, so we always drop our own
  // reference afterwards. On failure it sets an exception; propagate it.
  if (PyModule_AddObjectRef(mod, "_rust", rust) < 0) {
    Py_DECREF(rust);
    Py_DECREF(mod);
    return NULL;
  }
  Py_DECREF(rust);
  return mod;
}
