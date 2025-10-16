#pragma once

#ifdef _WIN32

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Use a void* to avoid exposing the internal _PyStackRef union on this
// translation unit
PyObject* Torch_PyStackRef_AsPyObjectBorrow(void* stackref);

#ifdef __cplusplus
}
#endif  // #ifdef __cplusplus

#endif  // #ifdef _WIN32