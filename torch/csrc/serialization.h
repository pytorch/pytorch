#ifndef THP_SERIALIZATION_INC
#define THP_SERIALIZATION_INC

#include "generic/serialization.h"
#include <TH/THGenerateAllTypes.h>

#include "generic/serialization.h"
#include <TH/THGenerateHalfType.h>

#include "python_serialization.h"

template <class io>
inline ssize_t doRead(io fildes, void* buf, size_t nbytes);

template <>
inline ssize_t doRead<int>(int fildes, void* buf, size_t nbytes) {
  return read(fildes, buf, nbytes);
}

template <>
inline ssize_t doRead<PyObject*>(PyObject* fildes, void* buf, size_t nbytes) {
  // Try to use fildes.readinto() instead of fildes.read()
  // because it is more memory efficient.
  auto has_readinto = PyObject_HasAttrString(fildes, "readinto") == 1;
  if (has_readinto) {
    return pyReadInto(fildes, buf, nbytes);
  }
  return pyReadBuffered(fildes, buf, nbytes);
}

template <class io>
inline ssize_t doWrite(io fildes, void* buf, size_t nbytes);

template <>
inline ssize_t doWrite<int>(int fildes, void* buf, size_t nbytes) {
  return write(fildes, buf, nbytes);
}

template <>
inline ssize_t doWrite<PyObject*>(PyObject* fildes, void* buf, size_t nbytes) {
  return pyWrite(fildes, buf, nbytes);
}

#endif
