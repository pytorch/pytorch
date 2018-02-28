#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/serialization.h"
#else

template <class io>
void THPStorage_(writeFileRaw)(THStorage *self, io fd);

template <class io>
THStorage * THPStorage_(readFileRaw)(io fd, THStorage *storage);

// Only define the following once (instead of once per storage class)
#ifndef _SERIALIZATION_H
#define _SERIALIZATION_H

#include "generic/python_serialization.h"

template <class io>
ssize_t doRead(io fildes, void* buf, size_t nbytes);

template <>
inline ssize_t doRead<int>(int fildes, void* buf, size_t nbytes) {
  return read(fildes, buf, nbytes);
}

template <>
inline ssize_t doRead<PyObject*>(PyObject* fildes, void* buf, size_t nbytes) {
  // fildews.readinto() is more efficient than filedes.read(),
  // so use it if it exists.
  auto has_readinto = PyObject_HasAttrString(fildes, "readinto") == 1;
  if (has_readinto) {
    return pyReadInto(fildes, buf, nbytes);
  }
  return pyReadBuffered(fildes, buf, nbytes);
}

template <class io>
ssize_t doWrite(io fildes, void* buf, size_t nbytes);

template <>
inline ssize_t doWrite<int>(int fildes, void* buf, size_t nbytes) {
  return write(fildes, buf, nbytes);
}

template <>
inline ssize_t doWrite<PyObject*>(PyObject* fildes, void* buf, size_t nbytes) {
  return pyWrite(fildes, buf, nbytes);
}

#endif // _SERIALIZATION_H
#endif
