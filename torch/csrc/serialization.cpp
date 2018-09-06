#include "torch/csrc/python_headers.h"
#include <system_error>

#include "THP.h"
#include "serialization.h"

static ssize_t doPythonReadBuffered(PyObject* fildes, void* buf, size_t nbytes);
static ssize_t doPythonReadInto(PyObject* fildes, void* buf, size_t nbytes);
static ssize_t doPythonWrite(PyObject* fildes, void* buf, size_t nbytes);

template <>
ssize_t doRead<int>(int fildes, void* buf, size_t nbytes) {
  return read(fildes, buf, nbytes);
}

template <>
ssize_t doRead<PyObject*>(PyObject* fildes, void* buf, size_t nbytes) {
  // Try to use fildes.readinto() instead of fildes.read()
  // because it is more memory efficient.
  auto has_readinto = PyObject_HasAttrString(fildes, "readinto") == 1;
  if (has_readinto) {
    return doPythonReadInto(fildes, buf, nbytes);
  }
  return doPythonReadBuffered(fildes, buf, nbytes);
}

template <>
ssize_t doWrite<int>(int fildes, void* buf, size_t nbytes) {
  return write(fildes, buf, nbytes);
}

template <>
ssize_t doWrite<PyObject*>(PyObject* fildes, void* buf, size_t nbytes) {
  return doPythonWrite(fildes, buf, nbytes);
}

static inline bool isUnsupportedOperation() {
  THPObjectPtr io(PyImport_ImportModule("io"));
  if (!io) throw python_error();
  THPObjectPtr exception(PyObject_GetAttrString(io, "UnsupportedOperation"));
  if (!exception) python_error();
  return PyErr_ExceptionMatches(exception.get());
}

// Call Python fildes.read(nbytes) and copy it to buf.
static inline ssize_t doPythonReadBuffered(PyObject* fildes, void* buf, size_t nbytes) {
  const size_t buffer_size = 262144;  // 2^18
  size_t read_bytes = 0;

  while (read_bytes < nbytes) {
    auto remaining = nbytes - read_bytes;
    auto to_read = remaining > buffer_size ? buffer_size : remaining;
    THPObjectPtr r(PyObject_CallMethod(fildes, "read", "i", to_read));
    if (!r) throw python_error();

    // read output is String (Python 2) / Bytes (Python 3)
#if PY_MAJOR_VERSION >= 3
    auto size = PyBytes_GET_SIZE(r.get());
    const void* bytes = PyBytes_AsString(r.get());
#else
    auto size = PyString_GET_SIZE(r.get());
    const void* bytes = PyString_AsString(r.get());
#endif

    // we read EOF
    if (size == 0) {
      return read_bytes;
    }

    memcpy(reinterpret_cast<char*>(buf) + read_bytes, bytes, size);
    read_bytes += size;
  } // Reading loop

  return read_bytes;
}

// Either does fildes.readinto(buf) or fildes.write(buf)
static inline ssize_t doPythonIO(PyObject* fildes, void* buf, size_t nbytes, bool is_read) {
#if PY_MAJOR_VERSION >= 3
  auto rw_flag = is_read ? PyBUF_WRITE : PyBUF_READ;
  THPObjectPtr memview(PyMemoryView_FromMemory(
      reinterpret_cast<char*>(buf), nbytes, rw_flag));
#else
  THPObjectPtr memview(PyBuffer_FromReadWriteMemory(buf, nbytes));
#endif
  if (!memview) throw python_error();

  char* method = "write";
  if (is_read) {
    method = "readinto";
  }
  THPObjectPtr r(PyObject_CallMethod(fildes, method, "O", memview.get()));
  if (r) {
    return PyLong_AsSsize_t(r.get());
  }

  // fildes.readinto can return UnsupportedOperation so fall back to fildes.read.
  if (is_read && isUnsupportedOperation()) {
    PyErr_Clear();
    return doPythonReadBuffered(fildes, buf, nbytes);
  }
  throw python_error();
}

// Call Python fildes.readinto(buf)
static ssize_t doPythonReadInto(PyObject* fildes, void* buf, size_t nbytes) {
  return doPythonIO(fildes, buf, nbytes, /* is_read */ true);
}

// Call Python fildes.write(buf)
static ssize_t doPythonWrite(PyObject* fildes, void* buf, size_t nbytes) {
  return doPythonIO(fildes, buf, nbytes, /* is_read */ false);
}

#include "generic/serialization.cpp"
#include <TH/THGenerateAllTypes.h>

#include "generic/serialization.cpp"
#include <TH/THGenerateHalfType.h>
