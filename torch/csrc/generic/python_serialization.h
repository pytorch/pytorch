#pragma once
#include <Python.h>

// Call Python fildes.read(nbytes) and copy it to buf.
// Requires a buffer for the bytes object returned from read
// so the reading is chunked.
inline ssize_t pyReadBuffered(PyObject* fildes, void* buf, size_t nbytes) {
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

static inline bool isUnsupportedOperation() {
  THPObjectPtr io(PyImport_ImportModule("io"));
  if (!io) throw python_error();
  THPObjectPtr exception(PyObject_GetAttrString(io, "UnsupportedOperation"));
  if (!exception) python_error();
  return PyErr_ExceptionMatches(exception.get());
}

// Either does fildes.readinto(buf) or fildes.write(buf)
static inline ssize_t io(PyObject* fildes, void* buf, size_t nbytes, bool is_read) {
#if PY_MAJOR_VERSION >= 3
  auto rw_flag = is_read ? PyBUF_WRITE : PyBUF_READ;
  THPObjectPtr memview(PyMemoryView_FromMemory(
      reinterpret_cast<char*>(buf), nbytes, rw_flag));
#else
  // PyMemoryView_FromMemory doesn't exist in Python 2.7, so we manually
  // create a Py_buffer that describes the memory and create a memoryview from it.
  auto readonly_flag = is_read ? 1 : 0;
  Py_buffer pyBuf;
  pyBuf.buf = buf;
  pyBuf.obj = nullptr;
  pyBuf.len = (Py_ssize_t)nbytes;
  pyBuf.itemsize = 1;
  pyBuf.readonly = readonly_flag;
  pyBuf.ndim = 0;
  pyBuf.format = nullptr;
  pyBuf.shape = nullptr;
  pyBuf.strides = nullptr;
  pyBuf.suboffsets = nullptr;
  pyBuf.internal = nullptr;

  THPObjectPtr memview(PyMemoryView_FromBuffer(&pyBuf));
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
    return pyReadBuffered(fildes, buf, nbytes);
  }
  throw python_error();
}

// Call Python fildes.readinto(buf)
inline ssize_t pyReadInto(PyObject* fildes, void* buf, size_t nbytes) {
  return io(fildes, buf, nbytes, /* is_read */ true);
}

// Call Python fildes.write(buf)
inline ssize_t pyWrite(PyObject* fildes, void* buf, size_t nbytes) {
  return io(fildes, buf, nbytes, /* is_read */ false);
}
