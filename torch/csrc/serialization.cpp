#include <torch/csrc/python_headers.h>
#include <system_error>

#include <torch/csrc/THP.h>
#include <torch/csrc/serialization.h>

template <class io>
ssize_t doPartialRead(io fildes, void* buf, size_t nbytes);

template <class io>
ssize_t doPartialWrite(io fildes, void* buf, size_t nbytes);

static ssize_t doPartialPythonReadBuffered(PyObject* fildes, void* buf, size_t nbytes);
static ssize_t doPartialPythonReadInto(PyObject* fildes, void* buf, size_t nbytes);
static ssize_t doPartialPythonWrite(PyObject* fildes, void* buf, size_t nbytes);

template <>
ssize_t doPartialRead<int>(int fildes, void* buf, size_t nbytes) {
  return read(fildes, buf, nbytes);
}

template <>
ssize_t doPartialRead<PyObject*>(PyObject* fildes, void* buf, size_t nbytes) {
  // Try to use fildes.readinto() instead of fildes.read()
  // because it is more memory efficient.
  // TODO: Stop calling PyObject_HasAttrString() in a loop on our read loop
  auto has_readinto = PyObject_HasAttrString(fildes, "readinto") == 1;
  if (has_readinto) {
    return doPartialPythonReadInto(fildes, buf, nbytes);
  }
  return doPartialPythonReadBuffered(fildes, buf, nbytes);
}

template <>
ssize_t doPartialWrite<int>(int fildes, void* buf, size_t nbytes) {
  return write(fildes, buf, nbytes);
}

template <>
ssize_t doPartialWrite<PyObject*>(PyObject* fildes, void* buf, size_t nbytes) {
  return doPartialPythonWrite(fildes, buf, nbytes);
}

static inline bool isUnsupportedOperation() {
  THPObjectPtr io(PyImport_ImportModule("io"));
  if (!io) throw python_error();
  THPObjectPtr exception(PyObject_GetAttrString(io, "UnsupportedOperation"));
  if (!exception) throw python_error();
  return PyErr_ExceptionMatches(exception.get());
}

// Call Python fildes.read(nbytes) and copy it to buf.
static inline ssize_t doPartialPythonReadBuffered(PyObject* fildes, void* buf, size_t raw_nbytes) {
  // If we request a large amount of data, f.read() will internally try to
  // allocate a buffer of that size.  This is counterproductive, because
  // it's not the buffer we ultimately want to write the data into.  Read
  // less than that and avoid allocating too much extra memory.
  // TODO: Maybe 260 KB is a bit small...
  const size_t nbytes = std::min<size_t>(raw_nbytes, 262144u); // 2^18 (~260 KB)

  THPObjectPtr r(PyObject_CallMethod(fildes, "read", "i", nbytes));
  if (!r) throw python_error();

  // read output is String (Python 2) / Bytes (Python 3)
#if PY_MAJOR_VERSION >= 3
  auto size = PyBytes_GET_SIZE(r.get());
  const void* py_buf = PyBytes_AsString(r.get());
#else
  auto size = PyString_GET_SIZE(r.get());
  const void* py_buf = PyString_AsString(r.get());
#endif

  // we read EOF
  if (size == 0) {
    return 0;
  }

  // Slurp it into the buffer we actually want
  memcpy(buf, py_buf, size);

  return size;
}

// Either does fildes.readinto(buf) or fildes.write(buf)
static inline ssize_t doPartialPythonIO(PyObject* fildes, void* buf, size_t nbytes, bool is_read) {
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
    return doPartialPythonReadBuffered(fildes, buf, nbytes);
  }
  throw python_error();
}

// Call Python fildes.readinto(buf)
static ssize_t doPartialPythonReadInto(PyObject* fildes, void* buf, size_t nbytes) {
  return doPartialPythonIO(fildes, buf, nbytes, /* is_read */ true);
}

// Call Python fildes.write(buf)
static ssize_t doPartialPythonWrite(PyObject* fildes, void* buf, size_t nbytes) {
  return doPartialPythonIO(fildes, buf, nbytes, /* is_read */ false);
}

// Requires that we read EXACTLY nbytes; fails if we don't.
template <typename io>
void doRead(io fildes, void* raw_buf, size_t nbytes) {
  char* buf = static_cast<char*>(raw_buf);
  while (nbytes > 0) {
    errno = 0; // doPartialRead may not set errno
    // we read in 1GB blocks to avoid bugs on Mac OS X Lion
    // see https://github.com/pytorch/pytorch/issues/1031 for more details
    ssize_t r = doPartialRead(fildes, buf, std::min<size_t>(nbytes, 1073741824));
    if (r < 0) {
      int err = errno;
      AT_ASSERTM(err != 0, "read(): impossible! r < 0, but no errno was set");
      AT_ASSERTM(err != EAGAIN, "read(): non-blocking fd ", fildes,
                                " read EAGAIN; cowardly refusing to spin-wait");
      if (err == EINTR) {
        continue;
      } else {
        AT_ERROR("read(): fd ", fildes, " failed with ", strerror(err));
      }
    } else if (r == 0) {
      break;
    }
    buf += r;
    // This is guaranteed by POSIX, but I just want to be double-sure
    // to not underflow a signed integer.
    AT_ASSERT(static_cast<size_t>(r) <= nbytes);
    nbytes -= r;
  }
  if (nbytes != 0) {
    AT_ERROR("unexpected EOF, expected ", nbytes, " more bytes. The file might be corrupted.");
  }
}

template <typename io>
void doWrite(io fildes, void* raw_buf, size_t nbytes) {
  char* buf = static_cast<char*>(raw_buf);
  while (nbytes > 0) {
    errno = 0; // doPartialWrite may not set errno
    // we write in 1GB blocks to avoid bugs on Mac OS X Lion
    // see https://github.com/pytorch/pytorch/issues/1031 for more details
    ssize_t r = doPartialWrite(fildes, buf, std::min<size_t>(nbytes, 1073741824));
    if (r < 0) {
      int err = errno;
      AT_ASSERTM(err != 0, "write(): impossible! r < 0, but no errno was set");
      AT_ASSERTM(err != EAGAIN, "write(): non-blocking fd ", fildes,
                                " read EAGAIN; cowardly refusing to spin-wait");
      if (err == EINTR) {
        continue;
      } else {
        AT_ERROR("write(): fd ", fildes, " failed with ", strerror(err));
      }
    }
    buf += r;
    AT_ASSERT(static_cast<size_t>(r) <= nbytes);
    nbytes -= r;
  }
}

#include <torch/csrc/generic/serialization.cpp>
#include <TH/THGenerateAllTypes.h>

#include <torch/csrc/generic/serialization.cpp>
#include <TH/THGenerateHalfType.h>

#include <torch/csrc/generic/serialization.cpp>
#include <TH/THGenerateBFloat16Type.h>

#include <torch/csrc/generic/serialization.cpp>
#include <TH/THGenerateBoolType.h>

#include <torch/csrc/generic/serialization.cpp>
#include <TH/THGenerateQTypes.h>
