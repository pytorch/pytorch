#include <torch/csrc/utils/python_compat.h>

namespace torch::impl {

struct OwnedPyObjectPtr;

/// Represents a borrowed PyObject*. The object won't incref'd on acquire or
/// decref'd on release.
struct BorrowedPyObjectPtr {
  ~BorrowedPyObjectPtr() = default;
  BorrowedPyObjectPtr() : m_value(nullptr) {}
  BorrowedPyObjectPtr(PyObject* p) : m_value(p) {}
  BorrowedPyObjectPtr(const BorrowedPyObjectPtr& o) = default;
  BorrowedPyObjectPtr(BorrowedPyObjectPtr&& o) = default;
  BorrowedPyObjectPtr& operator=(const BorrowedPyObjectPtr& o) = default;
  BorrowedPyObjectPtr& operator=(BorrowedPyObjectPtr&& o) = default;

  static BorrowedPyObjectPtr none() {
    return BorrowedPyObjectPtr(Py_None);
  }

  static BorrowedPyObjectPtr false_() {
    return BorrowedPyObjectPtr(Py_False);
  }

  static BorrowedPyObjectPtr true_() {
    return BorrowedPyObjectPtr(Py_True);
  }

  PyObject* ptr() const {
    return m_value;
  }

  /// Take ownership of the borrowed reference.
  OwnedPyObjectPtr own() const;

 private:
  PyObject* m_value;
};

/// Represents an owned PyObject*. You must be explicit on acquire (either with
/// OwnedPyObjectPtr::own() or OwnedPyObjectPtr::steal()). Releases the
/// reference on release.
struct OwnedPyObjectPtr {
  ~OwnedPyObjectPtr() {
    Py_CLEAR(m_value);
  }

  OwnedPyObjectPtr() : m_value(nullptr) {}

  OwnedPyObjectPtr(const OwnedPyObjectPtr& o)
      : m_value(Py_XNewRef(o.m_value)) {}

  OwnedPyObjectPtr(OwnedPyObjectPtr&& o) noexcept : m_value(nullptr) {
    std::swap(m_value, o.m_value);
  }

  OwnedPyObjectPtr& operator=(const OwnedPyObjectPtr& o) {
    if (this != &o) {
      Py_XSETREF(m_value, Py_XNewRef(o.m_value));
    }
    return *this;
  }

  OwnedPyObjectPtr& operator=(OwnedPyObjectPtr&& o) noexcept {
    std::swap(m_value, o.m_value);
    Py_CLEAR(o.m_value);
    return *this;
  }

  OwnedPyObjectPtr(const BorrowedPyObjectPtr& o)
      : m_value(Py_XNewRef(o.ptr())) {}

  OwnedPyObjectPtr& operator=(const BorrowedPyObjectPtr& o) {
    Py_XSETREF(m_value, Py_XNewRef(o.ptr()));
    return *this;
  }

  /// Take ownership of the passed PyObject*.
  static OwnedPyObjectPtr steal(PyObject* o) {
    return OwnedPyObjectPtr(o);
  }

  /// Take ownership of the passed PyObject* (incrementing the refcount).
  static OwnedPyObjectPtr own(PyObject* o) {
    return OwnedPyObjectPtr(Py_XNewRef(o));
  }

  static OwnedPyObjectPtr none() {
    return OwnedPyObjectPtr(Py_None);
  }

  static OwnedPyObjectPtr false_() {
    return OwnedPyObjectPtr(Py_False);
  }

  static OwnedPyObjectPtr true_() {
    return OwnedPyObjectPtr(Py_True);
  }

  PyObject* ptr() const {
    return m_value;
  }

  PyObject* release() {
    PyObject* tmp = nullptr;
    std::swap(tmp, m_value);
    return tmp;
  }

 private:
  PyObject* m_value;

  OwnedPyObjectPtr(PyObject* value) : m_value(value) {}
};

inline OwnedPyObjectPtr BorrowedPyObjectPtr::own() const {
  return OwnedPyObjectPtr::own(m_value);
}

} // namespace torch::impl
