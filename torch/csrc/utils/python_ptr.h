#include <torch/csrc/utils/python_compat.h>
#include <stdexcept>

namespace torch::pyptr {

struct Owned;

/// Represents a borrowed PyObject*. The object won't incref'd on acquire or
/// decref'd on release.
struct Borrowed {
  ~Borrowed() = default;
  Borrowed() : m_value(nullptr) {}
  Borrowed(PyObject* p) : m_value(p) {}
  Borrowed(const Borrowed& o) = default;
  Borrowed& operator=(const Borrowed& o) = default;

  Borrowed(Borrowed&& o) = delete;
  Borrowed& operator=(Borrowed&& o) = delete;

  static Borrowed none() {
    return Borrowed(Py_None);
  }

  static Borrowed false_() {
    return Borrowed(Py_False);
  }

  static Borrowed true_() {
    return Borrowed(Py_True);
  }

  PyObject* ptr() const {
    return m_value;
  }

  Py_hash_t hash() const {
    return PyObject_Hash(m_value);
  }

  bool equal(Borrowed other) const {
    auto result = PyObject_RichCompareBool(m_value, other.m_value, Py_EQ);
    switch (result) {
      case 0:
        return false;
      case 1:
        return true;
      default:
        throw std::runtime_error("PyObject_RichCompareBool failed");
    }
  }

  /// Take ownership of the borrowed reference.
  Owned to_owned() const;

 protected:
  PyObject* m_value;
};

/// Represents an owned PyObject*. You must be explicit on acquire (either with
/// Owned::own() or Owned::steal()). Releases the
/// reference on release.
struct Owned : Borrowed {
  ~Owned() {
    Py_CLEAR(m_value);
  }

  Owned() = default;

  Owned(const Owned& o) : Borrowed(o) {
    Py_XINCREF(m_value);
  }

  Owned(Owned&& o) noexcept : Borrowed() {
    std::swap(m_value, o.m_value);
  }

  Owned& operator=(const Owned& o) {
    if (this != &o) {
      Py_XSETREF(m_value, Py_XNewRef(o.m_value));
    }
    return *this;
  }

  Owned& operator=(Owned&& o) noexcept {
    std::swap(m_value, o.m_value);
    Py_CLEAR(o.m_value);
    return *this;
  }

  /// Take ownership of the passed PyObject* without incrementing the refcount.
  static Owned from_owned_ptr(PyObject* o) {
    return Owned(o);
  }

  /// Take ownership of the passed PyObject* incrementing the refcount.
  static Owned from_borrowed_ptr(PyObject* o) {
    return Owned(Py_XNewRef(o));
  }

  static Owned none() {
    return Borrowed::none().to_owned();
  }

  static Owned false_() {
    return Borrowed::false_().to_owned();
  }

  static Owned true_() {
    return Borrowed::true_().to_owned();
  }

  /// Release the internal PyObject* without modifying the refcount.
  PyObject* release() {
    PyObject* tmp = nullptr;
    std::swap(tmp, m_value);
    return tmp;
  }

 private:
  Owned(PyObject* value) : Borrowed(value) {}
};

inline Owned Borrowed::to_owned() const {
  return Owned::from_borrowed_ptr(m_value);
}

} // namespace torch::pyptr
