#include <torch/csrc/python_dimname.h>
#include <torch/csrc/Exceptions.h>
#include <torch/csrc/utils/python_strings.h>
#include <c10/util/flat_hash_map.h>

namespace torch {

struct InternedStringsTable {
  InternedStringsTable() = default;
  ~InternedStringsTable();
  InternedStringsTable(const InternedStringsTable &) = delete;
  InternedStringsTable& operator =(InternedStringsTable const&) = delete;
  InternedStringsTable(InternedStringsTable&&) = delete;
  InternedStringsTable& operator=(InternedStringsTable&&) = delete;

  at::optional<at::Dimname> lookup(PyObject* obj);
  // Precondition: obj is an interned python string.
  void addMapping(PyObject* obj, at::Dimname dimname);
 private:
  ska::flat_hash_map<PyObject*,at::Dimname> py_interned_string_to_dimname_;
};

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
InternedStringsTable kPyInternedStringToDimname;

InternedStringsTable::~InternedStringsTable() {
  for (auto it = py_interned_string_to_dimname_.begin();
      it != py_interned_string_to_dimname_.end(); ++it) {
    // See Note [References to python interned strings]
    Py_DECREF(it->first);
  }
}

at::optional<at::Dimname> InternedStringsTable::lookup(PyObject* obj) {
  auto it = py_interned_string_to_dimname_.find(obj);
  if (it == py_interned_string_to_dimname_.end()) {
    return at::nullopt;
  }
  return it->second;
}


void InternedStringsTable::addMapping(PyObject* obj, at::Dimname dimname) {
  // Note [References to python interned strings]
  // If a Python interned string has no references to it, then it gets
  // deallocated, invalidating this mapping. Let's immortalize the string by
  // holding a refcount to it and releasing it in the destructor
  Py_INCREF(obj);
  py_interned_string_to_dimname_.emplace(obj, dimname);
}

} // namespace torch

bool THPUtils_checkDimname(PyObject* obj) {
  return obj == Py_None || THPUtils_checkString(obj);
}

// To avoid ambiguity with IntArrayRef, we parse obj as a DimnameList if
// it is a list or tuple and its first elt is a Dimname
bool THPUtils_checkDimnameList(PyObject* obj) {
  auto tuple = PyTuple_Check(obj);
  if (!tuple && !PyList_Check(obj)) {
    return false;
  }
  // NOLINTNEXTLINE(bugprone-branch-clone)
  auto size = tuple ? PyTuple_GET_SIZE(obj) : PyList_GET_SIZE(obj);
  if (size == 0) {
    return true;
  }
  PyObject* first_elt = tuple ? PyTuple_GET_ITEM(obj, 0) : PyList_GET_ITEM(obj, 0);
  return THPUtils_checkDimname(first_elt);
}

at::Dimname THPDimname_parse(PyObject* obj) {
  if (obj == Py_None) {
    return at::Dimname::wildcard();
  }

  if (!THPUtils_checkString(obj)) {
    throw torch::TypeError("expected None or string for Dimname but got %s", Py_TYPE(obj)->tp_name);
  }

  if (!THPUtils_isInterned(obj)) {
    // internStringInPlace decrefs obj and increfs the result. Because we're
    // not actually returning the result to the user, we need to undo these.
    // See https://docs.python.org/3/c-api/unicode.html#c.PyUnicode_InternInPlace
    Py_INCREF(obj);
    THPUtils_internStringInPlace(&obj);
    Py_DECREF(obj);
  }

  auto maybeDimname = torch::kPyInternedStringToDimname.lookup(obj);
  if (maybeDimname) {
    return *maybeDimname;
  }

  const auto name = THPUtils_unpackString(obj);
  auto dimname = at::Dimname::fromSymbol(at::Symbol::dimname(name));
  torch::kPyInternedStringToDimname.addMapping(obj, dimname);
  return dimname;
}
