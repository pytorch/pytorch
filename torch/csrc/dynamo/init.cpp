#include <c10/util/Exception.h>
#include <torch/csrc/dynamo/init.h>
#include <torch/csrc/dynamo/utils.h>

#include <pybind11/stl_bind.h>
#include <torch/csrc/Exceptions.h>
#include <torch/csrc/dynamo/cache_entry.h>
#include <torch/csrc/dynamo/cpython_defs.h>
#include <torch/csrc/dynamo/eval_frame.h>
#include <torch/csrc/dynamo/extra_state.h>
#include <torch/csrc/dynamo/guards.h>
#include <torch/csrc/dynamo/python_compiled_autograd.h>
#include <torch/csrc/utils/pybind.h>
#include <torch/csrc/utils/python_compat.h>
#include <torch/csrc/utils/python_numbers.h>
#include <torch/csrc/utils/python_strings.h>

static struct PyModuleDef _module =
    {PyModuleDef_HEAD_INIT, "torch._C._dynamo", "", -1, nullptr};

PYBIND11_MAKE_OPAQUE(std::vector<uint8_t>)

namespace torch::dynamo {

std::vector<uint8_t> _PyOpcode_Caches_vec;

using torch::dynamo::autograd::torch_c_dynamo_compiled_autograd_init;

namespace {

struct StripFunctionCall {
  template <typename T>
  static bool unicode_is_literal_none(const T* start, const T* end) {
    if (end != start + 4) {
      return false;
    }

    return start[0] == 'N' && start[1] == 'o' && start[2] == 'n' &&
        start[3] == 'e';
  }

  // Takes a raw unicode pointer and length in code points and returns a
  // new/owned reference. T will be one of Py_UCS1, Py_UCS2, Py_UCS4.
  template <typename T>
  static THPObjectPtr apply(
      PyObject* original,
      const T* const start,
      size_t length) {
    // This function (based on the original python) is... not great.
    const T* const end = start + length;
    const T* curr = start;
    // All the code points we are interested in have the same values across UCS
    // types.
    for (auto p = start; p < end; ++p) {
      if (*p == ' ' || *p == '(') {
        curr = p + 1;
      } else if (*p == ')' || *p == ',' || *p == '[' || *p == ']') {
        if ((p > curr) && !unicode_is_literal_none(curr, p) &&
            (Py_UNICODE_ISALPHA(*curr) || *curr == '_')) {
          return apply(nullptr, curr, p - curr);
        }
        // The original code skipped adding these chars...
      }
    }

    // strip_getattr_getitem
    auto p = start;
    for (; p < end; ++p) {
      if (*p == '.' || *p == '[')
        break;
    }

    if (p == end && original) {
      return THPObjectPtr::dup(original);
    }

    return THPObjectPtr(
        PyUnicode_FromKindAndData(sizeof(*start), start, p - start));
  }
};

template <typename F>
THPObjectPtr _unicode_dispatch(PyObject* str) {
  if (!PyUnicode_Check(str)) {
    PyErr_SetString(PyExc_TypeError, "String expected");
    return THPObjectPtr();
  }

  // Remove this when we're 3.10+
  if (PyUnicode_READY(str) != 0) {
    // Returns -1 with an exception set on failure
    return THPObjectPtr();
  }

  auto length = PyUnicode_GET_LENGTH(str);

  switch (PyUnicode_KIND(str)) {
    case PyUnicode_1BYTE_KIND:
      return F::apply(str, PyUnicode_1BYTE_DATA(str), length);
    case PyUnicode_2BYTE_KIND:
      return F::apply(str, PyUnicode_2BYTE_DATA(str), length);
    case PyUnicode_4BYTE_KIND:
      return F::apply(str, PyUnicode_4BYTE_DATA(str), length);
    default:
      // This should be impossible - throw to make the compiler happy.
      TORCH_CHECK(false, "unreachable");
  }
}

bool _checkParamCount(size_t nargs, size_t expected) {
  if (nargs < expected) {
    PyErr_SetString(PyExc_TypeError, "Too few parameters");
    return false;
  }
  if (nargs > expected) {
    PyErr_SetString(PyExc_TypeError, "Too many parameters");
    return false;
  }
  return true;
}

struct IsValidVarName {
  // Takes a raw unicode pointer and length in code points and returns a
  // new/owned reference. T will be one of Py_UCS1, Py_UCS2, Py_UCS4.
  template <typename T>
  static THPObjectPtr apply(PyObject* original, const T* start, size_t length) {
    if (length < 1)
      return THPObjectPtr::dup(Py_False);

    // TODO: the original code is a bit odd... check it. It just checked that
    // the string starts with alnum. Then if it's all digits then it logs a
    // warning.

    if (!Py_UNICODE_ISALNUM(*start))
      return THPObjectPtr::dup(Py_False);
    while (length-- > 0) {
      if (!Py_UNICODE_ISDIGIT(*start++)) {
        return THPObjectPtr::dup(Py_True);
      }
    }

    // 2 == warning
    return THPObjectPtr(THPUtils_packInt32(2));
  }
};

PyObject* _strip_function_call(
    PyObject* self,
    PyObject* const* args,
    Py_ssize_t nargs) {
  if (!_checkParamCount(nargs, 1)) {
    return nullptr;
  }
  auto result = _unicode_dispatch<StripFunctionCall>(args[0]);
  return result.release();
}

PyObject* _is_valid_var_name(
    PyObject* self,
    PyObject* const* args,
    Py_ssize_t nargs) {
  if (!_checkParamCount(nargs, 1)) {
    return nullptr;
  }
  auto result = _unicode_dispatch<IsValidVarName>(args[0]);
  return result.release();
}

#define PYC_FN(x) ((PyCFunction)(void (*)()) & x)

void _register_functions(PyObject* mod) {
  static std::array<PyMethodDef, 3> fns = {
      PyMethodDef{
          "strip_function_call",
          PYC_FN(_strip_function_call),
          METH_FASTCALL,
          nullptr},
      PyMethodDef{
          "is_valid_var_name",
          PYC_FN(_is_valid_var_name),
          METH_FASTCALL,
          nullptr},
      PyMethodDef{nullptr, nullptr, 0, nullptr},
  };
  PyModule_AddFunctions(mod, fns.data());
}

struct __FixedLengthArray {
  /*
  A fixed length stack of VariableTracker objects with fast append and pop.
  The stack is initialized with a fixed capacity, and holds a pointer to the
  next free slot. This allows O(1) append/pop operations without resizing the
  underlying list. Random access is also O(1).

  CPython implements the stack as a contiguous C array with a fixed capacity.
  When the interpreter executes a PUSH operation, it writes the value to the
  next free slot and increments the stack pointer. When it executes a POP
  operation, it decrements the stack pointer and returns the value.
  The capacity is computed beforehand as "co_nlocalsplus + co_stacksize" in:
  https://github.com/python/cpython/blob/32e1e0699ffda8ec1dd5a0eb178b052352ab7d31/Objects/frameobject.c#L2122-L2139
  */

  std::vector<py::object> items;
  Py_ssize_t top;

  __FixedLengthArray(Py_ssize_t size) : items(size), top(0) {}
  ~__FixedLengthArray() = default;

  void append(py::object obj) {
    if (top >= (Py_ssize_t)items.size()) {
      throw std::runtime_error("stack overflow");
    }
    items[top++] = obj;
  }

  py::object __str__() const {
    std::string r = "[";
    for (Py_ssize_t i = 0; i < top; i++) {
      if (i != 0) {
        r += ", ";
      }
      r += py::str(items[i]);
    }
    r += "]";
    return py::str(r);
  }

  py::object __repr__() const {
    std::string r = "[";
    for (Py_ssize_t i = 0; i < top; i++) {
      if (i != 0) {
        r += ", ";
      }
      r += py::repr(items[i]);
    }
    r += "]";
    return py::str(r);
  }

  py::iterator __iter__() {
    return py::make_iterator(items.begin(), items.begin() + top);
  }

  void clear() {
    for (Py_ssize_t i = 0; i < top; i++) {
      items[i] = py::none();
    }
    top = 0;
  }

  py::object pop() {
    if (top <= 0) {
      throw python_error();
    }
    top--;
    return items[top];
  }

  Py_ssize_t __len__() const {
    return top;
  }

  py::object __getitem__(Py_ssize_t idx) {
    if (idx < 0) {
      idx += top;
    }

    if (idx < 0 || idx >= top) {
      throw python_error();
    }

    return items[idx];
  }

  void __setitem__(Py_ssize_t idx, py::object obj) {
    if (idx < 0) {
      idx += top;
    }

    if (idx < 0 || idx >= top) {
      throw python_error();
    }

    items[idx] = obj;
  }
};

struct FixedLengthArrayImpl {
  FixedLengthArrayImpl(Py_ssize_t size) : _top(0) {
    _items.reserve(size);
  }
  ~FixedLengthArrayImpl() {
    clear();
  }

  PyObject* append(PyObject* obj) {
    Py_INCREF(obj);
    _items.push_back(obj);
    Py_RETURN_NONE;
  }

  PyObject* pop() {
    if (len() <= 0) {
      throw python_error();
    }
    PyObject* obj = _items.back();
    _items.pop_back();
    return obj;
  }

  PyObject* clear() {
    for (Py_ssize_t i = 0; i < len(); i++) {
      Py_DECREF(_items[i]);
      _items[i] = nullptr;
    }
    _items.clear();
    Py_RETURN_NONE;
  }

  PyObject* getitem(Py_ssize_t idx) {
    if (idx < 0) {
      idx += len();
    }

    if (idx < 0 || idx >= len()) {
      throw python_error();
    }

    Py_INCREF(_items[idx]);
    return _items[idx];
  }

  int setitem(Py_ssize_t idx, PyObject* obj) {
    if (idx < 0) {
      idx += len();
    }

    if (idx < 0 || idx >= len()) {
      throw python_error();
    }

    Py_INCREF(obj);
    Py_DECREF(_items[idx]);
    _items[idx] = obj;
    return 0;
  }

  PyObject* repr() {
    std::string r = "[";
    Py_ssize_t sz = len();
    for (Py_ssize_t i = 0; i < sz; i++) {
      if (i != 0) {
        r += ", ";
      }
      auto s = THPObjectPtr(PyObject_Repr(_items[i]));
      r += THPUtils_unpackString(s);
    }
    r += "]";
    return THPUtils_packString(r);
  }

  PyObject* str() {
    std::string r = "[";
    Py_ssize_t sz = len();
    for (Py_ssize_t i = 0; i < sz; i++) {
      if (i != 0) {
        r += ", ";
      }
      auto s = THPObjectPtr(PyObject_Str(_items[i]));
      r += THPUtils_unpackString(s);
    }
    r += "]";
    return THPUtils_packString(r);
  }

  Py_ssize_t len() {
    return _items.size();
  }

  PyObject* iter() {
    auto lst = THPObjectPtr(PyList_New(len()));
    for (Py_ssize_t i = 0; i < len(); i++) {
      Py_INCREF(_items[i]);
      PyList_SET_ITEM(lst.get(), i, _items[i]);
    }
    PyObject* iter = PySeqIter_New(lst);
    return iter;
  }

  std::vector<PyObject*> _items;
  Py_ssize_t _top = 0;
};

} // anonymous namespace

typedef struct {
  PyObject_HEAD
  FixedLengthArrayImpl* impl;
} FixedLengthArray;

static int FixedLengthArray_init(
    PyObject* self,
    PyObject* args,
    PyObject* kwds) {
  Py_ssize_t size;
  if (!PyArg_ParseTuple(args, "n", &size))
    return -1;

  ((FixedLengthArray*)self)->impl = new FixedLengthArrayImpl(size);
  return 0;
}

static PyObject* FixedLengthArray_append(PyObject* self, PyObject* obj) {
  return ((FixedLengthArray*)self)->impl->append(obj);
}

static PyObject* FixedLengthArray_pop(
    PyObject* self,
    PyObject* Py_UNUSED(ignored)) {
  return ((FixedLengthArray*)self)->impl->pop();
}

static PyObject* FixedLengthArray_clear(
    PyObject* self,
    PyObject* Py_UNUSED(ignored)) {
  return ((FixedLengthArray*)self)->impl->clear();
}

static Py_ssize_t FixedLengthArray_len(PyObject* self) {
  return ((FixedLengthArray*)self)->impl->len();
}

static PyObject* FixedLengthArray_iter(PyObject* self) {
  return ((FixedLengthArray*)self)->impl->iter();
}

static PyObject* FixedLengthArray_getitem(PyObject* self, Py_ssize_t idx) {
  return ((FixedLengthArray*)self)->impl->getitem(idx);
}

static PyObject* FixedLengthArray_repr(PyObject* self) {
  return ((FixedLengthArray*)self)->impl->repr();
}

static PyObject* FixedLengthArray_str(PyObject* self) {
  return ((FixedLengthArray*)self)->impl->str();
}

static int FixedLengthArray_setitem(
    PyObject* self,
    Py_ssize_t idx,
    PyObject* obj) {
  return ((FixedLengthArray*)self)->impl->setitem(idx, obj);
}

static PyMethodDef FixedLengthArray_methods[] = {
    {"append", (PyCFunction)FixedLengthArray_append, METH_O, nullptr},
    {"pop", (PyCFunction)FixedLengthArray_pop, METH_NOARGS, nullptr},
    {"clear", (PyCFunction)FixedLengthArray_clear, METH_NOARGS, nullptr},
    {NULL, NULL, 0, NULL}};

static PySequenceMethods FixedLengthArray_as_sequence = {
    .sq_length = (lenfunc)FixedLengthArray_len,
    .sq_item = (ssizeargfunc)FixedLengthArray_getitem,
    .sq_ass_item = (ssizeobjargproc)FixedLengthArray_setitem,
};

static PyTypeObject FixedLengthArrayType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "FixedLengthArray",
    .tp_basicsize = sizeof(FixedLengthArray),
    .tp_itemsize = 0,
    .tp_repr = (reprfunc)FixedLengthArray_repr,
    // .tp_dealloc = (destructor)FixedLengthArray_dealloc,
    .tp_as_sequence = &FixedLengthArray_as_sequence,
    .tp_str = (reprfunc)FixedLengthArray_str,
    .tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,
    .tp_doc = "Fixed length array stack",
    .tp_iter = (getiterfunc)FixedLengthArray_iter,
    .tp_methods = FixedLengthArray_methods,
    .tp_init = (initproc)FixedLengthArray_init,
    .tp_new = PyType_GenericNew,
};

void initDynamoBindings(PyObject* torch) {
  PyObject* dynamo = PyModule_Create(&_module);
  if (dynamo == nullptr || PyModule_AddObject(torch, "_dynamo", dynamo) != 0) {
    throw python_error();
  }
#ifdef Py_GIL_DISABLED
  PyUnstable_Module_SetGIL(dynamo, Py_MOD_GIL_NOT_USED);
#endif

  PyObject* eval_frame = torch_c_dynamo_eval_frame_init();
  if (eval_frame == nullptr ||
      PyModule_AddObject(dynamo, "eval_frame", eval_frame) != 0) {
    throw python_error();
  }

  PyObject* utils = torch_c_dynamo_utils_init();
  if (utils == nullptr || PyModule_AddObject(dynamo, "utils", utils) != 0) {
    throw python_error();
  }

  PyObject* guards = torch_c_dynamo_guards_init();
  if (guards == nullptr || PyModule_AddObject(dynamo, "guards", guards) != 0) {
    throw python_error();
  }

  PyObject* compiled_autograd = torch_c_dynamo_compiled_autograd_init();
  if (compiled_autograd == nullptr ||
      PyModule_AddObject(dynamo, "compiled_autograd", compiled_autograd) != 0) {
    throw python_error();
  }

  PyModule_AddType(dynamo, &FixedLengthArrayType);

  py::class_<__FixedLengthArray>(dynamo, "Stack_pybind11")
      .def(py::init<Py_ssize_t>())
      .def("append", &__FixedLengthArray::append)
      .def("pop", &__FixedLengthArray::pop)
      .def("clear", &__FixedLengthArray::clear)
      .def("__getitem__", &__FixedLengthArray::__getitem__)
      .def("__setitem__", &__FixedLengthArray::__setitem__)
      .def("__iter__", &__FixedLengthArray::__iter__)
      .def("__str__", &__FixedLengthArray::__str__)
      .def("__repr__", &__FixedLengthArray::__repr__)
      .def("__len__", &__FixedLengthArray::__len__);

  auto m = py::handle(eval_frame).cast<py::module>();

  py::class_<CacheEntry>(m, "_CacheEntry")
      .def_readonly("guard_manager", &CacheEntry::guard_manager)
      .def_readonly("code", &CacheEntry::code)
      .def_readonly("compile_id", &CacheEntry::compile_id)
      .def_readonly("trace_annotation", &CacheEntry::trace_annotation)
      .def_property_readonly("next", &CacheEntry::next)
      .def(
          "update_diff_guard_root_manager",
          &CacheEntry::update_diff_guard_root_manager);

  py::class_<PrecompileEntry>(m, "_PrecompileEntry")
      .def_readonly("guard_manager", &PrecompileEntry::guard_manager);

  py::class_<ExtraState>(m, "_ExtraState")
      .def("invalidate", &ExtraState::invalidate);

  py::enum_<FrameAction>(m, "_FrameAction")
      .value("DEFAULT", FrameAction::DEFAULT)
      .value("SKIP", FrameAction::SKIP)
      .value("RUN_ONLY", FrameAction::RUN_ONLY);

  py::class_<FrameExecStrategy>(m, "_FrameExecStrategy")
      .def(py::init([]() {
        return FrameExecStrategy{FrameAction::SKIP, FrameAction::DEFAULT};
      }))
      .def(py::init([](FrameAction cur_action, FrameAction recursive_action) {
        return FrameExecStrategy{cur_action, recursive_action};
      }))
      .def_readwrite("cur_action", &FrameExecStrategy::cur_action)
      .def_readwrite("recursive_action", &FrameExecStrategy::recursive_action);

  m.def("_debug_get_cache_entry_list", &_debug_get_cache_entry_list);
  m.def("_reset_precompile_entries", &_reset_precompile_entries);
  m.def("_load_precompile_entry", &_load_precompile_entry);
  m.def("_debug_get_precompile_entries", &_debug_get_precompile_entries);
  py::bind_vector<std::vector<uint8_t>>(m, "VectorUInt8");
  init_THPCaches();
  if (THP_PyOpcode_Caches != nullptr) {
    _PyOpcode_Caches_vec.insert(
        _PyOpcode_Caches_vec.end(),
        THP_PyOpcode_Caches,
        THP_PyOpcode_Caches + THP_PyOpcode_Caches_size);
  }
  m.attr("py_opcode_caches") = _PyOpcode_Caches_vec;
  m.def("code_framelocals_names", &code_framelocals_names);
  _register_functions(dynamo);
}

} // namespace torch::dynamo
