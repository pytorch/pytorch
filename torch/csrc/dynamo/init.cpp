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

static struct PyModuleDef _module =
    {PyModuleDef_HEAD_INIT, "torch._C._dynamo", "", -1, nullptr};

PYBIND11_MAKE_OPAQUE(std::vector<uint8_t>)

namespace torch::dynamo {

#if IS_PYTHON_3_11_PLUS

std::vector<uint8_t> _PyOpcode_Caches_vec(
    THP_PyOpcode_Caches,
    THP_PyOpcode_Caches + THP_PyOpcode_Caches_size);

#else

std::vector<uint8_t> _PyOpcode_Caches_vec;

#endif

using torch::dynamo::autograd::torch_c_dynamo_compiled_autograd_init;

namespace {

std::map<std::pair<PyTypeObject*, PyTypeObject*>, bool> _isinstance_cache;
std::vector<PyObject*> _all_subclasses;
PyTypeObject* VT_tp = nullptr;
PyTypeObject* LazyVT_tp = nullptr;

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
      throw std::runtime_error("unreachable");
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

static int _isinstance_check(PyObject* obj, PyTypeObject* cls_tp) {
  PyTypeObject* obj_tp = Py_TYPE(obj);

  auto key = std::make_pair(cls_tp, obj_tp);
  if (_isinstance_cache.count(key) == 0) {
    _isinstance_cache[key] = PyObject_TypeCheck(obj, cls_tp);
  }
  return _isinstance_cache[key];
}

PyObject* _cached_isinstance_check(PyObject* cls, PyObject* obj) {
  // We cache LazyVT and VT types here to avoid importing it every time.
  // They are freed in the metaclass dealloc function.
  if (LazyVT_tp == nullptr || VT_tp == nullptr) {
    auto base_mod =
        THPObjectPtr(PyImport_ImportModule("torch._dynamo.variables.base"));
    PyObject* VT = PyObject_GetAttrString(base_mod.get(), "VariableTracker");
    VT_tp = (PyTypeObject*)VT;

    auto Lazy_mod =
        THPObjectPtr(PyImport_ImportModule("torch._dynamo.variables.lazy"));
    PyObject* LazyVT =
        PyObject_GetAttrString(Lazy_mod.get(), "LazyVariableTracker");
    LazyVT_tp = (PyTypeObject*)LazyVT;
  }

  PyTypeObject* cls_tp = (PyTypeObject*)cls;

  int r = -1;

  if (PyObject_TypeCheck(obj, LazyVT_tp) &&
      !(cls_tp == VT_tp || cls_tp == LazyVT_tp)) {
    // Realize the lazy object if cls is a VariableTracker subclass but not
    // VariableTracker or LazyVariableTracker itself
    auto realize = THPObjectPtr(PyUnicode_FromString("realize"));
    auto new_obj = THPObjectPtr(PyObject_CallMethodNoArgs(obj, realize.get()));

    if (!new_obj) {
      // possible graph break?
      return nullptr;
    }
    r = _isinstance_check(new_obj.get(), cls_tp);
  } else {
    r = _isinstance_check(obj, cls_tp);
  }

  if (r == 1) {
    Py_RETURN_TRUE;
  } else if (r == 0) {
    Py_RETURN_FALSE;
  } else {
    PyErr_SetString(PyExc_TypeError, "Unexpected result from isinstance check");
    return nullptr;
  }
}

static int VariableTrackerMeta_init_fn(
    PyObject* cls,
    PyObject* args,
    PyObject* kwargs) {
  if (PyType_Type.tp_init((PyObject*)cls, args, kwargs) < 0) {
    return -1;
  }
  _all_subclasses.push_back(cls);
  return 0;
}

static PyObject* _get_all_subclasses(PyObject* unused, PyObject* noargs) {
  auto lst = THPObjectPtr(PyList_New((int)_all_subclasses.size()));
  for (size_t i = 0; i < _all_subclasses.size(); i++) {
    Py_INCREF(_all_subclasses[i]);
    PyList_SET_ITEM(lst.get(), i, _all_subclasses[i]);
  }
  return lst.release();
}

static void VariableTrackerMeta_dealloc(PyObject* self) {
  Py_XDECREF((PyObject*)VT_tp);
  Py_XDECREF((PyObject*)LazyVT_tp);
  PyObject_GC_UnTrack(self);
  PyType_Type.tp_dealloc(self);
}

static int VariableTrackerMeta_traverse(
    PyObject* self,
    visitproc visit,
    void* arg) {
  return PyType_Type.tp_traverse(self, visit, arg);
}

static int VariableTrackerMeta_clear(PyObject* self) {
  return PyType_Type.tp_clear(self);
}

// NOLINTNEXTLINE(*c-arrays)
static PyMethodDef VariableTrackerMetaMethods[] = {
    {"__instancecheck__",
     (PyCFunction)_cached_isinstance_check,
     METH_O,
     nullptr},
    {"get_all_subclasses",
     (PyCFunction)_get_all_subclasses,
     METH_NOARGS | METH_CLASS,
     nullptr},
    {nullptr}, // sentinel
};

// NOLINTNEXTLINE(*c-arrays)
static PyType_Slot VariableTrackerMetaSlots[] = {
    {Py_tp_methods, VariableTrackerMetaMethods},
    {Py_tp_dealloc, (void*)VariableTrackerMeta_dealloc},
    {Py_tp_init, (void*)VariableTrackerMeta_init_fn},
    {Py_tp_traverse, (void*)VariableTrackerMeta_traverse},
    {Py_tp_clear, (void*)VariableTrackerMeta_clear},
    {0, nullptr},
};

static PyType_Spec VariableTrackerMetaSpec = {
    .name = "torch._C.VariableTrackerMeta",
    .basicsize = sizeof(PyHeapTypeObject),
    .flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,
    .slots = VariableTrackerMetaSlots,
};

static PyObject* new_VariableTrackerMetaType() {
  auto bases = THPObjectPtr(PyTuple_Pack(1, (PyObject*)&PyType_Type));
  PyObject* type =
      PyType_FromSpecWithBases(&VariableTrackerMetaSpec, bases.get());
  return type;
}

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

} // anonymous namespace

void initDynamoBindings(PyObject* torch) {
  PyObject* dynamo = PyModule_Create(&_module);
  if (dynamo == nullptr || PyModule_AddObject(torch, "_dynamo", dynamo) != 0) {
    throw python_error();
  }
#ifdef Py_GIL_DISABLED
  PyUnstable_Module_SetGIL(dynamo, Py_MOD_GIL_NOT_USED);
#endif

  PyObject* vt_meta = new_VariableTrackerMetaType();
  if (vt_meta == nullptr) {
    throw python_error();
  }
  if (PyType_Ready((PyTypeObject*)vt_meta) < 0) {
    Py_DECREF(vt_meta);
  }

  PyModule_AddObject(dynamo, "VariableTrackerMeta", vt_meta);

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
  m.attr("py_opcode_caches") = _PyOpcode_Caches_vec;
  m.def("code_framelocals_names", &code_framelocals_names);
  _register_functions(dynamo);
}

} // namespace torch::dynamo
