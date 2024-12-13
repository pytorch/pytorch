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

template <typename T>
bool unicode_is_literal_none(const T* start, const T* end) {
  if (end != start + 4) {
    return false;
  }

  return start[0] == 'N' && start[1] == 'o' && start[2] == 'n' &&
      start[3] == 'e';
}

template <typename T>
THPObjectPtr strip_function_call_helper(
    PyObject* original,
    const T* const start,
    size_t length) {
  // This function is... not great.
  const T* const end = start + length;
  const T* curr = start;
  for (auto p = start; p < end; ++p) {
    if (*p == ' ' || *p == '(') {
      curr = p + 1;
    } else if (*p == ')' || *p == ',' || *p == '[' || *p == ']') {
      if ((p > curr) && !unicode_is_literal_none(curr, p) &&
          (Py_UNICODE_ISALPHA(*curr) || *curr == '_')) {
        return strip_function_call_helper(nullptr, curr, p - curr);
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

THPObjectPtr strip_function_call(PyObject* name) {
  if (!PyUnicode_Check(name)) {
    PyErr_SetString(PyExc_TypeError, "String expected");
    return THPObjectPtr::none();
  }

  if (PyUnicode_READY(name) != 0)
    return THPObjectPtr::none();

  auto length = PyUnicode_GET_LENGTH(name);
  switch (PyUnicode_KIND(name)) {
    case PyUnicode_1BYTE_KIND:
      return strip_function_call_helper(
          name, PyUnicode_1BYTE_DATA(name), length);
    case PyUnicode_2BYTE_KIND:
      throw std::runtime_error("unimplemented - 2byte");
    case PyUnicode_4BYTE_KIND:
      throw std::runtime_error("unimplemented - 4byte");
    default:
      throw std::runtime_error("unimplemented - bad value");
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

template <typename T>
THPObjectPtr is_valid_var_name_helper(const T* start, size_t length) {
  if (length < 1)
    return THPObjectPtr::dup(Py_False);

  // TODO: the original code is a bit odd... check it. It just checked that the
  // string starts with alnum. Then if it's all digits then it logs a warning.

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

THPObjectPtr is_valid_var_name(PyObject* name) {
  if (!PyUnicode_Check(name)) {
    PyErr_SetString(PyExc_TypeError, "String expected");
    return THPObjectPtr::none();
  }

  if (PyUnicode_READY(name) != 0) {
    return THPObjectPtr::none();
  }

  auto length = PyUnicode_GET_LENGTH(name);
  switch (PyUnicode_KIND(name)) {
    case PyUnicode_1BYTE_KIND:
      return is_valid_var_name_helper(PyUnicode_1BYTE_DATA(name), length);
    case PyUnicode_2BYTE_KIND:
      return is_valid_var_name_helper(PyUnicode_2BYTE_DATA(name), length);
    case PyUnicode_4BYTE_KIND:
      return is_valid_var_name_helper(PyUnicode_4BYTE_DATA(name), length);
    default:
      throw std::runtime_error("unimplemented - bad value");
  }
}

PyObject* _strip_function_call(
    PyObject* self,
    PyObject* const* args,
    Py_ssize_t nargs) {
  if (!_checkParamCount(nargs, 1)) {
    return THPObjectPtr::none().release();
  }
  return strip_function_call(args[0]).release();
}

PyObject* _is_valid_var_name(
    PyObject* self,
    PyObject* const* args,
    Py_ssize_t nargs) {
  if (!_checkParamCount(nargs, 1)) {
    return THPObjectPtr::none().release();
  }
  return is_valid_var_name(args[0]).release();
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

  py::class_<ExtraState>(m, "_ExtraState")
      .def("invalidate", &ExtraState::invalidate);

  m.def("_debug_get_cache_entry_list", &_debug_get_cache_entry_list);
  py::bind_vector<std::vector<uint8_t>>(m, "VectorUInt8");
  m.attr("py_opcode_caches") = _PyOpcode_Caches_vec;
  _register_functions(dynamo);
}

} // namespace torch::dynamo
