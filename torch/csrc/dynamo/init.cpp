#include <c10/util/Exception.h>
#include <torch/csrc/dynamo/init.h>
#include <torch/csrc/dynamo/utils.h>

#include <pybind11/stl_bind.h>
#include <torch/csrc/Exceptions.h>
#include <torch/csrc/dynamo/cache_entry.h>
#include <torch/csrc/dynamo/cpython_defs.h>
#include <torch/csrc/dynamo/eval_frame.h>
#include <torch/csrc/dynamo/eval_frame_cpp.h>
#include <torch/csrc/dynamo/extra_state.h>
#include <torch/csrc/dynamo/guards.h>
#include <torch/csrc/dynamo/python_compiled_autograd.h>
#include <torch/csrc/utils/python_numbers.h>

#include <Python.h>

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

// Slot bit position definitions (each int64_t has independent bit positions)

enum class PySequenceSlotBit : int64_t {
  SQ_LENGTH = 0,
  SQ_CONCAT = 1,
  SQ_REPEAT = 2,
  SQ_ITEM = 3,
  SQ_CONTAINS = 4,
  SQ_ASS_ITEM = 5,
  SQ_INPLACE_CONCAT = 6,
  SQ_INPLACE_REPEAT = 7,
};

enum class PyMappingSlotBit : int64_t {
  MP_LENGTH = 0,
  MP_SUBSCRIPT = 1,
  MP_ASS_SUBSCRIPT = 2,
};

enum class PyNumberSlotBit : int64_t {
  NB_ADD = 0,
  NB_SUBTRACT = 1,
  NB_MULTIPLY = 2,
  NB_REMAINDER = 3,
  NB_POWER = 4,
  NB_NEGATIVE = 5,
  NB_POSITIVE = 6,
  NB_ABSOLUTE = 7,
  NB_BOOL = 8,
  NB_INVERT = 9,
  NB_LSHIFT = 10,
  NB_RSHIFT = 11,
  NB_AND = 12,
  NB_XOR = 13,
  NB_OR = 14,
  NB_INT = 15,
  NB_FLOAT = 16,
  NB_INPLACE_ADD = 17,
  NB_INPLACE_SUBTRACT = 18,
  NB_INPLACE_MULTIPLY = 19,
  NB_INPLACE_REMAINDER = 20,
  NB_INPLACE_POWER = 21,
  NB_INPLACE_LSHIFT = 22,
  NB_INPLACE_RSHIFT = 23,
  NB_INPLACE_AND = 24,
  NB_INPLACE_XOR = 25,
  NB_INPLACE_OR = 26,
  NB_FLOOR_DIVIDE = 27,
  NB_TRUE_DIVIDE = 28,
  NB_INPLACE_FLOOR_DIVIDE = 29,
  NB_INPLACE_TRUE_DIVIDE = 30,
  NB_INDEX = 31,
  NB_MATRIX_MULTIPLY = 32,
  NB_INPLACE_MATRIX_MULTIPLY = 33,
};

enum class PyTypeSlotBit : int64_t {
  TP_HASH = 0,
  TP_ITER = 1,
  TP_ITERNEXT = 2,
  TP_CALL = 3,
  TP_REPR = 4,
  TP_RICHCOMPARE = 5,
  TP_GETATTRO = 6,
  TP_SETATTRO = 7,
  TP_DESCR_GET = 8,
  TP_DESCR_SET = 9,
};

int64_t get_pysequence_slots(PyTypeObject* type) {
  int64_t slots = 0;
  if (PyType_GetSlot(type, Py_sq_length) != nullptr)
    slots |= (1LL << static_cast<int>(PySequenceSlotBit::SQ_LENGTH));
  if (PyType_GetSlot(type, Py_sq_concat) != nullptr)
    slots |= (1LL << static_cast<int>(PySequenceSlotBit::SQ_CONCAT));
  if (PyType_GetSlot(type, Py_sq_repeat) != nullptr)
    slots |= (1LL << static_cast<int>(PySequenceSlotBit::SQ_REPEAT));
  if (PyType_GetSlot(type, Py_sq_item) != nullptr)
    slots |= (1LL << static_cast<int>(PySequenceSlotBit::SQ_ITEM));
  if (PyType_GetSlot(type, Py_sq_contains) != nullptr)
    slots |= (1LL << static_cast<int>(PySequenceSlotBit::SQ_CONTAINS));
  if (PyType_GetSlot(type, Py_sq_ass_item) != nullptr)
    slots |= (1LL << static_cast<int>(PySequenceSlotBit::SQ_ASS_ITEM));
  if (PyType_GetSlot(type, Py_sq_inplace_concat) != nullptr)
    slots |= (1LL << static_cast<int>(PySequenceSlotBit::SQ_INPLACE_CONCAT));
  if (PyType_GetSlot(type, Py_sq_inplace_repeat) != nullptr)
    slots |= (1LL << static_cast<int>(PySequenceSlotBit::SQ_INPLACE_REPEAT));
  return slots;
}

int64_t get_pymapping_slots(PyTypeObject* type) {
  int64_t slots = 0;
  if (PyType_GetSlot(type, Py_mp_length) != nullptr)
    slots |= (1LL << static_cast<int>(PyMappingSlotBit::MP_LENGTH));
  if (PyType_GetSlot(type, Py_mp_subscript) != nullptr)
    slots |= (1LL << static_cast<int>(PyMappingSlotBit::MP_SUBSCRIPT));
  if (PyType_GetSlot(type, Py_mp_ass_subscript) != nullptr)
    slots |= (1LL << static_cast<int>(PyMappingSlotBit::MP_ASS_SUBSCRIPT));
  return slots;
}

int64_t get_pynumber_slots(PyTypeObject* type) {
  int64_t slots = 0;
  if (PyType_GetSlot(type, Py_nb_add) != nullptr)
    slots |= (1LL << static_cast<int>(PyNumberSlotBit::NB_ADD));
  if (PyType_GetSlot(type, Py_nb_subtract) != nullptr)
    slots |= (1LL << static_cast<int>(PyNumberSlotBit::NB_SUBTRACT));
  if (PyType_GetSlot(type, Py_nb_multiply) != nullptr)
    slots |= (1LL << static_cast<int>(PyNumberSlotBit::NB_MULTIPLY));
  if (PyType_GetSlot(type, Py_nb_remainder) != nullptr)
    slots |= (1LL << static_cast<int>(PyNumberSlotBit::NB_REMAINDER));
  if (PyType_GetSlot(type, Py_nb_power) != nullptr)
    slots |= (1LL << static_cast<int>(PyNumberSlotBit::NB_POWER));
  if (PyType_GetSlot(type, Py_nb_negative) != nullptr)
    slots |= (1LL << static_cast<int>(PyNumberSlotBit::NB_NEGATIVE));
  if (PyType_GetSlot(type, Py_nb_positive) != nullptr)
    slots |= (1LL << static_cast<int>(PyNumberSlotBit::NB_POSITIVE));
  if (PyType_GetSlot(type, Py_nb_absolute) != nullptr)
    slots |= (1LL << static_cast<int>(PyNumberSlotBit::NB_ABSOLUTE));
  if (PyType_GetSlot(type, Py_nb_bool) != nullptr)
    slots |= (1LL << static_cast<int>(PyNumberSlotBit::NB_BOOL));
  if (PyType_GetSlot(type, Py_nb_invert) != nullptr)
    slots |= (1LL << static_cast<int>(PyNumberSlotBit::NB_INVERT));
  if (PyType_GetSlot(type, Py_nb_lshift) != nullptr)
    slots |= (1LL << static_cast<int>(PyNumberSlotBit::NB_LSHIFT));
  if (PyType_GetSlot(type, Py_nb_rshift) != nullptr)
    slots |= (1LL << static_cast<int>(PyNumberSlotBit::NB_RSHIFT));
  if (PyType_GetSlot(type, Py_nb_and) != nullptr)
    slots |= (1LL << static_cast<int>(PyNumberSlotBit::NB_AND));
  if (PyType_GetSlot(type, Py_nb_xor) != nullptr)
    slots |= (1LL << static_cast<int>(PyNumberSlotBit::NB_XOR));
  if (PyType_GetSlot(type, Py_nb_or) != nullptr)
    slots |= (1LL << static_cast<int>(PyNumberSlotBit::NB_OR));
  if (PyType_GetSlot(type, Py_nb_int) != nullptr)
    slots |= (1LL << static_cast<int>(PyNumberSlotBit::NB_INT));
  if (PyType_GetSlot(type, Py_nb_float) != nullptr)
    slots |= (1LL << static_cast<int>(PyNumberSlotBit::NB_FLOAT));
  if (PyType_GetSlot(type, Py_nb_inplace_add) != nullptr)
    slots |= (1LL << static_cast<int>(PyNumberSlotBit::NB_INPLACE_ADD));
  if (PyType_GetSlot(type, Py_nb_inplace_subtract) != nullptr)
    slots |= (1LL << static_cast<int>(PyNumberSlotBit::NB_INPLACE_SUBTRACT));
  if (PyType_GetSlot(type, Py_nb_inplace_multiply) != nullptr)
    slots |= (1LL << static_cast<int>(PyNumberSlotBit::NB_INPLACE_MULTIPLY));
  if (PyType_GetSlot(type, Py_nb_inplace_remainder) != nullptr)
    slots |= (1LL << static_cast<int>(PyNumberSlotBit::NB_INPLACE_REMAINDER));
  if (PyType_GetSlot(type, Py_nb_inplace_power) != nullptr)
    slots |= (1LL << static_cast<int>(PyNumberSlotBit::NB_INPLACE_POWER));
  if (PyType_GetSlot(type, Py_nb_inplace_lshift) != nullptr)
    slots |= (1LL << static_cast<int>(PyNumberSlotBit::NB_INPLACE_LSHIFT));
  if (PyType_GetSlot(type, Py_nb_inplace_rshift) != nullptr)
    slots |= (1LL << static_cast<int>(PyNumberSlotBit::NB_INPLACE_RSHIFT));
  if (PyType_GetSlot(type, Py_nb_inplace_and) != nullptr)
    slots |= (1LL << static_cast<int>(PyNumberSlotBit::NB_INPLACE_AND));
  if (PyType_GetSlot(type, Py_nb_inplace_xor) != nullptr)
    slots |= (1LL << static_cast<int>(PyNumberSlotBit::NB_INPLACE_XOR));
  if (PyType_GetSlot(type, Py_nb_inplace_or) != nullptr)
    slots |= (1LL << static_cast<int>(PyNumberSlotBit::NB_INPLACE_OR));
  if (PyType_GetSlot(type, Py_nb_floor_divide) != nullptr)
    slots |= (1LL << static_cast<int>(PyNumberSlotBit::NB_FLOOR_DIVIDE));
  if (PyType_GetSlot(type, Py_nb_true_divide) != nullptr)
    slots |= (1LL << static_cast<int>(PyNumberSlotBit::NB_TRUE_DIVIDE));
  if (PyType_GetSlot(type, Py_nb_inplace_floor_divide) != nullptr)
    slots |=
        (1LL << static_cast<int>(PyNumberSlotBit::NB_INPLACE_FLOOR_DIVIDE));
  if (PyType_GetSlot(type, Py_nb_inplace_true_divide) != nullptr)
    slots |= (1LL << static_cast<int>(PyNumberSlotBit::NB_INPLACE_TRUE_DIVIDE));
  if (PyType_GetSlot(type, Py_nb_index) != nullptr)
    slots |= (1LL << static_cast<int>(PyNumberSlotBit::NB_INDEX));
  if (PyType_GetSlot(type, Py_nb_matrix_multiply) != nullptr)
    slots |= (1LL << static_cast<int>(PyNumberSlotBit::NB_MATRIX_MULTIPLY));
  if (PyType_GetSlot(type, Py_nb_inplace_matrix_multiply) != nullptr)
    slots |=
        (1LL << static_cast<int>(PyNumberSlotBit::NB_INPLACE_MATRIX_MULTIPLY));
  return slots;
}

// Helper function to check if a type has a specific method defined in its MRO
static bool type_has_method(PyTypeObject* type, const char* method_name) {
  // Check if the method is defined in the type or any of its base classes
  return PyObject_HasAttrString((PyObject*)type, method_name);
}

int64_t get_pytype_slots(PyTypeObject* type) {
  int64_t slots = 0;
  if (PyType_GetSlot(type, Py_tp_hash) != nullptr)
    slots |= (1LL << static_cast<int>(PyTypeSlotBit::TP_HASH));
  // For tp_iter, only set the bit if __iter__ is actually defined in the type's
  // MRO. CPython automatically fills tp_iter with a slot wrapper even if
  // __iter__ is not defined.
  if (PyType_GetSlot(type, Py_tp_iter) != nullptr &&
      type_has_method(type, "__iter__"))
    slots |= (1LL << static_cast<int>(PyTypeSlotBit::TP_ITER));
  // For tp_iternext, only set the bit if __next__ is actually defined in the
  // type's MRO. CPython automatically fills tp_iternext with a slot wrapper
  // even if __next__ is not defined, so we need to check if the method is truly
  // implemented.
  if (PyType_GetSlot(type, Py_tp_iternext) != nullptr &&
      type_has_method(type, "__next__"))
    slots |= (1LL << static_cast<int>(PyTypeSlotBit::TP_ITERNEXT));
  if (PyType_GetSlot(type, Py_tp_call) != nullptr)
    slots |= (1LL << static_cast<int>(PyTypeSlotBit::TP_CALL));
  if (PyType_GetSlot(type, Py_tp_repr) != nullptr)
    slots |= (1LL << static_cast<int>(PyTypeSlotBit::TP_REPR));
  if (PyType_GetSlot(type, Py_tp_richcompare) != nullptr)
    slots |= (1LL << static_cast<int>(PyTypeSlotBit::TP_RICHCOMPARE));
  if (PyType_GetSlot(type, Py_tp_getattro) != nullptr)
    slots |= (1LL << static_cast<int>(PyTypeSlotBit::TP_GETATTRO));
  if (PyType_GetSlot(type, Py_tp_setattro) != nullptr)
    slots |= (1LL << static_cast<int>(PyTypeSlotBit::TP_SETATTRO));
  if (PyType_GetSlot(type, Py_tp_descr_get) != nullptr)
    slots |= (1LL << static_cast<int>(PyTypeSlotBit::TP_DESCR_GET));
  if (PyType_GetSlot(type, Py_tp_descr_set) != nullptr)
    slots |= (1LL << static_cast<int>(PyTypeSlotBit::TP_DESCR_SET));
  return slots;
}

PyObject* _get_type_slots(
    PyObject* self,
    PyObject* const* args,
    Py_ssize_t nargs) {
  if (!_checkParamCount(nargs, 1)) {
    return nullptr;
  }

  PyObject* arg = args[0];
  PyTypeObject* type = PyType_Check(arg) ? (PyTypeObject*)arg : Py_TYPE(arg);

  int64_t seq_slots = get_pysequence_slots(type);
  int64_t map_slots = get_pymapping_slots(type);
  int64_t num_slots = get_pynumber_slots(type);
  int64_t type_slots = get_pytype_slots(type);

  PyObject* tuple = PyTuple_New(4);
  if (tuple == nullptr) {
    return nullptr;
  }
  PyTuple_SetItem(tuple, 0, PyLong_FromLongLong(seq_slots));
  PyTuple_SetItem(tuple, 1, PyLong_FromLongLong(map_slots));
  PyTuple_SetItem(tuple, 2, PyLong_FromLongLong(num_slots));
  PyTuple_SetItem(tuple, 3, PyLong_FromLongLong(type_slots));
  return tuple;
}

#define PYC_FN(x) ((PyCFunction)(void (*)()) & x)

void _register_functions(PyObject* mod) {
  static std::array<PyMethodDef, 4> fns = {
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
      PyMethodDef{
          "get_type_slots", PYC_FN(_get_type_slots), METH_FASTCALL, nullptr},
      PyMethodDef{nullptr, nullptr, 0, nullptr},
  };
  PyModule_AddFunctions(mod, fns.data());
}

} // anonymous namespace

void initDynamoBindings(PyObject* torch) {
  PyObject* dynamo = PyModule_Create(&_module);
  if (dynamo == nullptr || PyModule_AddObject(torch, "_dynamo", dynamo) != 0) {
    throw python_error(); // @allow-raw-throw
  }
#ifdef Py_GIL_DISABLED
  PyUnstable_Module_SetGIL(dynamo, Py_MOD_GIL_NOT_USED);
#endif

  PyObject* eval_frame = torch_c_dynamo_eval_frame_init();
  if (eval_frame == nullptr ||
      PyModule_AddObject(dynamo, "eval_frame", eval_frame) != 0) {
    throw python_error(); // @allow-raw-throw
  }

  PyObject* utils = torch_c_dynamo_utils_init();
  if (utils == nullptr || PyModule_AddObject(dynamo, "utils", utils) != 0) {
    throw python_error(); // @allow-raw-throw
  }

  PyObject* guards = torch_c_dynamo_guards_init();
  if (guards == nullptr || PyModule_AddObject(dynamo, "guards", guards) != 0) {
    throw python_error(); // @allow-raw-throw
  }

  PyObject* compiled_autograd = torch_c_dynamo_compiled_autograd_init();
  if (compiled_autograd == nullptr ||
      PyModule_AddObject(dynamo, "compiled_autograd", compiled_autograd) != 0) {
    throw python_error(); // @allow-raw-throw
  }

  auto m = py::handle(eval_frame).cast<py::module>();

  py::class_<CacheEntry>(m, "_CacheEntry")
      .def_readonly("guard_manager", &CacheEntry::guard_manager)
      .def_readonly("code", &CacheEntry::code)
      .def_readonly("compile_id", &CacheEntry::compile_id)
      .def_readonly("trace_annotation", &CacheEntry::trace_annotation)
      .def_readonly("backend", &CacheEntry::backend)
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

  m.def("set_c_recursion_limit", &dynamo_set_c_recursion_limit);
  m.def("get_c_recursion_limit", &dynamo_get_c_recursion_limit);

  m.def("_debug_get_cache_entry_list", &_debug_get_cache_entry_list);
  m.def("_reset_precompile_entries", &_reset_precompile_entries);
  m.def("_load_precompile_entry", &_load_precompile_entry);
  m.def("_debug_get_precompile_entries", &_debug_get_precompile_entries);
  m.def("_set_lru_cache", &_set_lru_cache);
  m.def(
      "_get_frame_value_stack_with_depth", &_get_frame_value_stack_with_depth);
  m.def("set_bytecode_debugger_callback", &set_bytecode_debugger_callback);
  m.def("get_bytecode_debugger_callback", &get_bytecode_debugger_callback);
  m.def("register_breakpoint_code", &register_breakpoint_code);

  // NullStackValue - sentinel for NULL stack values
  py::class_<NullStackValue>(m, "NullStackValue")
      .def("__repr__", [](const NullStackValue&) { return "<NULL>"; });
  m.attr("NULL_STACK_VALUE") = get_null_stack_value();

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

  auto dynamo_module = py::handle(dynamo).cast<py::module>();
  dynamo_module.def("has_slot", [](int64_t slots, py::object slot_bit_obj) {
    // Convert slot_bit to int - handle both int and pybind11 enums
    int64_t slot_bit = py::cast<int64_t>(slot_bit_obj.attr("__index__")());
    return (slots & (1LL << slot_bit)) != 0;
  });
  py::enum_<PySequenceSlotBit>(dynamo_module, "PySequenceSlots")
      .value("SQ_LENGTH", PySequenceSlotBit::SQ_LENGTH)
      .value("SQ_CONCAT", PySequenceSlotBit::SQ_CONCAT)
      .value("SQ_REPEAT", PySequenceSlotBit::SQ_REPEAT)
      .value("SQ_ITEM", PySequenceSlotBit::SQ_ITEM)
      .value("SQ_CONTAINS", PySequenceSlotBit::SQ_CONTAINS)
      .value("SQ_ASS_ITEM", PySequenceSlotBit::SQ_ASS_ITEM)
      .value("SQ_INPLACE_CONCAT", PySequenceSlotBit::SQ_INPLACE_CONCAT)
      .value("SQ_INPLACE_REPEAT", PySequenceSlotBit::SQ_INPLACE_REPEAT);

  py::enum_<PyMappingSlotBit>(dynamo_module, "PyMappingSlots")
      .value("MP_LENGTH", PyMappingSlotBit::MP_LENGTH)
      .value("MP_SUBSCRIPT", PyMappingSlotBit::MP_SUBSCRIPT)
      .value("MP_ASS_SUBSCRIPT", PyMappingSlotBit::MP_ASS_SUBSCRIPT);

  py::enum_<PyNumberSlotBit>(dynamo_module, "PyNumberSlots")
      .value("NB_ADD", PyNumberSlotBit::NB_ADD)
      .value("NB_SUBTRACT", PyNumberSlotBit::NB_SUBTRACT)
      .value("NB_MULTIPLY", PyNumberSlotBit::NB_MULTIPLY)
      .value("NB_REMAINDER", PyNumberSlotBit::NB_REMAINDER)
      .value("NB_POWER", PyNumberSlotBit::NB_POWER)
      .value("NB_NEGATIVE", PyNumberSlotBit::NB_NEGATIVE)
      .value("NB_POSITIVE", PyNumberSlotBit::NB_POSITIVE)
      .value("NB_ABSOLUTE", PyNumberSlotBit::NB_ABSOLUTE)
      .value("NB_BOOL", PyNumberSlotBit::NB_BOOL)
      .value("NB_INVERT", PyNumberSlotBit::NB_INVERT)
      .value("NB_LSHIFT", PyNumberSlotBit::NB_LSHIFT)
      .value("NB_RSHIFT", PyNumberSlotBit::NB_RSHIFT)
      .value("NB_AND", PyNumberSlotBit::NB_AND)
      .value("NB_XOR", PyNumberSlotBit::NB_XOR)
      .value("NB_OR", PyNumberSlotBit::NB_OR)
      .value("NB_INT", PyNumberSlotBit::NB_INT)
      .value("NB_FLOAT", PyNumberSlotBit::NB_FLOAT)
      .value("NB_INPLACE_ADD", PyNumberSlotBit::NB_INPLACE_ADD)
      .value("NB_INPLACE_SUBTRACT", PyNumberSlotBit::NB_INPLACE_SUBTRACT)
      .value("NB_INPLACE_MULTIPLY", PyNumberSlotBit::NB_INPLACE_MULTIPLY)
      .value("NB_INPLACE_REMAINDER", PyNumberSlotBit::NB_INPLACE_REMAINDER)
      .value("NB_INPLACE_POWER", PyNumberSlotBit::NB_INPLACE_POWER)
      .value("NB_INPLACE_LSHIFT", PyNumberSlotBit::NB_INPLACE_LSHIFT)
      .value("NB_INPLACE_RSHIFT", PyNumberSlotBit::NB_INPLACE_RSHIFT)
      .value("NB_INPLACE_AND", PyNumberSlotBit::NB_INPLACE_AND)
      .value("NB_INPLACE_XOR", PyNumberSlotBit::NB_INPLACE_XOR)
      .value("NB_INPLACE_OR", PyNumberSlotBit::NB_INPLACE_OR)
      .value("NB_FLOOR_DIVIDE", PyNumberSlotBit::NB_FLOOR_DIVIDE)
      .value("NB_TRUE_DIVIDE", PyNumberSlotBit::NB_TRUE_DIVIDE)
      .value(
          "NB_INPLACE_FLOOR_DIVIDE", PyNumberSlotBit::NB_INPLACE_FLOOR_DIVIDE)
      .value("NB_INPLACE_TRUE_DIVIDE", PyNumberSlotBit::NB_INPLACE_TRUE_DIVIDE)
      .value("NB_INDEX", PyNumberSlotBit::NB_INDEX)
      .value("NB_MATRIX_MULTIPLY", PyNumberSlotBit::NB_MATRIX_MULTIPLY)
      .value(
          "NB_INPLACE_MATRIX_MULTIPLY",
          PyNumberSlotBit::NB_INPLACE_MATRIX_MULTIPLY);

  py::enum_<PyTypeSlotBit>(dynamo_module, "PyTypeSlots")
      .value("TP_HASH", PyTypeSlotBit::TP_HASH)
      .value("TP_ITER", PyTypeSlotBit::TP_ITER)
      .value("TP_ITERNEXT", PyTypeSlotBit::TP_ITERNEXT)
      .value("TP_CALL", PyTypeSlotBit::TP_CALL)
      .value("TP_REPR", PyTypeSlotBit::TP_REPR)
      .value("TP_RICHCOMPARE", PyTypeSlotBit::TP_RICHCOMPARE)
      .value("TP_GETATTRO", PyTypeSlotBit::TP_GETATTRO)
      .value("TP_SETATTRO", PyTypeSlotBit::TP_SETATTRO)
      .value("TP_DESCR_GET", PyTypeSlotBit::TP_DESCR_GET)
      .value("TP_DESCR_SET", PyTypeSlotBit::TP_DESCR_SET);
}

} // namespace torch::dynamo
