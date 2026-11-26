// Opcode semantics for the CFG executor (torch/_rust/src/exec), transcribed
// from CPython 3.12's Python/bytecodes.c and Python/ceval.c. Most of the
// helpers there are static, hence the copies.
//
// Conventions: inputs are borrowed (the executor owns its SSA values) and
// outputs are new references. Functions returning int give 0 on success, -1
// with an exception set (the frame gets a traceback entry, like the default
// loop's `error:` label) or -2 with an exception being re-raised (no new
// traceback entry, like `exception_unwind:`).

#include <Python.h>

#undef _PyGC_FINALIZED
#define Py_BUILD_CORE
#define Py_BUILD_CORE_MODULE
#include <internal/pycore_frame.h>
#undef Py_BUILD_CORE
#undef Py_BUILD_CORE_MODULE

#include <opcode.h>

#if PY_VERSION_HEX < 0x030C0000 || PY_VERSION_HEX >= 0x030D0000
#error "exec_ops.c supports CPython 3.12 only"
#endif

#define NAME_ERROR_MSG "name '%.200s' is not defined"
#define UNBOUNDLOCAL_ERROR_MSG \
  "cannot access local variable '%s' where it is not associated with a value"
#define UNBOUNDFREE_ERROR_MSG                                                  \
  "cannot access free variable '%s' where it is not associated with a value" \
  " in enclosing scope"
#define CANNOT_CATCH_MSG \
  "catching classes that do not inherit from BaseException is not allowed"
#define CANNOT_EXCEPT_STAR_EG \
  "catching ExceptionGroup with except* is not allowed. Use except instead."

// From frame_shim.c's dependencies (torch/csrc/dynamo/cpython_defs.c).
PyFrameObject* THP_PyFrame_MakeAndSetFrameObject(_PyInterpreterFrame* frame);
PyAPI_FUNC(PyObject*) _PyTraceBack_FromFrame(PyObject* tb_next, PyFrameObject* frame);

// Non-core builds map this name to the _Py_Identifier variant; we want the
// exported PyObject* one (declared in pycore_object.h).
#undef _PyObject_LookupSpecial
PyObject* _PyObject_LookupSpecial(PyObject* self, PyObject* attr);

// Interned identifier, created on first use (the stand-in for _Py_ID).
#define ID(name)                                       \
  ({                                                   \
    static PyObject* id_;                              \
    if (id_ == NULL) {                                 \
      id_ = PyUnicode_InternFromString(#name);         \
    }                                                  \
    id_;                                               \
  })

static PyObject* co_name(_PyInterpreterFrame* f, int i) {
  return PyTuple_GET_ITEM(f->f_code->co_names, i);
}

static PyObject* slot_name(_PyInterpreterFrame* f, int slot) {
  return PyTuple_GET_ITEM(f->f_code->co_localsplusnames, slot);
}

static void format_exc_check_arg(PyObject* exc, const char* format_str, PyObject* obj) {
  if (!obj) {
    return;
  }
  const char* obj_str = PyUnicode_AsUTF8(obj);
  if (!obj_str) {
    return;
  }
  PyErr_Format(exc, format_str, obj_str);
  if (exc == PyExc_NameError) {
    // Include the name in the NameError exceptions to offer suggestions later.
    PyObject* raised = PyErr_GetRaisedException();
    if (PyErr_GivenExceptionMatches(raised, PyExc_NameError)) {
      if (((PyNameErrorObject*)raised)->name == NULL) {
        (void)PyObject_SetAttrString(raised, "name", obj);
      }
    }
    PyErr_SetRaisedException(raised);
  }
}

void tr_name_error(PyObject* name) {
  format_exc_check_arg(PyExc_NameError, NAME_ERROR_MSG, name);
}

void tr_unbound_local(_PyInterpreterFrame* f, int slot) {
  format_exc_check_arg(PyExc_UnboundLocalError, UNBOUNDLOCAL_ERROR_MSG, slot_name(f, slot));
}

// format_exc_unbound: for LOAD_DEREF / DELETE_DEREF on an empty cell.
void tr_unbound_deref(_PyInterpreterFrame* f, int slot) {
  if (PyErr_Occurred()) {
    return;
  }
  PyCodeObject* co = f->f_code;
  PyObject* name = slot_name(f, slot);
  if (slot < co->co_nlocalsplus - co->co_nfreevars) {
    format_exc_check_arg(PyExc_UnboundLocalError, UNBOUNDLOCAL_ERROR_MSG, name);
  } else {
    format_exc_check_arg(PyExc_NameError, UNBOUNDFREE_ERROR_MSG, name);
  }
}

static PyObject* load_global(_PyInterpreterFrame* f, PyObject* name) {
  PyObject* globals = f->f_globals;
  PyObject* builtins = f->f_builtins;
  PyObject* v;
  if (PyDict_CheckExact(globals) && PyDict_CheckExact(builtins)) {
    v = PyDict_GetItemWithError(globals, name);
    if (v == NULL && !PyErr_Occurred()) {
      v = PyDict_GetItemWithError(builtins, name);
    }
    if (v == NULL) {
      if (!PyErr_Occurred()) {
        tr_name_error(name);
      }
      return NULL;
    }
    return Py_NewRef(v);
  }
  // Slow path if globals or builtins is not a dict.
  v = PyObject_GetItem(globals, name);
  if (v == NULL) {
    if (!PyErr_ExceptionMatches(PyExc_KeyError)) {
      return NULL;
    }
    PyErr_Clear();
    v = PyObject_GetItem(builtins, name);
    if (v == NULL) {
      if (PyErr_ExceptionMatches(PyExc_KeyError)) {
        tr_name_error(name);
      }
      return NULL;
    }
  }
  return v;
}

// A LOAD_GLOBAL's last result, valid while both dicts keep their versions
// (globally unique, changed by every mutation), as in CPython's
// LOAD_GLOBAL_MODULE/BUILTIN specializations. `value` is borrowed: an
// unchanged dict still holds it.
typedef struct {
  PyObject* globals;
  uint64_t globals_version;
  PyObject* builtins;
  uint64_t builtins_version;
  PyObject* value;
} tr_global_cache;

PyObject* tr_load_global(_PyInterpreterFrame* f, int namei, tr_global_cache* c) {
  PyObject* globals = f->f_globals;
  PyObject* builtins = f->f_builtins;
  if (c->value != NULL && c->globals == globals && c->builtins == builtins &&
      ((PyDictObject*)globals)->ma_version_tag == c->globals_version &&
      ((PyDictObject*)builtins)->ma_version_tag == c->builtins_version) {
    return Py_NewRef(c->value);
  }
  PyObject* v = load_global(f, co_name(f, namei));
  if (v != NULL && PyDict_CheckExact(globals) && PyDict_CheckExact(builtins)) {
    c->globals = globals;
    c->globals_version = ((PyDictObject*)globals)->ma_version_tag;
    c->builtins = builtins;
    c->builtins_version = ((PyDictObject*)builtins)->ma_version_tag;
    c->value = v;
  }
  return v;
}

// Method-form LOAD_ATTR: (owner -- meth|NULL, self|attr).
int tr_load_method(PyObject* name, PyObject* owner, PyObject** out) {
  PyObject* meth = NULL;
  if (_PyObject_GetMethod(owner, name, &meth)) {
    out[0] = meth;
    out[1] = Py_NewRef(owner);
    return 0;
  }
  if (meth == NULL) {
    return -1;
  }
  out[0] = NULL;
  out[1] = meth;
  return 0;
}

// descrobject.c's wrapperobject: a slot wrapper bound to an object.
typedef struct {
  PyObject_HEAD
  PyWrapperDescrObject* descr;
  PyObject* self;
} tr_wrapperobject;

// What a bound method is bound to (its `__self__`), if `callable` is exactly
// one; else NULL. Borrowed: the method holds it. Runs no Python.
PyObject* tr_bound_self(PyObject* callable) {
  return Py_IS_TYPE(callable, &PyMethod_Type) ? PyMethod_GET_SELF(callable) : NULL;
}

// The callable a functools.partial calls (its `func`), if `callable` is
// exactly one; else NULL. Borrowed: the partial holds it. Runs no Python.
PyObject* tr_partial_func(PyObject* callable) {
  static PyTypeObject* partial = NULL;
  if (partial == NULL) {
    PyObject* name = PyUnicode_FromString("functools");
    PyObject* module = name == NULL ? NULL : PyImport_GetModule(name);
    Py_XDECREF(name);
    PyObject* type = module == NULL ? NULL : PyObject_GetAttrString(module, "partial");
    Py_XDECREF(module);
    if (type == NULL || !PyType_Check(type)) {
      Py_XDECREF(type);
      PyErr_Clear();
      return NULL;
    }
    partial = (PyTypeObject*)type;
  }
  if (!Py_IS_TYPE(callable, partial)) {
    return NULL;
  }
  PyObject* func = PyObject_GetAttrString(callable, "func");
  if (func == NULL) {
    PyErr_Clear();
    return NULL;
  }
  Py_DECREF(func);
  return func;
}

// CALL: in = [method|NULL, self|callable, args...].
// The function a call to `callable` runs, if it's a bound method, or the
// descriptor of a bound slot wrapper or builtin method. Borrowed.
// The method descriptor of `ml` among the classes tp inherits from, whichever
// of them the name resolves to (`super().m` reaches one a subclass hides).
static PyObject* tr_descr_of(
    PyTypeObject* tp, PyMethodDef* ml, PyTypeObject* kind) {
  PyObject* mro = tp->tp_mro;
  if (mro == NULL || !PyTuple_Check(mro)) {
    return NULL;
  }
  for (Py_ssize_t i = 0; i < PyTuple_GET_SIZE(mro); i++) {
    PyObject* dict = PyType_GetDict((PyTypeObject*)PyTuple_GET_ITEM(mro, i));
    if (dict == NULL) {
      PyErr_Clear();
      continue;
    }
    PyObject* d = PyDict_GetItemString(dict, ml->ml_name);
    Py_DECREF(dict);
    if (d != NULL && Py_IS_TYPE(d, kind) &&
        ((PyMethodDescrObject*)d)->d_method == ml) {
      return d;
    }
  }
  return NULL;
}

PyObject* tr_bound_function(PyObject* callable) {
  if (Py_IS_TYPE(callable, &_PyMethodWrapper_Type)) {
    return (PyObject*)((tr_wrapperobject*)callable)->descr;
  }
  if (PyCFunction_Check(callable)) {
    // Bound to an object (a method), to a class (a classmethod), or to a
    // class by its own class (`super().__instancecheck__` in a metaclass).
    PyObject* self = PyCFunction_GET_SELF(callable);
    if (self != NULL && !PyModule_Check(self)) {
      PyMethodDef* ml = ((PyCFunctionObject*)callable)->m_ml;
      PyObject* d = NULL;
      if (PyType_Check(self)) {
        d = tr_descr_of((PyTypeObject*)self, ml, &PyClassMethodDescr_Type);
      }
      if (d == NULL) {
        d = tr_descr_of(Py_TYPE(self), ml, &PyMethodDescr_Type);
      }
      if (d != NULL) {
        return d;
      }
    }
    return callable;
  }
  return PyMethod_Check(callable) ? PyMethod_GET_FUNCTION(callable) : callable;
}

// The Python function obj's type runs for obj(...) (its __call__) or, with
// getattr, for an attribute the generic lookup misses (its __getattr__), if
// obj's type is a class defined in Python and that's a plain function.
// Borrowed. Runs no Python code.
PyObject* tr_type_function(PyObject* obj, int getattr) {
  static PyObject* names[2];
  if (names[0] == NULL) {
    names[0] = PyUnicode_InternFromString("__call__");
    names[1] = PyUnicode_InternFromString("__getattr__");
    if (names[0] == NULL || names[1] == NULL) {
      PyErr_Clear();
      return NULL;
    }
  }
  PyTypeObject* tp = Py_TYPE(obj);
  if (!PyType_HasFeature(tp, Py_TPFLAGS_HEAPTYPE)) {
    return NULL;
  }
  PyObject* f = _PyType_Lookup(tp, names[getattr ? 1 : 0]);
  return f != NULL && PyFunction_Check(f) ? f : NULL;
}

// Whether obj's class runs f for `name` (looked up on the class only).
int tr_type_attr_is(PyObject* obj, PyObject* name, PyObject* f) {
  return _PyType_Lookup(Py_TYPE(obj), name) == f;
}

// Whether loading `name` from obj falls back to its class's __getattr__,
// which is f: the class has object's __getattribute__ and no attribute by
// that name, and obj's __dict__ (if any) doesn't have it either. Runs no
// Python code; 0 with no exception set if not (or on error).
int tr_getattr_falls_back(PyObject* obj, PyObject* name, PyObject* f) {
  static PyObject *getattr_name, *getattribute_name, *generic;
  if (generic == NULL) {
    getattr_name = PyUnicode_InternFromString("__getattr__");
    getattribute_name = PyUnicode_InternFromString("__getattribute__");
    if (getattr_name == NULL || getattribute_name == NULL) {
      PyErr_Clear();
      return 0;
    }
    generic = _PyType_Lookup(&PyBaseObject_Type, getattribute_name);
  }
  PyTypeObject* tp = Py_TYPE(obj);
  if (_PyType_Lookup(tp, getattr_name) != f || _PyType_Lookup(tp, getattribute_name) != generic ||
      _PyType_Lookup(tp, name) != NULL) {
    return 0;
  }
  if (tp->tp_dictoffset == 0) {
    return 1;
  }
  PyObject* dict = PyObject_GenericGetDict(obj, NULL);
  if (dict == NULL) {
    PyErr_Clear();
    return 0;
  }
  int has = PyDict_Contains(dict, name);
  Py_DECREF(dict);
  if (has < 0) {
    PyErr_Clear();
  }
  return has == 0;
}

PyObject* tr_call(PyObject* const* in, int argc, PyObject* kwnames) {
  PyObject* callable = in[0];
  PyObject* const* args = in + 1;
  int total = argc + 1;
  if (callable == NULL) {
    callable = in[1];
    args = in + 2;
    total = argc;
  }
  Py_ssize_t nkw = kwnames ? PyTuple_GET_SIZE(kwnames) : 0;
  return PyObject_Vectorcall(callable, args, total - nkw, kwnames);
}

PyObject* tr_binary_op(int op, PyObject* lhs, PyObject* rhs) {
  switch (op) {
    case NB_ADD: return PyNumber_Add(lhs, rhs);
    case NB_AND: return PyNumber_And(lhs, rhs);
    case NB_FLOOR_DIVIDE: return PyNumber_FloorDivide(lhs, rhs);
    case NB_LSHIFT: return PyNumber_Lshift(lhs, rhs);
    case NB_MATRIX_MULTIPLY: return PyNumber_MatrixMultiply(lhs, rhs);
    case NB_MULTIPLY: return PyNumber_Multiply(lhs, rhs);
    case NB_REMAINDER: return PyNumber_Remainder(lhs, rhs);
    case NB_OR: return PyNumber_Or(lhs, rhs);
    case NB_POWER: return PyNumber_Power(lhs, rhs, Py_None);
    case NB_RSHIFT: return PyNumber_Rshift(lhs, rhs);
    case NB_SUBTRACT: return PyNumber_Subtract(lhs, rhs);
    case NB_TRUE_DIVIDE: return PyNumber_TrueDivide(lhs, rhs);
    case NB_XOR: return PyNumber_Xor(lhs, rhs);
    case NB_INPLACE_ADD: return PyNumber_InPlaceAdd(lhs, rhs);
    case NB_INPLACE_AND: return PyNumber_InPlaceAnd(lhs, rhs);
    case NB_INPLACE_FLOOR_DIVIDE: return PyNumber_InPlaceFloorDivide(lhs, rhs);
    case NB_INPLACE_LSHIFT: return PyNumber_InPlaceLshift(lhs, rhs);
    case NB_INPLACE_MATRIX_MULTIPLY: return PyNumber_InPlaceMatrixMultiply(lhs, rhs);
    case NB_INPLACE_MULTIPLY: return PyNumber_InPlaceMultiply(lhs, rhs);
    case NB_INPLACE_REMAINDER: return PyNumber_InPlaceRemainder(lhs, rhs);
    case NB_INPLACE_OR: return PyNumber_InPlaceOr(lhs, rhs);
    case NB_INPLACE_POWER: return PyNumber_InPlacePower(lhs, rhs, Py_None);
    case NB_INPLACE_RSHIFT: return PyNumber_InPlaceRshift(lhs, rhs);
    case NB_INPLACE_SUBTRACT: return PyNumber_InPlaceSubtract(lhs, rhs);
    case NB_INPLACE_TRUE_DIVIDE: return PyNumber_InPlaceTrueDivide(lhs, rhs);
    case NB_INPLACE_XOR: return PyNumber_InPlaceXor(lhs, rhs);
  }
  PyErr_Format(PyExc_SystemError, "bad BINARY_OP oparg %d", op);
  return NULL;
}

// MAKE_FUNCTION: in = [defaults?, kwdefaults?, annotations?, closure?, code].
// Fields are set directly, as bytecodes.c does: in 3.12 annotations are a flat
// tuple that PyFunction_SetAnnotations would reject.
PyObject* tr_make_function(_PyInterpreterFrame* f, int flags, PyObject* const* in) {
  int n = 0;
  PyObject* defaults = (flags & 0x01) ? in[n++] : NULL;
  PyObject* kwdefaults = (flags & 0x02) ? in[n++] : NULL;
  PyObject* annotations = (flags & 0x04) ? in[n++] : NULL;
  PyObject* closure = (flags & 0x08) ? in[n++] : NULL;
  PyFunctionObject* func = (PyFunctionObject*)PyFunction_New(in[n], f->f_globals);
  if (func == NULL) {
    return NULL;
  }
  Py_XSETREF(func->func_closure, Py_XNewRef(closure));
  Py_XSETREF(func->func_annotations, Py_XNewRef(annotations));
  Py_XSETREF(func->func_kwdefaults, Py_XNewRef(kwdefaults));
  Py_XSETREF(func->func_defaults, Py_XNewRef(defaults));
  return (PyObject*)func;
}

PyObject* tr_load_from_dict_or_deref(_PyInterpreterFrame* f, PyObject* class_dict, int slot) {
  PyObject* name = slot_name(f, slot);
  PyObject* value;
  if (PyDict_CheckExact(class_dict)) {
    value = PyDict_GetItemWithError(class_dict, name);
    if (value != NULL) {
      return Py_NewRef(value);
    }
    if (PyErr_Occurred()) {
      return NULL;
    }
  } else {
    value = PyObject_GetItem(class_dict, name);
    if (value != NULL) {
      return value;
    }
    if (!PyErr_ExceptionMatches(PyExc_KeyError)) {
      return NULL;
    }
    PyErr_Clear();
  }
  value = PyCell_GET(f->localsplus[slot]);
  if (value == NULL) {
    tr_unbound_deref(f, slot);
    return NULL;
  }
  return Py_NewRef(value);
}

// RAISE_VARARGS (do_raise). exc/cause are borrowed and may be NULL.
int tr_do_raise(PyObject* exc, PyObject* cause) {
  PyObject *type = NULL, *value = NULL;
  if (exc == NULL) {
    // Reraise.
    PyObject* handled = PyErr_GetHandledException();
    if (handled == NULL) {
      PyErr_SetString(PyExc_RuntimeError, "No active exception to reraise");
      return -1;
    }
    PyErr_SetRaisedException(handled);
    return -2;
  }
  if (PyExceptionClass_Check(exc)) {
    type = Py_NewRef(exc);
    value = PyObject_CallNoArgs(exc);
    if (value == NULL) {
      goto raise_error;
    }
    if (!PyExceptionInstance_Check(value)) {
      PyErr_Format(
          PyExc_TypeError,
          "calling %R should have returned an instance of BaseException, not %R",
          type, Py_TYPE(value));
      goto raise_error;
    }
  } else if (PyExceptionInstance_Check(exc)) {
    value = Py_NewRef(exc);
    type = Py_NewRef(PyExceptionInstance_Class(exc));
  } else {
    PyErr_SetString(PyExc_TypeError, "exceptions must derive from BaseException");
    goto raise_error;
  }
  if (cause) {
    PyObject* fixed_cause;
    if (PyExceptionClass_Check(cause)) {
      fixed_cause = PyObject_CallNoArgs(cause);
      if (fixed_cause == NULL) {
        goto raise_error;
      }
    } else if (PyExceptionInstance_Check(cause)) {
      fixed_cause = Py_NewRef(cause);
    } else if (Py_IsNone(cause)) {
      fixed_cause = NULL;
    } else {
      PyErr_SetString(PyExc_TypeError, "exception causes must derive from BaseException");
      goto raise_error;
    }
    PyException_SetCause(value, fixed_cause);
  }
  PyErr_SetObject(type, value);
raise_error:
  Py_XDECREF(value);
  Py_XDECREF(type);
  return -1;
}

static int check_except_type_valid(PyObject* right) {
  if (PyTuple_Check(right)) {
    for (Py_ssize_t i = 0; i < PyTuple_GET_SIZE(right); i++) {
      if (!PyExceptionClass_Check(PyTuple_GET_ITEM(right, i))) {
        PyErr_SetString(PyExc_TypeError, CANNOT_CATCH_MSG);
        return -1;
      }
    }
  } else if (!PyExceptionClass_Check(right)) {
    PyErr_SetString(PyExc_TypeError, CANNOT_CATCH_MSG);
    return -1;
  }
  return 0;
}

static int check_except_star_type_valid(PyObject* right) {
  if (check_except_type_valid(right) < 0) {
    return -1;
  }
  int is_subclass = 0;
  if (PyTuple_Check(right)) {
    for (Py_ssize_t i = 0; i < PyTuple_GET_SIZE(right); i++) {
      is_subclass = PyObject_IsSubclass(PyTuple_GET_ITEM(right, i), PyExc_BaseExceptionGroup);
      if (is_subclass < 0) {
        return -1;
      }
      if (is_subclass) {
        break;
      }
    }
  } else {
    is_subclass = PyObject_IsSubclass(right, PyExc_BaseExceptionGroup);
    if (is_subclass < 0) {
      return -1;
    }
  }
  if (is_subclass) {
    PyErr_SetString(PyExc_TypeError, CANNOT_EXCEPT_STAR_EG);
    return -1;
  }
  return 0;
}

static int exception_group_match(
    _PyInterpreterFrame* f,
    PyObject* exc_value,
    PyObject* match_type,
    PyObject** match,
    PyObject** rest) {
  if (Py_IsNone(exc_value)) {
    *match = Py_NewRef(Py_None);
    *rest = Py_NewRef(Py_None);
    return 0;
  }
  if (PyErr_GivenExceptionMatches(exc_value, match_type)) {
    // Full match of exc itself.
    if (PyObject_TypeCheck(exc_value, (PyTypeObject*)PyExc_BaseExceptionGroup)) {
      *match = Py_NewRef(exc_value);
    } else {
      // Naked exception: wrap it (_PyExc_CreateExceptionGroup).
      PyObject* wrapped = PyObject_CallFunction(PyExc_BaseExceptionGroup, "s(O)", "", exc_value);
      if (wrapped == NULL) {
        return -1;
      }
      PyFrameObject* frame_obj = f->frame_obj ? f->frame_obj : THP_PyFrame_MakeAndSetFrameObject(f);
      if (frame_obj != NULL) {
        PyObject* tb = _PyTraceBack_FromFrame(NULL, frame_obj);
        if (tb == NULL) {
          Py_DECREF(wrapped);
          return -1;
        }
        PyException_SetTraceback(wrapped, tb);
        Py_DECREF(tb);
      }
      *match = wrapped;
    }
    *rest = Py_NewRef(Py_None);
    return 0;
  }
  // Check for a partial match if it's an exception group.
  if (PyObject_TypeCheck(exc_value, (PyTypeObject*)PyExc_BaseExceptionGroup)) {
    PyObject* pair = PyObject_CallMethod(exc_value, "split", "(O)", match_type);
    if (pair == NULL) {
      return -1;
    }
    if (!PyTuple_CheckExact(pair)) {
      PyErr_Format(
          PyExc_TypeError, "%.200s.split must return a tuple, not %.200s",
          Py_TYPE(exc_value)->tp_name, Py_TYPE(pair)->tp_name);
      Py_DECREF(pair);
      return -1;
    }
    // Tuples longer than 2 are allowed for backwards compatibility.
    if (PyTuple_GET_SIZE(pair) < 2) {
      PyErr_Format(
          PyExc_TypeError, "%.200s.split must return a 2-tuple, got tuple of size %zd",
          Py_TYPE(exc_value)->tp_name, PyTuple_GET_SIZE(pair));
      Py_DECREF(pair);
      return -1;
    }
    *match = Py_NewRef(PyTuple_GET_ITEM(pair, 0));
    *rest = Py_NewRef(PyTuple_GET_ITEM(pair, 1));
    Py_DECREF(pair);
    return 0;
  }
  *match = Py_NewRef(Py_None);
  *rest = Py_NewRef(exc_value);
  return 0;
}

// Writes the unpacked values to out[0..] in Python order (first item first).
static int unpack_iterable(PyObject* v, int argcnt, int argcntafter, PyObject** out) {
  int i = 0;
  PyObject* it = PyObject_GetIter(v);
  if (it == NULL) {
    if (PyErr_ExceptionMatches(PyExc_TypeError) && Py_TYPE(v)->tp_iter == NULL &&
        !PySequence_Check(v)) {
      PyErr_Format(
          PyExc_TypeError, "cannot unpack non-iterable %.200s object", Py_TYPE(v)->tp_name);
    }
    return -1;
  }
  for (; i < argcnt; i++) {
    PyObject* w = PyIter_Next(it);
    if (w == NULL) {
      if (!PyErr_Occurred()) {
        if (argcntafter == -1) {
          PyErr_Format(
              PyExc_ValueError,
              "not enough values to unpack (expected %d, got %d)", argcnt, i);
        } else {
          PyErr_Format(
              PyExc_ValueError,
              "not enough values to unpack (expected at least %d, got %d)",
              argcnt + argcntafter, i);
        }
      }
      goto error;
    }
    out[i] = w;
  }
  if (argcntafter == -1) {
    PyObject* w = PyIter_Next(it);
    if (w == NULL) {
      if (PyErr_Occurred()) {
        goto error;
      }
      Py_DECREF(it);
      return 0;
    }
    Py_DECREF(w);
    PyErr_Format(PyExc_ValueError, "too many values to unpack (expected %d)", argcnt);
    goto error;
  }
  PyObject* l = PySequence_List(it);
  if (l == NULL) {
    goto error;
  }
  out[i++] = l;
  Py_ssize_t ll = PyList_GET_SIZE(l);
  if (ll < argcntafter) {
    PyErr_Format(
        PyExc_ValueError,
        "not enough values to unpack (expected at least %d, got %zd)",
        argcnt + argcntafter, argcnt + ll);
    goto error;
  }
  for (int j = 0; j < argcntafter; j++) {
    out[i++] = Py_NewRef(PyList_GET_ITEM(l, ll - argcntafter + j));
  }
  if (PyList_SetSlice(l, ll - argcntafter, ll, NULL) < 0) {
    goto error;
  }
  Py_DECREF(it);
  return 0;
error:
  for (int j = 0; j < i; j++) {
    Py_CLEAR(out[j]);
  }
  Py_XDECREF(it);
  return -1;
}

static PyObject* import_name(
    _PyInterpreterFrame* f,
    PyObject* name,
    PyObject* fromlist,
    PyObject* level) {
  PyObject* import_func = PyObject_GetItem(f->f_builtins, ID(__import__));
  if (import_func == NULL) {
    if (PyErr_ExceptionMatches(PyExc_KeyError)) {
      PyErr_SetString(PyExc_ImportError, "__import__ not found");
    }
    return NULL;
  }
  // CPython calls PyImport_ImportModuleLevelObject directly when this is the
  // default __import__, which is what that function does anyway.
  PyObject* locals = f->f_locals;
  PyObject* stack[5] = {name, f->f_globals, locals == NULL ? Py_None : locals, fromlist, level};
  PyObject* res = PyObject_Vectorcall(import_func, stack, 5, NULL);
  Py_DECREF(import_func);
  return res;
}

static PyObject* import_from(PyObject* v, PyObject* name) {
  PyObject *x, *fullmodname, *pkgname, *pkgpath, *pkgname_or_unknown, *errmsg;
  if (_PyObject_LookupAttr(v, name, &x) != 0) {
    return x;
  }
  // Issue #17636: in case this failed because of a circular relative import,
  // try to fall back on reading the module directly from sys.modules.
  pkgname = PyObject_GetAttr(v, ID(__name__));
  if (pkgname == NULL) {
    goto error;
  }
  if (!PyUnicode_Check(pkgname)) {
    Py_CLEAR(pkgname);
    goto error;
  }
  fullmodname = PyUnicode_FromFormat("%U.%U", pkgname, name);
  if (fullmodname == NULL) {
    Py_DECREF(pkgname);
    return NULL;
  }
  x = PyImport_GetModule(fullmodname);
  Py_DECREF(fullmodname);
  if (x == NULL && !PyErr_Occurred()) {
    goto error;
  }
  Py_DECREF(pkgname);
  return x;
error:
  pkgpath = PyModule_GetFilenameObject(v);
  if (pkgname == NULL) {
    pkgname_or_unknown = PyUnicode_FromString("<unknown module name>");
    if (pkgname_or_unknown == NULL) {
      Py_XDECREF(pkgpath);
      return NULL;
    }
  } else {
    pkgname_or_unknown = pkgname;
  }
  PyObject* path_for_error = NULL;
  if (pkgpath == NULL || !PyUnicode_Check(pkgpath)) {
    PyErr_Clear();
    errmsg = PyUnicode_FromFormat(
        "cannot import name %R from %R (unknown location)", name, pkgname_or_unknown);
  } else {
    PyObject* spec = PyObject_GetAttr(v, ID(__spec__));
    int initializing = 0;
    if (spec == NULL) {
      PyErr_Clear();
    } else {
      PyObject* value;
      if (_PyObject_LookupAttr(spec, ID(_initializing), &value) < 0) {
        PyErr_Clear();
      } else if (value != NULL) {
        initializing = PyObject_IsTrue(value);
        Py_DECREF(value);
        if (initializing < 0) {
          PyErr_Clear();
          initializing = 0;
        }
      }
      Py_DECREF(spec);
    }
    const char* fmt = initializing
        ? "cannot import name %R from partially initialized module %R "
          "(most likely due to a circular import) (%S)"
        : "cannot import name %R from %R (%S)";
    errmsg = PyUnicode_FromFormat(fmt, name, pkgname_or_unknown, pkgpath);
    path_for_error = pkgpath;
  }
  if (errmsg != NULL) {
    // _PyErr_SetImportErrorWithNameFrom.
    PyErr_SetImportError(errmsg, pkgname, path_for_error);
    PyObject* exc = PyErr_GetRaisedException();
    if (PyErr_GivenExceptionMatches(exc, PyExc_ImportError)) {
      (void)PyObject_SetAttrString(exc, "name_from", name);
    }
    PyErr_SetRaisedException(exc);
  }
  Py_XDECREF(errmsg);
  Py_XDECREF(pkgname_or_unknown);
  Py_XDECREF(pkgpath);
  return NULL;
}

static int import_all_from(PyObject* locals, PyObject* v) {
  PyObject *all, *dict, *name, *value;
  int skip_leading_underscores = 0;
  int pos, err;
  if (_PyObject_LookupAttr(v, ID(__all__), &all) < 0) {
    return -1;
  }
  if (all == NULL) {
    if (_PyObject_LookupAttr(v, ID(__dict__), &dict) < 0) {
      return -1;
    }
    if (dict == NULL) {
      PyErr_SetString(PyExc_ImportError, "from-import-* object has no __dict__ and no __all__");
      return -1;
    }
    all = PyMapping_Keys(dict);
    Py_DECREF(dict);
    if (all == NULL) {
      return -1;
    }
    skip_leading_underscores = 1;
  }
  for (pos = 0, err = 0;; pos++) {
    name = PySequence_GetItem(all, pos);
    if (name == NULL) {
      if (!PyErr_ExceptionMatches(PyExc_IndexError)) {
        err = -1;
      } else {
        PyErr_Clear();
      }
      break;
    }
    if (!PyUnicode_Check(name)) {
      PyObject* modname = PyObject_GetAttr(v, ID(__name__));
      if (modname == NULL) {
        Py_DECREF(name);
        err = -1;
        break;
      }
      if (!PyUnicode_Check(modname)) {
        PyErr_Format(
            PyExc_TypeError, "module __name__ must be a string, not %.100s",
            Py_TYPE(modname)->tp_name);
      } else {
        PyErr_Format(
            PyExc_TypeError, "%s in %U.%s must be str, not %.100s",
            skip_leading_underscores ? "Key" : "Item", modname,
            skip_leading_underscores ? "__dict__" : "__all__", Py_TYPE(name)->tp_name);
      }
      Py_DECREF(modname);
      Py_DECREF(name);
      err = -1;
      break;
    }
    if (skip_leading_underscores && PyUnicode_GET_LENGTH(name) > 0 &&
        PyUnicode_READ_CHAR(name, 0) == '_') {
      Py_DECREF(name);
      continue;
    }
    value = PyObject_GetAttr(v, name);
    if (value == NULL) {
      err = -1;
    } else if (PyDict_CheckExact(locals)) {
      err = PyDict_SetItem(locals, name, value);
    } else {
      err = PyObject_SetItem(locals, name, value);
    }
    Py_DECREF(name);
    Py_XDECREF(value);
    if (err < 0) {
      break;
    }
  }
  Py_DECREF(all);
  return err;
}

static PyObject* match_keys(PyObject* map, PyObject* keys) {
  Py_ssize_t nkeys = PyTuple_GET_SIZE(keys);
  if (!nkeys) {
    return PyTuple_New(0);
  }
  PyObject* seen = NULL;
  PyObject* dummy = NULL;
  PyObject* values = NULL;
  PyObject* get = NULL;
  // map.get(key, default): atomically check and get without triggering
  // __missing__ (e.g. collections.defaultdict).
  int meth_found = _PyObject_GetMethod(map, ID(get), &get);
  if (get == NULL) {
    goto fail;
  }
  seen = PySet_New(NULL);
  if (seen == NULL) {
    goto fail;
  }
  dummy = PyObject_CallNoArgs((PyObject*)&PyBaseObject_Type);
  if (dummy == NULL) {
    goto fail;
  }
  values = PyTuple_New(nkeys);
  if (values == NULL) {
    goto fail;
  }
  for (Py_ssize_t i = 0; i < nkeys; i++) {
    PyObject* key = PyTuple_GET_ITEM(keys, i);
    if (PySet_Contains(seen, key) || PySet_Add(seen, key)) {
      if (!PyErr_Occurred()) {
        PyErr_Format(PyExc_ValueError, "mapping pattern checks duplicate key (%R)", key);
      }
      goto fail;
    }
    PyObject* args[] = {map, key, dummy};
    PyObject* value = meth_found ? PyObject_Vectorcall(get, args, 3, NULL)
                                 : PyObject_Vectorcall(get, &args[1], 2, NULL);
    if (value == NULL) {
      goto fail;
    }
    if (value == dummy) {
      // Key not in map: the result is None.
      Py_DECREF(value);
      Py_DECREF(values);
      values = Py_NewRef(Py_None);
      goto done;
    }
    PyTuple_SET_ITEM(values, i, value);
  }
done:
  Py_DECREF(get);
  Py_DECREF(seen);
  Py_DECREF(dummy);
  return values;
fail:
  Py_XDECREF(get);
  Py_XDECREF(seen);
  Py_XDECREF(dummy);
  Py_XDECREF(values);
  return NULL;
}

static PyObject* match_class_attr(PyObject* subject, PyObject* type, PyObject* name, PyObject* seen) {
  if (PySet_Contains(seen, name) || PySet_Add(seen, name)) {
    if (!PyErr_Occurred()) {
      PyErr_Format(
          PyExc_TypeError, "%s() got multiple sub-patterns for attribute %R",
          ((PyTypeObject*)type)->tp_name, name);
    }
    return NULL;
  }
  PyObject* attr;
  (void)_PyObject_LookupAttr(subject, name, &attr);
  return attr;
}

// On a match, a tuple of extracted attributes; otherwise NULL (check
// PyErr_Occurred to tell a failed match from an error).
static PyObject* match_class(PyObject* subject, PyObject* type, Py_ssize_t nargs, PyObject* kwargs) {
  if (!PyType_Check(type)) {
    PyErr_SetString(PyExc_TypeError, "called match pattern must be a class");
    return NULL;
  }
  if (PyObject_IsInstance(subject, type) <= 0) {
    return NULL;
  }
  PyObject* seen = PySet_New(NULL);
  if (seen == NULL) {
    return NULL;
  }
  PyObject* attrs = PyList_New(0);
  if (attrs == NULL) {
    Py_DECREF(seen);
    return NULL;
  }
  PyObject* match_args = NULL;
  if (nargs) {
    int match_self = 0;
    match_args = PyObject_GetAttrString(type, "__match_args__");
    if (match_args) {
      if (!PyTuple_CheckExact(match_args)) {
        PyErr_Format(
            PyExc_TypeError, "%s.__match_args__ must be a tuple (got %s)",
            ((PyTypeObject*)type)->tp_name, Py_TYPE(match_args)->tp_name);
        goto fail;
      }
    } else if (PyErr_ExceptionMatches(PyExc_AttributeError)) {
      PyErr_Clear();
      // _Py_TPFLAGS_MATCH_SELF only counts if the type doesn't define
      // __match_args__.
      match_args = PyTuple_New(0);
      match_self = PyType_HasFeature((PyTypeObject*)type, _Py_TPFLAGS_MATCH_SELF);
    } else {
      goto fail;
    }
    Py_ssize_t allowed = match_self ? 1 : PyTuple_GET_SIZE(match_args);
    if (allowed < nargs) {
      const char* plural = (allowed == 1) ? "" : "s";
      PyErr_Format(
          PyExc_TypeError, "%s() accepts %d positional sub-pattern%s (%d given)",
          ((PyTypeObject*)type)->tp_name, allowed, plural, nargs);
      goto fail;
    }
    if (match_self) {
      if (PyList_Append(attrs, subject) < 0) {
        goto fail;
      }
    } else {
      for (Py_ssize_t i = 0; i < nargs; i++) {
        PyObject* name = PyTuple_GET_ITEM(match_args, i);
        if (!PyUnicode_CheckExact(name)) {
          PyErr_Format(
              PyExc_TypeError, "__match_args__ elements must be strings (got %s)",
              Py_TYPE(name)->tp_name);
          goto fail;
        }
        PyObject* attr = match_class_attr(subject, type, name, seen);
        if (attr == NULL) {
          goto fail;
        }
        if (PyList_Append(attrs, attr) < 0) {
          Py_DECREF(attr);
          goto fail;
        }
        Py_DECREF(attr);
      }
    }
    Py_CLEAR(match_args);
  }
  for (Py_ssize_t i = 0; i < PyTuple_GET_SIZE(kwargs); i++) {
    PyObject* name = PyTuple_GET_ITEM(kwargs, i);
    PyObject* attr = match_class_attr(subject, type, name, seen);
    if (attr == NULL) {
      goto fail;
    }
    if (PyList_Append(attrs, attr) < 0) {
      Py_DECREF(attr);
      goto fail;
    }
    Py_DECREF(attr);
  }
  Py_SETREF(attrs, PyList_AsTuple(attrs));
  Py_DECREF(seen);
  return attrs;
fail:
  Py_XDECREF(match_args);
  Py_DECREF(seen);
  Py_DECREF(attrs);
  return NULL;
}

static int check_args_iterable(PyObject* func, PyObject* args) {
  if (Py_TYPE(args)->tp_iter == NULL && !PySequence_Check(args)) {
    PyObject* funcstr = _PyObject_FunctionStr(func);
    if (funcstr != NULL) {
      PyErr_Format(
          PyExc_TypeError, "%U argument after * must be an iterable, not %.200s",
          funcstr, Py_TYPE(args)->tp_name);
      Py_DECREF(funcstr);
    }
    return -1;
  }
  return 0;
}

static void format_kwargs_error(PyObject* func, PyObject* kwargs) {
  // _PyDict_MergeEx raises AttributeError (percolated from getting 'keys')
  // instead of TypeError if its second argument is not a mapping.
  if (PyErr_ExceptionMatches(PyExc_AttributeError)) {
    PyErr_Clear();
    PyObject* funcstr = _PyObject_FunctionStr(func);
    if (funcstr != NULL) {
      PyErr_Format(
          PyExc_TypeError, "%U argument after ** must be a mapping, not %.200s",
          funcstr, Py_TYPE(kwargs)->tp_name);
      Py_DECREF(funcstr);
    }
  } else if (PyErr_ExceptionMatches(PyExc_KeyError)) {
    PyObject* exc = PyErr_GetRaisedException();
    PyObject* args = ((PyBaseExceptionObject*)exc)->args;
    if (exc && PyTuple_Check(args) && PyTuple_GET_SIZE(args) == 1) {
      PyObject* funcstr = _PyObject_FunctionStr(func);
      if (funcstr != NULL) {
        PyObject* key = PyTuple_GET_ITEM(args, 0);
        PyErr_Format(
            PyExc_TypeError, "%U got multiple values for keyword argument '%S'",
            funcstr, key);
        Py_DECREF(funcstr);
      }
      Py_XDECREF(exc);
    } else {
      PyErr_SetRaisedException(exc);
    }
  }
}


// --- Generators and coroutines ---

static int gen_is_coroutine(PyObject* o) {
  if (PyGen_CheckExact(o)) {
    _PyInterpreterFrame* frame = (_PyInterpreterFrame*)((PyGenObject*)o)->gi_iframe;
    if (frame->f_code->co_flags & CO_ITERABLE_COROUTINE) {
      return 1;
    }
  }
  return 0;
}

// _PyCoro_GetAwaitableIter.
static PyObject* coro_get_awaitable_iter(PyObject* o) {
  if (PyCoro_CheckExact(o) || gen_is_coroutine(o)) {
    return Py_NewRef(o);
  }
  PyTypeObject* ot = Py_TYPE(o);
  unaryfunc getter = ot->tp_as_async != NULL ? ot->tp_as_async->am_await : NULL;
  if (getter != NULL) {
    PyObject* res = (*getter)(o);
    if (res != NULL) {
      if (PyCoro_CheckExact(res) || gen_is_coroutine(res)) {
        // __await__ must return an *iterator*, not a coroutine or another
        // awaitable (see PEP 492).
        PyErr_SetString(PyExc_TypeError, "__await__() returned a coroutine");
        Py_CLEAR(res);
      } else if (!PyIter_Check(res)) {
        PyErr_Format(
            PyExc_TypeError, "__await__() returned non-iterator of type '%.100s'",
            Py_TYPE(res)->tp_name);
        Py_CLEAR(res);
      }
    }
    return res;
  }
  PyErr_Format(PyExc_TypeError, "object %.100s can't be used in 'await' expression", ot->tp_name);
  return NULL;
}

// _PyGen_yf: what a suspended generator is delegating to via yield from /
// await (the receiver on top of its value stack), or NULL.
static PyObject* gen_yf(PyGenObject* gen) {
  if (gen->gi_frame_state < FRAME_CLEARED) {
    _PyInterpreterFrame* frame = (_PyInterpreterFrame*)gen->gi_iframe;
    if (gen->gi_frame_state == FRAME_CREATED) {
      return NULL;
    }
    _Py_CODEUNIT next = frame->prev_instr[1];
    if ((next.op.code != RESUME && next.op.code != INSTRUMENTED_RESUME) || next.op.arg < 2) {
      return NULL;
    }
    return Py_NewRef(frame->localsplus[frame->stacktop - 1]);
  }
  return NULL;
}

static void format_awaitable_error(PyTypeObject* type, int oparg) {
  if (type->tp_as_async == NULL || type->tp_as_async->am_await == NULL) {
    if (oparg == 1) {
      PyErr_Format(
          PyExc_TypeError,
          "'async with' received an object from __aenter__ that does not implement "
          "__await__: %.100s",
          type->tp_name);
    } else if (oparg == 2) {
      PyErr_Format(
          PyExc_TypeError,
          "'async with' received an object from __aexit__ that does not implement "
          "__await__: %.100s",
          type->tp_name);
    }
  }
}

static PyObject* stopiteration_error(_PyInterpreterFrame* f, PyObject* exc) {
  int flags = f->f_code->co_flags;
  const char* msg = NULL;
  if (PyErr_GivenExceptionMatches(exc, PyExc_StopIteration)) {
    msg = "generator raised StopIteration";
    if (flags & CO_ASYNC_GENERATOR) {
      msg = "async generator raised StopIteration";
    } else if (flags & CO_COROUTINE) {
      msg = "coroutine raised StopIteration";
    }
  } else if ((flags & CO_ASYNC_GENERATOR) &&
             PyErr_GivenExceptionMatches(exc, PyExc_StopAsyncIteration)) {
    msg = "async generator raised StopAsyncIteration";
  }
  if (msg != NULL) {
    PyObject* message = PyUnicode_FromString(msg);
    if (message == NULL) {
      return NULL;
    }
    PyObject* error = PyObject_CallOneArg(PyExc_RuntimeError, message);
    Py_DECREF(message);
    if (error == NULL) {
      return NULL;
    }
    PyException_SetCause(error, Py_NewRef(exc));
    PyException_SetContext(error, Py_NewRef(exc));
    return error;
  }
  return Py_NewRef(exc);
}

// SEND: 0 if the receiver yielded *out, 1 if it returned *out, -1 on error.
int tr_send(PyObject* receiver, PyObject* v, PyObject** out) {
  PyObject* retval;
  if (Py_IsNone(v) && PyIter_Check(receiver)) {
    retval = Py_TYPE(receiver)->tp_iternext(receiver);
  } else {
    retval = PyObject_CallMethodOneArg(receiver, ID(send), v);
  }
  if (retval == NULL) {
    if (_PyGen_FetchStopIterationValue(&retval) == 0) {
      *out = retval;
      return 1;
    }
    return -1;
  }
  *out = retval;
  return 0;
}

// genobject.c's _PyAsyncGenWrappedValue.
typedef struct {
  PyObject_HEAD
  PyObject* agw_val;
} AsyncGenWrappedValue;

static PyObject* call_intrinsic_1(_PyInterpreterFrame* f, int oparg, PyObject* value) {
  switch (oparg) {
    case 0:
      PyErr_SetString(PyExc_SystemError, "invalid intrinsic function");
      return NULL;
    case 1: {  // INTRINSIC_PRINT
      PyObject* hook = PySys_GetObject("displayhook");
      if (hook == NULL) {
        PyErr_SetString(PyExc_RuntimeError, "lost sys.displayhook");
        return NULL;
      }
      Py_INCREF(hook);
      PyObject* res = PyObject_CallOneArg(hook, value);
      Py_DECREF(hook);
      return res;
    }
    case 2: {  // INTRINSIC_IMPORT_STAR (module and class scope only)
      if (f->f_locals == NULL) {
        PyErr_SetString(PyExc_SystemError, "no locals found during 'import *'");
        return NULL;
      }
      if (import_all_from(f->f_locals, value) < 0) {
        return NULL;
      }
      Py_RETURN_NONE;
    }
    case 3:  // INTRINSIC_STOPITERATION_ERROR
      return stopiteration_error(f, value);
    case 4: {  // INTRINSIC_ASYNC_GEN_WRAP (_PyAsyncGenValueWrapperNew)
      AsyncGenWrappedValue* o = PyObject_GC_New(AsyncGenWrappedValue, &_PyAsyncGenWrappedValue_Type);
      if (o == NULL) {
        return NULL;
      }
      o->agw_val = Py_NewRef(value);
      PyObject_GC_Track((PyObject*)o);
      return (PyObject*)o;
    }
    case 5:  // INTRINSIC_UNARY_POSITIVE
      return PyNumber_Positive(value);
    case 6:  // INTRINSIC_LIST_TO_TUPLE
      return PyList_AsTuple(value);
  }
  PyErr_Format(
      PyExc_NotImplementedError, "CFG executor: CALL_INTRINSIC_1 %d is not supported", oparg);
  return NULL;
}

int tr_generic(_PyInterpreterFrame* f, int opcode, int oparg, PyObject* const* in, PyObject** out) {
  switch (opcode) {
    case UNARY_NEGATIVE:
      out[0] = PyNumber_Negative(in[0]);
      return out[0] ? 0 : -1;
    case UNARY_INVERT:
      out[0] = PyNumber_Invert(in[0]);
      return out[0] ? 0 : -1;
    case UNARY_NOT: {
      int err = PyObject_IsTrue(in[0]);
      if (err < 0) {
        return -1;
      }
      out[0] = Py_NewRef(err == 0 ? Py_True : Py_False);
      return 0;
    }
    case BINARY_SUBSCR:
      out[0] = PyObject_GetItem(in[0], in[1]);
      return out[0] ? 0 : -1;
    case BINARY_SLICE: {
      PyObject* slice = PySlice_New(in[1], in[2], NULL);
      if (slice == NULL) {
        return -1;
      }
      out[0] = PyObject_GetItem(in[0], slice);
      Py_DECREF(slice);
      return out[0] ? 0 : -1;
    }
    case STORE_SLICE: {
      // (v, container, start, stop -- )
      PyObject* slice = PySlice_New(in[2], in[3], NULL);
      if (slice == NULL) {
        return -1;
      }
      int err = PyObject_SetItem(in[1], slice, in[0]);
      Py_DECREF(slice);
      return err ? -1 : 0;
    }
    case STORE_SUBSCR:
      // (v, container, sub -- )
      return PyObject_SetItem(in[1], in[2], in[0]) ? -1 : 0;
    case DELETE_SUBSCR:
      return PyObject_DelItem(in[0], in[1]) ? -1 : 0;
    case GET_LEN: {
      Py_ssize_t len = PyObject_Length(in[0]);
      if (len < 0) {
        return -1;
      }
      out[0] = PyLong_FromSsize_t(len);
      return out[0] ? 0 : -1;
    }
    case MATCH_MAPPING:
      out[0] = Py_NewRef(Py_TYPE(in[0])->tp_flags & Py_TPFLAGS_MAPPING ? Py_True : Py_False);
      return 0;
    case MATCH_SEQUENCE:
      out[0] = Py_NewRef(Py_TYPE(in[0])->tp_flags & Py_TPFLAGS_SEQUENCE ? Py_True : Py_False);
      return 0;
    case MATCH_KEYS:
      out[0] = match_keys(in[0], in[1]);
      return out[0] ? 0 : -1;
    case MATCH_CLASS: {
      // (subject, type, names -- attrs|None)
      PyObject* attrs = match_class(in[0], in[1], oparg, in[2]);
      if (attrs == NULL) {
        if (PyErr_Occurred()) {
          return -1;
        }
        attrs = Py_NewRef(Py_None);
      }
      out[0] = attrs;
      return 0;
    }
    case PUSH_EXC_INFO: {
      // (new_exc -- prev_exc, new_exc). Only the current exc_info item, not
      // the topmost handled exception (PyErr_GetHandledException walks the
      // chain, e.g. out of a generator into its caller).
      _PyErr_StackItem* exc_info = PyThreadState_GET()->exc_info;
      out[0] = exc_info->exc_value != NULL ? exc_info->exc_value : Py_NewRef(Py_None);
      exc_info->exc_value = Py_NewRef(in[0]);
      return 0;
    }
    case POP_EXCEPT: {
      _PyErr_StackItem* exc_info = PyThreadState_GET()->exc_info;
      Py_XSETREF(exc_info->exc_value, Py_NewRef(in[0]));
      return 0;
    }
    case CHECK_EXC_MATCH: {
      // (left, right -- left, b)
      if (check_except_type_valid(in[1]) < 0) {
        return -1;
      }
      out[0] = Py_NewRef(PyErr_GivenExceptionMatches(in[0], in[1]) ? Py_True : Py_False);
      return 0;
    }
    case CHECK_EG_MATCH: {
      // (exc_value, match_type -- rest, match)
      if (check_except_star_type_valid(in[1]) < 0) {
        return -1;
      }
      PyObject *match = NULL, *rest = NULL;
      if (exception_group_match(f, in[0], in[1], &match, &rest) < 0) {
        return -1;
      }
      if (!Py_IsNone(match)) {
        PyErr_SetHandledException(match);
      }
      out[0] = rest;
      out[1] = match;
      return 0;
    }
    case WITH_EXCEPT_START: {
      // Inputs used: exit_func (stack[-4]) and val (TOS).
      PyObject* exit_func = in[0];
      PyObject* val = in[1];
      PyObject* exc = PyExceptionInstance_Class(val);
      PyObject* tb = PyException_GetTraceback(val);
      PyObject* stack[3] = {exc, val, tb ? tb : Py_None};
      out[0] = PyObject_Vectorcall(exit_func, stack, 3, NULL);
      Py_XDECREF(tb);
      return out[0] ? 0 : -1;
    }
    case BEFORE_WITH: {
      // (mgr -- exit, res)
      PyObject* mgr = in[0];
      PyObject* enter = _PyObject_LookupSpecial(mgr, ID(__enter__));
      if (enter == NULL) {
        if (!PyErr_Occurred()) {
          PyErr_Format(
              PyExc_TypeError,
              "'%.200s' object does not support the context manager protocol",
              Py_TYPE(mgr)->tp_name);
        }
        return -1;
      }
      PyObject* exit = _PyObject_LookupSpecial(mgr, ID(__exit__));
      if (exit == NULL) {
        if (!PyErr_Occurred()) {
          PyErr_Format(
              PyExc_TypeError,
              "'%.200s' object does not support the context manager protocol "
              "(missed __exit__ method)",
              Py_TYPE(mgr)->tp_name);
        }
        Py_DECREF(enter);
        return -1;
      }
      PyObject* res = PyObject_CallNoArgs(enter);
      Py_DECREF(enter);
      if (res == NULL) {
        Py_DECREF(exit);
        return -1;
      }
      out[0] = exit;
      out[1] = res;
      return 0;
    }
    case LOAD_BUILD_CLASS: {
      PyObject* bc;
      if (PyDict_CheckExact(f->f_builtins)) {
        bc = PyDict_GetItemWithError(f->f_builtins, ID(__build_class__));
        if (bc == NULL) {
          if (!PyErr_Occurred()) {
            PyErr_SetString(PyExc_NameError, "__build_class__ not found");
          }
          return -1;
        }
        Py_INCREF(bc);
      } else {
        bc = PyObject_GetItem(f->f_builtins, ID(__build_class__));
        if (bc == NULL) {
          if (PyErr_ExceptionMatches(PyExc_KeyError)) {
            PyErr_SetString(PyExc_NameError, "__build_class__ not found");
          }
          return -1;
        }
      }
      out[0] = bc;
      return 0;
    }
    case LOAD_ASSERTION_ERROR:
      out[0] = Py_NewRef(PyExc_AssertionError);
      return 0;
    case SETUP_ANNOTATIONS: {
      PyObject* locals = f->f_locals;
      if (locals == NULL) {
        PyErr_Format(PyExc_SystemError, "no locals found when setting up annotations");
        return -1;
      }
      PyObject* ann_dict;
      if (PyDict_CheckExact(locals)) {
        ann_dict = PyDict_GetItemWithError(locals, ID(__annotations__));
        if (ann_dict == NULL) {
          if (PyErr_Occurred()) {
            return -1;
          }
          ann_dict = PyDict_New();
          if (ann_dict == NULL) {
            return -1;
          }
          int err = PyDict_SetItem(locals, ID(__annotations__), ann_dict);
          Py_DECREF(ann_dict);
          return err ? -1 : 0;
        }
        return 0;
      }
      ann_dict = PyObject_GetItem(locals, ID(__annotations__));
      if (ann_dict == NULL) {
        if (!PyErr_ExceptionMatches(PyExc_KeyError)) {
          return -1;
        }
        PyErr_Clear();
        ann_dict = PyDict_New();
        if (ann_dict == NULL) {
          return -1;
        }
        int err = PyObject_SetItem(locals, ID(__annotations__), ann_dict);
        Py_DECREF(ann_dict);
        return err ? -1 : 0;
      }
      Py_DECREF(ann_dict);
      return 0;
    }
    case LOAD_LOCALS:
      if (f->f_locals == NULL) {
        PyErr_SetString(PyExc_SystemError, "no locals found");
        return -1;
      }
      out[0] = Py_NewRef(f->f_locals);
      return 0;
    case STORE_NAME: {
      PyObject* name = co_name(f, oparg);
      PyObject* ns = f->f_locals;
      if (ns == NULL) {
        PyErr_Format(PyExc_SystemError, "no locals found when storing %R", name);
        return -1;
      }
      int err = PyDict_CheckExact(ns) ? PyDict_SetItem(ns, name, in[0])
                                      : PyObject_SetItem(ns, name, in[0]);
      return err ? -1 : 0;
    }
    case DELETE_NAME: {
      PyObject* name = co_name(f, oparg);
      PyObject* ns = f->f_locals;
      if (ns == NULL) {
        PyErr_Format(PyExc_SystemError, "no locals when deleting %R", name);
        return -1;
      }
      if (PyObject_DelItem(ns, name) != 0) {
        format_exc_check_arg(PyExc_NameError, NAME_ERROR_MSG, name);
        return -1;
      }
      return 0;
    }
    case LOAD_FROM_DICT_OR_GLOBALS: {
      PyObject* name = co_name(f, oparg);
      PyObject* ns = in[0];
      PyObject* v;
      if (PyDict_CheckExact(ns)) {
        v = PyDict_GetItemWithError(ns, name);
        if (v != NULL) {
          Py_INCREF(v);
        } else if (PyErr_Occurred()) {
          return -1;
        }
      } else {
        v = PyObject_GetItem(ns, name);
        if (v == NULL) {
          if (!PyErr_ExceptionMatches(PyExc_KeyError)) {
            return -1;
          }
          PyErr_Clear();
        }
      }
      out[0] = v != NULL ? v : load_global(f, name);
      return out[0] ? 0 : -1;
    }
    case LOAD_NAME: {
      PyObject* name = co_name(f, oparg);
      PyObject* ns = f->f_locals;
      if (ns == NULL) {
        PyErr_SetString(PyExc_SystemError, "no locals found");
        return -1;
      }
      PyObject* v;
      if (PyDict_CheckExact(ns)) {
        v = PyDict_GetItemWithError(ns, name);
        if (v != NULL) {
          Py_INCREF(v);
        } else if (PyErr_Occurred()) {
          return -1;
        }
      } else {
        v = PyObject_GetItem(ns, name);
        if (v == NULL) {
          if (!PyErr_ExceptionMatches(PyExc_KeyError)) {
            return -1;
          }
          PyErr_Clear();
        }
      }
      if (v == NULL) {
        v = PyDict_GetItemWithError(f->f_globals, name);
        if (v != NULL) {
          Py_INCREF(v);
        } else if (PyErr_Occurred()) {
          return -1;
        } else if (PyDict_CheckExact(f->f_builtins)) {
          v = PyDict_GetItemWithError(f->f_builtins, name);
          if (v == NULL) {
            if (!PyErr_Occurred()) {
              format_exc_check_arg(PyExc_NameError, NAME_ERROR_MSG, name);
            }
            return -1;
          }
          Py_INCREF(v);
        } else {
          v = PyObject_GetItem(f->f_builtins, name);
          if (v == NULL) {
            if (PyErr_ExceptionMatches(PyExc_KeyError)) {
              format_exc_check_arg(PyExc_NameError, NAME_ERROR_MSG, name);
            }
            return -1;
          }
        }
      }
      out[0] = v;
      return 0;
    }
    case STORE_GLOBAL:
      return PyDict_SetItem(f->f_globals, co_name(f, oparg), in[0]) ? -1 : 0;
    case DELETE_GLOBAL: {
      PyObject* name = co_name(f, oparg);
      if (PyDict_DelItem(f->f_globals, name) != 0) {
        if (PyErr_ExceptionMatches(PyExc_KeyError)) {
          PyErr_Clear();
          format_exc_check_arg(PyExc_NameError, NAME_ERROR_MSG, name);
        }
        return -1;
      }
      return 0;
    }
    case DELETE_ATTR:
      return PyObject_DelAttr(in[0], co_name(f, oparg)) ? -1 : 0;
    case UNPACK_SEQUENCE: {
      // Outputs bottom first, i.e. the last item first.
      PyObject* items[oparg > 0 ? oparg : 1];
      if (unpack_iterable(in[0], oparg, -1, items) < 0) {
        return -1;
      }
      for (int i = 0; i < oparg; i++) {
        out[i] = items[oparg - 1 - i];
      }
      return 0;
    }
    case UNPACK_EX: {
      int before = oparg & 0xFF, after = oparg >> 8, n = before + 1 + after;
      PyObject* items[n];
      if (unpack_iterable(in[0], before, after, items) < 0) {
        return -1;
      }
      for (int i = 0; i < n; i++) {
        out[i] = items[n - 1 - i];
      }
      return 0;
    }
    case BUILD_LIST: {
      PyObject* list = PyList_New(oparg);
      if (list == NULL) {
        return -1;
      }
      for (int i = 0; i < oparg; i++) {
        PyList_SET_ITEM(list, i, Py_NewRef(in[i]));
      }
      out[0] = list;
      return 0;
    }
    case BUILD_SET: {
      PyObject* set = PySet_New(NULL);
      if (set == NULL) {
        return -1;
      }
      for (int i = 0; i < oparg; i++) {
        if (PySet_Add(set, in[i]) < 0) {
          Py_DECREF(set);
          return -1;
        }
      }
      out[0] = set;
      return 0;
    }
    case BUILD_MAP: {
      PyObject* map = PyDict_New();
      if (map == NULL) {
        return -1;
      }
      for (int i = 0; i < oparg; i++) {
        if (PyDict_SetItem(map, in[2 * i], in[2 * i + 1]) < 0) {
          Py_DECREF(map);
          return -1;
        }
      }
      out[0] = map;
      return 0;
    }
    case BUILD_CONST_KEY_MAP: {
      PyObject* keys = in[oparg];
      if (!PyTuple_CheckExact(keys) || PyTuple_GET_SIZE(keys) != (Py_ssize_t)oparg) {
        PyErr_SetString(PyExc_SystemError, "bad BUILD_CONST_KEY_MAP keys argument");
        return -1;
      }
      PyObject* map = PyDict_New();
      if (map == NULL) {
        return -1;
      }
      for (int i = 0; i < oparg; i++) {
        if (PyDict_SetItem(map, PyTuple_GET_ITEM(keys, i), in[i]) < 0) {
          Py_DECREF(map);
          return -1;
        }
      }
      out[0] = map;
      return 0;
    }
    case BUILD_SLICE:
      out[0] = PySlice_New(in[0], in[1], oparg == 3 ? in[2] : NULL);
      return out[0] ? 0 : -1;
    case BUILD_STRING: {
      PyObject* empty = PyUnicode_FromString("");
      if (empty == NULL) {
        return -1;
      }
      PyObject* pieces = PyTuple_New(oparg);
      if (pieces == NULL) {
        Py_DECREF(empty);
        return -1;
      }
      for (int i = 0; i < oparg; i++) {
        PyTuple_SET_ITEM(pieces, i, Py_NewRef(in[i]));
      }
      out[0] = PyUnicode_Join(empty, pieces);
      Py_DECREF(empty);
      Py_DECREF(pieces);
      return out[0] ? 0 : -1;
    }
    case LIST_APPEND:
      return PyList_Append(in[0], in[1]) ? -1 : 0;
    case SET_ADD:
      return PySet_Add(in[0], in[1]) ? -1 : 0;
    case MAP_ADD:
      return PyDict_SetItem(in[0], in[1], in[2]) ? -1 : 0;
    case LIST_EXTEND: {
      PyObject* none_val = _PyList_Extend((PyListObject*)in[0], in[1]);
      if (none_val == NULL) {
        if (PyErr_ExceptionMatches(PyExc_TypeError) && Py_TYPE(in[1])->tp_iter == NULL &&
            !PySequence_Check(in[1])) {
          PyErr_Clear();
          PyErr_Format(
              PyExc_TypeError, "Value after * must be an iterable, not %.200s",
              Py_TYPE(in[1])->tp_name);
        }
        return -1;
      }
      Py_DECREF(none_val);
      return 0;
    }
    case SET_UPDATE:
      return _PySet_Update(in[0], in[1]) < 0 ? -1 : 0;
    case DICT_UPDATE:
      if (PyDict_Update(in[0], in[1]) < 0) {
        if (PyErr_ExceptionMatches(PyExc_AttributeError)) {
          PyErr_Format(
              PyExc_TypeError, "'%.200s' object is not a mapping", Py_TYPE(in[1])->tp_name);
        }
        return -1;
      }
      return 0;
    case DICT_MERGE:
      // (func, unused, dict, unused[oparg-1], update): see opcodes.rs.
      if (_PyDict_MergeEx(in[1], in[2], 2) < 0) {
        format_kwargs_error(in[0], in[2]);
        return -1;
      }
      return 0;
    case IMPORT_NAME:
#ifdef EAGER_IMPORT_NAME
    // Meta's lazy-imports CPython: the decoder maps both to the eager import.
    case EAGER_IMPORT_NAME:
#endif
      // (level, fromlist -- res)
      out[0] = import_name(f, co_name(f, oparg), in[1], in[0]);
      return out[0] ? 0 : -1;
    case IMPORT_FROM:
      out[0] = import_from(in[0], co_name(f, oparg));
      return out[0] ? 0 : -1;
    case IS_OP:
      out[0] = Py_NewRef((Py_Is(in[0], in[1]) ^ oparg) ? Py_True : Py_False);
      return 0;
    case CONTAINS_OP: {
      int res = PySequence_Contains(in[1], in[0]);
      if (res < 0) {
        return -1;
      }
      out[0] = Py_NewRef((res ^ oparg) ? Py_True : Py_False);
      return 0;
    }
    case FORMAT_VALUE: {
      PyObject* value = in[0];
      PyObject* fmt_spec = (oparg & 0x04) ? in[1] : NULL;
      PyObject* converted = NULL;
      switch (oparg & 0x03) {
        case 1: converted = PyObject_Str(value); break;
        case 2: converted = PyObject_Repr(value); break;
        case 3: converted = PyObject_ASCII(value); break;
      }
      if ((oparg & 0x03) != 0) {
        if (converted == NULL) {
          return -1;
        }
        value = converted;
      }
      if (PyUnicode_CheckExact(value) && fmt_spec == NULL) {
        out[0] = Py_NewRef(value);
      } else {
        out[0] = PyObject_Format(value, fmt_spec);
      }
      Py_XDECREF(converted);
      return out[0] ? 0 : -1;
    }
    case CALL_FUNCTION_EX: {
      // (unused, func, callargs, kwargs?)
      PyObject* func = in[0];
      PyObject* callargs = in[1];
      PyObject* kwargs = (oparg & 1) ? in[2] : NULL;
      PyObject* tuple;
      if (PyTuple_CheckExact(callargs)) {
        tuple = Py_NewRef(callargs);
      } else {
        if (check_args_iterable(func, callargs) < 0) {
          return -1;
        }
        tuple = PySequence_Tuple(callargs);
        if (tuple == NULL) {
          return -1;
        }
      }
      out[0] = PyObject_Call(func, tuple, kwargs);
      Py_DECREF(tuple);
      return out[0] ? 0 : -1;
    }
    case CALL_INTRINSIC_1:
      out[0] = call_intrinsic_1(f, oparg, in[0]);
      return out[0] ? 0 : -1;
    case CALL_INTRINSIC_2:
      if (oparg == 1) {  // INTRINSIC_PREP_RERAISE_STAR
        out[0] = PyUnstable_Exc_PrepReraiseStar(in[0], in[1]);
        return out[0] ? 0 : -1;
      }
      if (oparg == 4) {  // INTRINSIC_SET_FUNCTION_TYPE_PARAMS
        PyFunctionObject* func = (PyFunctionObject*)in[0];
        Py_XSETREF(func->func_typeparams, Py_NewRef(in[1]));
        out[0] = Py_NewRef(in[0]);
        return 0;
      }
      PyErr_Format(
          PyExc_NotImplementedError, "CFG executor: CALL_INTRINSIC_2 %d is not supported", oparg);
      return -1;
    case GET_YIELD_FROM_ITER: {
      PyObject* iterable = in[0];
      if (PyCoro_CheckExact(iterable)) {
        if (!(f->f_code->co_flags & (CO_COROUTINE | CO_ITERABLE_COROUTINE))) {
          PyErr_SetString(
              PyExc_TypeError,
              "cannot 'yield from' a coroutine object in a non-coroutine generator");
          return -1;
        }
        out[0] = Py_NewRef(iterable);
      } else if (PyGen_CheckExact(iterable)) {
        out[0] = Py_NewRef(iterable);
      } else {
        out[0] = PyObject_GetIter(iterable);
      }
      return out[0] ? 0 : -1;
    }
    case GET_AWAITABLE: {
      PyObject* iter = coro_get_awaitable_iter(in[0]);
      if (iter == NULL) {
        format_awaitable_error(Py_TYPE(in[0]), oparg);
      }
      if (iter != NULL && PyCoro_CheckExact(iter)) {
        PyObject* yf = gen_yf((PyGenObject*)iter);
        if (yf != NULL) {
          Py_DECREF(yf);
          Py_CLEAR(iter);
          PyErr_SetString(PyExc_RuntimeError, "coroutine is being awaited already");
        }
      }
      out[0] = iter;
      return iter ? 0 : -1;
    }
    case GET_AITER: {
      PyTypeObject* type = Py_TYPE(in[0]);
      unaryfunc getter = type->tp_as_async != NULL ? type->tp_as_async->am_aiter : NULL;
      if (getter == NULL) {
        PyErr_Format(
            PyExc_TypeError, "'async for' requires an object with __aiter__ method, got %.100s",
            type->tp_name);
        return -1;
      }
      PyObject* iter = (*getter)(in[0]);
      if (iter == NULL) {
        return -1;
      }
      if (Py_TYPE(iter)->tp_as_async == NULL || Py_TYPE(iter)->tp_as_async->am_anext == NULL) {
        PyErr_Format(
            PyExc_TypeError,
            "'async for' received an object from __aiter__ that does not implement "
            "__anext__: %.100s",
            Py_TYPE(iter)->tp_name);
        Py_DECREF(iter);
        return -1;
      }
      out[0] = iter;
      return 0;
    }
    case GET_ANEXT: {
      // (aiter -- aiter, awaitable)
      PyObject* aiter = in[0];
      PyTypeObject* type = Py_TYPE(aiter);
      if (PyAsyncGen_CheckExact(aiter)) {
        out[0] = type->tp_as_async->am_anext(aiter);
        return out[0] ? 0 : -1;
      }
      unaryfunc getter = type->tp_as_async != NULL ? type->tp_as_async->am_anext : NULL;
      if (getter == NULL) {
        PyErr_Format(
            PyExc_TypeError, "'async for' requires an iterator with __anext__ method, got %.100s",
            type->tp_name);
        return -1;
      }
      PyObject* next_iter = (*getter)(aiter);
      if (next_iter == NULL) {
        return -1;
      }
      PyObject* awaitable = coro_get_awaitable_iter(next_iter);
      if (awaitable == NULL) {
        _PyErr_FormatFromCause(
            PyExc_TypeError, "'async for' received an invalid object from __anext__: %.100s",
            Py_TYPE(next_iter)->tp_name);
      }
      Py_DECREF(next_iter);
      out[0] = awaitable;
      return awaitable ? 0 : -1;
    }
    case END_ASYNC_FOR:
      // (awaitable, exc -- )
      if (PyErr_GivenExceptionMatches(in[1], PyExc_StopAsyncIteration)) {
        return 0;
      }
      PyErr_SetRaisedException(Py_NewRef(in[1]));
      return -2;
    case BEFORE_ASYNC_WITH: {
      PyObject* mgr = in[0];
      PyObject* enter = _PyObject_LookupSpecial(mgr, ID(__aenter__));
      if (enter == NULL) {
        if (!PyErr_Occurred()) {
          PyErr_Format(
              PyExc_TypeError,
              "'%.200s' object does not support the asynchronous context manager protocol",
              Py_TYPE(mgr)->tp_name);
        }
        return -1;
      }
      PyObject* exit = _PyObject_LookupSpecial(mgr, ID(__aexit__));
      if (exit == NULL) {
        if (!PyErr_Occurred()) {
          PyErr_Format(
              PyExc_TypeError,
              "'%.200s' object does not support the asynchronous context manager protocol "
              "(missed __aexit__ method)",
              Py_TYPE(mgr)->tp_name);
        }
        Py_DECREF(enter);
        return -1;
      }
      PyObject* res = PyObject_CallNoArgs(enter);
      Py_DECREF(enter);
      if (res == NULL) {
        Py_DECREF(exit);
        return -1;
      }
      out[0] = exit;
      out[1] = res;
      return 0;
    }
    case CLEANUP_THROW:
      // (sub_iter, last_sent_val, exc_value -- none, value); last_sent_val
      // may be NULL (see Op::Yield's handler args).
      if (PyErr_GivenExceptionMatches(in[2], PyExc_StopIteration)) {
        out[0] = Py_NewRef(Py_None);
        out[1] = Py_NewRef(((PyStopIterationObject*)in[2])->value);
        return 0;
      }
      PyErr_SetRaisedException(Py_NewRef(in[2]));
      return -2;
    case LOAD_SUPER_ATTR: {
      // (global_super, class, self -- NULL?, res)
      PyObject* name = co_name(f, oparg >> 2);
      PyObject* stack[2] = {in[1], in[2]};
      PyObject* super = PyObject_Vectorcall(in[0], stack, oparg & 2, NULL);
      if (super == NULL) {
        return -1;
      }
      PyObject* res = PyObject_GetAttr(super, name);
      Py_DECREF(super);
      if (res == NULL) {
        return -1;
      }
      if (oparg & 1) {
        out[0] = NULL;
        out[1] = res;
      } else {
        out[0] = res;
      }
      return 0;
    }
  }
  PyErr_Format(
      PyExc_NotImplementedError, "CFG executor: opcode %d is not supported", opcode);
  return -1;
}

// What a traced region read (torch/_rust/src/exec/reads.rs). None of these
// runs Python code.

// How a load of `name` from obj is guarded:
//   -1 it can't be: obj's class resolves attributes its own way (unless
//      `direct`: the load is object.__getattribute__'s own, whatever the
//      class does with other loads)
//    0 nothing to guard: obj's type can't change and obj has no __dict__
//    1 by obj's class alone (a method, a class attribute, the __getattr__ it
//      falls back to), and by `name` not being in *dict (if obj has one)
//    2 by *dict, obj's __dict__, which has it
//    3 by a data descriptor of obj's class (a slot, a getset, a property)
//    4 by *dict, the module obj's namespace
//    5 obj is a class: by that class
// *dict is a new reference.
int tr_attr_home(PyObject* obj, PyObject* name, PyObject** dict, int direct) {
  static PyObject *getattribute_name, *generic;
  *dict = NULL;
  if (generic == NULL) {
    getattribute_name = PyUnicode_InternFromString("__getattribute__");
    if (getattribute_name == NULL) {
      PyErr_Clear();
      return -1;
    }
    generic = _PyType_Lookup(&PyBaseObject_Type, getattribute_name);
  }
  if (PyType_Check(obj)) {
    return 5;
  }
  if (PyModule_Check(obj)) {
    PyObject* d = PyModule_GetDict(obj);
    if (d == NULL) {
      PyErr_Clear();
      return -1;
    }
    *dict = Py_NewRef(d);
    return 4;
  }
  PyTypeObject* tp = Py_TYPE(obj);
  if (!direct && tp->tp_getattro != PyObject_GenericGetAttr) {
    // A class with a __getattr__ has a slot of its own that still starts
    // with the generic lookup unless __getattribute__ is overridden too.
    PyObject* g = _PyType_Lookup(tp, getattribute_name);
    if (g != generic &&
        !(g != NULL && Py_IS_TYPE(g, &PyWrapperDescr_Type) &&
          ((PyWrapperDescrObject*)g)->d_wrapped == (void*)PyObject_GenericGetAttr)) {
      return -1;
    }
  }
  PyObject* descr = _PyType_Lookup(tp, name);
  if (descr != NULL && Py_TYPE(descr)->tp_descr_set != NULL) {
    return 3;
  }
  if (tp->tp_dictoffset == 0) {
    return PyType_HasFeature(tp, Py_TPFLAGS_IMMUTABLETYPE) ? 0 : 1;
  }
  PyObject* d = PyObject_GenericGetDict(obj, NULL);
  if (d == NULL || !PyDict_Check(d)) {
    PyErr_Clear();
    Py_XDECREF(d);
    return -1;
  }
  int has = PyDict_Contains(d, name);
  if (has < 0) {
    PyErr_Clear();
    Py_DECREF(d);
    return -1;
  }
  *dict = d;
  return has ? 2 : 1;
}

// tp's version tag, which CPython changes when tp or one of its bases is
// modified; 0 if tp can't have one. An immutable type's never changes.
unsigned int tr_type_version(PyTypeObject* tp) {
  return PyUnstable_Type_AssignVersionTag(tp) ? tp->tp_version_tag : 0;
}

int tr_type_version_is(PyTypeObject* tp, unsigned int tag) {
  return PyType_HasFeature(tp, Py_TPFLAGS_VALID_VERSION_TAG) && tp->tp_version_tag == tag;
}

// What class tp (with its bases) has under `name` (borrowed), or NULL.
PyObject* tr_type_lookup(PyTypeObject* tp, PyObject* name) {
  return _PyType_Lookup(tp, name);
}

int tr_type_heap(PyTypeObject* tp) {
  return PyType_HasFeature(tp, Py_TPFLAGS_HEAPTYPE);
}

// The class after `cls` in the order obj's class resolves methods in (what
// super() in a method of `cls` starts from), borrowed; NULL if there is none.
PyObject* tr_super_next(PyObject* cls, PyObject* obj) {
  PyObject* mro = Py_TYPE(obj)->tp_mro;
  if (mro == NULL || !PyTuple_Check(mro)) {
    return NULL;
  }
  Py_ssize_t n = PyTuple_GET_SIZE(mro);
  for (Py_ssize_t i = 0; i + 1 < n; i++) {
    if (PyTuple_GET_ITEM(mro, i) == cls) {
      return PyTuple_GET_ITEM(mro, i + 1);
    }
  }
  return NULL;
}

int tr_type_immutable(PyTypeObject* tp) {
  return PyType_HasFeature(tp, Py_TPFLAGS_IMMUTABLETYPE);
}

// A dict's version tag: another one after every change to the dict.
uint64_t tr_dict_version(PyObject* d) {
  return ((PyDictObject*)d)->ma_version_tag;
}

// obj's __dict__ as it is now (borrowed), or NULL if it has none. No code
// of obj's class runs.
PyObject* tr_instance_dict(PyObject* obj) {
  PyObject** at = _PyObject_GetDictPtr(obj);
  return at == NULL ? NULL : *at;
}

// The dict a view (keys, values, items) is of (borrowed), or NULL.
PyObject* tr_view_dict(PyObject* o) {
  if (PyDictViewSet_Check(o) || PyDictValues_Check(o)) {
    return (PyObject*)((_PyDictViewObject*)o)->dv_dict;
  }
  return NULL;
}

// Whether o is an OrderedDict: a dict that keeps an order of its own, which
// is not the order of the dict's table (move_to_end changes only the former).
int tr_ordered(PyObject* o) {
  return PyODict_Check(o);
}

// A new iterator over the keys of an OrderedDict in its order: the type's
// own, so that no __iter__ of a subclass runs.
PyObject* tr_ordered_keys(PyObject* o) {
  return PyODict_Type.tp_iter(o);
}

// descrobject.c's mappingproxyobject.
typedef struct {
  PyObject_HEAD
  PyObject* mapping;
} tr_mappingproxyobject;

// The dict a mappingproxy (a class's __dict__) shows (borrowed), or NULL.
PyObject* tr_proxy_dict(PyObject* o) {
  if (Py_IS_TYPE(o, &PyDictProxy_Type)) {
    PyObject* m = ((tr_mappingproxyobject*)o)->mapping;
    return PyDict_Check(m) ? m : NULL;
  }
  return NULL;
}

// Whether descriptor d is a slot a class declares in `__slots__`: a member
// holding an object, which reads as an AttributeError while unset.
static int tr_is_slot(PyObject* d) {
  return Py_IS_TYPE(d, &PyMemberDescr_Type) &&
      ((PyMemberDescrObject*)d)->d_member->type == Py_T_OBJECT_EX &&
      !(((PyMemberDescrObject*)d)->d_member->flags & Py_READONLY);
}

// How `obj.name = value` or `del obj.name` (`del_`) is carried out. 1: the
// plain store into the instance's dict by the generic rule (the class has no
// __setattr__ of its own and no data descriptor under the name, and the
// object has a dict); with `generic`, the call is object.__setattr__ itself,
// whatever the class defines. 3: the generic store into a slot of the class
// (`__slots__`). 0: Python code of the class runs (its __setattr__, a
// property), whose own stores are what happens. 2: C code other than these
// (and the deletion of a slot). Runs no Python code.
int tr_store_attr_kind(PyTypeObject* tp, PyObject* name, int generic, int del_) {
  if (!generic) {
    PyObject* special = PyUnicode_InternFromString(del_ ? "__delattr__" : "__setattr__");
    PyObject* f = special == NULL ? NULL : _PyType_Lookup(tp, special);
    Py_XDECREF(special);
    if (f == NULL) {
      PyErr_Clear();
      return 2;
    }
    if (PyFunction_Check(f)) {
      return 0;
    }
    if (tp->tp_setattro != PyObject_GenericSetAttr) {
      return 2;
    }
  }
  PyObject* d = _PyType_Lookup(tp, name);
  if (d != NULL && tr_is_slot(d)) {
    return del_ ? 2 : 3;
  }
  if (d != NULL && Py_TYPE(d)->tp_descr_set != NULL) {
    return !generic && Py_IS_TYPE(d, &PyProperty_Type) ? 0 : 2;
  }
  return tp->tp_dictoffset != 0 || PyType_HasFeature(tp, Py_TPFLAGS_MANAGED_DICT) ? 1 : 2;
}

// What the slot `name` of obj's class holds for obj (a new reference), or
// NULL: nothing, or `name` is no slot. Runs no Python code.
PyObject* tr_slot_get(PyObject* obj, PyObject* name) {
  PyObject* d = _PyType_Lookup(Py_TYPE(obj), name);
  if (d == NULL || !tr_is_slot(d)) {
    return NULL;
  }
  PyObject* x = Py_TYPE(d)->tp_descr_get(d, obj, (PyObject*)Py_TYPE(obj));
  if (x == NULL) {
    PyErr_Clear();
  }
  return x;
}

// Whether subscript stores into (and deletions from) an instance of tp run
// Python code of the class (its __setitem__ / __delitem__).
int tr_python_setitem(PyTypeObject* tp, int del_) {
  PyObject* special = PyUnicode_InternFromString(del_ ? "__delitem__" : "__setitem__");
  PyObject* f = special == NULL ? NULL : _PyType_Lookup(tp, special);
  Py_XDECREF(special);
  if (f == NULL) {
    PyErr_Clear();
    return 0;
  }
  return PyFunction_Check(f);
}

// The name of a method or slot wrapper descriptor (borrowed).
PyObject* tr_descr_name(PyObject* descr) {
  return ((PyDescrObject*)descr)->d_name;
}

// The object a builtin method is bound to (borrowed), or NULL.
PyObject* tr_builtin_self(PyObject* callable) {
  if (Py_IS_TYPE(callable, &_PyMethodWrapper_Type)) {
    return ((tr_wrapperobject*)callable)->self;
  }
  if (PyCFunction_Check(callable)) {
    PyObject* self = PyCFunction_GET_SELF(callable);
    return self != NULL && !PyModule_Check(self) ? self : NULL;
  }
  return NULL;
}

// Whether `callable` is C code that may read what it is handed: a builtin
// function or method, a slot wrapper, or a type that isn't a Python class. A
// Python function, a method of one, a Python class and an instance called
// through its class's __call__ hand their arguments to a frame instead.
int tr_calls_builtin(PyObject* callable) {
  if (PyType_Check(callable)) {
    return !PyType_HasFeature((PyTypeObject*)callable, Py_TPFLAGS_HEAPTYPE);
  }
  return PyCFunction_Check(callable) || Py_IS_TYPE(callable, &PyMethodDescr_Type) ||
      Py_IS_TYPE(callable, &PyClassMethodDescr_Type) || Py_IS_TYPE(callable, &PyWrapperDescr_Type) ||
      Py_IS_TYPE(callable, &_PyMethodWrapper_Type);
}
