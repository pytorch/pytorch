// What a slot callee of a call list (torch/_rust/src/exec/calls) would run
// for operands of given exact classes, decided from the classes alone, on
// CPython 3.12. No code of the program runs here.
//
// A class that defines a special method in Python (or holds anything but a
// slot wrapper under its name) gets one of typeobject.c's dispatchers in the
// slot: slot_nb_add, slot_tp_richcompare, slot_mp_subscript and so on. Their
// addresses are read off a class made here that holds None under every name
// below. A slot that is one of them is resolved one step further, as the
// dispatcher does it: the attribute of the class under the method's name
// must be a method descriptor, a slot wrapper or a builtin function, or be
// absent (the dispatcher then answers NotImplemented or raises, in C).
//
// Covered, and nothing else:
//   binary        nb_<op> of both classes, nb_inplace_<op> of the left one
//                 (sq_concat, sq_repeat and their in-place forms are never a
//                 dispatcher: typeobject.c has no function for them)
//   compare       tp_richcompare of both classes (for != also __eq__, which
//                 object.__ne__ calls)
//   unary         nb_negative, nb_invert, nb_positive
//   subscripts    mp_subscript / sq_item, mp_ass_subscript / sq_ass_item of
//                 the container's class; a container that is itself a class
//                 is refused (__class_getitem__)
//   attributes    only a class whose tp_getattro / tp_setattro is object's,
//                 and then only a name that the class resolves to nothing,
//                 to a plain value, or to a descriptor of a C type listed in
//                 attribute(): a property or any other __get__ / __set__ is
//                 refused
// What the C function found does with the OTHER operands (a key's __index__,
// an iterable's __iter__, an operand's __torch_function__) is not decided
// here: that is the license of the callee, as for any C callable.

#include <Python.h>

#include <stddef.h>
#include <stdio.h>

#if PY_VERSION_HEX < 0x030C0000 || PY_VERSION_HEX >= 0x030D0000
#error "calls_slots.c supports CPython 3.12 only"
#endif

// In the order of BINARY_OP's oparg (NB_ADD .. NB_XOR; + 13: in place).
static const char* const BINARY[13][3] = {
    {"__add__", "__radd__", "__iadd__"},
    {"__and__", "__rand__", "__iand__"},
    {"__floordiv__", "__rfloordiv__", "__ifloordiv__"},
    {"__lshift__", "__rlshift__", "__ilshift__"},
    {"__matmul__", "__rmatmul__", "__imatmul__"},
    {"__mul__", "__rmul__", "__imul__"},
    {"__mod__", "__rmod__", "__imod__"},
    {"__or__", "__ror__", "__ior__"},
    {"__pow__", "__rpow__", "__ipow__"},
    {"__rshift__", "__rrshift__", "__irshift__"},
    {"__sub__", "__rsub__", "__isub__"},
    {"__truediv__", "__rtruediv__", "__itruediv__"},
    {"__xor__", "__rxor__", "__ixor__"},
};
#define NB(slot) offsetof(PyNumberMethods, slot)
static const size_t BINARY_SLOT[13][2] = {
    {NB(nb_add), NB(nb_inplace_add)},
    {NB(nb_and), NB(nb_inplace_and)},
    {NB(nb_floor_divide), NB(nb_inplace_floor_divide)},
    {NB(nb_lshift), NB(nb_inplace_lshift)},
    {NB(nb_matrix_multiply), NB(nb_inplace_matrix_multiply)},
    {NB(nb_multiply), NB(nb_inplace_multiply)},
    {NB(nb_remainder), NB(nb_inplace_remainder)},
    {NB(nb_or), NB(nb_inplace_or)},
    {NB(nb_power), NB(nb_inplace_power)},
    {NB(nb_rshift), NB(nb_inplace_rshift)},
    {NB(nb_subtract), NB(nb_inplace_subtract)},
    {NB(nb_true_divide), NB(nb_inplace_true_divide)},
    {NB(nb_xor), NB(nb_inplace_xor)},
};
// In the order of Py_LT .. Py_GE: the method, and the one of the other side.
static const char* const COMPARE[6][2] = {
    {"__lt__", "__gt__"}, {"__le__", "__ge__"}, {"__eq__", "__eq__"},
    {"__ne__", "__ne__"}, {"__gt__", "__lt__"}, {"__ge__", "__le__"},
};
static const char* const UNARY[3] = {"__neg__", "__invert__", "__pos__"};
static const size_t UNARY_SLOT[3] = {NB(nb_negative), NB(nb_invert), NB(nb_positive)};

// A class whose every slot named above is CPython's dispatcher, and the
// deallocator every struct sequence type has. Made once, kept.
static PyTypeObject* dispatchers;
static destructor structseq_dealloc;
static char why[200];

static int ready(void) {
  if (dispatchers != NULL) {
    return 1;
  }
  static PyStructSequence_Field fields[] = {{"item", NULL}, {NULL, NULL}};
  static PyStructSequence_Desc desc = {"torch._rust.Sequence", NULL, fields, 1};
  PyTypeObject* sequence = PyStructSequence_NewType(&desc);
  PyObject* names = PyDict_New();
  int failed = sequence == NULL || names == NULL;
  for (int i = 0; i < 13 && !failed; i++) {
    failed = PyDict_SetItemString(names, BINARY[i][0], Py_None) < 0 ||
        PyDict_SetItemString(names, BINARY[i][2], Py_None) < 0;
  }
  static const char* const more[] = {"__neg__", "__invert__", "__pos__", "__lt__", "__getitem__", "__setitem__"};
  for (int i = 0; i < 6 && !failed; i++) {
    failed = PyDict_SetItemString(names, more[i], Py_None) < 0;
  }
  PyObject* args = failed ? NULL : Py_BuildValue("(s()O)", "Dispatchers", names);
  // type's own constructor: it runs __set_name__ and __init_subclass__, and
  // None and object have none in Python.
  PyObject* made = args == NULL ? NULL : PyType_Type.tp_new(&PyType_Type, args, NULL);
  Py_XDECREF(args);
  Py_XDECREF(names);
  if (made == NULL) {
    Py_XDECREF(sequence);
    PyErr_Clear();
    return 0;
  }
  structseq_dealloc = sequence->tp_dealloc;
  dispatchers = (PyTypeObject*)made;
  return 1;
}

static const char* refuse(const char* what, const char* name, PyTypeObject* tp) {
  snprintf(why, sizeof(why), "%s%s%s of the class %.80s", what, name == NULL ? "" : " ", name == NULL ? "" : name, tp->tp_name);
  return why;
}

// NULL if what `tp` holds under `name` is C code or nothing.
static const char* method(PyTypeObject* tp, const char* name) {
  PyObject* key = PyUnicode_InternFromString(name);
  if (key == NULL) {
    PyErr_Clear();
    return refuse("no name for", name, tp);
  }
  PyObject* found = _PyType_Lookup(tp, key);
  Py_DECREF(key);
  if (found == NULL) {
    return NULL;
  }
  PyTypeObject* kind = Py_TYPE(found);
  if (kind == &PyMethodDescr_Type || kind == &PyWrapperDescr_Type || kind == &PyCFunction_Type || kind == &PyCMethod_Type) {
    return NULL;
  }
  return refuse("no C function:", name, tp);
}

static void* at(void* table, size_t offset) {
  return table == NULL ? NULL : *(void**)((char*)table + offset);
}

// A number slot: absent, the C function of a C type, or the dispatcher, which
// calls `name`.
static const char* number(PyTypeObject* tp, size_t offset, const char* name) {
  void* slot = at(tp->tp_as_number, offset);
  return slot == NULL || slot != at(dispatchers->tp_as_number, offset) ? NULL : method(tp, name);
}

static const char* compare(PyTypeObject* tp, int op, int other) {
  if (tp->tp_richcompare != dispatchers->tp_richcompare) {
    return NULL;
  }
  const char* r = method(tp, COMPARE[op][other]);
  return r != NULL || op != Py_NE ? r : method(tp, "__eq__");
}

static const char* subscript(PyTypeObject* tp, int store, const char* name) {
  if (PyType_HasFeature(tp, Py_TPFLAGS_TYPE_SUBCLASS)) {
    return refuse("a subscript of a class:", "__class_getitem__", tp);
  }
  PyMappingMethods* m = tp->tp_as_mapping;
  PySequenceMethods* s = tp->tp_as_sequence;
  PyMappingMethods* dm = dispatchers->tp_as_mapping;
  PySequenceMethods* ds = dispatchers->tp_as_sequence;
  int python = store
      ? ((m != NULL && m->mp_ass_subscript != NULL) ? m->mp_ass_subscript == dm->mp_ass_subscript
            : (s != NULL && s->sq_ass_item != NULL && s->sq_ass_item == ds->sq_ass_item))
      : ((m != NULL && m->mp_subscript != NULL) ? m->mp_subscript == dm->mp_subscript
            : (s != NULL && s->sq_item != NULL && s->sq_item == ds->sq_item));
  return python ? method(tp, name) : NULL;
}

static const char* attribute(PyTypeObject* tp, PyObject* name, int store) {
  if (name == NULL || !PyUnicode_CheckExact(name)) {
    return refuse("no attribute name for", NULL, tp);
  }
  if (store ? tp->tp_setattro != PyObject_GenericSetAttr : tp->tp_getattro != PyObject_GenericGetAttr) {
    return refuse(store ? "not covered: its own __setattr__," : "not covered: its own __getattribute__ or __getattr__,", NULL, tp);
  }
  PyObject* found = _PyType_Lookup(tp, name);
  if (found == NULL) {
    return NULL;
  }
  PyTypeObject* kind = Py_TYPE(found);
  if (store) {
    return kind->tp_descr_set == NULL || kind == &PyGetSetDescr_Type || kind == &PyMemberDescr_Type
        ? NULL : refuse("not covered: the __set__ of what the name is", NULL, tp);
  }
  return kind->tp_descr_get == NULL || kind == &PyGetSetDescr_Type || kind == &PyMemberDescr_Type ||
          kind == &PyMethodDescr_Type || kind == &PyWrapperDescr_Type || kind == &PyClassMethodDescr_Type ||
          kind == &PyFunction_Type || kind == &PyStaticMethod_Type
      ? NULL : refuse("not covered: the __get__ of what the name is", NULL, tp);
}

// kind: 0 binary (op: BINARY_OP's oparg), 1 compare (op: Py_LT ..), 2 unary
// (op: 0 "-", 1 "~", 2 "+"), 3 getitem, 4 setitem, 5 delitem, 6 getattr,
// 7 setattr (name). `a` is the class of the operand that decides (the left
// one, the container, the owner), `b` the right one's for 0 and 1.
// NULL: no Python-level method of a class is run by the slot itself. Else
// the reason, valid until the next call.
const char* tr_slot_python(int kind, int op, PyTypeObject* a, PyTypeObject* b, PyObject* name) {
  if (!ready()) {
    return "the table of CPython's dispatchers could not be made";
  }
  const char* r = NULL;
  switch (kind) {
    case 0: {
      if (op < 0 || op >= 26 || b == NULL) {
        break;
      }
      int base = op % 13;
      if (op >= 13 && (r = number(a, BINARY_SLOT[base][1], BINARY[base][2])) != NULL) {
        return r;
      }
      if ((r = number(a, BINARY_SLOT[base][0], BINARY[base][0])) != NULL) {
        return r;
      }
      return a == b ? NULL : number(b, BINARY_SLOT[base][0], BINARY[base][1]);
    }
    case 1:
      if (op < 0 || op >= 6 || b == NULL) {
        break;
      }
      if ((r = compare(a, op, 0)) != NULL) {
        return r;
      }
      return a == b ? NULL : compare(b, op, 1);
    case 2:
      if (op < 0 || op >= 3) {
        break;
      }
      return number(a, UNARY_SLOT[op], UNARY[op]);
    case 3:
      return subscript(a, 0, "__getitem__");
    case 4:
      return subscript(a, 1, "__setitem__");
    case 5:
      return subscript(a, 1, "__delitem__");
    case 6:
      return attribute(a, name, 0);
    case 7:
      return attribute(a, name, 1);
    default:
      break;
  }
  return "no slot of that kind";
}

// Whether instances of `tp` are tuples read by C code only: tuple itself, or
// a struct sequence type (those cannot be subclassed).
int tr_tuple_like(PyTypeObject* tp) {
  return tp == &PyTuple_Type || (ready() && tp->tp_dealloc == structseq_dealloc);
}
