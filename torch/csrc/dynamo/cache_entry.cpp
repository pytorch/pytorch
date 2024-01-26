#include <torch/csrc/dynamo/cache_entry.h>

#include <torch/csrc/dynamo/debug_macros.h>

#define DECLARE_CACHE_ENTRY_ATTR(name) \
static PyObject* CacheEntry_##name(CacheEntry* self, PyObject* _noargs) { \
  PyObject* res = (PyObject*)self->name; \
  Py_INCREF(res); \
  return res; \
}

DECLARE_CACHE_ENTRY_ATTR(check_fn)
DECLARE_CACHE_ENTRY_ATTR(code)
DECLARE_CACHE_ENTRY_ATTR(next)

static struct PyGetSetDef CacheEntry_properties[] = {
    {"check_fn", (getter)CacheEntry_check_fn, NULL, NULL, NULL},
    {"code", (getter)CacheEntry_code, NULL, NULL, NULL},
    {"next", (getter)CacheEntry_next, NULL, NULL, NULL},
    {NULL}};


static PyObject* cache_entry_new(PyTypeObject* type, PyObject* args, PyObject* kwargs) {
  CacheEntry *self = (CacheEntry*) type->tp_alloc(type, 0);
  if (self != NULL) {
    // The corresponding decrefs for Py_None are in cache_entry_init.
    Py_INCREF(Py_None);
    self->check_fn = Py_None;
    Py_INCREF(Py_None);
    self->code = (PyCodeObject*)Py_None;
    Py_INCREF(Py_None);
    self->next = (CacheEntry*)Py_None;
  }
  return (PyObject*)self;
}


static int cache_entry_init(CacheEntry* self, PyObject* args, PyObject* kwds) {
  PyObject* check_fn = NULL;
  PyCodeObject* code = NULL;
  CacheEntry* next = NULL;

  static char *kwlist[] = {"check_fn", "code", "next", NULL};

  int ret = PyArg_ParseTupleAndKeywords(
    args, kwds, "OOO", kwlist,
    &check_fn, &code, &next);

  if (!ret) return -1;

  if (check_fn) {
    PyObject* tmp = self->check_fn;
    Py_INCREF(check_fn);
    self->check_fn = check_fn;
    Py_XDECREF(tmp);
  }

  if (code) {
    PyCodeObject* tmp = self->code;
    Py_INCREF(code);
    self->code = code;
    Py_XDECREF(tmp);
  }

  if (next) {
    CacheEntry* tmp = self->next;
    Py_INCREF(next);
    self->next = next;
    Py_XDECREF(tmp);
  }
  return 0;
}

CacheEntry* create_cache_entry(
    CacheEntry* next,
    PyObject* guarded_code) {
  PyObject* check_fn = PyObject_GetAttrString(guarded_code, "check_fn"); // new reference
  PyCodeObject* code = (PyCodeObject*)PyObject_GetAttrString(guarded_code, "code"); // new reference

  // equivalent to CacheEntry(check_fn, code, next) in Python
  PyObject* args = Py_BuildValue("OOO", check_fn, code, next);
  CacheEntry* e = (CacheEntry*)PyObject_CallObject((PyObject*)&CacheEntryType, args); // new reference
  // CacheEntry e is the now the owner of old cachey entry next. This happens
  // when we incref the next pointer in cache_entry_init.
  Py_DECREF(next);
  Py_DECREF(check_fn);
  Py_DECREF(code);
  Py_DECREF(args);
  return e;
}

static void cache_entry_dealloc(CacheEntry* e) {
  Py_XDECREF(e->check_fn);
  Py_XDECREF(e->code);
  // This will recursively call cache_entry_dealloc for the next items in the
  // linked list.
  Py_XDECREF(e->next);
  Py_TYPE(e)->tp_free((PyObject*)e);
}

PyTypeObject CacheEntryType = {
  PyVarObject_HEAD_INIT(NULL, 0)
  .tp_name = "torch._C.dynamo.eval_frame.CacheEntryWrapper",
  .tp_basicsize = sizeof(CacheEntry),
  .tp_itemsize = 0,
  .tp_dealloc = (destructor)cache_entry_dealloc,
  .tp_flags = Py_TPFLAGS_DEFAULT,
  .tp_getset = CacheEntry_properties,
  .tp_init = (initproc)cache_entry_init,
  .tp_new = cache_entry_new,
};
