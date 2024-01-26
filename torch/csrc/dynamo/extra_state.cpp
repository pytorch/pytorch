#include <torch/csrc/dynamo/extra_state.h>

#include <torch/csrc/dynamo/cache_entry.h>
#include <torch/csrc/dynamo/debug_macros.h>

Py_ssize_t extra_index = -1;

CacheEntry* extract_cache_entry(ExtraState* extra_state) {
  if (extra_state == NULL || extra_state == SKIP_CODE) {
    return NULL;
  }
  return extra_state->cache_entry;
}

FrameState* extract_frame_state(ExtraState* extra_state) {
  if (extra_state == NULL || extra_state == SKIP_CODE) {
    return NULL;
  }
  return extra_state->frame_state;
}

ExtraState* get_extra_state(PyCodeObject* code) {
  ExtraState* extra = NULL;
  _PyCode_GetExtra((PyObject*)code, extra_index, (void**)&extra);
  return extra;
}

void destroy_extra_state(void* obj) {
  ExtraState* extra = (ExtraState*)obj;
  if (extra != NULL && extra != SKIP_CODE) {
    // Cpython gc will call cache_entry_dealloc on its own when the ref count
    // goes to 0.
    Py_XDECREF(extra->cache_entry);
    Py_XDECREF(extra->frame_state);
    free(extra);
  }
}

void set_extra_state(PyCodeObject* code, ExtraState* extra_state) {
  ExtraState* old_extra_state = get_extra_state(code);
  CHECK(old_extra_state == NULL || old_extra_state == SKIP_CODE || old_extra_state != extra_state);
  _PyCode_SetExtra((PyObject*)code, extra_index, extra_state);
}

ExtraState* init_and_set_extra_state(PyCodeObject* code) {
  // Invariant - Extra state should not have been set before, therefore it should be NULL.
  CHECK(get_extra_state(code) == NULL);
  ExtraState* extra_state = (ExtraState*)malloc(sizeof(ExtraState));
  DEBUG_NULL_CHECK(extra_state);
  // We set the last node in the linked list to Py_None. We incref the Py_None
  // here, the corresponding decref is in cache_entry_dealloc.
  Py_INCREF(Py_None);
  extra_state->cache_entry = (CacheEntry*)Py_None;
  extra_state->frame_state = PyDict_New();
  set_extra_state(code, extra_state);
  return extra_state;
}

PyObject* _debug_get_cache_entry_list(PyObject* self, PyObject* args) {
  PyObject* object = NULL;
  if (!PyArg_ParseTuple(args, "O", &object)) {
    return NULL;
  }
  if (!PyCode_Check(object)) {
    PyErr_SetString(PyExc_TypeError, "expected a code object!");
    return NULL;
  }
  PyCodeObject* code = (PyCodeObject*)object;

  ExtraState* extra = get_extra_state(code);
  CacheEntry* current_node = extract_cache_entry(extra);
  if (current_node == NULL)
  {
    Py_RETURN_NONE;
  }
  Py_INCREF(current_node);
  return (PyObject*)current_node;
}
