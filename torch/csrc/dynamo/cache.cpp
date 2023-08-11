// #include <python3.10/fileutils.h>
// #include <python3.10/object.h>
#include <torch/csrc/dynamo/cache.h>
#include <exception>
#include <iostream>
#include <unordered_map>
#include <memory>
using namespace std;


int foo(int i) {
    return 2 * i;
}

typedef std::unordered_map<PyObject*, CompileUnit*> CacheMap;
PyObject* sentinel_value = Py_True;

// Flag to just run a frame normally
#define SKIP_CODE ((void*)0x1)

// inline static PyObject* get_nn_module_if_frame_is_method_of_nn_module(PyObject** fastlocals, PyObject* var_index) {
//   size_t index = PyLong_AsSize_t(var_index);
//   PyObject* self_object = fastlocals[index];
//   if (self_object == NULL) {
//     return NULL;
//   }
//   return self_object;
// }



PyObject* get_nn_module_if_frame_is_method_of_nn_module(PyObject** fastlocals, PyObject* var_index) {
  if (var_index == Py_None || var_index == NULL) {
    return NULL;
  }
  size_t index = PyLong_AsSize_t(var_index);
  PyObject* self_object = fastlocals[index];
  if (self_object == NULL) {
    return NULL;
  }
  // fprintf(stderr, "Found nn module - %p\n", self_object);
  return self_object;
}

void* prepare_cache(CacheEntry* cache_entry, PyObject* frame_state, PyObject* maybe_nn_module) {
  CacheMap* cache_map = new CacheMap();
  CompileUnit* cache_value = new CompileUnit();
  cache_value->cache_entry = cache_entry;
  cache_value->frame_state = frame_state;
  if (maybe_nn_module == NULL) {
    cache_map->insert(make_pair(sentinel_value, cache_value));
  } else {
    // fprintf(stderr, "Inserting nn module - %p\n", maybe_nn_module);
    cache_map->insert(make_pair(maybe_nn_module, cache_value));
  }
  return (void*)cache_map;
}

CompileUnit* read_cache(void* cache_map) {
  CacheMap* cache_map_ptr = (CacheMap*)cache_map;
  return cache_map_ptr->at(sentinel_value);
}

CompileUnit* extract_compile_unit(PyObject** fastlocals, ExtraState* extra) {
  if (extra == NULL || extra == SKIP_CODE) {
    return NULL;
  }
  PyObject* guarded_nn_module_var_index = extract_guarded_nn_module_var_index(extra);
  PyObject* maybe_nn_module = get_nn_module_if_frame_is_method_of_nn_module(fastlocals,  guarded_nn_module_var_index);
  CacheMap* cache_map_ptr = (CacheMap*)extra->cache;
  if (maybe_nn_module == NULL) {
    return cache_map_ptr->at(sentinel_value);
  }
  if (cache_map_ptr->find(maybe_nn_module) == cache_map_ptr->end()) {
    return NULL;
  }
  return cache_map_ptr->at(maybe_nn_module);
}

CacheEntry* extract_cache_entry(CompileUnit* compile_unit) {
  // Helper to extra the cache_entry from the extra state.
  if (compile_unit == NULL) {
    return NULL;
  }
  return compile_unit->cache_entry;
}

PyObject* extract_frame_state(CompileUnit* compile_unit) {
  PyObject *frame_state = NULL;
  if (compile_unit != NULL && compile_unit->frame_state != NULL) {
    frame_state = compile_unit->frame_state;
  } else {
    frame_state = PyDict_New();
  }
  return frame_state;
}

PyObject* extract_guarded_nn_module_var_index(ExtraState* extra) {
  if (extra == NULL) {
    return NULL;
  }
  return extra->guarded_nn_module_var_index;
}


void destroy_cache_entry(CacheEntry* e) {
  if (e == NULL || e == SKIP_CODE) {
    return;
  }
  Py_XDECREF(e->check_fn);
  Py_XDECREF(e->code);
  destroy_cache_entry(e->next);
  free(e);
}

void destroy_cache(ExtraState* extra) {
  if (extra == NULL || extra == SKIP_CODE) {
    return;
  }
  CacheMap* cache_map_ptr = (CacheMap*)extra->cache;
  for (auto key_compile_unit = cache_map_ptr->begin(); key_compile_unit!= cache_map_ptr->end(); ++key_compile_unit) {
    PyObject* key = key_compile_unit->first;
    CompileUnit* compile_unit = key_compile_unit->second;
    CacheEntry* cache_entry = extract_cache_entry(compile_unit);
    destroy_cache_entry(cache_entry);
    PyObject* frame_state = extract_frame_state(compile_unit);
    Py_XDECREF(frame_state);
    // TODO - Uncommenting next line causes segfault. Investigate why?
    // Py_XDECREF(key);
    free(compile_unit);
  }
  free(cache_map_ptr);
}
