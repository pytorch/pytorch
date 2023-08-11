#pragma once
#include <Python.h>

#ifdef __cplusplus
extern "C" {
#endif

int foo(int i);

typedef struct cache_entry {
  // check the guards: lambda: <locals of user function>: bool
  PyObject* check_fn;
  // modified user bytecode (protected by check_fn's guards)
  PyCodeObject* code;
  // on a cache miss, linked list of next thing to try
  struct cache_entry* next;
} CacheEntry;

typedef struct {
  // Cache entry for the code object
  // TODO - void* because I cant forward declare an unordered_map as this header
  // runs with gcc compiler (not g++).
  void* cache;
  // Guarded nn module index in co_varnames
  PyObject* guarded_nn_module_var_index;
} ExtraState;

typedef struct {
  CacheEntry* cache_entry;
  // Frame state to detect dynamic shape dims
  PyObject* frame_state;
} CompileUnit;


void* prepare_cache(CacheEntry* cache_entry, PyObject* frame_state, PyObject* maybe_nn_module);
// CompileUnit* read_cache(void* cache_map);
CompileUnit* extract_compile_unit(PyObject** fastlocals, ExtraState* state);
CacheEntry* extract_cache_entry(CompileUnit* compile_unit);
PyObject* extract_frame_state(CompileUnit* compile_unit);
PyObject* extract_guarded_nn_module_var_index(ExtraState* extra);
void destroy_cache(ExtraState* extra);
PyObject* get_nn_module_if_frame_is_method_of_nn_module(PyObject** fastlocals, PyObject* var_index);
void destroy_cache_entry(CacheEntry* e);

#ifdef __cplusplus
}
#endif
