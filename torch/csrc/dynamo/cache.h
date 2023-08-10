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


void* prepare_cache(CacheEntry* cache_entry);
CacheEntry* read_cache(void* cache_map);

#ifdef __cplusplus
}
#endif
