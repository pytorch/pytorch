#include <torch/csrc/dynamo/cache.h>
#include <iostream>
#include <unordered_map>
#include <memory>
using namespace std;


int foo(int i) {
    return 2 * i;
}

typedef std::unordered_map<PyObject*, CacheEntry*> CacheMap;
PyObject* sentinel_value = Py_True;


void* prepare_cache(CacheEntry* cache_entry) {
  CacheMap* cache_map = new CacheMap();
  cache_map->insert(make_pair(sentinel_value, cache_entry));
  return (void*)cache_map;
}

CacheEntry* read_cache(void* cache_map) {
  CacheMap* cache_map_ptr = (CacheMap*)cache_map;
  return cache_map_ptr->at(sentinel_value);
}

// unordered_map<int, int> memo;
// typedef struct cache_entry {
//   // check the guards: lambda: <locals of user function>: bool
//   PyObject* check_fn;
//   // modified user bytecode (protected by check_fn's guards)
//   PyCodeObject* code;
//   // on a cache miss, linked list of next thing to try
//   struct cache_entry* next;
// } CacheEntry;

// #ifdef __cplusplus
// }
// #endif

// int main() {
//   std::cout << "Answere  = " << foo(2) << "\n";;
// }
