#ifndef THP_NUMPY_INC
#define THP_NUMPY_INC

#include <type_traits>
#include <memory>

#ifndef WITH_NUMPY_IMPORT_ARRAY
#define NO_IMPORT_ARRAY
#endif
#define PY_ARRAY_UNIQUE_SYMBOL __numpy_array_api
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

// Adapted from fblualib
class NumpyArrayAllocator {
public:
  NumpyArrayAllocator(PyObject *wrapped_array) {
      Py_INCREF(wrapped_array);
      array = wrapped_array;
  }

  void* malloc(long size);
  void* realloc(void* ptr, long size);
  void free(void* ptr);

  THPObjectPtr array;
};

extern THAllocator THNumpyArrayAllocator;

#endif
