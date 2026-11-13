#pragma once

#include <torch/csrc/python_headers.h>

#ifdef USE_NUMPY

#if !defined(NO_IMPORT_ARRAY) && !defined(WITH_NUMPY_IMPORT_ARRAY)
#define NO_IMPORT_ARRAY
#endif

#ifndef PY_ARRAY_UNIQUE_SYMBOL
#define PY_ARRAY_UNIQUE_SYMBOL __numpy_array_api
#endif

#ifndef NPY_NO_DEPRECATED_API
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#endif

#include <numpy/arrayobject.h>

#endif // USE_NUMPY
