#ifndef THP_H
#define THP_H

#include <stdbool.h>
#include <TH/TH.h>

// Back-compatibility macros, Thanks to http://cx-oracle.sourceforge.net/
// define PyInt_* macros for Python 3.x
#ifndef PyInt_Check
#define PyInt_Check             PyLong_Check
#define PyInt_FromLong          PyLong_FromLong
#define PyInt_AsLong            PyLong_AsLong
#define PyInt_Type              PyLong_Type
#endif

// By default, don't specify library state (TH doesn't use one)
#define LIBRARY_STATE
#define LIBRARY_STATE_NOARGS

#define THP_API extern "C"

#include "Exceptions.h"
#include "Generator.h"
#include "Storage.h"
#include "Tensor.h"
#include "Module.h"
#include "utils.h" // This requires defined Storage and Tensor types
#include "byte_order.h"

#ifdef _THP_CORE
#include "serialization.h"
#include "allocators.h"
#endif

#endif
