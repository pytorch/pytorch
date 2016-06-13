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

#include "Exceptions.h"

#include "Storage.h"
#include "Tensor.h"

// This requires defined Storage and Tensor types
#include "utils.h"
