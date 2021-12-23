#ifndef THP_H
#define THP_H

#include <torch/csrc/python_headers.h>
#include <torch/csrc/Export.h>

// Back-compatibility macros, Thanks to http://cx-oracle.sourceforge.net/
// define PyInt_* macros for Python 3.x.  NB: We must include Python.h first,
// otherwise we'll incorrectly conclude PyInt_Check isn't defined!
#ifndef PyInt_Check
#define PyInt_Check             PyLong_Check
#define PyInt_FromLong          PyLong_FromLong
#define PyInt_AsLong            PyLong_AsLong
#define PyInt_Type              PyLong_Type
#endif

#include <torch/csrc/Exceptions.h>
#include <torch/csrc/Generator.h>
#include <torch/csrc/Module.h>
#include <torch/csrc/Size.h>
#include <torch/csrc/Storage.h>
#include <torch/csrc/Types.h>
#include <torch/csrc/utils.h> // This requires defined Storage and Tensor types
#include <torch/csrc/utils/byte_order.h>

#include <torch/csrc/serialization.h>

#include <torch/csrc/autograd/python_autograd.h>

#endif
