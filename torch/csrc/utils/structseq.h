#pragma once

#include "torch/csrc/python_headers.h"

namespace torch { namespace utils {

#if PY_MAJOR_VERSION == 2
PyObject *make_tuple(PyStructSequence *obj);
#endif

PyObject *returned_structseq_repr(PyStructSequence *obj);

}}
