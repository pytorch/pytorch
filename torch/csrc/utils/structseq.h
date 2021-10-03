#pragma once

#include <torch/csrc/python_headers.h>

namespace torch { namespace utils {

PyObject *returned_structseq_repr(PyStructSequence *obj);

}}
