#pragma once

#include <torch/csrc/python_headers.h>

// This file contains utilities used for handling PyObject preservation

void clear_slots(PyTypeObject* type, PyObject* self);
