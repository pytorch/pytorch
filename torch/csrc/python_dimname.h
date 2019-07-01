#pragma once
#ifdef BUILD_NAMEDTENSOR
#include <torch/csrc/python_headers.h>
#include <ATen/Dimname.h>

at::Dimname THPDimname_parse(PyObject* obj);

#endif
