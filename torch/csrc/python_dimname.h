#pragma once
#include <ATen/Dimname.h>
#include <torch/csrc/python_headers.h>

at::Dimname THPDimname_parse(PyObject* obj);
bool THPUtils_checkDimname(PyObject* obj);
bool THPUtils_checkDimnameList(PyObject* obj);
