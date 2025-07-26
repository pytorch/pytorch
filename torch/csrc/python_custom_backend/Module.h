#pragma once
#include <torch/csrc/python_headers.h>


namespace torch::python_custom_backend {
// PyMethodDef* python_functions();
void initModule(PyObject* module);

}