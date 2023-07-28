#pragma once

#include <torch/csrc/utils/pybind.h>

void bind_cpartition(pybind11::module& m);
void bind_engine(pybind11::module& m);
void bind_graph(pybind11::module& m);
void bind_logical_tensor(pybind11::module& m);
void bind_op(pybind11::module& m);
void bind_partition(pybind11::module& m);
void bind_stream(pybind11::module& m);
void bind_tensor(pybind11::module& m);

void initOnednnPythonBindings(PyObject* module);
