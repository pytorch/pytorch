#pragma once

#include <Python.h>
#include <memory>

#include "torch/csrc/autograd/variable.h"

struct THPNode {
    PyObject_HEAD
    std::shared_ptr<torch::autograd::Node> cdata;
};

extern PyObject *THPNodeClass;

bool THPNode_initModule(PyObject *module);
PyObject * THPNode_Wrap(const std::shared_ptr<torch::autograd::Node>& node);
