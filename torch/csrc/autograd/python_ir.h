#pragma once

#include <Python.h>
#include <memory>

#include "torch/csrc/autograd/variable.h"

struct THPExpr {
    PyObject_HEAD
    std::shared_ptr<torch::autograd::Expr> cdata;
};

extern PyObject *THPExprClass;

PyObject * THPExpr_Wrap(const std::shared_ptr<torch::autograd::Expr>& node);

inline bool THPExpr_Check(PyObject *obj)
{
  return THPExprClass && PyObject_IsInstance(obj, THPExprClass);
}

struct THPArg {
    PyObject_HEAD
    std::shared_ptr<torch::autograd::Arg> cdata;
};

extern PyObject *THPArgClass;

PyObject * THPArg_Wrap(const std::shared_ptr<torch::autograd::Arg>& node);

bool THPIR_initModule(PyObject *module);
