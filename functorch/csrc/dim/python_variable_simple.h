// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once
// note: pytorch's python variable simple includes pybind which conflicts with minpybind
// so this file just reproduces the minimial API needed to extract Tensors from python objects.

#include <torch/csrc/python_headers.h>
#include <ATen/core/Tensor.h>
#include <torch/csrc/Export.h>

// Python object that backs torch.autograd.Variable
// NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
struct THPVariable {
  PyObject_HEAD;
  // Payload
  c10::MaybeOwned<at::Tensor> cdata;
  // Hooks to be run on backwards pass (corresponds to Python attr
  // '_backwards_hooks', set by 'register_hook')
  PyObject* backward_hooks = nullptr;
};

TORCH_PYTHON_API extern PyObject *THPVariableClass;
TORCH_PYTHON_API extern PyObject *ParameterClass;

TORCH_PYTHON_API PyObject * THPVariable_Wrap(at::TensorBase var);

inline bool THPVariable_Check(PyObject *obj)
{
  if (!THPVariableClass)
      return false;

  const auto result = PyObject_IsInstance(obj, THPVariableClass);
  AT_ASSERT(result != -1);
  return result;
}

inline const at::Tensor& THPVariable_Unpack(THPVariable* var) {
  return *var->cdata;
}

inline const at::Tensor& THPVariable_Unpack(PyObject* obj) {
  return THPVariable_Unpack(reinterpret_cast<THPVariable*>(obj));
}

TORCH_PYTHON_API c10::impl::PyInterpreter* getPyInterpreter();
