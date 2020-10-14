#pragma once

#include <Python.h>
#include <torch/csrc/WindowsTorchApiMacro.h>

PyCFunction convertPyCFunctionWithKeywords(PyCFunctionWithKeywords func);
