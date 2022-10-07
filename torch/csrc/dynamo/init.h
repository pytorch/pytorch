#pragma once
#include <Python.h>

namespace torch {
namespace dynamo {
void initDynamoBindings(PyObject* torch);
}
} // namespace torch
