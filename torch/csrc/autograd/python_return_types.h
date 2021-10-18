#pragma once

namespace torch { namespace autograd {

PyTypeObject* get_namedtuple(const char* name);
void initReturnTypes(PyObject* module);

}} // namespace torch::autograd
