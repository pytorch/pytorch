#pragma once

namespace torch {
namespace autograd {

PyTypeObject* get_namedtuple(std::string name);
void initReturnTypes(PyObject* module);

} // namespace autograd
} // namespace torch
