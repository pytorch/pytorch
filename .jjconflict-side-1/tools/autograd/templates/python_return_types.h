#pragma once

namespace torch {
namespace autograd {
namespace generated {

${py_return_types_declarations}

}

void initReturnTypes(PyObject* module);

} // namespace autograd
} // namespace torch
