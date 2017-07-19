#include <Python.h>

#include "THP.h"
#include "torch/csrc/jit/python_ir.h"


PyObject * THPJIT_initExtension(PyObject *_unused)
{
  PyObject *jit_module = PyImport_ImportModule("torch.jit");
  THPUtils_assert(jit_module, "class loader couldn't access "
          "torch.jit module");
  PyObject *jit_dict = PyModule_GetDict(jit_module);

  THPGraphClass = PyMapping_GetItemString(jit_dict,(char*)"Graph");
  THPUtils_assert(THPGraphClass, "couldn't find "
          "Graph class in torch.jit module");

  Py_RETURN_TRUE;
}
