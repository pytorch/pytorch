#include <Python.h>

#include "THP.h"


PyObject * THPAutograd_initExtension(PyObject *_unused)
{
  PyObject *autograd_module = PyImport_ImportModule("torch.autograd");
  THPUtils_assert(autograd_module, "class loader couldn't access "
          "torch.autograd module");
  PyObject *autograd_dict = PyModule_GetDict(autograd_module);

  THPVariableClass      = PyMapping_GetItemString(autograd_dict,(char*)"Variable");
  THPFunctionClass      = PyMapping_GetItemString(autograd_dict,(char*)"Function");

  PyObject *thnn_functions = PyImport_ImportModule("torch.nn._functions.thnn");
  THPUtils_assert(thnn_functions, "class loader couldn't access "
      "torch.nn._functions.thnn module");
  THPBatchNormBackwardBackwardFunction = PyObject_GetAttrString(thnn_functions,(char*)"batchnorm_double_backwards_fn");

  THPStochasticFunctionClass = PyMapping_GetItemString(autograd_dict,(char*)"StochasticFunction");
  THPUtils_assert(THPVariableClass, "couldn't find Variable class in "
          "torch.autograd module");
  THPUtils_assert(THPFunctionClass, "couldn't find Function class in "
          "torch.autograd module");
  THPUtils_assert(THPStochasticFunctionClass, "couldn't find "
          "StochasticFunction class in torch.autograd module");

  Py_RETURN_TRUE;
}
