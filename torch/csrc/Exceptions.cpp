#include <Python.h>

#include "THP.h"

PyObject *THPException_FatalError;

#define ASSERT_TRUE(cond) if (!(cond)) return false
bool THPException_init(PyObject *module)
{
  ASSERT_TRUE(THPException_FatalError = PyErr_NewException("torch.FatalError", NULL, NULL));
  ASSERT_TRUE(PyModule_AddObject(module, "FatalError", THPException_FatalError) == 0);
  return true;
}
