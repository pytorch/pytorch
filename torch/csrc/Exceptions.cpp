#include <Python.h>

#include "THP.h"

PyObject *THPException_FatalError;

bool THPException_init(PyObject *module)
{
  THPException_FatalError = PyErr_NewException("torch.FatalError",
      PyExc_BaseException, NULL);
  return THPException_FatalError != NULL;
}
