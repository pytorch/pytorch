#include "Module.h"

#include <Python.h>
#include "Types.h"
#include "Conv.h"
#include "CppWrapper.h"
#include "torch/csrc/THP.h"

bool THCUDNNModule_initModule(PyObject *module)
{
  return THPWrapper_init(module);
}
