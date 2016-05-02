#include <Python.h>

#include <stdbool.h>
#include <TH/TH.h>

#include "THP.h"

#define ASSERT_TRUE(cmd) if (!(cmd)) return NULL

static PyMethodDef TorchMethods[] = {
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef torchmodule = {
   PyModuleDef_HEAD_INIT,
   "torch.C",
   NULL,
   -1,
   TorchMethods
};

PyMODINIT_FUNC PyInit_C()
{
  PyObject* m;
  ASSERT_TRUE(m = PyModule_Create(&torchmodule));

  ASSERT_TRUE(THPDoubleStorage_init(m));
  ASSERT_TRUE(THPFloatStorage_init(m));
  ASSERT_TRUE(THPLongStorage_init(m));
  ASSERT_TRUE(THPIntStorage_init(m));
  ASSERT_TRUE(THPCharStorage_init(m));
  ASSERT_TRUE(THPByteStorage_init(m));

  ASSERT_TRUE(THPDoubleTensor_init(m));
  ASSERT_TRUE(THPFloatTensor_init(m));
  ASSERT_TRUE(THPLongTensor_init(m));
  ASSERT_TRUE(THPIntTensor_init(m));
  ASSERT_TRUE(THPCharTensor_init(m));
  ASSERT_TRUE(THPByteTensor_init(m));

  return m;
}
