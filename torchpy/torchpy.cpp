#include <torchpy.h>
#include <Python.h>
#include <torch/torch.h>
#include <iostream>

void torchpy::init() {
  Py_Initialize();
  PyRun_SimpleString(
      "from time import time,ctime\n"
      "print('Today is',ctime(time()))\n");
  Py_Finalize();
}
