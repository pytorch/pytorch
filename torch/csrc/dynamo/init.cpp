#include <torch/csrc/dynamo/init.h>

#include <torch/csrc/Exceptions.h>
#include <torch/csrc/dynamo/eval_frame.h>
#include <torch/csrc/dynamo/guards.h>
#include <torch/csrc/dynamo/python_compiled_autograd.h>

static struct PyModuleDef _module =
    {PyModuleDef_HEAD_INIT, "torch._C._dynamo", "", -1, NULL};

namespace torch {
namespace dynamo {

void initDynamoBindings(PyObject* torch) {
  PyObject* dynamo = PyModule_Create(&_module);
  if (dynamo == NULL || PyModule_AddObject(torch, "_dynamo", dynamo) != 0) {
    throw python_error();
  }

  PyObject* eval_frame = torch_c_dynamo_eval_frame_init();
  if (eval_frame == NULL ||
      PyModule_AddObject(dynamo, "eval_frame", eval_frame) != 0) {
    throw python_error();
  }

  PyObject* guards = torch_c_dynamo_guards_init();
  if (guards == NULL || PyModule_AddObject(dynamo, "guards", guards) != 0) {
    throw python_error();
  }

  PyObject* compiled_autograd = torch_c_dynamo_compiled_autograd_init();
  if (compiled_autograd == nullptr ||
      PyModule_AddObject(dynamo, "compiled_autograd", compiled_autograd) != 0) {
    throw python_error();
  }
}

} // namespace dynamo
} // namespace torch
