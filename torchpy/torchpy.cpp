#include <torchpy.h>
#include <Python.h>
#include <assert.h>
#include <torch/csrc/jit/api/module.h>
#include <torch/script.h>
#include <torch/torch.h>
#include <iostream>

void torchpy::init() {
  Py_Initialize();
  PyRun_SimpleString(
      "from time import time,ctime\n"
      "print('Today is',ctime(time()))\n");
  Py_Finalize();
}

// https://docs.python.org/3/extending/extending.html
// https://docs.python.org/3/c-api/
std::string torchpy::hello() {
  Py_Initialize();

  PyObject* globals = PyDict_New();
  assert(PyDict_Check(globals) == true);
  PyObject* builtins = PyEval_GetBuiltins();
  assert(NULL != builtins);
  PyDict_SetItemString(globals, "__builtins__", builtins);

  PyRun_String(
      "def hello():\n"
      "   return \"Hello Py\"\n"
      "print(hello())\n",
      Py_file_input,
      globals,
      globals);

  PyObject* hello = PyDict_GetItemString(globals, "hello");
  assert(PyFunction_Check(hello) == true);
  PyObject* pyresult = PyObject_CallObject(hello, NULL);
  auto c_str = PyUnicode_AsUTF8(pyresult);
  assert(NULL != c_str);
  auto result = std::string(c_str);

  Py_Finalize();
  return result;
}

torch::jit::Module torchpy::load(const std::string& filename) {
  return torch::jit::load(filename);
}

std::vector<torch::jit::IValue> torchpy::inputs(std::vector<int64_t> shape) {
  std::vector<torch::jit::IValue> inputs;
  auto at_shape = at::IntArrayRef(shape);
  inputs.push_back(torch::ones(at_shape));
  return inputs;
}
