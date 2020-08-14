#include <torchpy.h>
#include <Python.h>
#include <assert.h>
#include <stdio.h>
#include <torch/script.h>
#include <torch/torch.h>
#include <iostream>

namespace torchpy {

void init() {
  Py_Initialize();
}

void finalize() {
  Py_Finalize();
}
// https://docs.python.org/3/extending/extending.html
// https://docs.python.org/3/c-api/
std::string hello() {
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

  return result;
}

PyModule load(const std::string& filename) {
  PyObject* globals = PyDict_New();

  assert(PyDict_Check(globals) == true);
  PyObject* builtins = PyEval_GetBuiltins();
  assert(NULL != builtins);
  PyDict_SetItemString(globals, "__builtins__", builtins);
  FILE* fp = fopen("torchpy/loader.py", "r");
  // const char *filename = "loader.py";
  PyRun_File(fp, "loader.py", Py_file_input, globals, globals);

  PyObject* load = PyDict_GetItemString(globals, "load");
  assert(PyFunction_Check(load) == true);

  PyObject* args = Py_BuildValue("(s)", filename.c_str());
  PyObject* pyresult = PyObject_CallObject(load, args);
  assert(NULL != pyresult);
  // auto c_str = PyUnicode_AsUTF8(pyresult);
  // assert(NULL != c_str);
  // auto result = std::string(c_str);

  // return result;
  auto mod = PyModule();
  return mod;
}

std::vector<at::Tensor> inputs(std::vector<int64_t> shape) {
  std::vector<at::Tensor> inputs;
  auto at_shape = at::IntArrayRef(shape);
  inputs.push_back(torch::ones(at_shape));
  return inputs;
}

at::Tensor PyModule::forward(std::vector<at::Tensor> inputs) {
  return inputs.at(0);
}

} // namespace torchpy