// Note(jiayq): the import_array function is done inside caffe2_python.cc.
// Read http://docs.scipy.org/doc/numpy-1.10.1/reference/c-api.array.html#miscellaneous
// for more details.
#define NO_IMPORT_ARRAY

#include "caffe2/python/caffe2_python.h"

#include "caffe2/core/context_gpu.h"

namespace caffe2 {
REGISTER_BLOB_FETCHER(
    (TypeMeta::Id<TensorCUDA>()),
    TensorFetcher<CUDAContext>);
REGISTER_BLOB_FEEDER(
    CUDA,
    TensorFeeder<CUDAContext>);
}  // namespace caffe2

extern "C" {

// Here are functions that are purely GPU-based functions to be filled.

PyObject* NumCudaDevices(PyObject* self, PyObject* args) {
  int num_devices = caffe2::NumCudaDevices();
  return Py_BuildValue("i", num_devices);
}

PyObject* SetDefaultGPUID(PyObject* self, PyObject* args) {
  int device_id;
  if (!PyArg_ParseTuple(args, "i", &device_id)) {
    PyErr_SetString(PyExc_ValueError, "Incorrect arguments: must pass an int.");
    return nullptr;
  }
  caffe2::SetDefaultGPUID(device_id);
  Py_RETURN_TRUE;
}

PyObject* GetDefaultGPUID(PyObject* self, PyObject* args) {
  int device_id = caffe2::GetDefaultGPUID();
  return Py_BuildValue("i", device_id);
}

PyObject* GetCudaPeerAccessPattern(PyObject* self, PyObject* args) {
  std::vector<std::vector<bool> > pattern;
  if (!caffe2::GetCudaPeerAccessPattern(&pattern)) {
    PyErr_SetString(PyExc_RuntimeError,
                    "Error in running caffe2::GetCudaPeerAccessPattern.");
    return nullptr;
  }
  std::vector<npy_intp> npy_dims;
  int num_devices = pattern.size();
  npy_dims.push_back(num_devices);
  npy_dims.push_back(num_devices);

  PyObject* array = PyArray_SimpleNew(2, npy_dims.data(), NPY_BOOL);
  bool* npy_data = static_cast<bool*>(
      PyArray_DATA(reinterpret_cast<PyArrayObject*>(array)));
  for (int i = 0; i < num_devices; ++i) {
    for (int j = 0; j < num_devices; ++j) {
      *(npy_data++) = pattern[i][j];
    }
  }
  return array;
}


#define _PYNAME(name) {#name, name, METH_VARARGS, ""}
static PyMethodDef* GetCaffe2PythonGPUMethods() {
  static std::vector<PyMethodDef> methods{
    // TODO(Yangqing): write the methods string.
    // Note(Yangqing): For any function that we are going to override in the
    // python file, we prepend "cc_" here.
    _PYNAME(NumCudaDevices),
    _PYNAME(SetDefaultGPUID),
    _PYNAME(GetDefaultGPUID),
    _PYNAME(GetCudaPeerAccessPattern),
  };

  static bool method_initialized = false;
  if (!method_initialized) {
    // Add the methods inherited from caffe2_python_cpu.
    method_initialized = true;
    PyMethodDef* cpu_methods = GetCaffe2PythonMethods();
    while (cpu_methods->ml_name != nullptr) {
      methods.push_back(*cpu_methods);
      cpu_methods++;
    }
    // Add the termination mark.
    methods.push_back({nullptr, nullptr, 0, nullptr});
  }
  return methods.data();
}
#undef _PYNAME

// The initialization code.
#if PY_MAJOR_VERSION >= 3

struct module_state {
  PyObject* error;
};

inline static struct module_state* ModuleGetState(PyObject* module) {
  return (struct module_state*)PyModule_GetState(module);
}
static int caffe2_python_gpu_traverse(PyObject* m, visitproc visit, void* arg) {
  Py_VISIT(ModuleGetState(m)->error);
  return 0;
}

static int caffe2_python_gpu_clear(PyObject* m) {
  Py_CLEAR(ModuleGetState(m)->error);
  return 0;
}

static struct PyModuleDef gModuleDef = {
  PyModuleDef_HEAD_INIT,
  "libcaffe2_python_gpu",
  nullptr,
  sizeof(struct module_state),
  GetCaffe2PythonGPUMethods(),
  nullptr,
  caffe2_python_gpu_traverse,
  caffe2_python_gpu_clear,
  nullptr
};

PyObject* PyInit_libcaffe2_python_gpu(void) {
  PyObject* module = PyModule_Create(&gModuleDef);
  if (module == nullptr) {
    return nullptr;
  }
  struct module_state* st = ModuleGetState(module);
  st->error = PyErr_NewException(
      "libcaffe2_python_gpu.Error", nullptr, nullptr);
  if (st->error == nullptr) {
    Py_DECREF(module);
    return nullptr;
  }
  common_init_libcaffe2_python_cpu();
  return module;
}

#else  // PY_MAJOR_VERSION >= 3

void initlibcaffe2_python_gpu(void) {
  PyObject* module = Py_InitModule(
      "libcaffe2_python_gpu", GetCaffe2PythonGPUMethods());
  if (module == nullptr) {
    return;
  }
  common_init_libcaffe2_python_cpu();
}

#endif  // PY_MAJOR_VERSION >= 3

}  // extern "C"
