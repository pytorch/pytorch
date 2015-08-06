#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "caffe2/core/context.h"
#ifndef PYCAFFE2_CPU_ONLY
#include "caffe2/core/context_gpu.h"
#endif  // PYCAFFE2_CPU_ONLY
#include "caffe2/core/net.h"
#include "caffe2/core/workspace.h"
#include "caffe2/proto/caffe2.pb.h"
#include "glog/logging.h"

using std::map;
using std::string;
using std::unique_ptr;
using std::vector;
using namespace caffe2;  // NOLINT

// gWorkspaces allows us to define and switch between multiple workspaces in
// Python.
static map<string, unique_ptr<Workspace> > gWorkspaces;
// gWorkspace is the pointer to the current workspace. The ownership is kept
// by the gWorkspaces map.
static Workspace* gWorkspace = nullptr;
static string gCurrentWorkspaceName;

namespace {

bool SwitchWorkspaceInternal(const string& name, const bool create_if_missing) {
  if (gWorkspaces.count(name)) {
    gCurrentWorkspaceName = name;
    gWorkspace = gWorkspaces[name].get();
    return true;
  } else if (create_if_missing) {
    std::unique_ptr<Workspace> new_workspace(new Workspace());
    gWorkspace = new_workspace.get();
    gWorkspaces.insert(std::make_pair(name, std::move(new_workspace)));
    gCurrentWorkspaceName = name;
    return true;
  } else {
    return false;
  }
}

inline string PyStringToStdString(PyObject* pystring) {
  return string(PyString_AsString(pystring), PyString_Size(pystring));
}

inline PyObject* StdStringToPyString(const string& str) {
  return PyString_FromStringAndSize(str.c_str(), str.size());
}

static_assert(sizeof(int) == sizeof(int32_t),
              "Yangqing made a loose assumption that int will always be int32 "
              "for numpy type mapping");

template <typename T> struct NumpyTypeWrapper;
template<> struct NumpyTypeWrapper<float> {
  static const int type = NPY_FLOAT;
};
template<> struct NumpyTypeWrapper<int> {
  static const int type = NPY_INT32;
};

template <typename T, class DeviceContext>
PyObject* FetchTensor(const Blob& blob) {
  DeviceContext context;
  const Tensor<T, DeviceContext>& tensor =
      blob.Get<Tensor<T, DeviceContext> >();
  CHECK_GT(tensor.size(), 0);
  vector<npy_intp> npy_dims;
  for (const int dim : tensor.dims()) {
    npy_dims.push_back(dim);
  }
  PyObject* array = PyArray_SimpleNew(
      tensor.ndim(), npy_dims.data(), NumpyTypeWrapper<T>::type);
  // Now, copy the data to the tensor.
  // TODO(Yangqing): Is there an easier way to convert PyObject to
  // PyArrayObject?
  context.template Copy<T, DeviceContext, CPUContext>(
      tensor.size(), tensor.data(),
      static_cast<T*>(PyArray_DATA(reinterpret_cast<PyArrayObject*>(array))));
  return array;
}

template <typename T, class DeviceContext>
PyObject* FeedTensor(const DeviceOption& option, PyArrayObject* original_array,
                     Blob* blob) {
  PyArrayObject* array = PyArray_GETCONTIGUOUS(original_array);
  DeviceContext context(option);
  Tensor<T, DeviceContext>* tensor =
      blob->GetMutable<Tensor<T, DeviceContext> >();
  // numpy requires long int as its dims.
  int ndim = PyArray_NDIM(array);
  npy_intp* npy_dims = PyArray_DIMS(array);
  vector<int> dims;
  for (int i = 0; i < ndim; ++i) {
    dims.push_back(npy_dims[i]);
  }
  tensor->Reshape(dims);
  // Now, copy the data to the tensor.
  context.template Copy<T, CPUContext, DeviceContext>(
      tensor->size(), static_cast<T*>(PyArray_DATA(array)),
      tensor->mutable_data());
  Py_XDECREF(array);
  Py_RETURN_TRUE;
}

}  // namespace

extern "C" {

// The InitGoogleLogging function is provided so one can initialize google logging
// from python. You should make sure it is not called twice.
PyObject* InitGoogleLogging(PyObject* self, PyObject* args) {
  char binary_name[] = "python";
  google::InitGoogleLogging(binary_name);
  Py_RETURN_TRUE;
}

PyObject* SwitchWorkspace(PyObject* self, PyObject* args) {
  PyObject* name = nullptr;
  PyObject* create_if_missing = nullptr;
  if (!PyArg_ParseTuple(args, "S|O", &name, &create_if_missing)) {
    PyErr_SetString(PyExc_ValueError,
                    "SwitchWorkspace takes in a workspace name, and "
                    "an optional boolean value that specifies whether "
                    "we want to create the workspace if it is missing.");
    return NULL;
  }
  bool success = SwitchWorkspaceInternal(
      PyStringToStdString(name),
      (create_if_missing != nullptr) && PyObject_IsTrue(create_if_missing));
  if (!success) {
    PyErr_SetString(
        PyExc_RuntimeError,
        "Workspace of the given name does not exist, and I am not instructed "
        "to create it either.");
    return NULL;
  }
  Py_RETURN_TRUE;
}

PyObject* CurrentWorkspace(PyObject* self, PyObject* args) {
  return StdStringToPyString(gCurrentWorkspaceName);
}

PyObject* Workspaces(PyObject* self, PyObject* args) {
  PyObject* list = PyList_New(gWorkspaces.size());
  int i = 0;
  for (auto const & it : gWorkspaces) {
    CHECK_EQ(PyList_SetItem(list, i, StdStringToPyString(it.first)), 0);
    i += 1;
  }
  return list;
}

PyObject* ResetWorkspace(PyObject* self, PyObject* args) {
  PyObject* root_folder = nullptr;
  if (!PyArg_ParseTuple(args, "|S", &root_folder)) {
    PyErr_SetString(PyExc_ValueError,
                    "ResetWorkspace takes in either no argument, or a string "
                    "specifying the root folder of the workspace.");
    return NULL;
  }
  LOG(INFO) << "Resetting workspace.";
  if (root_folder == nullptr) {
    gWorkspaces[gCurrentWorkspaceName].reset(
        new Workspace());
  } else {
    gWorkspaces[gCurrentWorkspaceName].reset(
        new Workspace(PyStringToStdString(root_folder)));
  }
  gWorkspace = gWorkspaces[gCurrentWorkspaceName].get();
  Py_RETURN_TRUE;
}

PyObject* RootFolder(PyObject* self, PyObject* args) {
  return StdStringToPyString(gWorkspace->RootFolder());
}

// This function should not be called by the user - only used during the
// destruction of the module.
PyObject* OnModuleExit(PyObject* self, PyObject* args) {
  gWorkspaces.clear();
  Py_RETURN_TRUE;
}

PyObject* Blobs(PyObject* self, PyObject* args) {
  vector<string> blob_strings = gWorkspace->Blobs();
  PyObject* list = PyList_New(blob_strings.size());
  for (int i = 0; i < blob_strings.size(); ++i) {
    CHECK_EQ(PyList_SetItem(list, i, StdStringToPyString(blob_strings[i])), 0);
  }
  return list;
}

PyObject* HasBlob(PyObject* self, PyObject* args) {
  char* name;
  if (!PyArg_ParseTuple(args, "s", &name)) {
    return NULL;
  }
  if (gWorkspace->HasBlob(string(name))) {
    Py_RETURN_TRUE;
  } else {
    Py_RETURN_FALSE;
  }
}

PyObject* CreateNet(PyObject* self, PyObject* args) {
  PyObject* proto_string;
  if (!PyArg_ParseTuple(args, "S", &proto_string)) {
    return NULL;
  }
  caffe2::NetDef proto;
  if (!proto.ParseFromString(PyStringToStdString(proto_string))) {
    PyErr_SetString(PyExc_ValueError, "Cannot parse input net string.");
    return NULL;
  }
  if (!gWorkspace->CreateNet(proto)) {
    PyErr_SetString(
        PyExc_RuntimeError,
        "Cannot create network. See console log for error messages.");
    return NULL;
  }
  Py_RETURN_TRUE;
}

PyObject* RunNet(PyObject* self, PyObject* args) {
  char* name;
  if (!PyArg_ParseTuple(args, "s", &name)) {
    PyErr_SetString(PyExc_ValueError,
                    "Incorrect argument. Must pass in a single string.");
    return NULL;
  }
  if (!gWorkspace->RunNet(string(name))) {
    PyErr_SetString(
        PyExc_RuntimeError,
        "Cannot run network. See console log for error messages.");
    return NULL;
  }
  Py_RETURN_TRUE;
}

PyObject* DeleteNet(PyObject* self, PyObject* args) {
  char* name;
  if (!PyArg_ParseTuple(args, "s", &name)) {
    PyErr_SetString(PyExc_ValueError,
                    "Incorrect argument. Must pass in a single string.");
    return NULL;
  }
  gWorkspace->DeleteNet(string(name));
  Py_RETURN_TRUE;
}

PyObject* Nets(PyObject* self, PyObject* args) {
  vector<string> net_strings = gWorkspace->Nets();
  PyObject* list = PyList_New(net_strings.size());
  for (int i = 0; i < net_strings.size(); ++i) {
    CHECK_EQ(PyList_SetItem(list, i, StdStringToPyString(net_strings[i])), 0);
  }
  return list;
}

PyObject* RunOperatorOnce(PyObject* self, PyObject* args) {
  PyObject* proto_string;
  if (!PyArg_ParseTuple(args, "S", &proto_string)) {
    PyErr_SetString(PyExc_ValueError,
                    "Incorrect argument. Must pass in a single string.");
    return NULL;
  }
  caffe2::OperatorDef proto;
  if (!proto.ParseFromString(PyStringToStdString(proto_string))) {
    PyErr_SetString(PyExc_ValueError, "Cannot parse input operator proto.");
    return NULL;
  }
  if (!gWorkspace->RunOperatorOnce(proto)) {
    PyErr_SetString(
        PyExc_RuntimeError,
        "Cannot run operator. See console log for error messages.");
    return NULL;
  }
  Py_RETURN_TRUE;
}

PyObject* RunNetOnce(PyObject* self, PyObject* args) {
  PyObject* proto_string;
  if (!PyArg_ParseTuple(args, "S", &proto_string)) {
    PyErr_SetString(PyExc_ValueError,
                    "Incorrect argument. Must pass in a single string.");
    return NULL;
  }
  caffe2::NetDef proto;
  if (!proto.ParseFromString(PyStringToStdString(proto_string))) {
    PyErr_SetString(PyExc_ValueError, "Cannot parse input net proto.");
    return NULL;
  }
  if (!gWorkspace->RunNetOnce(proto)) {
    PyErr_SetString(
        PyExc_RuntimeError,
        "Cannot run net. See console log for error messages.");
    return NULL;
  }
  Py_RETURN_TRUE;
}

PyObject* RunPlan(PyObject* self, PyObject* args) {
  PyObject* proto_string;
  if (!PyArg_ParseTuple(args, "S", &proto_string)) {
    PyErr_SetString(PyExc_ValueError,
                    "Incorrect argument. Must pass in a single string.");
    return NULL;
  }
  caffe2::PlanDef proto;
  if (!proto.ParseFromString(PyStringToStdString(proto_string))) {
    PyErr_SetString(PyExc_ValueError, "Cannot parse input plan proto.");
    return NULL;
  }
  if (!gWorkspace->RunPlan(proto)) {
    PyErr_SetString(
        PyExc_RuntimeError,
        "Cannot run plan. See console log for error messages.");
    return NULL;
  }
  Py_RETURN_TRUE;
}

PyObject* CreateBlob(PyObject* self, PyObject* args) {
  char* name_char;
  if (!PyArg_ParseTuple(args, "s", &name_char)) {
    PyErr_SetString(PyExc_ValueError, "Incorrect arguments.");
    return NULL;
  }
  string name(name_char);
  Blob* blob = gWorkspace->CreateBlob(name);
  Py_RETURN_TRUE;
}

#define RETURN_TENSOR_IF_FORMAT(dtype, context)                                \
  if (blob.IsType<caffe2::Tensor<dtype, context> >()) {                        \
    return FetchTensor<dtype, context>(blob);                                  \
  }

PyObject* FetchBlob(PyObject* self, PyObject* args) {
  char* name;
  if (!PyArg_ParseTuple(args, "s", &name)) {
    PyErr_SetString(PyExc_ValueError, "Incorrect arguments.");
    return NULL;
  }
  if (!gWorkspace->HasBlob(string(name))) {
    PyErr_SetString(PyExc_ValueError, "Requested blob does not exist.");
    return NULL;
  }
  const caffe2::Blob& blob = *(gWorkspace->GetBlob(string(name)));
  // We only support a subset of exporting capabilities.
  RETURN_TENSOR_IF_FORMAT(float, CPUContext)
  RETURN_TENSOR_IF_FORMAT(int, CPUContext)
#ifndef PYCAFFE2_CPU_ONLY
  RETURN_TENSOR_IF_FORMAT(float, CUDAContext)
  RETURN_TENSOR_IF_FORMAT(int, CUDAContext)
#endif  // PYCAFFE2_CPU_ONLY

  // If all branches failed, we should throw an error.
  LOG(ERROR) << "Blob" << string(name) << " has unsupported data type: "
             << blob.TypeName();
  PyErr_SetString(PyExc_TypeError, "Unsupported data type.");
  return NULL;
}


PyObject* FeedBlob(PyObject* self, PyObject* args) {
  char* name_char;
  PyArrayObject* array = nullptr;
  PyObject* device_option_string = nullptr;
  if (!PyArg_ParseTuple(args, "sO!|O", &name_char, &PyArray_Type, &array,
                        &device_option_string)) {
    PyErr_SetString(PyExc_ValueError, "Incorrect arguments.");
    return NULL;
  }
  string name(name_char);
  DeviceOption option;
  if (device_option_string != nullptr) {
    // If we have a device option passed in, read it.
    if (!option.ParseFromString(PyStringToStdString(device_option_string))) {
      PyErr_SetString(PyExc_ValueError, "Cannot parse device option string.");
      return NULL;
    }
  }
  Blob* blob = gWorkspace->CreateBlob(name);
  int data_type = PyArray_TYPE(array);

  // Since there is really no polymorphism, we will have to do so...
  switch (option.device_type()) {
  case CPU:
    switch (data_type) {
      case NPY_INT:
        return FeedTensor<int, CPUContext>(option, array, blob);
      case NPY_FLOAT:
        return FeedTensor<float, CPUContext>(option, array, blob);
      default:
        PyErr_SetString(PyExc_TypeError, "Unsupported numpy data type.");
        return NULL;
    }
#ifndef PYCAFFE2_CPU_ONLY
  case CUDA:
    switch (data_type) {
      case NPY_INT:
        return FeedTensor<int, CUDAContext>(option, array, blob);
      case NPY_FLOAT:
        return FeedTensor<float, CUDAContext>(option, array, blob);
      default:
        PyErr_SetString(PyExc_TypeError, "Unsupported numpy data type.");
        return NULL;
    }
#endif  // PYCAFFE2_CPU_ONLY
  default:
    PyErr_SetString(PyExc_TypeError, "Unknown device type.");
    return NULL;
  }
}

// A simple macro to avoid writing repeated symbols.
#define _PYNAME(name) {#name, name, METH_VARARGS}

static PyMethodDef gPycaffe2Methods[] = {
  // TODO(Yangqing): write the methods string.
  // Note(Yangqing): For any function that we are going to override in the
  // python file, we prepend "cc_" here.
  _PYNAME(InitGoogleLogging),
  _PYNAME(SwitchWorkspace),
  _PYNAME(CurrentWorkspace),
  _PYNAME(Workspaces),
  {"cc_ResetWorkspace", ResetWorkspace, METH_VARARGS},
  _PYNAME(RootFolder),
  _PYNAME(OnModuleExit),
  _PYNAME(Blobs),
  _PYNAME(HasBlob),
  {"cc_CreateNet", CreateNet, METH_VARARGS},
  _PYNAME(RunNet),
  _PYNAME(DeleteNet),
  _PYNAME(Nets),
  {"cc_RunOperatorOnce", RunOperatorOnce, METH_VARARGS},
  {"cc_RunNetOnce", RunNetOnce, METH_VARARGS},
  {"cc_RunPlan", RunPlan, METH_VARARGS},
  _PYNAME(CreateBlob),
  _PYNAME(FetchBlob),
  {"cc_FeedBlob", FeedBlob, METH_VARARGS},
  {NULL, NULL},  // end of python methods.
};
#undef _PYNAME

#ifdef PYCAFFE2_CPU_ONLY
void initlibcaffe2_python_nogpu(void) {
  (void) Py_InitModule("libcaffe2_python_nogpu", gPycaffe2Methods);
#else
void initlibcaffe2_python(void) {
  (void) Py_InitModule("libcaffe2_python", gPycaffe2Methods);
#endif
  import_array();  // for numpy
  // We will create a default workspace for us to run stuff.
  SwitchWorkspaceInternal("default", true);
  gCurrentWorkspaceName = "default";
}

}  // extern "C"

