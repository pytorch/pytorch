#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

#include <cstdint>
#include <memory>
#include <set>
#include <string>
#include <sstream>
#include <vector>

#include "caffe2/core/context.h"
#ifndef PYCAFFE2_CPU_ONLY
#include "caffe2/core/context_gpu.h"
#endif  // PYCAFFE2_CPU_ONLY
#include "caffe2/core/init.h"
#include "caffe2/core/net.h"
#include "caffe2/core/operator.h"
#include "caffe2/core/workspace.h"
#include "caffe2/proto/caffe2.pb.h"

//using namespace caffe2;  // NOLINT
using caffe2::Blob;
using caffe2::DeviceOption;
using caffe2::Tensor;
using caffe2::Workspace;
using caffe2::CPUContext;
using caffe2::OperatorDef;

#ifndef PYCAFFE2_CPU_ONLY
using caffe2::CUDAContext;
#endif  // PYCAFFE2_CPU_ONLY

// gWorkspaces allows us to define and switch between multiple workspaces in
// Python.
static std::map<std::string, std::unique_ptr<Workspace> > gWorkspaces;
// gWorkspace is the pointer to the current workspace. The ownership is kept
// by the gWorkspaces map.
static Workspace* gWorkspace = nullptr;
static std::string gCurrentWorkspaceName;

namespace {

using caffe2::string;

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

template <typename T>
inline void MakeStringInternal(std::stringstream& ss, const T& t) {
  ss << t;
}

template <typename T, typename ... Args>
inline void MakeStringInternal(std::stringstream& ss, const T& t, const Args&... args) {
  MakeStringInternal(ss, t);
  MakeStringInternal(ss, args...);
}

template <typename... Args>
string MakeString(const Args&... args) {
  std::stringstream ss;
  MakeStringInternal(ss, args...);
  return string(ss.str());
}


inline void PyErr_SetString(PyObject* type, const string& str) {
  PyErr_SetString(type, str.c_str());
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
PyObject* FetchTensor(const Tensor<DeviceContext>& tensor) {
  DeviceContext context;
  CAFFE_CHECK_GT(tensor.size(), 0);
  std::vector<npy_intp> npy_dims;
  for (const int dim : tensor.dims()) {
    npy_dims.push_back(dim);
  }
  PyObject* array = PyArray_SimpleNew(
      tensor.ndim(), npy_dims.data(), NumpyTypeWrapper<T>::type);
  // Now, copy the data to the tensor.
  // TODO(Yangqing): Is there an easier way to convert PyObject to
  // PyArrayObject?
  context.template Copy<T, DeviceContext, caffe2::CPUContext>(
      tensor.size(), tensor.template data<T>(),
      static_cast<T*>(PyArray_DATA(reinterpret_cast<PyArrayObject*>(array))));
  return array;
}

template <typename T, class DeviceContext>
PyObject* FeedTensor(const DeviceOption& option, PyArrayObject* original_array,
                     Blob* blob) {
  PyArrayObject* array = PyArray_GETCONTIGUOUS(original_array);
  DeviceContext context(option);
  Tensor<DeviceContext>* tensor =
      blob->GetMutable<Tensor<DeviceContext> >();
  // numpy requires long int as its dims.
  int ndim = PyArray_NDIM(array);
  npy_intp* npy_dims = PyArray_DIMS(array);
  std::vector<int> dims;
  for (int i = 0; i < ndim; ++i) {
    dims.push_back(npy_dims[i]);
  }
  tensor->Reshape(dims);
  // Now, copy the data to the tensor.
  context.template Copy<T, caffe2::CPUContext, DeviceContext>(
      tensor->size(), static_cast<T*>(PyArray_DATA(array)),
      tensor->template mutable_data<T>());
  Py_XDECREF(array);
  Py_RETURN_TRUE;
}

}  // namespace

extern "C" {

PyObject* GlobalInit(PyObject* self, PyObject* args) {
  static bool global_init_called = false;
  if (global_init_called) {
    PyErr_SetString(PyExc_RuntimeError, "GlobalInit already called.");
    return NULL;
  }
  PyObject* list;
  if (!PyArg_ParseTuple(args, "O!", &PyList_Type, &list)) {
    PyErr_SetString(PyExc_ValueError, "Incorrect arguments.");
    return NULL;
  }
  int argc = PyList_Size(list);
  std::unique_ptr<char*> argv(new char*[std::max(argc, 1)]);
  char** raw_argv = argv.get();
  for (int i = 0; i < argc; ++i) {
    // Get the pointer to the string
    raw_argv[i] = PyString_AsString(PyList_GetItem(list, i));
  }
  // Special case for argc = 0: in this case, we will simply add a dummy
  // argv to call caffe2's underlying code.
  if (argc == 0) {
    ++argc;
    raw_argv[0] = "python";
  }
  global_init_called = true;
  if (!caffe2::GlobalInit(&argc, raw_argv)) {
    PyErr_SetString(PyExc_RuntimeError, "Error in global init.");
    return NULL;
  }
  Py_RETURN_TRUE;
}

PyObject* RegisteredOperators(PyObject* self, PyObject* args) {
  std::set<string> all_keys;
  // CPU operators
  for (const auto& name : caffe2::CPUOperatorRegistry()->Keys()) {
    all_keys.insert(name);
  }
  // CUDA operators
  for (const auto& name : caffe2::CUDAOperatorRegistry()->Keys()) {
    all_keys.insert(name);
  }
  // Now, add it to the list
  PyObject* list = PyList_New(all_keys.size());
  int idx = 0;
  for (const string& name : all_keys) {
    CAFFE_CHECK_EQ(PyList_SetItem(list, idx, StdStringToPyString(name)), 0);
    ++idx;
  }
  return list;
}

PyObject* GetGradientDefs(PyObject* self, PyObject* args) {
  PyObject* def_string = nullptr;
  if (!PyArg_ParseTuple(args, "|S", &def_string)) {
    PyErr_SetString(PyExc_ValueError,
                    "GetGradientDefs requires an input that is a serialized "
                    "OperatorDef protobuffer.");
    return NULL;
  }
  OperatorDef def;
  if (!def.ParseFromString(PyStringToStdString(def_string))) {
    PyErr_SetString(PyExc_ValueError,
                    "Provided string is not a valid OperatorDef protobuffer.");
    return NULL;
  }
  std::unique_ptr<std::vector<OperatorDef> > grad_defs(GetGradientDefs(def));
  if (grad_defs.get() == nullptr) {
    PyErr_SetString(
        PyExc_ValueError,
        ("Gradient not registered for operator type " + def.type()).c_str());
    return NULL;
  }
  PyObject* list = PyList_New(grad_defs->size());
  int i = 0;
  for (const OperatorDef & grad_def : *grad_defs) {
    CAFFE_CHECK_EQ(PyList_SetItem(
        list, i, StdStringToPyString(grad_def.SerializeAsString())), 0);
    ++i;
  }
  return list;
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
    CAFFE_CHECK_EQ(PyList_SetItem(list, i, StdStringToPyString(it.first)), 0);
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
  CAFFE_VLOG(1) << "Resetting workspace.";
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
  std::vector<caffe2::string> blob_strings = gWorkspace->Blobs();
  PyObject* list = PyList_New(blob_strings.size());
  for (int i = 0; i < blob_strings.size(); ++i) {
    CAFFE_CHECK_EQ(
        PyList_SetItem(list, i, StdStringToPyString(blob_strings[i])), 0);
  }
  return list;
}

PyObject* HasBlob(PyObject* self, PyObject* args) {
  char* name;
  if (!PyArg_ParseTuple(args, "s", &name)) {
    return NULL;
  }
  if (gWorkspace->HasBlob(caffe2::string(name))) {
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
  if (!gWorkspace->RunNet(caffe2::string(name))) {
    PyErr_SetString(
        PyExc_RuntimeError,
        "Cannot run network. See console log for error messages.");
    return NULL;
  }
  Py_RETURN_TRUE;
}


PyObject* BenchmarkNet(PyObject* self, PyObject* args) {
  char* name;
  int warmup_runs = 0;
  int main_runs = 0;
  PyObject* run_individual = nullptr;
  if (!PyArg_ParseTuple(args, "siiO", &name, &warmup_runs,
                        &main_runs, &run_individual)) {
    PyErr_SetString(PyExc_ValueError,
                    "Incorrect argument.");
    return NULL;
  }
  caffe2::NetBase* net = gWorkspace->GetNet(caffe2::string(name));
  if (net == nullptr) {
    PyErr_SetString(PyExc_RuntimeError, "Cannot find network.");
    return NULL;
  }
  net->TEST_Benchmark(warmup_runs, main_runs,
                      PyObject_IsTrue(run_individual));
  Py_RETURN_TRUE;
}

PyObject* DeleteNet(PyObject* self, PyObject* args) {
  char* name;
  if (!PyArg_ParseTuple(args, "s", &name)) {
    PyErr_SetString(PyExc_ValueError,
                    "Incorrect argument. Must pass in a single string.");
    return NULL;
  }
  gWorkspace->DeleteNet(caffe2::string(name));
  Py_RETURN_TRUE;
}

PyObject* Nets(PyObject* self, PyObject* args) {
  std::vector<caffe2::string> net_strings = gWorkspace->Nets();
  PyObject* list = PyList_New(net_strings.size());
  for (int i = 0; i < net_strings.size(); ++i) {
    CAFFE_CHECK_EQ(PyList_SetItem(list, i, StdStringToPyString(net_strings[i])), 0);
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
  caffe2::string name(name_char);
  (void) gWorkspace->CreateBlob(name);
  Py_RETURN_TRUE;
}

PyObject* FetchBlob(PyObject* self, PyObject* args) {
  char* name;
  if (!PyArg_ParseTuple(args, "s", &name)) {
    PyErr_SetString(PyExc_ValueError, "Incorrect arguments.");
    return NULL;
  }
  if (!gWorkspace->HasBlob(caffe2::string(name))) {
    PyErr_SetString(PyExc_ValueError, "Requested blob does not exist.");
    return NULL;
  }
  const caffe2::Blob& blob = *(gWorkspace->GetBlob(caffe2::string(name)));
  if (blob.IsType<Tensor<CPUContext> >()) {
    const Tensor<CPUContext>& tensor = blob.Get<Tensor<CPUContext> >();
    if (tensor.IsType<float>()) {
      return FetchTensor<float, CPUContext>(tensor);
    } else if (tensor.IsType<int>()) {
      return FetchTensor<int, CPUContext>(tensor);
    }
  }
#ifndef PYCAFFE2_CPU_ONLY
  if (blob.IsType<Tensor<CUDAContext> >()) {
    const Tensor<CUDAContext>& tensor = blob.Get<Tensor<CUDAContext> >();
    if (tensor.IsType<float>()) {
      return FetchTensor<float, CUDAContext>(tensor);
    } else if (tensor.IsType<int>()) {
      return FetchTensor<int, CUDAContext>(tensor);
    }
  }
#endif  // !PYCAFFE2_CPU_ONLY
  // If all branches failed, we will return a metainfo string.
  std::stringstream ss;
  ss << caffe2::string(name) << ", a C++ native class of type "
     << blob.TypeName() << ".";
  return StdStringToPyString(ss.str());
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
  caffe2::string name(name_char);
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
  case caffe2::CPU:
    switch (data_type) {
      case NPY_LONG:
        if (sizeof(long) != sizeof(int)) {
          CAFFE_LOG_FATAL << "On this platform NPY_LONG does not equal to "
                             "NPY_INT and such type is not supported yet.";
        } else {
          return FeedTensor<int, caffe2::CPUContext>(option, array, blob);
        }
      case NPY_INT:
        return FeedTensor<int, caffe2::CPUContext>(option, array, blob);
      case NPY_FLOAT:
        return FeedTensor<float, caffe2::CPUContext>(option, array, blob);
      default:
        PyErr_SetString(PyExc_TypeError,
                        MakeString("Unsupported numpy data type: ", data_type, "."));
        return NULL;
    }
#ifndef PYCAFFE2_CPU_ONLY
  case caffe2::CUDA:
    switch (data_type) {
      case NPY_LONG:
        if (sizeof(long) != sizeof(int)) {
          CAFFE_LOG_FATAL << "On this platform NPY_LONG does not equal to "
                             "NPY_INT and such type is not supported yet.";
        } else {
          return FeedTensor<int, caffe2::CUDAContext>(option, array, blob);
        }
      case NPY_INT:
        return FeedTensor<int, caffe2::CUDAContext>(option, array, blob);
      case NPY_FLOAT:
        return FeedTensor<float, caffe2::CUDAContext>(option, array, blob);
      default:
        PyErr_SetString(PyExc_TypeError,
                        MakeString("Unsupported numpy data type: ", data_type, "."));
        return NULL;
    }
#endif  // !PYCAFFE2_CPU_ONLY
  default:
    PyErr_SetString(PyExc_TypeError, "Unknown device type.");
    return NULL;
  }
}

PyObject* HasGPUSupport(PyObject* self, PyObject* args) {
#ifdef PYCAFFE2_CPU_ONLY
  return Py_BuildValue("i", 0);
#else  // PYCAFFE2_CPU_ONLY
  return Py_BuildValue("i", 1);
#endif  // PYCAFFE2_CPU_ONLY
}

#ifndef PYCAFFE2_CPU_ONLY
// Here are functions that are purely GPU-based functions to be filled.

PyObject* NumberOfGPUs(PyObject* self, PyObject* args) {
  int num_devices = 0;
  auto err = cudaGetDeviceCount(&num_devices);
  if (err == cudaErrorNoDevice || err == cudaErrorInsufficientDriver) {
    return Py_BuildValue("i", 0);
  } else if (err != cudaSuccess) {
    PyErr_SetString(PyExc_RuntimeError, "Runtime CUDA error.");
    return NULL;
  }
  return Py_BuildValue("i", num_devices);
}

PyObject* SetDefaultGPUID(PyObject* self, PyObject* args) {
  int device_id;
  if (!PyArg_ParseTuple(args, "i", &device_id)) {
    PyErr_SetString(PyExc_ValueError, "Incorrect arguments: must pass an int.");
    return NULL;
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
    return NULL;
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

#endif  // !PYCAFFE2_CPU_ONLY

// A simple macro to avoid writing repeated symbols.
#define _PYNAME(name) {#name, name, METH_VARARGS, ""}


static PyMethodDef gPycaffe2Methods[] = {
  // TODO(Yangqing): write the methods string.
  // Note(Yangqing): For any function that we are going to override in the
  // python file, we prepend "cc_" here.
  _PYNAME(GlobalInit),
  _PYNAME(RegisteredOperators),
  {"cc_GetGradientDefs", GetGradientDefs, METH_VARARGS, ""},
  _PYNAME(SwitchWorkspace),
  _PYNAME(CurrentWorkspace),
  _PYNAME(Workspaces),
  {"cc_ResetWorkspace", ResetWorkspace, METH_VARARGS, ""},
  _PYNAME(RootFolder),
  _PYNAME(OnModuleExit),
  _PYNAME(Blobs),
  _PYNAME(HasBlob),
  {"cc_CreateNet", CreateNet, METH_VARARGS, ""},
  _PYNAME(RunNet),
  _PYNAME(BenchmarkNet),
  _PYNAME(DeleteNet),
  _PYNAME(Nets),
  {"cc_RunOperatorOnce", RunOperatorOnce, METH_VARARGS, ""},
  {"cc_RunNetOnce", RunNetOnce, METH_VARARGS, ""},
  {"cc_RunPlan", RunPlan, METH_VARARGS, ""},
  _PYNAME(CreateBlob),
  _PYNAME(FetchBlob),
  {"cc_FeedBlob", FeedBlob, METH_VARARGS, ""},
  _PYNAME(HasGPUSupport),
#ifndef PYCAFFE2_CPU_ONLY
  _PYNAME(NumberOfGPUs),
  _PYNAME(SetDefaultGPUID),
  _PYNAME(GetDefaultGPUID),
  _PYNAME(GetCudaPeerAccessPattern),
#endif   // !PYCAFFE2_CPU_ONLY
  {NULL, NULL, 0, NULL},  // end of python methods.
};
#undef _PYNAME


#ifdef PYCAFFE2_CPU_ONLY
void initlibcaffe2_python_nogpu(void) {
  (void) Py_InitModule("libcaffe2_python_nogpu", gPycaffe2Methods);
#else  // !PYCAFFE2_CPU_ONPY
void initlibcaffe2_python(void) {
  (void) Py_InitModule("libcaffe2_python", gPycaffe2Methods);
#endif  // PYCAFFE2_CPU_ONPY
  import_array();  // for numpy
  // We will create a default workspace for us to run stuff.
  SwitchWorkspaceInternal("default", true);
  gCurrentWorkspaceName = "default";
}

}  // extern "C"

