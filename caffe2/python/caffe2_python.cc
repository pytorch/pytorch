#include "caffe2/python/caffe2_python.h"

// TODO(Yangqing): avoid carpet-bombing "using namespace".
using namespace caffe2;  // NOLINT

using caffe2::Blob;
using caffe2::DeviceOption;
using caffe2::Tensor;
using caffe2::Workspace;
using caffe2::CPUContext;
using caffe2::OperatorDef;

// gWorkspaces allows us to define and switch between multiple workspaces in
// Python.
static std::map<std::string, std::unique_ptr<Workspace> > gWorkspaces;
// gWorkspace is the pointer to the current workspace. The ownership is kept
// by the gWorkspaces map.
static Workspace* gWorkspace = nullptr;
static std::string gCurrentWorkspaceName;

namespace caffe2 {

BlobFetcherBase::~BlobFetcherBase() {}
BlobFeederBase::~BlobFeederBase() {}

CAFFE_DEFINE_TYPED_REGISTRY(
    BlobFetcherRegistry,
    CaffeTypeId,
    BlobFetcherBase);
CAFFE_DEFINE_TYPED_REGISTRY(
    BlobFeederRegistry,
    int,
    BlobFeederBase);

REGISTER_BLOB_FETCHER(
    (TypeMeta::Id<TensorCPU>()),
    TensorFetcher<CPUContext>);
REGISTER_BLOB_FEEDER(
    CPU,
    TensorFeeder<CPUContext>);

class StringFetcher : public BlobFetcherBase {
 public:
  PyObject* Fetch(const Blob& blob) override {
    return StdStringToPyBytes(blob.Get<string>());
  }
};
REGISTER_BLOB_FETCHER(
    (TypeMeta::Id<string>()),
    StringFetcher);

static_assert(sizeof(int) == sizeof(int32_t),
              "We make an assumption that int is always int32 for numpy "
              "type mapping.");
int CaffeToNumpyType(const TypeMeta& meta) {
  static std::map<CaffeTypeId, int> numpy_type_map {
    {TypeMeta::Id<bool>(), NPY_BOOL},
    {TypeMeta::Id<double>(), NPY_DOUBLE},
    {TypeMeta::Id<float>(), NPY_FLOAT},
    {TypeMeta::Id<float16>(), NPY_FLOAT16},
    {TypeMeta::Id<int>(), NPY_INT},
    {TypeMeta::Id<int8_t>(), NPY_INT8},
    {TypeMeta::Id<int16_t>(), NPY_INT16},
    {TypeMeta::Id<int64_t>(), NPY_LONGLONG},
    {TypeMeta::Id<uint8_t>(), NPY_UINT8},
    {TypeMeta::Id<uint16_t>(), NPY_UINT16},
    {TypeMeta::Id<std::string>(), NPY_OBJECT},
    // Note: Add more types here.
  };
  const auto it = numpy_type_map.find(meta.id());
  return it == numpy_type_map.end() ? -1 : it->second;
}

const TypeMeta& NumpyTypeToCaffe(int numpy_type) {
  static std::map<int, TypeMeta> caffe_type_map {
    {NPY_BOOL, TypeMeta::Make<bool>()},
    {NPY_DOUBLE, TypeMeta::Make<double>()},
    {NPY_FLOAT, TypeMeta::Make<float>()},
    {NPY_FLOAT16, TypeMeta::Make<float16>()},
    {NPY_INT, TypeMeta::Make<int>()},
    {NPY_INT8, TypeMeta::Make<int8_t>()},
    {NPY_INT16, TypeMeta::Make<int16_t>()},
    {NPY_INT64, TypeMeta::Make<int64_t>()},
    {NPY_LONG, sizeof(long) == sizeof(int) ?
               TypeMeta::Make<int>() : TypeMeta::Make<int64_t>()},
    {NPY_LONGLONG, TypeMeta::Make<int64_t>()},
    {NPY_UINT8, TypeMeta::Make<uint8_t>()},
    {NPY_UINT16, TypeMeta::Make<uint16_t>()},
    {NPY_OBJECT, TypeMeta::Make<std::string>()},
    // Note: Add more types here.
  };
  static TypeMeta unknown_type;
  const auto it = caffe_type_map.find(numpy_type);
  return it == caffe_type_map.end() ? unknown_type : it->second;
}

}  // namespace caffe2

extern "C" {

static bool SwitchWorkspaceInternal(
    const string& name, const bool create_if_missing) {
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

PyObject* GlobalInit(PyObject* self, PyObject* args) {
  PyObject* list;
  if (!PyArg_ParseTuple(args, "O!", &PyList_Type, &list)) {
    PyErr_SetString(PyExc_ValueError, "Incorrect arguments.");
    return nullptr;
  }
  int argc = PyList_Size(list);
  std::unique_ptr<char*[]> argv(new char*[std::max(argc, 1)]);
  char** raw_argv = argv.get();
  for (int i = 0; i < argc; ++i) {
    // Get the pointer to the string
    raw_argv[i] = PyBytes_AsString(PyList_GetItem(list, i));
  }
  // Special case for argc = 0: in this case, we will simply add a dummy
  // argv to call caffe2's underlying code.
  if (argc == 0) {
    ++argc;
    static char dummy_argv[] = "python";
    raw_argv[0] = dummy_argv;
  }
  if (!caffe2::GlobalInit(&argc, &raw_argv)) {
    PyErr_SetString(PyExc_RuntimeError, "Error in global init.");
    return nullptr;
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
    CHECK_EQ(PyList_SetItem(list, idx, StdStringToPyBytes(name)), 0);
    ++idx;
  }
  return list;
}

static bool GradientWrappersFromPyList(
    PyObject* g_output_py, vector<GradientWrapper>* pgrad) {
  // Just to be safe... we clear and resize grad. If the grad is passed in
  // empty, this won't hurt much performance either.
  vector<GradientWrapper>& grad = *pgrad;
  grad.clear();
  int size = PyList_Size(g_output_py);
  grad.resize(size);
  for (int i = 0; i < size; ++i) {
    PyObject* obj = PyList_GetItem(g_output_py, i);
    if (obj == Py_None) {
      // No gradient info provided.
      continue;
    } else if (PyTuple_Check(obj)) {
      // Is tuple: should be sparse containing indices and gradients.
      if (PyTuple_Size(obj) != 2) {
        PyErr_SetString(PyExc_TypeError,
                       "Encountered a gradient tuple that is not of size 2");
        return false;
      }
      grad[i].indices_ =
          PyString_AsString(PyObject_Str(PyTuple_GetItem(obj, 0)));
      grad[i].values_ =
          PyString_AsString(PyObject_Str(PyTuple_GetItem(obj, 1)));
    } else {
      // Is dense type.
      // TODO(jiayq): this could go really wrong because PyObject_Str can do
      // any object. Consider sanity check?
      grad[i].dense_ = PyString_AsString(PyObject_Str(obj));
    }
  }
  return true;
}

static PyObject* PyListFromGradientWrappers(
    const vector<GradientWrapper>& grad) {
  PyObject* g_output_py = PyList_New(grad.size());
  for (int i = 0; i < grad.size(); ++i) {
    PyObject* obj = nullptr;
    if (grad[i].IsEmpty()) {
      // Return None
      obj = Py_BuildValue("");
    } else if (grad[i].IsDense()) {
      // Return dense string
      obj = StdStringToPyUnicode(grad[i].dense_);
    } else {
      // Return sparse tuple
      obj = PyTuple_Pack(2, StdStringToPyUnicode(grad[i].indices_),
                         StdStringToPyUnicode(grad[i].values_));
    }
    CHECK_EQ(PyList_SetItem(g_output_py, i, obj), 0);
  }
  //TODO(jiayq): implement
  return g_output_py;
}

PyObject* GetGradientDefs(PyObject* self, PyObject* args) {
  PyObject* def_string = nullptr;
  PyObject* g_output_py = nullptr;
  if (!PyArg_ParseTuple(args, "SO!", &def_string, &PyList_Type, &g_output_py)) {
    PyErr_SetString(PyExc_ValueError,
                    "GetGradientDefs requires an input that is a serialized "
                    "OperatorDef protobuffer, and a list containing the "
                    "gradient of the original op's output.");
    return nullptr;
  }
  OperatorDef def;
  if (!def.ParseFromString(PyBytesToStdString(def_string))) {
    PyErr_SetString(PyExc_ValueError,
                    "Provided string is not a valid OperatorDef protobuffer.");
    return nullptr;
  }
  if (!caffe2::GradientRegistry()->Has(def.type())) {
    PyErr_SetString(PyExc_KeyError, "Gradient not registered.");
    return nullptr;
  }
  vector<GradientWrapper> g_output;
  if (!GradientWrappersFromPyList(g_output_py, &g_output)) {
    return nullptr;
  }
  GradientOpsMeta meta = GetGradientForOp(def, g_output);
  PyObject* grad_ops = PyList_New(meta.ops_.size());
  for (int i = 0; i < meta.ops_.size(); ++i) {
    CHECK_EQ(PyList_SetItem(
        grad_ops, i, StdStringToPyBytes(meta.ops_[i].SerializeAsString())), 0);
  }
  PyObject* g_input_py = PyListFromGradientWrappers(meta.g_input_);
  return PyTuple_Pack(2, grad_ops, g_input_py);
}

PyObject* SwitchWorkspace(PyObject* self, PyObject* args) {
  PyObject* name = nullptr;
  PyObject* create_if_missing = nullptr;
  if (!PyArg_ParseTuple(args, "S|O", &name, &create_if_missing)) {
    PyErr_SetString(PyExc_ValueError,
                    "SwitchWorkspace takes in a workspace name, and "
                    "an optional boolean value that specifies whether "
                    "we want to create the workspace if it is missing.");
    return nullptr;
  }
  bool success = SwitchWorkspaceInternal(
      PyBytesToStdString(name),
      (create_if_missing != nullptr) && PyObject_IsTrue(create_if_missing));
  if (!success) {
    PyErr_SetString(
        PyExc_RuntimeError,
        "Workspace of the given name does not exist, and I am not instructed "
        "to create it either.");
    return nullptr;
  }
  Py_RETURN_TRUE;
}

PyObject* CurrentWorkspace(PyObject* self, PyObject* args) {
  return StdStringToPyBytes(gCurrentWorkspaceName);
}

PyObject* Workspaces(PyObject* self, PyObject* args) {
  PyObject* list = PyList_New(gWorkspaces.size());
  int i = 0;
  for (auto const & it : gWorkspaces) {
    CHECK_EQ(PyList_SetItem(list, i, StdStringToPyBytes(it.first)), 0);
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
    return nullptr;
  }
  VLOG(1) << "Resetting workspace.";
  if (root_folder == nullptr) {
    gWorkspaces[gCurrentWorkspaceName].reset(
        new Workspace());
  } else {
    gWorkspaces[gCurrentWorkspaceName].reset(
        new Workspace(PyBytesToStdString(root_folder)));
  }
  gWorkspace = gWorkspaces[gCurrentWorkspaceName].get();
  Py_RETURN_TRUE;
}

PyObject* RootFolder(PyObject* self, PyObject* args) {
  return StdStringToPyBytes(gWorkspace->RootFolder());
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
    CHECK_EQ(
        PyList_SetItem(list, i, StdStringToPyBytes(blob_strings[i])), 0);
  }
  return list;
}

PyObject* HasBlob(PyObject* self, PyObject* args) {
  char* name;
  if (!PyArg_ParseTuple(args, "s", &name)) {
    return nullptr;
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
    return nullptr;
  }
  caffe2::NetDef proto;
  if (!proto.ParseFromString(PyBytesToStdString(proto_string))) {
    PyErr_SetString(PyExc_ValueError, "Cannot parse input net string.");
    return nullptr;
  }
  if (!gWorkspace->CreateNet(proto)) {
    PyErr_SetString(
        PyExc_RuntimeError,
        "Cannot create network. See console log for error messages.");
    return nullptr;
  }
  Py_RETURN_TRUE;
}

PyObject* RunNet(PyObject* self, PyObject* args) {
  char* cname;
  if (!PyArg_ParseTuple(args, "s", &cname)) {
    PyErr_SetString(PyExc_ValueError,
                    "Incorrect argument. Must pass in a single string.");
    return nullptr;
  }
  caffe2::string name(cname);

  bool result;
  BEGIN_CAFFE2_PY_EXCEPTION_HANDLING_WITH_GUARD;
  result = gWorkspace->RunNet(name);
  END_CAFFE2_PY_EXCEPTION_HANDLING;

  if (!result) {
    PyErr_SetString(
        PyExc_RuntimeError,
        "Cannot run network. See console log for error messages.");
    return nullptr;
  }
  Py_RETURN_TRUE;
}


PyObject* BenchmarkNet(PyObject* self, PyObject* args) {
  char* name;
  int warmup_runs = 0;
  int main_runs = 0;
  PyObject* run_individual_obj = nullptr;
  if (!PyArg_ParseTuple(args, "siiO", &name, &warmup_runs,
                        &main_runs, &run_individual_obj)) {
    PyErr_SetString(PyExc_ValueError,
                    "Incorrect argument.");
    return nullptr;
  }
  caffe2::NetBase* net = gWorkspace->GetNet(caffe2::string(name));
  if (net == nullptr) {
    PyErr_SetString(PyExc_RuntimeError, "Cannot find network.");
    return nullptr;
  }
  bool run_individual = PyObject_IsTrue(run_individual_obj);

  BEGIN_CAFFE2_PY_EXCEPTION_HANDLING_WITH_GUARD;
  net->TEST_Benchmark(warmup_runs, main_runs, run_individual);
  END_CAFFE2_PY_EXCEPTION_HANDLING;

  Py_RETURN_TRUE;
}

PyObject* DeleteNet(PyObject* self, PyObject* args) {
  char* name;
  if (!PyArg_ParseTuple(args, "s", &name)) {
    PyErr_SetString(PyExc_ValueError,
                    "Incorrect argument. Must pass in a single string.");
    return nullptr;
  }
  gWorkspace->DeleteNet(caffe2::string(name));
  Py_RETURN_TRUE;
}

PyObject* Nets(PyObject* self, PyObject* args) {
  std::vector<caffe2::string> net_strings = gWorkspace->Nets();
  PyObject* list = PyList_New(net_strings.size());
  for (int i = 0; i < net_strings.size(); ++i) {
    CHECK_EQ(PyList_SetItem(list, i, StdStringToPyBytes(net_strings[i])), 0);
  }
  return list;
}

PyObject* RunOperatorOnce(PyObject* self, PyObject* args) {
  PyObject* proto_string;
  if (!PyArg_ParseTuple(args, "S", &proto_string)) {
    PyErr_SetString(PyExc_ValueError,
                    "Incorrect argument. Must pass in a single string.");
    return nullptr;
  }
  caffe2::OperatorDef proto;
  if (!proto.ParseFromString(PyBytesToStdString(proto_string))) {
    PyErr_SetString(PyExc_ValueError, "Cannot parse input operator proto.");
    return nullptr;
  }

  bool result;
  BEGIN_CAFFE2_PY_EXCEPTION_HANDLING_WITH_GUARD;
  result = gWorkspace->RunOperatorOnce(proto);
  END_CAFFE2_PY_EXCEPTION_HANDLING;

  if (!result) {
    PyErr_SetString(
        PyExc_RuntimeError,
        "Cannot run operator. See console log for error messages.");
    return nullptr;
  }
  Py_RETURN_TRUE;
}

PyObject* RunNetOnce(PyObject* self, PyObject* args) {
  PyObject* proto_string;
  if (!PyArg_ParseTuple(args, "S", &proto_string)) {
    PyErr_SetString(PyExc_ValueError,
                    "Incorrect argument. Must pass in a single string.");
    return nullptr;
  }
  caffe2::NetDef proto;
  if (!proto.ParseFromString(PyBytesToStdString(proto_string))) {
    PyErr_SetString(PyExc_ValueError, "Cannot parse input net proto.");
    return nullptr;
  }

  bool result;
  BEGIN_CAFFE2_PY_EXCEPTION_HANDLING_WITH_GUARD;
  result = gWorkspace->RunNetOnce(proto);
  END_CAFFE2_PY_EXCEPTION_HANDLING;

  if (!result) {
    PyErr_SetString(
        PyExc_RuntimeError,
        "Cannot run net. See console log for error messages.");
    return nullptr;
  }
  Py_RETURN_TRUE;
}

PyObject* RunPlan(PyObject* self, PyObject* args) {
  PyObject* proto_string;
  if (!PyArg_ParseTuple(args, "S", &proto_string)) {
    PyErr_SetString(PyExc_ValueError,
                    "Incorrect argument. Must pass in a single string.");
    return nullptr;
  }
  caffe2::PlanDef proto;
  if (!proto.ParseFromString(PyBytesToStdString(proto_string))) {
    PyErr_SetString(PyExc_ValueError, "Cannot parse input plan proto.");
    return nullptr;
  }

  bool result;
  BEGIN_CAFFE2_PY_EXCEPTION_HANDLING_WITH_GUARD;
  result = gWorkspace->RunPlan(proto);
  END_CAFFE2_PY_EXCEPTION_HANDLING;

  if (!result) {
    PyErr_SetString(
        PyExc_RuntimeError,
        "Cannot run plan. See console log for error messages.");
    return nullptr;
  }
  Py_RETURN_TRUE;
}

PyObject* CreateBlob(PyObject* self, PyObject* args) {
  char* name_char;
  if (!PyArg_ParseTuple(args, "s", &name_char)) {
    PyErr_SetString(PyExc_ValueError, "Incorrect arguments.");
    return nullptr;
  }
  caffe2::string name(name_char);
  (void) gWorkspace->CreateBlob(name);
  Py_RETURN_TRUE;
}

PyObject* SerializeBlob(PyObject* self, PyObject* args) {
  char* name;
  if (!PyArg_ParseTuple(args, "s", &name)) {
    PyErr_SetString(PyExc_ValueError, "Incorrect arguments.");
    return nullptr;
  }
  if (!gWorkspace->HasBlob(caffe2::string(name))) {
    PyErr_SetString(PyExc_KeyError, "Requested blob does not exist.");
    return nullptr;
  }
  const caffe2::Blob& blob = *(gWorkspace->GetBlob(caffe2::string(name)));
  return StdStringToPyBytes(blob.Serialize(caffe2::string(name)));
}

PyObject* DeserializeBlob(PyObject* self, PyObject* args) {
  char* name;
  char* serialized;
  int serialized_len;
  if (!PyArg_ParseTuple(args, "ss#", &name, &serialized, &serialized_len)) {
    PyErr_SetString(PyExc_ValueError, "Incorrect arguments.");
    return nullptr;
  }
  caffe2::Blob* blob = gWorkspace->CreateBlob(caffe2::string(name));
  if (!blob->Deserialize(std::string(serialized, serialized_len))) {
    PyErr_SetString(PyExc_ValueError, "Deserialization failure.");
    return nullptr;
  }
  Py_RETURN_TRUE;
}

PyObject* FetchBlob(PyObject* self, PyObject* args) {
  char* name;
  if (!PyArg_ParseTuple(args, "s", &name)) {
    PyErr_SetString(PyExc_ValueError, "Incorrect arguments.");
    return nullptr;
  }
  if (!gWorkspace->HasBlob(caffe2::string(name))) {
    PyErr_SetString(
        PyExc_KeyError,
        MakeString("Requested blob does not exist: ", name));
    return nullptr;
  }
  const caffe2::Blob& blob = *(gWorkspace->GetBlob(caffe2::string(name)));
  auto fetcher = CreateFetcher(blob.meta().id());
  if (fetcher) {
    return fetcher->Fetch(blob);
  } else {
    // If there is no fetcher registered, return a metainfo string.
    // If all branches failed, we will return a metainfo string.
    std::stringstream ss;
    ss << caffe2::string(name) << ", a C++ native class of type "
       << blob.TypeName() << ".";
    return StdStringToPyBytes(ss.str());
  }
}

PyObject* FeedBlob(PyObject* self, PyObject* args) {
  char* name_char;
  PyObject* arg = nullptr;
  PyObject* device_option_string = nullptr;
  if (!PyArg_ParseTuple(
          args, "sO|O", &name_char, &arg, &device_option_string)) {
    PyErr_SetString(PyExc_ValueError, "Incorrect arguments.");
    return nullptr;
  }
  caffe2::string name(name_char);
  DeviceOption option;
  if (device_option_string != nullptr) {
    // If we have a device option passed in, read it.
    if (!option.ParseFromString(PyBytesToStdString(device_option_string))) {
      PyErr_SetString(PyExc_ValueError, "Cannot parse device option string.");
      return nullptr;
    }
  }
  Blob* blob = gWorkspace->CreateBlob(name);

  if (PyArray_Check(arg)) { // numpy array
    PyArrayObject* array = reinterpret_cast<PyArrayObject*>(arg);
    auto feeder = CreateFeeder(option.device_type());
    if (!feeder) {
      PyErr_SetString(
          PyExc_TypeError, "Unknown device type encountered in FeedBlob.");
      return nullptr;
    }
    return feeder->Feed(option, array, blob);
  } else if (PyString_Check(arg)) { // string
    *blob->GetMutable<std::string>() = PyBytesToStdString(arg);
    Py_RETURN_TRUE;
  } else {
    PyErr_SetString(
        PyExc_ValueError,
        "Unexpected type of argument - only numpy array or string are "
        "supported for feeding");
    return nullptr;
  }
}

// A simple macro to avoid writing repeated symbols.
#define _PYNAME(name) {#name, name, METH_VARARGS, ""}
PyMethodDef* GetCaffe2PythonMethods() {
  static PyMethodDef gCaffe2PythonMethods[] = {
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
      _PYNAME(SerializeBlob),
      _PYNAME(DeserializeBlob),
      {"cc_FetchBlob", FetchBlob, METH_VARARGS, ""},
      {"cc_FeedBlob", FeedBlob, METH_VARARGS, ""},
      {nullptr, nullptr, 0, nullptr}, // end of python methods.
  };
  return gCaffe2PythonMethods;
}
#undef _PYNAME

// This is a workaround so we can deal with numpy's import_array behavior.
// Despite the fact that you may think import_array() is a function call,
// it is defined as a macro (as of 1.10). As a result, we wrap it inside a
// function to make everythings safe, as well as dealing with  the different
// behaviors in python 2 and 3.
#if PY_MAJOR_VERSION >= 3
#define CAFFE2_NUMPY_RETURN_TYPE int
#else
#define CAFFE2_NUMPY_RETURN_TYPE void
#endif

static CAFFE2_NUMPY_RETURN_TYPE import_array_wrapper() {
  import_array();
}

void common_init_libcaffe2_python_cpu() {
  import_array_wrapper();
  static bool initialized = false;
  if (initialized) {
    return;
  }
  // We will create a default workspace for us to run stuff.
  SwitchWorkspaceInternal("default", true);
  gCurrentWorkspaceName = "default";
  initialized = true;
}

// The initialization code.
#if PY_MAJOR_VERSION >= 3

struct module_state {
  PyObject* error;
};

inline static struct module_state* ModuleGetState(PyObject* module) {
  return (struct module_state*)PyModule_GetState(module);
}
static int caffe2_python_traverse(PyObject* m, visitproc visit, void* arg) {
  Py_VISIT(ModuleGetState(m)->error);
  return 0;
}

static int caffe2_python_clear(PyObject* m) {
  Py_CLEAR(ModuleGetState(m)->error);
  return 0;
}

static struct PyModuleDef gModuleDef = {
  PyModuleDef_HEAD_INIT,
  "libcaffe2_python_cpu",
  nullptr,
  sizeof(struct module_state),
  GetCaffe2PythonMethods(),
  nullptr,
  caffe2_python_traverse,
  caffe2_python_clear,
  nullptr
};

PyObject* PyInit_libcaffe2_python_cpu(void) {
  PyObject* module = PyModule_Create(&gModuleDef);
  if (module == nullptr) {
    return nullptr;
  }
  struct module_state* st = ModuleGetState(module);
  st->error = PyErr_NewException(
      "libcaffe2_python_cpu.Error", nullptr, nullptr);
  if (st->error == nullptr) {
    Py_DECREF(module);
    return nullptr;
  }
  common_init_libcaffe2_python_cpu();
  return module;
}

#else  // PY_MAJOR_VERSION >= 3

void initlibcaffe2_python_cpu(void) {
  PyObject* module = Py_InitModule(
      "libcaffe2_python_cpu", GetCaffe2PythonMethods());
  if (module == nullptr) {
    return;
  }
  common_init_libcaffe2_python_cpu();
}

#endif  // PY_MAJOR_VERSION >= 3

}  // extern "C"
