#include "pybind_state.h"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "caffe2/core/asan.h"
#include "caffe2/core/db.h"
#include "caffe2/core/operator.h"
#include "caffe2/core/predictor.h"
#include "caffe2/utils/mkl_utils.h"
#include "caffe2/utils/string_utils.h"
#include "google/protobuf/io/coded_stream.h"
#include "google/protobuf/io/zero_copy_stream_impl_lite.h"

#if defined(_MSC_VER)
#include "caffe2/utils/windows_cpu_supports.h"
#endif

namespace caffe2 {
namespace python {

// A dummy variable to overcome the pybind11 py::arg::operator= ambiguity
// for some earlier versions of pybind11.
constexpr bool kPyBindFalse = false;

namespace py = pybind11;

// gWorkspaces allows us to define and switch between multiple workspaces in
// Python.
static std::map<std::string, std::unique_ptr<Workspace>> gWorkspaces;
// gWorkspace is the pointer to the current workspace. The ownership is kept
// by the gWorkspaces map.
static Workspace* gWorkspace = nullptr;
static std::string gCurrentWorkspaceName;

BlobFetcherBase::~BlobFetcherBase() {}
BlobFeederBase::~BlobFeederBase() {}

CAFFE_DEFINE_TYPED_REGISTRY(BlobFetcherRegistry, CaffeTypeId, BlobFetcherBase);
CAFFE_DEFINE_TYPED_REGISTRY(BlobFeederRegistry, int, BlobFeederBase);

REGISTER_BLOB_FETCHER((TypeMeta::Id<TensorCPU>()), TensorFetcher<CPUContext>);
REGISTER_BLOB_FEEDER(CPU, TensorFeeder<CPUContext>);

class StringFetcher : public BlobFetcherBase {
 public:
  py::object Fetch(const Blob& blob) override {
    return py::bytes(blob.Get<string>());
  }
};
REGISTER_BLOB_FETCHER((TypeMeta::Id<string>()), StringFetcher);

static_assert(
    sizeof(int) == sizeof(int32_t),
    "We make an assumption that int is always int32 for numpy "
    "type mapping.");
int CaffeToNumpyType(const TypeMeta& meta) {
  static std::map<CaffeTypeId, int> numpy_type_map{
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
  static std::map<int, TypeMeta> caffe_type_map{
      {NPY_BOOL, TypeMeta::Make<bool>()},
      {NPY_DOUBLE, TypeMeta::Make<double>()},
      {NPY_FLOAT, TypeMeta::Make<float>()},
      {NPY_FLOAT16, TypeMeta::Make<float16>()},
      {NPY_INT, TypeMeta::Make<int>()},
      {NPY_INT8, TypeMeta::Make<int8_t>()},
      {NPY_INT16, TypeMeta::Make<int16_t>()},
      {NPY_INT64, TypeMeta::Make<int64_t>()},
      {NPY_LONG,
       sizeof(long) == sizeof(int) ? TypeMeta::Make<int>()
                                   : TypeMeta::Make<int64_t>()},
      {NPY_LONGLONG, TypeMeta::Make<int64_t>()},
      {NPY_UINT8, TypeMeta::Make<uint8_t>()},
      {NPY_UINT16, TypeMeta::Make<uint16_t>()},
      {NPY_OBJECT, TypeMeta::Make<std::string>()},
      {NPY_UNICODE, TypeMeta::Make<std::string>()},
      {NPY_STRING, TypeMeta::Make<std::string>()},
      // Note: Add more types here.
  };
  static TypeMeta unknown_type;
  const auto it = caffe_type_map.find(numpy_type);
  return it == caffe_type_map.end() ? unknown_type : it->second;
}

template <typename Registry>
std::function<const char*(const string&)> DefinitionGetter(
    const Registry* registry) {
  return [registry](const string& name) { return registry->HelpMessage(name); };
}

void switchWorkspaceInternal(const std::string& name, bool create_if_missing) {
  if (gWorkspaces.count(name)) {
    gCurrentWorkspaceName = name;
    gWorkspace = gWorkspaces[name].get();
    return;
  }

  CAFFE_ENFORCE(create_if_missing);
  std::unique_ptr<Workspace> new_workspace(new Workspace());
  gWorkspace = new_workspace.get();
  gWorkspaces.insert(std::make_pair(name, std::move(new_workspace)));
  gCurrentWorkspaceName = name;
}

namespace python_detail {

// Python Op implementations.
struct Func {
  py::object py_func;
  bool needs_workspace;
};
using FuncRegistery = std::unordered_map<std::string, Func>;

FuncRegistery& gRegistery() {
  // Always leak the objects registered here.
  static FuncRegistery* r = new FuncRegistery();
  return *r;
}

const Func& getOpFunc(const std::string& token) {
  CAFFE_ENFORCE(
      gRegistery().count(token),
      "Python operator for ",
      token,
      " is not available. If you use distributed training it probably means "
      "that python implementation has to be registered in each of the workers");
  return gRegistery()[token];
}

const Func& getGradientFunc(const std::string& token) {
  return getOpFunc(token + "_gradient");
}

py::object fetchBlob(Workspace* ws, const std::string& name) {
  CAFFE_ENFORCE(ws->HasBlob(name), "Can't find blob: ", name);
  const caffe2::Blob& blob = *(ws->GetBlob(name));
  auto fetcher = CreateFetcher(blob.meta().id());
  if (fetcher) {
    return fetcher->Fetch(blob);
  } else {
    // If there is no fetcher registered, return a metainfo string.
    // If all branches failed, we will return a metainfo string.
    std::stringstream ss;
    ss << caffe2::string(name) << ", a C++ native class of type "
       << blob.TypeName() << ".";
    return py::bytes(ss.str());
  }
}
}

void printPythonStackTrace() {
  PyObject *type = nullptr, *value = nullptr, *trace = nullptr;
  PyErr_Fetch(&type, &value, &trace);
  PyTracebackObject* traceback = reinterpret_cast<PyTracebackObject*>(trace);
  vector<PyTracebackObject*> trace_vec;
  while (traceback) {
    trace_vec.push_back(traceback);
    traceback = traceback->tb_next;
  }
  for (int i = trace_vec.size() - 1; i >= 0; --i) {
    int line = trace_vec[i]->tb_lineno;
    const char* filename;
    const char* funcname;
    if (PyUnicode_Check(trace_vec[i]->tb_frame->f_code->co_filename)) {
      auto encoded = PyUnicode_AsEncodedString(
          trace_vec[i]->tb_frame->f_code->co_filename, "ASCII", "replace");
      if (encoded != nullptr) {
        filename = strdup(PyBytes_AS_STRING(encoded));
        Py_DECREF(encoded);
      } else {
        filename = "<unknown>";
      }
    } else {
      filename = PyBytes_AsString(trace_vec[i]->tb_frame->f_code->co_filename);
    }
    if (PyUnicode_Check(trace_vec[i]->tb_frame->f_code->co_name)) {
      auto encoded = PyUnicode_AsEncodedString(
          trace_vec[i]->tb_frame->f_code->co_name, "ASCII", "replace");
      if (encoded != nullptr) {
        funcname = strdup(PyBytes_AS_STRING(encoded));
        Py_DECREF(encoded);
      } else {
        funcname = "<unknown>";
      }
    } else {
      funcname = PyBytes_AsString(trace_vec[i]->tb_frame->f_code->co_name);
    }

    LOG(ERROR) << "    # " << trace_vec.size() - i - 1 << "  " << filename
               << " (" << line << "): " << funcname;
  }
  Py_XDECREF(type);
  Py_XDECREF(value);
  Py_XDECREF(trace);
}

PythonOpBase::PythonOpBase(
    const OperatorDef& operator_def,
    Workspace* ws,
    const std::string& pickled_builder_arg_name)
    : Operator(operator_def, ws),
      ws_(ws),
      token_(OperatorBase::GetSingleArgument<std::string>("token", "")) {
  using namespace python_detail;
  auto pickled = GetSingleArgument<string>(pickled_builder_arg_name, "");
  CAFFE_ENFORCE(
      !pickled.empty() || !token_.empty(),
      "PythonOp requires either pickled_builder or token arg.");
  if (!pickled.empty()) {
    py::gil_scoped_acquire g;
    try {
      auto pickle =
          py::object(PyImport_ImportModule("pickle"), /* borrowed */ false);
      CAFFE_ENFORCE(pickle);
      auto loads = pickle.attr("loads").cast<py::object>();
      CAFFE_ENFORCE(loads);
      auto builder_call = loads(py::bytes(pickled)).cast<py::tuple>();
      CAFFE_ENFORCE(builder_call);
      CAFFE_ENFORCE_EQ(py::len(builder_call), 3);
      auto func = builder_call[0].cast<py::object>();
      auto args = builder_call[1].cast<py::tuple>();
      auto kwargs = builder_call[2].cast<py::dict>();
      auto built_func = func(*args, **kwargs);
      CAFFE_ENFORCE(built_func);
      built_func_.reset(new Func{
          built_func, GetSingleArgument<bool>("pass_workspace", false)});
    } catch (const py::error_already_set& e) {
      LOG(ERROR) << "Python exception encountered while creating PythonOp: "
                 << e.what() << "\nTraceback: ";
      printPythonStackTrace();
      CAFFE_THROW("Python exception encountered while creating PythonOp.");
    }
  }
}

PythonOpBase::~PythonOpBase() {
  if (built_func_) {
    // since it may trigger python interpreter when refcount reaches zero
    py::gil_scoped_acquire g;
    built_func_.reset();
  }
}

bool PythonOpBase::RunOnDevice() {
  std::vector<TensorCPU*> inputs;
  inputs.reserve(InputSize());
  for (auto i = 0; i < InputSize(); ++i) {
    inputs.push_back(const_cast<TensorCPU*>(&Input(i)));
  }
  std::vector<TensorCPU*> outputs;
  outputs.reserve(OutputSize());
  for (auto i = 0; i < OutputSize(); ++i) {
    outputs.push_back(Output(i));
  }
  auto* pyFunc = built_func_ ? built_func_.get() : &getFunc(token_);
  CAFFE_ENFORCE(pyFunc);
  {
    // Acquire GIL for call to Python runtime.
    py::gil_scoped_acquire g;
    try {
      if (pyFunc->needs_workspace) {
        pyFunc->py_func(inputs, outputs, ws_);
      } else {
        pyFunc->py_func(inputs, outputs);
      }
    } catch (const py::error_already_set& e) {
      LOG(ERROR) << "Exception encountered running PythonOp function: "
                 << e.what() << "\nTraceback: ";
      printPythonStackTrace();
      return false;
    }
  }
  return true;
}

const python_detail::Func& PythonOp::getFunc(const std::string& token) {
  return python_detail::getOpFunc(token);
}

const python_detail::Func& PythonGradientOp::getFunc(const std::string& token) {
  return python_detail::getGradientFunc(token);
}

struct GetPythonGradient : public GradientMakerBase {
  using GradientMakerBase::GradientMakerBase;
  std::vector<OperatorDef> GetGradientDefs() override {
    ArgumentHelper helper(Def());
    auto gradOutputIndices =
        helper.GetRepeatedArgument<int>("grad_output_indices");
    auto gradInputIndices =
        helper.GetRepeatedArgument<int>("grad_input_indices");
    std::vector<std::string> gradientInputs;
    for (int i = 0; i < def_.input_size(); ++i) {
      gradientInputs.push_back(I(i));
    }
    for (int i = 0; i < def_.output_size(); ++i) {
      gradientInputs.push_back(O(i));
    }
    if (gradOutputIndices.size() > 0) {
      for (int i = 0; i < gradOutputIndices.size(); ++i) {
        int GO_i = gradOutputIndices[i];
        gradientInputs.push_back(GO(GO_i));
      }
    } else {
      for (int i = 0; i < def_.output_size(); ++i) {
        gradientInputs.push_back(GO(i));
      }
    }
    std::vector<std::string> gradientOutputs;
    if (gradInputIndices.size() > 0) {
      for (int i = 0; i < gradInputIndices.size(); ++i) {
        int GI_i = gradInputIndices[i];
        gradientOutputs.push_back(GI(GI_i));
      }
    } else {
      for (int i = 0; i < def_.input_size(); ++i) {
        gradientOutputs.push_back(GI(i));
      }
    }

    return SingleGradientDef(
        "PythonGradient", "", gradientInputs, gradientOutputs);
  }
};

REGISTER_CPU_OPERATOR(Python, PythonOp);
REGISTER_CPU_OPERATOR(PythonGradient, PythonGradientOp);
// Always allow running in-place
OPERATOR_SCHEMA(Python).AllowInplace([](int, int) { return true; });
OPERATOR_SCHEMA(PythonGradient).AllowInplace([](int, int) { return true; });

REGISTER_GRADIENT(Python, GetPythonGradient);

static bool ParseProtobufFromLargeString(const string& str, Message* proto) {
  ::google::protobuf::io::ArrayInputStream input_stream(str.data(), str.size());
  ::google::protobuf::io::CodedInputStream coded_stream(&input_stream);
  // Set PlanDef message size limit to 1G.
  coded_stream.SetTotalBytesLimit(1024LL << 20, 512LL << 20);
  return proto->ParseFromCodedStream(&coded_stream);
}

void addObjectMethods(py::module& m) {
  py::class_<NetBase>(m, "Net").def("run", [](NetBase* net) {
    py::gil_scoped_release g;
    CAFFE_ENFORCE(net->Run());
  });

  py::class_<Blob>(m, "Blob")
      .def(
          "serialize",
          [](const Blob& blob, const std::string& name) -> py::bytes {
            return blob.Serialize(name);
          })
      .def(
          "deserialize",
          [](Blob* blob, py::bytes serialized) {
            blob->Deserialize(serialized);
          })
      .def(
          "fetch",
          [](const Blob& blob) {
            auto fetcher = CreateFetcher(blob.meta().id());
            CAFFE_ENFORCE(
                fetcher,
                "Could not fetch for blob of type: ",
                blob.meta().name());
            return fetcher->Fetch(blob);
          })
      .def(
          "tensor",
          [](Blob* blob) {
            auto t = blob->GetMutable<TensorCPU>();
            return py::cast(t, py::return_value_policy::reference_internal);
          })
      .def(
          "_feed",
          [](Blob* blob,
             const py::object& arg,
             const py::object device_option) {
            DeviceOption option;
            if (device_option != py::none()) {
              // If we have a device option passed in, read it.
              CAFFE_ENFORCE(ParseProtobufFromLargeString(
                  py::bytes(device_option).cast<std::string>(), &option));
            }
            if (PyArray_Check(arg.ptr())) { // numpy array
              PyArrayObject* array =
                  reinterpret_cast<PyArrayObject*>(arg.ptr());
              auto feeder = CreateFeeder(option.device_type());
              CAFFE_ENFORCE(
                  feeder, "Unknown device type encountered in FeedBlob.");
              feeder->Feed(option, array, blob);
              return true;
            }

            if (PyBytes_Check(arg.ptr()) || PyUnicode_Check(arg.ptr())) {
              *blob->GetMutable<std::string>() = arg.cast<std::string>();
              return true;
            }
            CAFFE_THROW(
                "Unexpected type of argument - only numpy array or string are "
                "supported for feeding");
          },
          "Feed an input array or string, with the (optional) DeviceOption",
          py::arg("arg"),
          py::arg("device_option") = py::none());

  py::class_<TensorCPU>(m, "TensorCPU")
      .def_property_readonly(
          "data",
          [](TensorCPU* t) -> py::object {
            if (t->meta() == TypeMeta{}) {
              // keep this behavior for backward compatibility
              t->mutable_data<float>();
            }
            auto res = TensorFetcher<CPUContext>().FetchTensor(*t, false);
            return res.obj;
          },
          "Return numpy array pointing to this tensor's data if possible. "
          "Otherwise (e.g. for strings) copies the data (same as fetch).")
      .def(
          "feed",
          [](TensorCPU* t, py::object obj) {
            if (!PyArray_Check(obj.ptr())) {
              CAFFE_THROW(
                  "Unexpected type of argument -- expected numpy array");
            }
            TensorFeeder<CPUContext>().FeedTensor(
                DeviceOption{}, reinterpret_cast<PyArrayObject*>(obj.ptr()), t);
          },
          "Copy data from given numpy array into this tensor.")
      .def(
          "fetch",
          [](TensorCPU* t) {
            auto res = TensorFetcher<CPUContext>().FetchTensor(*t, true);
            return res.obj;
          },
          "Copy data from this tensor into a new numpy array.")
      .def(
          "init",
          [](TensorCPU* t, std::vector<TIndex> dims, int caffe_type) {
            const auto& meta =
                DataTypeToTypeMeta((TensorProto::DataType)caffe_type);
            CAFFE_ENFORCE(
                !TensorFetcher<CPUContext>().NeedsCopy(meta),
                "Cannot init tensor of this type. Use `feed` instead.");
            t->Resize(dims);
            t->raw_mutable_data(meta);
          },
          "Initialize this tensor to given shape and data type. "
          "Fail if the given data type cannot be accessed from python.")
      .def_property_readonly(
          "_shape", [](const TensorCPU& t) { return t.dims(); })
      .def("_reshape", [](TensorCPU* t, std::vector<TIndex> dims) {
        t->Resize(dims);
      });

  py::class_<Workspace>(m, "Workspace")
      .def(py::init<>())
      .def(py::init<Workspace*>())
      .def_property_readonly(
          "nets",
          [](Workspace* self) {
            CHECK_NOTNULL(self);
            std::map<std::string, py::object> nets;
            for (const auto& name : self->Nets()) {
              LOG(INFO) << "name: " << name;
              nets[name] = py::cast(
                  self->GetNet(name),
                  py::return_value_policy::reference_internal);
            }
            return nets;
          })
      .def_property_readonly(
          "blobs",
          [](Workspace* self) {
            CHECK_NOTNULL(self);
            std::map<std::string, py::object> blobs;
            for (const auto& name : self->Blobs()) {
              blobs[name] = py::cast(
                  self->GetBlob(name),
                  py::return_value_policy::reference_internal);
            }
            return blobs;
          })
      .def(
          "_create_net",
          [](Workspace* self, py::bytes def, bool overwrite) -> py::object {
            caffe2::NetDef proto;
            CAFFE_ENFORCE(
                ParseProtobufFromLargeString(def.cast<std::string>(), &proto));
            auto* net = self->CreateNet(proto, overwrite);
            CAFFE_ENFORCE(net);
            return py::cast(net, py::return_value_policy::reference_internal);
          },
          py::arg("def"),
          py::arg("overwrite") = kPyBindFalse)
      .def(
          "create_blob",
          [](Workspace* self, const std::string& name) -> py::object {
            auto* blob = self->CreateBlob(name);
            return py::cast(blob, py::return_value_policy::reference_internal);
          })
      .def("fetch_blob", &python_detail::fetchBlob)
      .def(
          "has_blob",
          [](Workspace* self, const std::string& name) {
            return self->HasBlob(name);
          })
      .def(
          "_run_net",
          [](Workspace* self, py::bytes def) {
            caffe2::NetDef proto;
            CAFFE_ENFORCE(
                ParseProtobufFromLargeString(def.cast<std::string>(), &proto));
            py::gil_scoped_release g;
            CAFFE_ENFORCE(self->RunNetOnce(proto));
          })
      .def(
          "_run_operator",
          [](Workspace* self, py::bytes def) {
            caffe2::OperatorDef proto;
            CAFFE_ENFORCE(
                ParseProtobufFromLargeString(def.cast<std::string>(), &proto));
            py::gil_scoped_release g;
            CAFFE_ENFORCE(self->RunOperatorOnce(proto));
          })
      .def(
          "_run_plan",
          [](Workspace* self, py::bytes def) {
            caffe2::PlanDef proto;
            CAFFE_ENFORCE(
                ParseProtobufFromLargeString(def.cast<std::string>(), &proto));
            py::gil_scoped_release g;
            CAFFE_ENFORCE(self->RunPlan(proto));
          })
      .def(
          "_last_failed_op_net_position",
          [](Workspace* self) {
            CAFFE_ENFORCE(self);
            return (int)self->last_failed_op_net_position;
          })
      .def_property_readonly_static("current", [](py::object /* type */) {
        auto ws = gWorkspaces.find(gCurrentWorkspaceName);
        CAFFE_ENFORCE(ws != gWorkspaces.end());
        CAFFE_ENFORCE(ws->second.get());
        return py::cast(ws->second.get(), py::return_value_policy::reference);
      });

  // Gradients
  py::class_<GradientWrapper>(m, "GradientWrapper")
      .def(py::init<>())
      .def_readwrite("dense", &GradientWrapper::dense_)
      .def_readwrite("indices", &GradientWrapper::indices_)
      .def_readwrite("values", &GradientWrapper::values_)
      .def("is_sparse", &GradientWrapper::IsSparse)
      .def("is_dense", &GradientWrapper::IsDense)
      .def("is_empty", &GradientWrapper::IsEmpty);

  m.def(
      "get_gradient_defs",
      [](py::bytes op_def, std::vector<GradientWrapper> output_gradients) {
        OperatorDef def;
        CAFFE_ENFORCE(
            ParseProtobufFromLargeString(op_def.cast<std::string>(), &def));
        CAFFE_ENFORCE(caffe2::GradientRegistry()->Has(def.type()));
        const auto& meta = GetGradientForOp(def, output_gradients);
        std::vector<py::bytes> grad_ops;
        for (const auto& op : meta.ops_) {
          grad_ops.push_back(op.SerializeAsString());
        }
        return std::pair<std::vector<py::bytes>, std::vector<GradientWrapper>>{
            grad_ops, meta.g_input_};
      });

  // DB
  py::class_<db::Transaction>(m, "Transaction")
      .def("put", &db::Transaction::Put)
      .def("commit", &db::Transaction::Commit);
  py::class_<db::Cursor>(m, "Cursor")
      .def("supports_seek", &db::Cursor::SupportsSeek)
      .def("seek_to_first", &db::Cursor::SeekToFirst)
      .def("next", &db::Cursor::Next)
      .def("key", [](db::Cursor* self) -> py::bytes { return self->key(); })
      .def("value", [](db::Cursor* self) -> py::bytes { return self->value(); })
      .def("valid", &db::Cursor::Valid);
  py::enum_<db::Mode>(m, "Mode")
      .value("read", db::Mode::READ)
      .value("write", db::Mode::WRITE)
      .value("new", db::Mode::NEW)
      .export_values();
  py::class_<db::DB /*, std::unique_ptr<DB>*/>(m, "DB")
      .def("new_transaction", &db::DB::NewTransaction)
      .def("new_cursor", &db::DB::NewCursor)
      .def("close", &db::DB::Close);
  m.def("create_db", &db::CreateDB);
  m.def("registered_dbs", []() {
    return caffe2::db::Caffe2DBRegistry()->Keys();
  });


  // OpSchema
  py::class_<OpSchema>(m, "OpSchema")
      .def_property_readonly("file", &OpSchema::file)
      .def_property_readonly("line", &OpSchema::line)
      .def_property_readonly("private", &OpSchema::private_op)
      .def_property_readonly(
          "doc", &OpSchema::doc, py::return_value_policy::reference)
      .def_property_readonly("arg_desc", &OpSchema::arg_desc)
      .def_property_readonly("input_desc", &OpSchema::input_desc)
      .def_property_readonly("output_desc", &OpSchema::output_desc)
      // Note: this does not work yet, we will need to figure out how to pass
      // protobuf objects.
      .def("infer_tensor", &OpSchema::InferTensor)
      .def_static(
          "get", &OpSchemaRegistry::Schema, py::return_value_policy::reference)
      .def_static(
          "get_cpu_impl",
          DefinitionGetter(CPUOperatorRegistry()),
          py::return_value_policy::reference)
      .def_static(
          "get_cuda_impl",
          DefinitionGetter(CUDAOperatorRegistry()),
          py::return_value_policy::reference)
      .def_static(
          "get_gradient_impl",
          DefinitionGetter(GradientRegistry()),
          py::return_value_policy::reference);

  py::class_<Predictor>(m, "Predictor")
      .def(
          "__init__",
          [](Predictor& instance, py::bytes init_net, py::bytes predict_net) {
            CAFFE_ENFORCE(gWorkspace);
            NetDef init_net_, predict_net_;
            CAFFE_ENFORCE(ParseProtobufFromLargeString(
                init_net.cast<std::string>(), &init_net_));
            CAFFE_ENFORCE(ParseProtobufFromLargeString(
                predict_net.cast<std::string>(), &predict_net_));
            new (&instance) Predictor(init_net_, predict_net_, gWorkspace);
          })
      .def(
          "run",
          [](Predictor& instance,
             std::vector<py::object> inputs) -> std::vector<py::object> {
            std::vector<TensorCPU*> tensors;
            std::vector<TensorCPU> tensors_data(inputs.size());
            for (auto i = 0; i < inputs.size(); ++i) {
              auto input = inputs[i];
              CAFFE_ENFORCE(
                  PyArray_Check(input.ptr()),
                  "Input must be of type numpy array.");
              PyArrayObject* array =
                  reinterpret_cast<PyArrayObject*>(input.ptr());
              TensorFeeder<CPUContext>().FeedTensor(
                  DeviceOption(), array, &(tensors_data[i]));
              tensors.push_back(&(tensors_data[i]));
            }
            std::vector<TensorCPU*> out;
            instance.run(tensors, &out);
            std::vector<py::object> pyout;
            for (auto t : out) {
              pyout.push_back(
                  TensorFetcher<CPUContext>().FetchTensor(*t, true).obj);
            }
            return pyout;
          });
}

void addGlobalMethods(py::module& m) {
  m.attr("is_asan") = py::bool_(CAFFE2_ASAN_ENABLED);

  m.attr("has_mkldnn") = py::bool_(
#ifdef CAFFE2_HAS_MKL_DNN
      true
#else // CAFFE2_HAS_MKL_DNN
      false
#endif // CAFFE2_HAS_MKL_DNN
      );

  m.def("global_init", [](std::vector<std::string> args) -> void {
    int argc = args.size();
    std::vector<char*> argv;
    for (auto& arg : args) {
      argv.push_back(const_cast<char*>(arg.data()));
    }
    char** pargv = argv.data();
    CAFFE_ENFORCE(caffe2::GlobalInit(&argc, &pargv));
  });

  m.def("registered_operators", []() {
    std::set<string> all_keys;

    // CPU operators
    for (const auto& name : caffe2::CPUOperatorRegistry()->Keys()) {
      all_keys.insert(name);
    }
    // CUDA operators
    for (const auto& name : caffe2::CUDAOperatorRegistry()->Keys()) {
      all_keys.insert(name);
    }
    // Ensure we are lexicographically ordered.
    std::vector<std::string> keys;
    for (const auto& key : all_keys) {
      keys.push_back(key);
    }
    return keys;
  });
  m.def("on_module_exit", []() { gWorkspaces.clear(); });
  // create_if_missing not used by necessary for pybind to do
  // properly do function overloading.
  m.def("switch_workspace", [](Workspace* ws, py::object create_if_missing) {
    gWorkspace = ws;
  });
  m.def(
      "switch_workspace",
      [](const std::string& name, const py::object create_if_missing) {
        if (create_if_missing == py::none()) {
          return switchWorkspaceInternal(name, false);
        }
        return switchWorkspaceInternal(name, create_if_missing.cast<bool>());
      },
      "Switch to the specified workspace, creating if necessary",
      py::arg("name"),
      py::arg("create_if_missing") = py::none());
  m.def(
      "reset_workspace",
      [](const py::object& root_folder) {
        VLOG(1) << "Resetting workspace.";
        if (root_folder == py::none()) {
          gWorkspaces[gCurrentWorkspaceName].reset(new Workspace());
        } else {
          gWorkspaces[gCurrentWorkspaceName].reset(
              new Workspace(root_folder.cast<std::string>()));
        }
        gWorkspace = gWorkspaces[gCurrentWorkspaceName].get();
        return true;
      },
      "Reset the workspace",
      py::arg("root_folder") = py::none());

  m.def("root_folder", []() {
    CAFFE_ENFORCE(gWorkspace);
    return gWorkspace->RootFolder();
  });
  m.def("current_workspace", []() { return gCurrentWorkspaceName; });
  m.def("workspaces", []() {
    std::vector<std::string> names;
    for (const auto& kv : gWorkspaces) {
      names.push_back(kv.first);
    }
    return names;
  });
  m.def("nearby_opnames", [](const std::string& name) {
    std::vector<std::string> alternatives;
    int editTolerance = 3;
    for (auto it : caffe2::CPUOperatorRegistry()->Keys()) {
      if(editDistance(it, name, editTolerance) < editTolerance + 1) {
        alternatives.push_back(it);
      }
    }
    return alternatives;
  });
  m.def("local_blobs", []() {
    CAFFE_ENFORCE(gWorkspace);
    return gWorkspace->LocalBlobs();
  });
  m.def("blobs", []() {
    CAFFE_ENFORCE(gWorkspace);
    return gWorkspace->Blobs();
  });
  m.def("has_blob", [](const std::string& name) {
    CAFFE_ENFORCE(gWorkspace);
    return gWorkspace->HasBlob(name);
  });
  m.def(
      "create_net",
      [](py::bytes net_def, bool overwrite) {
        CAFFE_ENFORCE(gWorkspace);
        caffe2::NetDef proto;
        CAFFE_ENFORCE(
            ParseProtobufFromLargeString(net_def.cast<std::string>(), &proto),
            "Can't parse net proto: ",
            net_def.cast<std::string>());
        CAFFE_ENFORCE(
            gWorkspace->CreateNet(proto, overwrite),
            "Error creating net with proto: ",
            net_def.cast<std::string>());
        return true;
      },
      py::arg("net_def"),
      py::arg("overwrite") = kPyBindFalse);
  m.def("run_net", [](const std::string& name, int num_iter, bool allow_fail) {
    CAFFE_ENFORCE(gWorkspace);
    CAFFE_ENFORCE(gWorkspace->GetNet(name), "Can't find net ", name);
    py::gil_scoped_release g;
    for (int i = 0; i < num_iter; i++) {
      bool success = gWorkspace->RunNet(name);
      if (!allow_fail) {
        CAFFE_ENFORCE(success, "Error running net ", name);
      } else {
        if (!success) {
          return false;
        }
      }
    }
    return true;
  });
  m.def(
      "benchmark_net",
      [](const std::string& name,
         size_t warmup_runs,
         size_t main_runs,
         bool run_individual) {
        CAFFE_ENFORCE(gWorkspace);
        auto* net = gWorkspace->GetNet(name);
        CAFFE_ENFORCE(net, "Didn't find net: ", name);
        py::gil_scoped_release g;
        vector<float> stat =
            net->TEST_Benchmark(warmup_runs, main_runs, run_individual);
        return stat;
      });

  m.def("delete_net", [](const std::string& name) {
    CAFFE_ENFORCE(gWorkspace);
    gWorkspace->DeleteNet(name);
    return true;
  });
  m.def("nets", []() { return gWorkspace->Nets(); });
  m.def("run_operator_once", [](const py::bytes& op_def) {
    CAFFE_ENFORCE(gWorkspace);
    OperatorDef def;
    CAFFE_ENFORCE(
        ParseProtobufFromLargeString(op_def.cast<std::string>(), &def));
    py::gil_scoped_release g;
    CAFFE_ENFORCE(gWorkspace->RunOperatorOnce(def));
    return true;
  });
  m.def("run_net_once", [](const py::bytes& net_def) {
    CAFFE_ENFORCE(gWorkspace);
    NetDef def;
    CAFFE_ENFORCE(
        ParseProtobufFromLargeString(net_def.cast<std::string>(), &def));
    py::gil_scoped_release g;
    CAFFE_ENFORCE(gWorkspace->RunNetOnce(def));
    return true;
  });
  m.def("run_plan", [](const py::bytes& plan_def) {
    CAFFE_ENFORCE(gWorkspace);
    PlanDef def;
    CAFFE_ENFORCE(
        ParseProtobufFromLargeString(plan_def.cast<std::string>(), &def));
    py::gil_scoped_release g;
    CAFFE_ENFORCE(gWorkspace->RunPlan(def));
    return true;
  });
  m.def(
      "infer_shapes_and_types_from_workspace",
      [](const std::vector<py::bytes>& net_protos) {
        CAFFE_ENFORCE(gWorkspace);

        // Parse protobuffers to NetDefs
        std::vector<std::unique_ptr<caffe2::NetDef>> nets;
        for (auto proto : net_protos) {
          std::unique_ptr<NetDef> def(new NetDef());
          CAFFE_ENFORCE(def.get()->ParseFromString(proto));
          nets.push_back(std::move(def));
        }

        auto blob_info = InferBlobShapesAndTypesFromWorkspace(gWorkspace, nets);

        std::string protob;
        CAFFE_ENFORCE(blob_info.SerializeToString(&protob));
        return py::bytes(protob);
      });
  m.def(
      "infer_shapes_and_types_from_map",
      [](const std::vector<py::bytes>& net_protos,
         const std::map<std::string, std::vector<TIndex>> blob_dimensions) {
        // Parse protobuffers to NetDefs
        std::vector<std::unique_ptr<caffe2::NetDef>> nets;
        for (auto proto : net_protos) {
          std::unique_ptr<NetDef> def(new NetDef());
          CAFFE_ENFORCE(def.get()->ParseFromString(proto));
          nets.push_back(std::move(def));
        }

        auto blob_info = InferBlobShapesAndTypesFromMap(blob_dimensions, nets);

        std::string protob;
        CAFFE_ENFORCE(blob_info.SerializeToString(&protob));
        return py::bytes(protob);
      });
  m.def("create_blob", [](const std::string& name) {
    CAFFE_ENFORCE(gWorkspace);
    CAFFE_ENFORCE(gWorkspace->CreateBlob(name));
    return true;
  });
  m.def("fetch_blob", [](const std::string& name) -> py::object {
    return python_detail::fetchBlob(gWorkspace, name);
  });
  m.def(
      "feed_blob",
      [](const std::string& name, py::object arg, py::object device_option) {
        DeviceOption option;
        if (device_option != py::none()) {
          // If we have a device option passed in, read it.
          CAFFE_ENFORCE(ParseProtobufFromLargeString(
              py::bytes(device_option).cast<std::string>(), &option));
        }
        auto* blob = gWorkspace->CreateBlob(name);
        if (PyArray_Check(arg.ptr())) { // numpy array
          PyArrayObject* array = reinterpret_cast<PyArrayObject*>(arg.ptr());
          auto feeder = CreateFeeder(option.device_type());
          CAFFE_ENFORCE(feeder, "Unknown device type encountered in FeedBlob.");
          feeder->Feed(option, array, blob);
          return true;
        }
        if (PyBytes_Check(arg.ptr()) || PyUnicode_Check(arg.ptr())) { // string
          *blob->GetMutable<std::string>() = arg.cast<std::string>();
          return true;
        }
        CAFFE_THROW(
            "Unexpected type of argument - only numpy array or string are "
            "supported for feeding");
        return false;
      },
      "",
      py::arg("name"),
      py::arg("arg"),
      py::arg("device_option") = py::none());
  m.def("serialize_blob", [](const std::string& name) {
    CAFFE_ENFORCE(gWorkspace);
    auto* blob = gWorkspace->GetBlob(name);
    CAFFE_ENFORCE(blob);
    return py::bytes(blob->Serialize(name));
  });
  m.def(
      "deserialize_blob",
      [](const std::string& name, const py::bytes& serialized) {
        CAFFE_ENFORCE(gWorkspace);
        auto* blob = gWorkspace->CreateBlob(name);
        blob->Deserialize(serialized.cast<std::string>());
      });

  // we support 2 possible signatures of python op: (inputs, outputs) or
  // (inputs, outputs, workspace)
  m.def(
      "register_python_op",
      [](py::object func, bool pass_workspace, std::string name) {
        using namespace python_detail;
        CAFFE_ENFORCE(func != py::none());
        if (!name.empty()) {
          name += ":";
        }
        name += func.attr("__name__").cast<std::string>();
        std::string token = name;
        for (int i = 1; gRegistery().count(token) > 0; ++i) {
          token = name + ":" + to_string(i);
        }
        gRegistery()[token] = Func{func, pass_workspace};
        return token;
      });
  m.def(
      "register_python_gradient_op",
      [](const std::string& token, py::object func) {
        using namespace python_detail;
        CAFFE_ENFORCE(func != py::none());
        CAFFE_ENFORCE(gRegistery().find(token) != gRegistery().end());
        // For global sanity gradient ops shouldn't access workspace
        gRegistery()[token + "_gradient"] = Func{func, false};
      });
  m.def("infer_op_input_output_device", [](const py::bytes& op) {
    std::unique_ptr<caffe2::OperatorDef> def(new caffe2::OperatorDef());
    CAFFE_ENFORCE(def.get()->ParseFromString(op));
    // device_info is a pair of vector of DeviceOption.
    // `first` is for inputs, `second` is for outputs.
    auto device_info = InferOpInputOutputDevice(*def);

    std::vector<py::bytes> in_res;
    std::vector<py::bytes> out_res;
    for (auto& in_dev : device_info.first) {
      std::string protob;
      CAFFE_ENFORCE(in_dev.SerializeToString(&protob));
      in_res.push_back(py::bytes(protob));
    }
    for (auto& out_dev : device_info.second) {
      std::string protob;
      CAFFE_ENFORCE(out_dev.SerializeToString(&protob));
      out_res.push_back(py::bytes(protob));
    }
    return std::make_pair(in_res, out_res);
  });

#define CAFFE2_CPU_FEATURE_SUPPORT(feature)      \
  m.def("builtin_cpu_supports_" #feature, []() { \
    return __builtin_cpu_supports(#feature);     \
  })

// Clang does not support __builtin_cpu_supports until
// revision r240994:
// http://lists.llvm.org/pipermail/cfe-commits/Week-of-Mon-20150629/131941.html
#if (                                                                 \
    __clang__ && ((__apple_build_version__ &&                         \
                   ((__clang_major__ == 8 && __clang_minor__ == 0) || \
                    (__clang_major__ <= 7))) ||                       \
                  (!__apple_build_version__ &&                        \
                   ((__clang_major__ == 3 && __clang_minor__ < 7) ||  \
                    (__clang_major__ <= 2)))))
#warning \
    "Compiling without AVX2. Please consider upgrading your version of Clang."
  // Provide a dummy avx2 flag.
  m.def("builtin_cpu_supports_avx2", []() { return false; });
#elif defined(CAFFE2_NO_BUILTIN_CPU_SUPPORTS) && !defined(__AVX2__)
  // If the compile does not support builtin_cpu_supports, and avx2 is not
  // manually specified, we mark it as not-supported.
  m.def("builtin_cpu_supports_avx2", []() { return false; });
#else
  CAFFE2_CPU_FEATURE_SUPPORT(avx2);
#endif

#undef CAFFE2_CPU_FEATURE_SUPPORT

  auto initialize = [&]() {
    // Initialization of the module
    ([]() -> void {
      // import_array1() forces a void return value.
      import_array1();
    })();
    // Single threaded, so safe
    static bool initialized = false;
    if (initialized) {
      return;
    }
    // We will create a default workspace for us to run stuff.
    switchWorkspaceInternal("default", true);
    gCurrentWorkspaceName = "default";
    initialized = true;
  };

  initialize();
};

PYBIND11_PLUGIN(caffe2_pybind11_state) {
  py::module m(
      "caffe2_pybind11_state",
      "pybind11 stateful interface to Caffe2 workspaces");

  addGlobalMethods(m);
  addObjectMethods(m);
  return m.ptr();
}

} // namespace python
} // namespace caffe2
