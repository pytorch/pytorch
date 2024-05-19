#include "pybind_state.h"

#include <chrono>
#include <future>
#include <memory>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <c10/macros/Macros.h>

#include "caffe2/core/blob_serialization.h"
#include "caffe2/core/blob_stats.h"
#include "caffe2/core/common.h"
#include "caffe2/core/db.h"
#include "caffe2/core/numa.h"
#include "caffe2/core/operator.h"
#include "caffe2/core/stats.h"
#include "caffe2/core/transform.h"
#include "caffe2/observers/profile_observer.h"
#include "caffe2/observers/runcnt_observer.h"
#include "caffe2/observers/time_observer.h"
#include "caffe2/onnx/backend.h"
#include "caffe2/onnx/helper.h"
#include "caffe2/onnx/offline_tensor.h"
#include "caffe2/onnx/onnx_exporter.h"
#include "caffe2/opt/converter.h"
#include "caffe2/opt/fakefp16_transform.h"
#include "caffe2/opt/fusion.h"
#include "caffe2/opt/mobile.h"
#include "caffe2/opt/onnxifi_transformer.h"
#include "caffe2/opt/optimize_ideep.h"
#include "caffe2/opt/passes.h"
#include "caffe2/opt/shape_info.h"
#include "caffe2/predictor/emulator/data_filler.h"
#include "caffe2/predictor/predictor.h"
#include "caffe2/proto/caffe2_pb.h"
#include "caffe2/proto/torch.pb.h"
#include "caffe2/python/pybind_state_registry.h"
#include "caffe2/python/pybind_workspace.h"
#include "caffe2/utils/cpuid.h"
#include "caffe2/utils/string_utils.h"
#include "torch/csrc/autograd/variable.h"
#include "torch/csrc/jit/python/module_python.h"

// Because of CMake setup, we can't depend on script module here just yet -
// it pulls in generated files from a different directory and it
// probabilistically breaks the build.
// TODO: enable if once shared libraries are unified in CMake
#ifdef FBCODE_CAFFE2
#include "torch/script.h"
#endif

namespace caffe2 {
namespace python {

// A dummy variable to overcome the pybind11 py::arg::operator= ambiguity
// for some earlier versions of pybind11.
constexpr bool kPyBindFalse = false;

namespace py = pybind11;

// NOLINTNEXTLINE(modernize-use-equals-default)
BlobFeederBase::~BlobFeederBase() {}

C10_DEFINE_TYPED_REGISTRY(
    BlobFeederRegistry,
    caffe2::DeviceType,
    BlobFeederBase,
    std::unique_ptr);

REGISTER_BLOB_FETCHER((TypeMeta::Id<Tensor>()), TensorFetcher);
REGISTER_BLOB_FEEDER(CPU, TensorFeeder<CPUContext>);

class StringFetcher : public BlobFetcherBase {
 public:
  py::object Fetch(const Blob& blob) override {
    return py::bytes(blob.Get<string>());
  }
};
REGISTER_BLOB_FETCHER((TypeMeta::Id<string>()), StringFetcher);

#ifdef FBCODE_CAFFE2
class ScriptModuleFetcher : public BlobFetcherBase {
 public:
  pybind11::object Fetch(const Blob& blob) override {
    return py::cast(*blob.Get<std::unique_ptr<torch::jit::Module>>());
  }
};

REGISTER_BLOB_FETCHER(
    (TypeMeta::Id<std::unique_ptr<torch::jit::Module>>()),
    caffe2::python::ScriptModuleFetcher);
#endif

static_assert(
    sizeof(int) == sizeof(int32_t),
    "We make an assumption that int is always int32 for numpy "
    "type mapping.");
int CaffeToNumpyType(const TypeMeta meta) {
#ifdef USE_NUMPY
  static std::map<TypeIdentifier, int> numpy_type_map{
      {TypeMeta::Id<bool>(), NPY_BOOL},
      {TypeMeta::Id<double>(), NPY_DOUBLE},
      {TypeMeta::Id<float>(), NPY_FLOAT},
      {TypeMeta::Id<std::complex<double>>(), NPY_COMPLEX128},
      {TypeMeta::Id<std::complex<float>>(), NPY_COMPLEX64},
      {TypeMeta::Id<at::Half>(), NPY_FLOAT16},
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
#else
  CAFFE_THROW("Caffe2 compiled without NumPy support.");
#endif // USE_NUMPY
}

const TypeMeta NumpyTypeToCaffe(int numpy_type) {
#ifdef USE_NUMPY
  static std::map<int, TypeMeta> caffe_type_map{
      {NPY_BOOL, TypeMeta::Make<bool>()},
      {NPY_DOUBLE, TypeMeta::Make<double>()},
      {NPY_FLOAT, TypeMeta::Make<float>()},
      {NPY_FLOAT16, TypeMeta::Make<at::Half>()},
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
#else
  CAFFE_THROW("Caffe2 compiled without NumPy support.");
#endif // USE_NUMPY
}

template <typename Registry>
std::function<const char*(const string&)> DefinitionGetter(
    const Registry* registry) {
  return [registry](const string& name) { return registry->HelpMessage(name); };
}

namespace python_detail {
// Python Op implementations.
using FuncRegistry = std::unordered_map<std::string, Func>;

FuncRegistry& gRegistry() {
  // Always leak the objects registered here.
  static FuncRegistry* r = new FuncRegistry();
  return *r;
}

const Func& getOpFunc(const std::string& token) {
  CAFFE_ENFORCE(
      gRegistry().count(token),
      "Python operator for ",
      token,
      " is not available. If you use distributed training it probably means "
      "that python implementation has to be registered in each of the workers");
  return gRegistry()[token];
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
    ss << std::string(name) << ", a C++ native class of type "
       << blob.TypeName() << ".";
    return py::bytes(ss.str());
  }
}

// This function can only return true, but keeping it for backward compatibility
bool feedBlob(
    Blob* blob,
    const py::object& arg,
    const py::object device_option) {
  DeviceOption option;
  if (!device_option.is_none()) {
    // If we have a device option passed in, read it.
    CAFFE_ENFORCE(ParseProtoFromLargeString(
        py::bytes(device_option).cast<std::string>(), &option));
  }
#ifdef USE_NUMPY
  if (PyArray_Check(arg.ptr())) { // numpy array
    PyArrayObject* array = reinterpret_cast<PyArrayObject*>(arg.ptr());
    auto feeder = CreateFeeder(option.device_type());
    CAFFE_ENFORCE(feeder, "Unknown device type encountered in FeedBlob.");
    feeder->Feed(option, array, blob, true); /* default to inplace feed */
    return true;
  }
#else
  CAFFE_THROW("Caffe2 compiled without NumPy support.");
#endif // USE_NUMPY
  if (PyBytes_Check(arg.ptr()) || PyUnicode_Check(arg.ptr())) {
    *blob->GetMutable<std::string>() = arg.cast<std::string>();
    return true;
  }
#ifdef FBCODE_CAFFE2
  if (auto module = torch::jit::as_module(arg)) {
    blob->GetMutable<std::unique_ptr<torch::jit::Module>>()->reset(
        new torch::jit::Module(*module));
    return true;
  }
#endif
  CAFFE_THROW(
      "Unexpected type of argument - only numpy array or string are "
      "supported for feeding");
  return false;
}

Blob deserializeBlob(const string& content) {
  Blob blob;
  DeserializeBlob(content, &blob);
  return blob;
}
} // namespace python_detail

class GetPythonGradient : public GradientMakerBase {
 public:
  using GradientMakerBase::GradientMakerBase;
  std::vector<OperatorDef> GetGradientDefs() override {
    CAFFE_ENFORCE(Def().type() == "Python" || Def().type() == "PythonDLPack");
    ArgumentHelper helper(Def());
    auto gradOutputIndices =
        helper.GetRepeatedArgument<int>("grad_output_indices");
    auto gradInputIndices =
        helper.GetRepeatedArgument<int>("grad_input_indices");
    std::vector<std::string> gradientInputs;
    for (int i = 0; i < def_.input_size(); ++i) {
      // NOLINTNEXTLINE(performance-inefficient-vector-operation)
      gradientInputs.push_back(I(i));
    }
    for (int i = 0; i < def_.output_size(); ++i) {
      gradientInputs.push_back(O(i));
    }
    if (gradOutputIndices.size() > 0) {
      // NOLINTNEXTLINE(modernize-loop-convert)
      for (unsigned i = 0; i < gradOutputIndices.size(); ++i) {
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
      // NOLINTNEXTLINE(modernize-loop-convert)
      for (unsigned i = 0; i < gradInputIndices.size(); ++i) {
        int GI_i = gradInputIndices[i];
        gradientOutputs.push_back(GI(GI_i));
      }
    } else {
      for (int i = 0; i < def_.input_size(); ++i) {
        gradientOutputs.push_back(GI(i));
      }
    }

    std::string grad_op_name = "PythonGradient";
    if (Def().type() == "PythonDLPack") {
      grad_op_name = "PythonDLPackGradient";
    }
    return SingleGradientDef(grad_op_name, "", gradientInputs, gradientOutputs);
  }
};

REGISTER_CPU_OPERATOR(Python, PythonOp<CPUContext, false>);
REGISTER_CPU_OPERATOR(PythonGradient, PythonGradientOp<CPUContext, false>);
// Always allow running in-place
OPERATOR_SCHEMA(Python).AllowInplace([](int, int) { return true; });
OPERATOR_SCHEMA(PythonGradient).AllowInplace([](int, int) { return true; });
REGISTER_GRADIENT(Python, GetPythonGradient);

REGISTER_CPU_OPERATOR(PythonDLPack, PythonOp<CPUContext, true>);
REGISTER_CPU_OPERATOR(PythonDLPackGradient, PythonGradientOp<CPUContext, true>);
OPERATOR_SCHEMA(PythonDLPack).AllowInplace([](int, int) { return true; });
OPERATOR_SCHEMA(PythonDLPackGradient).AllowInplace([](int, int) {
  return true;
});
REGISTER_GRADIENT(PythonDLPack, GetPythonGradient);

class BackgroundPlan {
 public:
  // NOLINTNEXTLINE(modernize-pass-by-value)
  BackgroundPlan(Workspace* ws, PlanDef def) : ws_(ws), def_(def) {}

  void run() {
    fut_ =
        std::async(std::launch::async, [this]() { return ws_->RunPlan(def_); });
  }

  bool isDone() {
    CAFFE_ENFORCE(fut_.valid());
    auto status = fut_.wait_for(std::chrono::milliseconds(0));
    return status == std::future_status::ready;
  }

  bool isSucceeded() {
    CAFFE_ENFORCE(isDone());
    return fut_.get();
  }

 private:
  Workspace* ws_;
  PlanDef def_;

  std::future<bool> fut_;
};

void addObjectMethods(py::module& m) {
  py::class_<NetBase>(m, "Net")
      .def(
          "run",
          [](NetBase* net) {
            py::gil_scoped_release g;
            CAFFE_ENFORCE(net->Run());
          })
      .def("cancel", [](NetBase* net) {
        py::gil_scoped_release g;
        net->Cancel();
      });

  py::class_<ObserverBase<NetBase>>(m, "Observer")
      .def(
          "average_time",
          [](ObserverBase<NetBase>* ob) {
            auto* cast_ob = dynamic_cast_if_rtti<TimeObserver*>(ob);
            CAFFE_ENFORCE(
                cast_ob, "Observer does not implement this function.");
            return cast_ob->average_time();
          })
      .def(
          "average_time_children",
          [](ObserverBase<NetBase>* ob) {
            auto* cast_ob = dynamic_cast_if_rtti<TimeObserver*>(ob);
            CAFFE_ENFORCE(
                cast_ob, "Observer does not implement this function.");
            return cast_ob->average_time_children();
          })
      .def("debug_info", [](ObserverBase<NetBase>* ob) {
        return ob->debugInfo();
      });

  py::class_<Blob>(m, "Blob")
      .def(
          "serialize",
          [](const Blob& blob, const std::string& name) -> py::bytes {
            return SerializeBlob(blob, name);
          })
      .def(
          "deserialize",
          [](Blob* blob, py::bytes serialized) {
            DeserializeBlob(serialized, blob);
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
      .def("is_tensor", [](Blob* blob) { return blob->IsType<Tensor>(); })
      // return any device Tensor
      .def(
          "as_tensor",
          [](Blob* blob) {
            CAFFE_ENFORCE(
                blob->IsType<Tensor>(),
                "Passed in blob doesn't contain Tensor and instead has ",
                blob->meta());
            return py::cast(&blob->Get<Tensor>());
          },
          py::return_value_policy::reference_internal)
      // legacy API that resets tensor to CPUTensor if it's not already
      .def(
          "tensor",
          [](Blob* blob) { return py::cast(BlobGetMutableTensor(blob, CPU)); },
          py::return_value_policy::reference_internal)
      .def(
          "_feed",
          &python_detail::feedBlob,
          "Feed an input array or string, with the (optional) DeviceOption",
          py::arg("arg"),
          py::arg("device_option") = py::none())
      .def("_wrap_tensor_impl", [](Blob* blob, void* ptr) {
        auto p = c10::intrusive_ptr<c10::TensorImpl, at::UndefinedTensorImpl>::
            unsafe_reclaim_from_nonowning(static_cast<c10::TensorImpl*>(ptr));
        TORCH_CHECK(p.defined(), "Can't wrap undefined tensor");
        TORCH_CHECK(
            !p->requires_grad(), "Can wrap only non-requires-grad tensor");
        auto at_tensor = at::Tensor::wrap_tensor_impl(std::move(p));
        BlobSetTensor(blob, Tensor(std::move(at_tensor)));
      });

  py::class_<DLPackWrapper<CPUContext>>(m, "DLPackTensorCPU")
      .def_property_readonly(
          "data",
          [](DLPackWrapper<CPUContext>* t) -> py::object {
            CAFFE_ENFORCE_EQ(
                t->device_option.device_type(),
                PROTO_CPU,
                "Expected CPU device option for CPU tensor");
            return t->data();
          },
          "Return DLPack tensor with tensor's data.")
      .def(
          "feed",
          [](DLPackWrapper<CPUContext>* t, py::object obj) {
            CAFFE_ENFORCE_EQ(
                t->device_option.device_type(),
                PROTO_CPU,
                "Expected CPU device option for CPU tensor");
            t->feed(obj);
          },
          "Copy data from given DLPack tensor into this tensor.")
      .def_property_readonly(
          "_shape",
          [](const DLPackWrapper<CPUContext>& t) {
            auto* tensor = t.tensor;
            // TODO: This is marginally less efficient than it could
            // be, since we're doing an extra allocation we didn't
            // need to do.  But I don't remember how to clue in
            // pybind11 how to convert ArrayRef to vector.
            return tensor->sizes().vec();
          })
      .def(
          "_reshape",
          [](DLPackWrapper<CPUContext>* t, std::vector<int64_t> dims) {
            auto* tensor = t->tensor;
            tensor->Resize(dims);
          });

  py::class_<TensorCPU>(m, "TensorCPU")
      .def_property_readonly(
          "data",
          [](TensorCPU* t) -> py::object {
            if (t->dtype() == TypeMeta{}) {
              // keep this behavior for backward compatibility
              t->mutable_data<float>();
            }
            auto res = TensorFetcher().FetchTensor(*t, false);
            return res.obj;
          },
          "Return numpy array pointing to this tensor's data if possible. "
          "Otherwise (e.g. for strings) copies the data (same as fetch).")
      .def(
          "feed",
          [](TensorCPU* t, py::object obj) {
#ifdef USE_NUMPY
            if (!PyArray_Check(obj.ptr())) {
              CAFFE_THROW(
                  "Unexpected type of argument -- expected numpy array");
            }
            *t = TensorFeeder<CPUContext>().FeedTensor(
                DeviceOption{}, reinterpret_cast<PyArrayObject*>(obj.ptr()));
#else
            CAFFE_THROW("Caffe2 compiled without NumPy support.");
#endif // USE_NUMPY
          },
          "Copy data from given numpy array into this tensor.")
      .def(
          "fetch",
          [](TensorCPU* t) {
            auto res = TensorFetcher().FetchTensor(*t, true);
            return res.obj;
          },
          "Copy data from this tensor into a new numpy array.")
      .def(
          "init",
          [](Tensor* t, std::vector<int64_t> dims, int caffe_type) {
            const auto& meta =
                DataTypeToTypeMeta((TensorProto::DataType)caffe_type);
            CAFFE_ENFORCE(
                !TensorFetcher().NeedsCopy(t, meta),
                "Cannot init tensor of this type. Use `feed` instead.");
            t->Resize(dims);
            t->raw_mutable_data(meta);
          },
          "Initialize this tensor to given shape and data type. "
          "Fail if the given data type cannot be accessed from python.")
      .def(
          "_tensor_impl_raw_handle",
          [](TensorCPU* t) -> void* {
            // NOLINTNEXTLINE(performance-unnecessary-copy-initialization)
            auto p = t->getIntrusivePtr();
            // We return a raw non-owning pointer here, we rely on surrounding
            // code to keep the original tensor alive
            return p.get();
          })
      .def_property_readonly(
          "_shape", [](const TensorCPU& t) { return t.sizes().vec(); })
      .def("_reshape", [](TensorCPU* t, std::vector<int64_t> dims) {
        t->Resize(dims);
      });

  py::class_<Workspace>(m, "Workspace")
      .def(py::init<>())
      .def(py::init<Workspace*>())
      .def_property_readonly(
          "nets",
          [](Workspace* self) {
            TORCH_CHECK_NOTNULL(self);
            std::map<std::string, py::object> nets;
            for (const auto& name : self->Nets()) {
              LOG(INFO) << "name: " << name;
              nets[name] = py::cast(self->GetNet(name));
            }
            return nets;
          },
          py::return_value_policy::reference_internal)
      .def_property_readonly(
          "blobs",
          [](Workspace* self) {
            TORCH_CHECK_NOTNULL(self);
            std::map<std::string, py::object> blobs;
            for (const auto& name : self->Blobs()) {
              blobs[name] = py::cast(self->GetBlob(name));
            }
            return blobs;
          },
          py::return_value_policy::reference_internal)
      .def(
          "_create_net",
          [](Workspace* self, py::bytes def, bool overwrite) -> py::object {
            caffe2::NetDef proto;
            CAFFE_ENFORCE(
                ParseProtoFromLargeString(def.cast<std::string>(), &proto));
            NetBase* net = self->CreateNet(proto, overwrite);
            CAFFE_ENFORCE(net);
            return py::cast(net);
          },
          py::return_value_policy::reference_internal,
          py::arg("def"),
          py::arg("overwrite") = kPyBindFalse)
      .def(
          "create_blob",
          [](Workspace* self, const std::string& name) -> py::object {
            return py::cast(self->CreateBlob(name));
          },
          py::return_value_policy::reference_internal)
      .def(
          "_remove_blob",
          [](Workspace* self, const std::string& name) -> py::bool_ {
            return self->RemoveBlob(name);
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
                ParseProtoFromLargeString(def.cast<std::string>(), &proto));
            py::gil_scoped_release g;
            CAFFE_ENFORCE(self->RunNetOnce(proto));
          })
      .def(
          "_run_operator",
          [](Workspace* self, py::bytes def) {
            caffe2::OperatorDef proto;
            CAFFE_ENFORCE(
                ParseProtoFromLargeString(def.cast<std::string>(), &proto));
            py::gil_scoped_release g;
            CAFFE_ENFORCE(self->RunOperatorOnce(proto));
          })
      .def(
          "_run_plan",
          [](Workspace* self, py::bytes def) {
            caffe2::PlanDef proto;
            CAFFE_ENFORCE(
                ParseProtoFromLargeString(def.cast<std::string>(), &proto));
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
        auto ws = caffe2::python::GetCurrentWorkspace();
        CAFFE_ENFORCE(ws);
        return py::cast(ws, py::return_value_policy::reference);
      });

  py::class_<BackgroundPlan, std::shared_ptr<BackgroundPlan>>(
      m, "BackgroundPlan")
      .def("is_done", &BackgroundPlan::isDone)
      .def("is_succeeded", &BackgroundPlan::isSucceeded);

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
            ParseProtoFromLargeString(op_def.cast<std::string>(), &def));
        CAFFE_ENFORCE(caffe2::GradientRegistry()->Has(def.type()));
        const auto& meta = GetGradientForOp(def, output_gradients);
        std::vector<py::bytes> grad_ops;
        for (const auto& op : meta.ops_) {
          // NOLINTNEXTLINE(modernize-use-emplace)
          grad_ops.push_back(
              SerializeAsString_EnforceCheck(op, "addObjectMethods"));
        }
        return std::pair<std::vector<py::bytes>, std::vector<GradientWrapper>>{
            grad_ops, meta.g_input_};
      },
      pybind11::return_value_policy::copy);

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
  py::class_<OpSchema> op_schema(m, "OpSchema");
  op_schema.def_property_readonly("file", &OpSchema::file)
      .def_property_readonly("line", &OpSchema::line)
      .def_property_readonly("private", &OpSchema::private_op)
      .def_property_readonly(
          "doc", &OpSchema::doc, py::return_value_policy::reference)
      .def_property_readonly("args", &OpSchema::args)
      .def_property_readonly("input_desc", &OpSchema::input_desc)
      .def_property_readonly("output_desc", &OpSchema::output_desc)
      .def_property_readonly("max_input", &OpSchema::max_input)
      .def_property_readonly("max_output", &OpSchema::max_output)
      .def_property_readonly("min_input", &OpSchema::min_input)
      .def_property_readonly("min_output", &OpSchema::min_output)
      .def_property_readonly("inf", &OpSchema::inf)
      // Note: this does not work yet, we will need to figure out how to pass
      // protobuf objects.
      .def("infer_tensor", &OpSchema::InferTensor)
      .def("CalculateOutput", &OpSchema::CalculateOutput)
      .def("inplace_enforced", &OpSchema::inplace_enforced)
      .def("num_inputs_allowed", &OpSchema::num_inputs_allowed)
      .def("num_outputs_allowed", &OpSchema::num_outputs_allowed)
      .def("num_inputs_outputs_allowed", &OpSchema::num_inputs_outputs_allowed)
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

  py::class_<OpSchema::Argument>(op_schema, "Argument")
      .def_property_readonly("name", &OpSchema::Argument::name)
      .def_property_readonly("description", &OpSchema::Argument::description)
      .def_property_readonly("required", &OpSchema::Argument::is_required);

  py::class_<caffe2::onnx::Caffe2Ops>(m, "Caffe2Ops")
      .def(py::init([](const std::vector<py::bytes>& init_ops,
                       const std::vector<py::bytes>& ops,
                       const std::vector<std::string>& interface_blobs) {
        auto* c2ops = new caffe2::onnx::Caffe2Ops();
        for (const auto& s : init_ops) {
          ParseProtoFromLargeString(
              s.cast<std::string>(), c2ops->init_ops.Add());
        }
        for (const auto& s : ops) {
          ParseProtoFromLargeString(s.cast<std::string>(), c2ops->ops.Add());
        }
        for (const auto& s : interface_blobs) {
          auto* tmp = c2ops->interface_blobs.Add();
          *tmp = s;
        }
        return c2ops;
      }));

  py::class_<caffe2::onnx::DummyName>(m, "DummyName")
      .def(py::init<>())
      .def(
          "reset",
          [](caffe2::onnx::DummyName& instance, const py::object& args) {
            if (args.is_none()) {
              instance.Reset(std::unordered_set<std::string>());
            } else {
              instance.Reset(args.cast<std::unordered_set<std::string>>());
            }
          },
          "Reset the dummy name generator",
          py::arg("args") = py::none())
      .def(
          "new_dummy_name",
          [](caffe2::onnx::DummyName& instance) -> std::string {
            return instance.NewDummyName();
          });

  py::class_<caffe2::onnx::Caffe2BackendRep>(m, "Caffe2BackenRep")
      .def(py::init<>())
      .def(
          "init_net",
          [](caffe2::onnx::Caffe2BackendRep& instance) {
            const auto& init_net = instance.init_net();
            std::string out;
            init_net.SerializeToString(&out);
            return py::bytes(out);
          })

      .def(
          "pred_net",
          [](caffe2::onnx::Caffe2BackendRep& instance) {
            const auto& pred_net = instance.pred_net();
            std::string out;
            pred_net.SerializeToString(&out);
            return py::bytes(out);
          })
      .def(
          "external_outputs",
          [](caffe2::onnx::Caffe2BackendRep& instance) {
            std::vector<std::string> outputs;
            for (const auto& o : instance.pred_net().external_output()) {
              outputs.emplace_back(o);
            }
            return outputs;
          })
      .def(
          "external_inputs",
          [](caffe2::onnx::Caffe2BackendRep& instance) {
            std::vector<std::string> inputs;
            for (const auto& o : instance.pred_net().external_input()) {
              inputs.emplace_back(o);
            }
            return inputs;
          })
      .def(
          "uninitialized_inputs",
          [](caffe2::onnx::Caffe2BackendRep& instance) {
            return instance.uninitialized_inputs();
          })
      .def(
          "run",
          [](caffe2::onnx::Caffe2BackendRep& instance,
             std::map<std::string, py::object> inputs)
              -> std::vector<py::object> {
            caffe2::Predictor::TensorMap tensors_data{};
            for (const auto& pair : inputs) {
              const auto& name = pair.first;
              const auto& input = pair.second;
#ifdef USE_NUMPY
              CAFFE_ENFORCE(
                  PyArray_Check(input.ptr()),
                  "Input must be of type numpy array.");
              PyArrayObject* array =
                  reinterpret_cast<PyArrayObject*>(input.ptr());
              tensors_data.emplace(
                  name,
                  TensorFeeder<CPUContext>().FeedTensor(DeviceOption(), array));
#else
              CAFFE_THROW("Caffe2 was compiled without NumPy support.");
#endif // USE_NUMPY
            }
            caffe2::Predictor::TensorList out;
            instance.RunMap(tensors_data, &out);
            std::vector<py::object> pyout;
            for (auto& t : out) {
              pyout.push_back(TensorFetcher().FetchTensor(t, true).obj);
            }
            return pyout;
          })
      .def(
          "run",
          [](caffe2::onnx::Caffe2BackendRep& instance,
             std::vector<py::object> inputs) -> std::vector<py::object> {
            std::vector<TensorCPU> tensors_data;
#ifdef USE_NUMPY
            // NOLINTNEXTLINE(modernize-loop-convert)
            for (auto i = 0U; i < inputs.size(); ++i) {
              auto input = inputs[i];
              CAFFE_ENFORCE(
                  PyArray_Check(input.ptr()),
                  "Input must be of type numpy array.");
              PyArrayObject* array =
                  reinterpret_cast<PyArrayObject*>(input.ptr());
              tensors_data.push_back(
                  TensorFeeder<CPUContext>().FeedTensor(DeviceOption(), array));
            }
#else
            CAFFE_THROW("Caffe2 was compiled without NumPy support.");
#endif // USE_NUMPY
            std::vector<TensorCPU> out;
            instance.Run(tensors_data, &out);
            std::vector<py::object> pyout;
            for (auto& t : out) {
              // NOLINTNEXTLINE(performance-inefficient-vector-operation)
              pyout.push_back(TensorFetcher().FetchTensor(t, true).obj);
            }
            return pyout;
          });

  py::class_<caffe2::onnx::Caffe2Backend>(m, "Caffe2Backend")
      .def(py::init<>())
      .def(py::init<caffe2::onnx::DummyName*>())
      .def(
          "support_onnx_import",
          [](caffe2::onnx::Caffe2Backend& instance,
             const std::string& op) -> bool { return instance.SupportOp(op); })
      .def(
          "prepare",
          [](caffe2::onnx::Caffe2Backend& instance,
             const py::bytes& onnx_model_str,
             const std::string& device,
             const std::vector<caffe2::onnx::Caffe2Ops>& extras) {
            auto* rep = instance.Prepare(
                onnx_model_str.cast<std::string>(), device, extras);
            return rep;
          })
      .def(
          "convert_node",
          [](caffe2::onnx::Caffe2Backend& instance,
             const py::bytes& node_str,
             const std::vector<py::bytes>& value_infos_bytes,
             int opset_version) -> std::vector<std::vector<py::bytes>> {
            // Note that we return two lists of serialized ops. The first set is
            // init_ops and the second set is ops for pred net. When converting
            // RNN related op, it is possible that we will create ops in the
            // init_net. Hence the return structure here
            caffe2::onnx::ValueInfoMap value_infos{};
            for (const auto& vi_bytes : value_infos_bytes) {
              ::ONNX_NAMESPACE::ValueInfoProto vi{};
              vi.ParseFromString(vi_bytes);
              auto name = vi.name();
              value_infos.emplace(std::move(name), std::move(vi));
            }
            auto c2ops = instance.ConvertNode(
                node_str.cast<std::string>(), {value_infos, opset_version});
            std::vector<std::vector<py::bytes>> vals;
            vals.emplace_back();
            auto& init_vals = vals.back();
            for (const auto& init_op : c2ops.init_ops) {
              std::string out;
              init_op.SerializeToString(&out);
              init_vals.emplace_back(py::bytes(out));
            }
            vals.emplace_back();
            auto& normal_vals = vals.back();
            for (const auto& op : c2ops.ops) {
              std::string out;
              op.SerializeToString(&out);
              normal_vals.emplace_back(py::bytes(out));
            }
            return vals;
          },
          py::arg("node_str"),
          py::arg("value_infos_bytes") = std::vector<py::bytes>{},
          py::arg("opset_version") = kKnownOpsetVersion)
      .def(
          "_build_tensor_filling_op",
          [](caffe2::onnx::Caffe2Backend& instance,
             const py::bytes& tensor_proto_str,
             const std::string& name = "") -> py::bytes {
            caffe2::OperatorDef op;
            ::ONNX_NAMESPACE::TensorProto tp;
            ParseProtoFromLargeString(tensor_proto_str, &tp);
            instance.BuildTensorFillingOp(&op, tp, name);
            std::string out;
            op.SerializeToString(&out);
            return py::bytes(out);
          });

  py::class_<Predictor>(m, "Predictor")
      .def(py::init([](py::bytes init_net, py::bytes predict_net) {
        Workspace* workspace = caffe2::python::GetCurrentWorkspace();
        CAFFE_ENFORCE(workspace);
        NetDef init_net_, predict_net_;
        CAFFE_ENFORCE(ParseProtoFromLargeString(
            init_net.cast<std::string>(), &init_net_));
        CAFFE_ENFORCE(ParseProtoFromLargeString(
            predict_net.cast<std::string>(), &predict_net_));
        return new Predictor(
            makePredictorConfig(init_net_, predict_net_, workspace));
      }))
      .def(
          "run",
          [](Predictor& instance,
             std::vector<py::object> inputs) -> std::vector<py::object> {
            std::vector<Tensor> tensors_data;
#ifdef USE_NUMPY
            // NOLINTNEXTLINE(modernize-loop-convert)
            for (auto i = 0U; i < inputs.size(); ++i) {
              auto input = inputs[i];
              CAFFE_ENFORCE(
                  PyArray_Check(input.ptr()),
                  "Input must be of type numpy array.");
              PyArrayObject* array =
                  reinterpret_cast<PyArrayObject*>(input.ptr());
              tensors_data.push_back(
                  TensorFeeder<CPUContext>().FeedTensor(DeviceOption(), array));
            }
#else
            CAFFE_THROW("Caffe2 was compiled without NumPy support.");
#endif // USE_NUMPY
            std::vector<TensorCPU> out;
            instance(tensors_data, &out);
            std::vector<py::object> pyout;
            for (auto& t : out) {
              // NOLINTNEXTLINE(performance-inefficient-vector-operation)
              pyout.push_back(TensorFetcher().FetchTensor(t, true).obj);
            }
            return pyout;
          })
      .def(
          "run",
          [](Predictor& instance, std::map<std::string, py::object> inputs)
              -> std::vector<py::object> {
            Predictor::TensorMap tensors_data;
#ifdef USE_NUMPY
            for (const auto& pair : inputs) {
              const auto& name = pair.first;
              const auto& input = pair.second;
              CAFFE_ENFORCE(
                  PyArray_Check(input.ptr()),
                  "Input must be of type numpy array.");
              PyArrayObject* array =
                  reinterpret_cast<PyArrayObject*>(input.ptr());
              tensors_data.emplace(
                  name,
                  TensorFeeder<CPUContext>().FeedTensor(DeviceOption(), array));
            }
#else
            CAFFE_THROW("Caffe2 was compiled without NumPy support.");
#endif // USE_NUMPY
            Predictor::TensorList out;
            instance(tensors_data, &out);
            std::vector<py::object> pyout;
            for (auto& t : out) {
              pyout.push_back(TensorFetcher().FetchTensor(t, true).obj);
            }
            return pyout;
          });
}

void addGlobalMethods(py::module& m) {
  m.attr("is_asan") = py::bool_(C10_ASAN_ENABLED);
  m.attr("has_fbgemm") = py::bool_(
#ifdef USE_FBGEMM
      true
#else
      false
#endif
  );
  m.def("get_build_options", []() { return GetBuildOptions(); });

  // The old mkl backend has been removed permanently, but we
  // keep this Python attribute for BC
  m.attr("has_mkldnn") = py::bool_(false);

  m.attr("use_mkldnn") = py::bool_(
#ifdef USE_MKLDNN
      true
#else // USE_MKLDNN
      false
#endif // USE_MKLDNN
  );

  // if the binary is built with USE_ROCM, this is a ROCm build
  // and therefore we need to ignore dyndep failures (because the module
  // may not have a ROCm equivalent yet e.g. nccl)
  m.attr("use_rocm") = py::bool_(
#if defined(USE_ROCM)
      true
#else // USE_ROCM
      false
#endif // USE_ROCM
  );

  m.attr("use_trt") = py::bool_(
#ifdef CAFFE2_USE_TRT
      true
#else // CAFFE2_USE_TRT
      false
#endif // CAFFE2_USE_TRT
  );

  m.attr("define_caffe2_no_operator_schema") = py::bool_(
#ifdef CAFFE2_NO_OPERATOR_SCHEMA
      true
#else // CAFFE2_NO_OPERATOR_SCHEMA
      false
#endif // CAFFE2_NO_OPERATOR_SCHEMA
  );

  m.def("set_per_op_engine_pref", [](const PerOpEnginePrefType& pref) -> void {
    caffe2::SetPerOpEnginePref(pref);
  });

  m.def("set_global_engine_pref", [](const GlobalEnginePrefType& pref) -> void {
    caffe2::SetGlobalEnginePref(pref);
  });
  m.def(
      "set_engine_pref",
      [](const PerOpEnginePrefType& per_op_pref,
         const GlobalEnginePrefType& global_pref) -> void {
        caffe2::SetEnginePref(per_op_pref, global_pref);
      });
  m.def(
      "set_op_engine_pref",
      [](const std::string& op_type,
         const CaffeMap<DeviceType, EnginePrefType>& op_pref) -> void {
        caffe2::SetOpEnginePref(op_type, op_pref);
      });

  m.def(
      "op_registry_key",
      [](const std::string& op_type,
         const std::string& engine) -> const std::string {
        return caffe2::OpRegistryKey(op_type, engine);
      });
  m.def("global_init", [](std::vector<std::string> args) -> void {
    int argc = args.size();
    std::vector<char*> argv;
    for (auto& arg : args) {
      // NOLINTNEXTLINE(performance-inefficient-vector-operation,cppcoreguidelines-pro-type-const-cast)
      argv.push_back(const_cast<char*>(arg.data()));
    }
    char** pargv = argv.data();
    CAFFE_ENFORCE(caffe2::GlobalInit(&argc, &pargv));
  });

  m.def("registered_operators", []() {
    std::set<string> all_keys = caffe2::GetRegisteredOperators();

    // Ensure we are lexicographically ordered.
    std::vector<std::string> keys;
    for (const auto& key : all_keys) {
      // NOLINTNEXTLINE(performance-inefficient-vector-operation)
      keys.push_back(key);
    }
    return keys;
  });
  m.def("on_module_exit", []() { caffe2::python::ClearWorkspaces(); });
  // create_if_missing not used by necessary for pybind to do
  // properly do function overloading.
  m.def(
      "switch_workspace", [](Workspace* ws, py::object /*create_if_missing*/) {
        // TODO
        caffe2::python::SetCurrentWorkspace(ws);
      });
  m.def(
      "create_child_workspace",
      [](const std::string& parent_ws_name, const std::string& child_ws_name) {
        auto parent_gws = caffe2::python::GetWorkspaceByName(parent_ws_name);
        CAFFE_ENFORCE(parent_gws, "Parent ws does not exist.");
        std::unique_ptr<Workspace> child_ws(new Workspace(parent_gws));
        caffe2::python::InsertWorkspace(child_ws_name, std::move(child_ws));
      },
      "Create and register child ws, sharing existing blobs in parent ws.",
      py::arg("parent_ws_name"),
      py::arg("child_ws_name"));
  m.def(
      "switch_workspace",
      [](const std::string& name, const py::object create_if_missing) {
        if (create_if_missing.is_none()) {
          return caffe2::python::SwitchWorkspaceInternal(name, false);
        }
        return caffe2::python::SwitchWorkspaceInternal(
            name, create_if_missing.cast<bool>());
      },
      "Switch to the specified workspace, creating if necessary",
      py::arg("name"),
      py::arg("create_if_missing") = py::none());
  m.def(
      "reset_workspace",
      [](const py::object& root_folder) {
        VLOG(1) << "Resetting workspace.";
        if (root_folder.is_none()) {
          caffe2::python::ResetWorkspace(new Workspace());
        } else {
          caffe2::python::ResetWorkspace(
              new Workspace(root_folder.cast<std::string>()));
        }
        return true;
      },
      "Reset the workspace",
      py::arg("root_folder") = py::none());

  m.def("root_folder", []() {
    Workspace* workspace = caffe2::python::GetCurrentWorkspace();
    CAFFE_ENFORCE(workspace);
    return workspace->RootFolder();
  });
  m.def("current_workspace", []() {
    return caffe2::python::GetCurrentWorkspaceName();
  });
  m.def("workspaces", []() {
    std::vector<std::string> names;
    caffe2::python::GetWorkspaceNames(names);
    return names;
  });
  m.def("nearby_opnames", [](const std::string& name) {
    std::vector<std::string> alternatives;
    unsigned editTolerance = 3;
    // NOLINTNEXTLINE(performance-for-range-copy)
    for (auto it : caffe2::CPUOperatorRegistry()->Keys()) {
      if (editDistance(it, name, editTolerance) < editTolerance + 1) {
        alternatives.push_back(it);
      }
    }
    return alternatives;
  });
  m.def("local_blobs", []() {
    Workspace* workspace = caffe2::python::GetCurrentWorkspace();
    CAFFE_ENFORCE(workspace);
    return workspace->LocalBlobs();
  });
  m.def("blobs", []() {
    Workspace* workspace = caffe2::python::GetCurrentWorkspace();
    CAFFE_ENFORCE(workspace);
    return workspace->Blobs();
  });
  m.def("has_blob", [](const std::string& name) {
    Workspace* workspace = caffe2::python::GetCurrentWorkspace();
    CAFFE_ENFORCE(workspace);
    return workspace->HasBlob(name);
  });
  m.def(
      "fill_random_network_inputs",
      [](const py::bytes& net_def,
         const std::vector<std::vector<std::vector<int64_t>>>& inputDims,
         const std::vector<std::vector<std::string>>& inputTypes) {
        Workspace* workspace = caffe2::python::GetCurrentWorkspace();
        CAFFE_ENFORCE(workspace);
        py::gil_scoped_release g;
        NetDef net;
        CAFFE_ENFORCE(
            ParseProtoFromLargeString(net_def.cast<std::string>(), &net));
        caffe2::emulator::fillRandomNetworkInputs(
            net, inputDims, inputTypes, workspace);
      });
  m.def(
      "create_net",
      [](py::bytes net_def, bool overwrite) {
        Workspace* workspace = caffe2::python::GetCurrentWorkspace();
        CAFFE_ENFORCE(workspace);
        caffe2::NetDef proto;
        CAFFE_ENFORCE(
            ParseProtoFromLargeString(net_def.cast<std::string>(), &proto),
            "Can't parse net proto: ",
            net_def.cast<std::string>());
        CAFFE_ENFORCE(
            workspace->CreateNet(proto, overwrite),
            "Error creating net with proto: ",
            net_def.cast<std::string>());
        return true;
      },
      py::arg("net_def"),
      py::arg("overwrite") = kPyBindFalse);
  m.def("run_net", [](const std::string& name, int num_iter, bool allow_fail) {
    Workspace* workspace = caffe2::python::GetCurrentWorkspace();
    CAFFE_ENFORCE(workspace);
    CAFFE_ENFORCE(workspace->GetNet(name), "Can't find net ", name);
    py::gil_scoped_release g;
    for (int i = 0; i < num_iter; i++) {
      bool success = workspace->RunNet(name);
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
      "add_observer_to_net",
      [](const std::string& net_name, const std::string& observer_type) {
        Workspace* workspace = caffe2::python::GetCurrentWorkspace();
        CAFFE_ENFORCE(workspace);
        CAFFE_ENFORCE(workspace->GetNet(net_name), "Can't find net ", net_name);
        py::gil_scoped_release g;

        NetBase* net = workspace->GetNet(net_name);
        const Observable<NetBase>::Observer* observer = nullptr;

#define REGISTER_PYTHON_EXPOSED_OBSERVER(ob_type)             \
  {                                                           \
    if (observer_type.compare(#ob_type) == 0) {               \
      unique_ptr<ob_type> net_ob = make_unique<ob_type>(net); \
      observer = net->AttachObserver(std::move(net_ob));      \
    }                                                         \
  }

        REGISTER_PYTHON_EXPOSED_OBSERVER(ProfileObserver);
        REGISTER_PYTHON_EXPOSED_OBSERVER(TimeObserver);
#undef REGISTER_PYTHON_EXPOSED_OBSERVER

        if (observer_type.compare("RunCountObserver") == 0) {
          unique_ptr<RunCountNetObserver> net_ob =
              make_unique<RunCountNetObserver>(net);
          observer = net->AttachObserver(std::move(net_ob));
        }

        CAFFE_ENFORCE(observer != nullptr);
        return py::cast(observer);
      });
  m.def(
      "remove_observer_from_net",
      [](const std::string& net_name, const ObserverBase<NetBase>* observer) {
        Workspace* workspace = caffe2::python::GetCurrentWorkspace();
        CAFFE_ENFORCE(workspace);
        CAFFE_ENFORCE(workspace->GetNet(net_name), "Can't find net ", net_name);
        py::gil_scoped_release g;

        NetBase* net = workspace->GetNet(net_name);
        net->DetachObserver(observer);
      });
  m.def("clear_global_net_observer", []() {
    py::gil_scoped_release g;
    caffe2::ClearGlobalNetObservers();
  });
  m.def("num_observers_on_net", [](const std::string& net_name) {
    Workspace* workspace = caffe2::python::GetCurrentWorkspace();
    CAFFE_ENFORCE(workspace);
    CAFFE_ENFORCE(workspace->GetNet(net_name), "Can't find net ", net_name);
    py::gil_scoped_release g;

    NetBase* net = workspace->GetNet(net_name);
    return net->NumObservers();
  });
  m.def(
      "benchmark_net",
      [](const std::string& name,
         size_t warmup_runs,
         size_t main_runs,
         bool run_individual) {
        Workspace* workspace = caffe2::python::GetCurrentWorkspace();
        CAFFE_ENFORCE(workspace);
        auto* net = workspace->GetNet(name);
        CAFFE_ENFORCE(net, "Didn't find net: ", name);
        py::gil_scoped_release g;
        vector<float> stat =
            net->TEST_Benchmark(warmup_runs, main_runs, run_individual);
        return stat;
      });
  m.def("benchmark_net_once", [](const std::string& name) {
    Workspace* workspace = caffe2::python::GetCurrentWorkspace();
    CAFFE_ENFORCE(workspace);
    auto* net = workspace->GetNet(name);
    CAFFE_ENFORCE(net, "Didn't find net: ", name);
    py::gil_scoped_release g;
    float stat = net->TEST_Benchmark_One_Run();
    return stat;
  });

  m.def("delete_net", [](const std::string& name) {
    Workspace* workspace = caffe2::python::GetCurrentWorkspace();
    CAFFE_ENFORCE(workspace);
    workspace->DeleteNet(name);
    return true;
  });
  m.def("nets", []() {
    Workspace* workspace = caffe2::python::GetCurrentWorkspace();
    CAFFE_ENFORCE(workspace);
    return workspace->Nets();
  });
  m.def("run_operator_once", [](const py::bytes& op_def) {
    Workspace* workspace = caffe2::python::GetCurrentWorkspace();
    CAFFE_ENFORCE(workspace);
    OperatorDef def;
    CAFFE_ENFORCE(ParseProtoFromLargeString(op_def.cast<std::string>(), &def));
    py::gil_scoped_release g;
    CAFFE_ENFORCE(workspace->RunOperatorOnce(def));
    return true;
  });
  // Run an operator multiple times.
  // This is needed for microbenchmarking as we want the benchmark loop to be in
  // C++ to minimize overhead.
  m.def("run_operator_multiple", [](const py::bytes& op_def, int num_runs) {
    Workspace* workspace = caffe2::python::GetCurrentWorkspace();
    CAFFE_ENFORCE(workspace);
    OperatorDef def;
    CAFFE_ENFORCE(ParseProtoFromLargeString(op_def.cast<std::string>(), &def));
    py::gil_scoped_release g;
    std::unique_ptr<OperatorBase> op(CreateOperator(def, workspace));
    for (int i = 0; i < num_runs; i++) {
      if (!op->Run()) {
        return false;
      }
    }
    return true;
  });
  m.def(
      "get_operator_cost",
      [](const py::bytes& op_def, const std::vector<string>& input_blobs) {
        Workspace* workspace = caffe2::python::GetCurrentWorkspace();
        CAFFE_ENFORCE(workspace);
        OperatorDef def;
        CAFFE_ENFORCE(
            ParseProtoFromLargeString(op_def.cast<std::string>(), &def),
            "Couldn't parse operator proto.");
        const auto op_type = def.type();
        auto* schema = OpSchemaRegistry::Schema(op_type);
        CAFFE_ENFORCE(schema);
        vector<TensorShape> shapes;
        for (const auto& blob_name : input_blobs) {
          auto* blob = workspace->GetBlob(blob_name);
          shapes.emplace_back(GetTensorShapeOfBlob(blob));
        }
        const auto c = schema->InferCost(def, shapes);
        return std::make_tuple(c.flops, c.bytes_written, c.bytes_read);
      });
  m.def("run_net_once", [](const py::bytes& net_def) {
    Workspace* workspace = caffe2::python::GetCurrentWorkspace();
    CAFFE_ENFORCE(workspace);
    NetDef def;
    CAFFE_ENFORCE(ParseProtoFromLargeString(net_def.cast<std::string>(), &def));
    py::gil_scoped_release g;
    CAFFE_ENFORCE(workspace->RunNetOnce(def));
    return true;
  });
  m.def("run_plan", [](const py::bytes& plan_def) {
    Workspace* workspace = caffe2::python::GetCurrentWorkspace();
    CAFFE_ENFORCE(workspace);
    PlanDef def;
    CAFFE_ENFORCE(
        ParseProtoFromLargeString(plan_def.cast<std::string>(), &def));
    py::gil_scoped_release g;
    CAFFE_ENFORCE(workspace->RunPlan(def));
    return true;
  });
  m.def("run_plan_in_background", [](const py::bytes& plan_def) {
    Workspace* workspace = caffe2::python::GetCurrentWorkspace();
    CAFFE_ENFORCE(workspace);
    PlanDef def;
    CAFFE_ENFORCE(
        ParseProtoFromLargeString(plan_def.cast<std::string>(), &def));
    py::gil_scoped_release g;

    auto background_plan = std::make_shared<BackgroundPlan>(workspace, def);
    background_plan->run();
    return background_plan;
  });
  m.def(
      "apply_transform",
      [](const string& transform_key, const py::bytes& net_def) {
        NetDef def;
        CAFFE_ENFORCE(
            ParseProtoFromLargeString(net_def.cast<std::string>(), &def));
        py::gil_scoped_release g;

        auto transformed_net = ApplyTransform(transform_key, def);

        std::string protob;
        CAFFE_ENFORCE(transformed_net.SerializeToString(&protob));
        return py::bytes(protob);
      });
  m.def(
      "apply_transform_if_faster",
      [](const string& transform_key,
         const py::bytes& net_def_bytes,
         const py::bytes& init_def_bytes,
         int warmup_runs,
         int main_runs,
         double improvement_threshold) {
        NetDef def;
        CAFFE_ENFORCE(
            ParseProtoFromLargeString(net_def_bytes.cast<std::string>(), &def));
        NetDef init_def;
        CAFFE_ENFORCE(ParseProtoFromLargeString(
            init_def_bytes.cast<std::string>(), &init_def));
        py::gil_scoped_release g;

        std::string protob;

        auto transformed_net = ApplyTransformIfFaster(
            transform_key,
            def,
            init_def,
            warmup_runs,
            main_runs,
            improvement_threshold);

        CAFFE_ENFORCE(transformed_net.SerializeToString(&protob));
        return py::bytes(protob);
      });
  m.def(
      "memonger_compute_blob_recycling_for_dag",
      [](const py::bytes& net_def,
         const std::vector<string>& input_blobs,
         const std::vector<int>& op_indices,
         const std::unordered_set<string>& shareable_blob_names,
         const string& namescope,
         const std::unordered_set<string>& dont_share_blob_names,
         const std::unordered_map<string, vector<int>>& blob_shapes) {
        py::gil_scoped_release g;
        NetDef net;
        CAFFE_ENFORCE(
            ParseProtoFromLargeString(net_def.cast<std::string>(), &net));
        NetDef optimized_proto =
            caffe2::memonger::compute_blob_recycling_for_dag(
                net,
                input_blobs,
                op_indices,
                shareable_blob_names,
                namescope,
                dont_share_blob_names,
                blob_shapes);
        std::string protob;
        CAFFE_ENFORCE(optimized_proto.SerializeToString(&protob));
        return py::bytes(protob);
      });
  m.def(
      "memonger_optimize_inference_net",
      [](const py::bytes& net_def,
         const std::vector<std::string>& static_blobs) {
        NetDef def;
        CAFFE_ENFORCE(
            ParseProtoFromLargeString(net_def.cast<std::string>(), &def));
        py::gil_scoped_release g;

        std::set<string> static_blobs_set(
            static_blobs.begin(), static_blobs.end());
        NetDef optimized =
            caffe2::memonger::optimize_inference_net(def, static_blobs_set);

        std::string protob;
        CAFFE_ENFORCE(optimized.SerializeToString(&protob));
        return py::bytes(protob);
      });
  m.def(
      "infer_shapes_and_types_from_workspace",
      [](const std::vector<py::bytes>& net_protos) {
        Workspace* workspace = caffe2::python::GetCurrentWorkspace();
        CAFFE_ENFORCE(workspace);

        // Parse protobuffers to NetDefs
        std::vector<std::unique_ptr<caffe2::NetDef>> nets;
        std::vector<caffe2::NetDef*> nets_ptr;
        // NOLINTNEXTLINE(performance-for-range-copy)
        for (auto proto : net_protos) {
          std::unique_ptr<NetDef> def(new NetDef());
          CAFFE_ENFORCE(def->ParseFromString(proto));
          nets_ptr.push_back(def.get());
          nets.push_back(std::move(def));
        }

        auto blob_info =
            InferBlobShapesAndTypesFromWorkspace(workspace, nets_ptr);

        std::string protob;
        CAFFE_ENFORCE(blob_info.SerializeToString(&protob));
        return py::bytes(protob);
      });
  m.def(
      "infer_shapes_and_types_from_map",
      [](const std::vector<py::bytes>& net_protos,
         const std::map<std::string, std::vector<int64_t>> blob_dimensions) {
        // Parse protobuffers to NetDefs
        std::vector<std::unique_ptr<caffe2::NetDef>> nets;
        std::vector<caffe2::NetDef*> nets_ptr;
        // NOLINTNEXTLINE(performance-for-range-copy)
        for (auto proto : net_protos) {
          std::unique_ptr<NetDef> def(new NetDef());
          CAFFE_ENFORCE(def->ParseFromString(proto));
          nets_ptr.push_back(def.get());
          nets.push_back(std::move(def));
        }

        auto blob_info =
            InferBlobShapesAndTypesFromMap(blob_dimensions, nets_ptr);

        std::string protob;
        CAFFE_ENFORCE(blob_info.SerializeToString(&protob));
        return py::bytes(protob);
      });
  m.def(
      "infer_shapes_and_types_from_map",
      [](const std::vector<py::bytes>& net_protos,
         const std::map<std::string, std::vector<int64_t>> blob_dimensions,
         const std::map<std::string, int> int_blob_types) {
        // Parse protobuffers to NetDefs
        std::vector<std::unique_ptr<caffe2::NetDef>> nets;
        std::vector<caffe2::NetDef*> nets_ptr;
        // NOLINTNEXTLINE(performance-for-range-copy)
        for (auto proto : net_protos) {
          std::unique_ptr<NetDef> def(new NetDef());
          CAFFE_ENFORCE(def->ParseFromString(proto));
          nets_ptr.push_back(def.get());
          nets.push_back(std::move(def));
        }
        std::map<std::string, TensorProto_DataType> blob_types;
        // NOLINTNEXTLINE(performance-for-range-copy)
        for (auto blob_type : int_blob_types) {
          blob_types[blob_type.first] =
              static_cast<TensorProto_DataType>(blob_type.second);
        }

        auto blob_info = InferBlobShapesAndTypesFromMap(
            blob_dimensions, blob_types, nets_ptr);

        std::string protob;
        CAFFE_ENFORCE(blob_info.SerializeToString(&protob));
        return py::bytes(protob);
      });
  m.def("ssa_rewrite", [](const py::bytes& net_proto) {
    auto net_def = std::make_unique<NetDef>();
    CAFFE_ENFORCE(net_def->ParseFromString(net_proto));
    onnx::SsaRewrite(nullptr, net_def.get());
    std::string output_net_proto;
    CAFFE_ENFORCE(net_def->SerializeToString(&output_net_proto));
    return py::bytes(output_net_proto);
  });
  m.def("create_blob", [](const std::string& name) {
    Workspace* workspace = caffe2::python::GetCurrentWorkspace();
    CAFFE_ENFORCE(workspace);
    CAFFE_ENFORCE(workspace->CreateBlob(name));
    return true;
  });
  m.def("reset_blob", [](const std::string& name) {
    Workspace* workspace = caffe2::python::GetCurrentWorkspace();
    CAFFE_ENFORCE(workspace);
    auto* b = workspace->GetBlob(name);
    CAFFE_ENFORCE(b);
    b->Reset();
  });
  m.def("fetch_blob", [](const std::string& name) -> py::object {
    Workspace* workspace = caffe2::python::GetCurrentWorkspace();
    return python_detail::fetchBlob(workspace, name);
  });
  m.def(
      "feed_blob",
      [](const std::string& name, py::object arg, py::object device_option) {
        Workspace* workspace = caffe2::python::GetCurrentWorkspace();
        auto* blob = workspace->CreateBlob(name);
        return python_detail::feedBlob(blob, arg, device_option);
      },
      "",
      py::arg("name"),
      py::arg("arg"),
      py::arg("device_option") = py::none());
  m.def("deserialize_blob", [](const string& content) {
    return python_detail::deserializeBlob(content);
  });
  m.def("serialize_blob", [](const std::string& name) {
    Workspace* workspace = caffe2::python::GetCurrentWorkspace();
    CAFFE_ENFORCE(workspace);
    auto* blob = workspace->GetBlob(name);
    CAFFE_ENFORCE(blob);
    return py::bytes(SerializeBlob(*blob, name));
  });
  m.def(
      "deserialize_blob",
      [](const std::string& name, const py::bytes& serialized) {
        Workspace* workspace = caffe2::python::GetCurrentWorkspace();
        CAFFE_ENFORCE(workspace);
        auto* blob = workspace->CreateBlob(name);
        DeserializeBlob(serialized.cast<std::string>(), blob);
      });

  // we support 2 possible signatures of python op: (inputs, outputs) or
  // (inputs, outputs, workspace)
  m.def(
      "register_python_op",
      [](py::object func, bool pass_workspace, std::string name) {
        using namespace python_detail;
        CAFFE_ENFORCE(!func.is_none());
        if (!name.empty()) {
          name += ":";
        }
        name += func.attr("__name__").cast<std::string>();
        std::string token = name;
        for (int i = 1; gRegistry().count(token) > 0; ++i) {
          token = name + ":" + to_string(i);
        }
        gRegistry()[token] = Func{func, pass_workspace};
        return token;
      });
  m.def(
      "register_python_gradient_op",
      [](const std::string& token, py::object func) {
        using namespace python_detail;
        CAFFE_ENFORCE(!func.is_none());
        CAFFE_ENFORCE(gRegistry().find(token) != gRegistry().end());
        // For global sanity gradient ops shouldn't access workspace
        gRegistry()[token + "_gradient"] = Func{func, false};
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
      // NOLINTNEXTLINE(modernize-use-emplace)
      in_res.push_back(py::bytes(protob));
    }
    for (auto& out_dev : device_info.second) {
      std::string protob;
      CAFFE_ENFORCE(out_dev.SerializeToString(&protob));
      // NOLINTNEXTLINE(modernize-use-emplace)
      out_res.push_back(py::bytes(protob));
    }
    return std::make_pair(in_res, out_res);
  });
  m.def("get_stats", []() {
    ExportedStatList stats;
    StatRegistry::get().publish(stats);
    std::unordered_map<std::string, int> stats_map;
    for (const auto& stat : stats) {
      stats_map[stat.key] = stat.value;
    }
    return stats_map;
  });
  m.def("is_numa_enabled", []() { return IsNUMAEnabled(); });
  m.def("get_num_numa_nodes", []() { return GetNumNUMANodes(); });
  m.def("get_blob_numa_node", [](const std::string& blob_name) {
    Workspace* workspace = caffe2::python::GetCurrentWorkspace();
    CAFFE_ENFORCE(workspace);
    auto* blob = workspace->GetBlob(blob_name);
    CAFFE_ENFORCE(blob);
    const TensorCPU& tensor = blob->Get<TensorCPU>();
    const void* raw_data = tensor.raw_data();
    CAFFE_ENFORCE(raw_data);
    return GetNUMANode(raw_data);
  });
  m.def("get_blob_size_bytes", [](const std::string& blob_name) {
    Workspace* workspace = caffe2::python::GetCurrentWorkspace();
    CAFFE_ENFORCE(workspace);
    auto* blob = workspace->GetBlob(blob_name);
    CAFFE_ENFORCE(blob);
    return BlobStat::sizeBytes(*blob);
  });
  m.def("support_onnx_export", [](const std::string& op) -> bool {
    const OpSchema* schema = caffe2::OpSchemaRegistry::Schema(op);
    if (!schema) {
      return false;
    }
    return !schema->onnx_schema().empty();
  });
  m.def(
      "export_to_onnx",
      [](caffe2::onnx::DummyName* dummy,
         const py::bytes& c2op,
         const std::unordered_map<std::string, std::vector<int>>& shapes)
          -> std::pair<std::vector<py::bytes>, std::vector<py::bytes>> {
        OperatorDef op;
        CAFFE_ENFORCE(ParseProtoFromLargeString(c2op.cast<std::string>(), &op));
        const auto& type = op.type();
        const OpSchema* schema = caffe2::OpSchemaRegistry::Schema(type);
        CAFFE_ENFORCE(schema);
        std::unordered_map<std::string, TensorShape> tensor_shapes;
        for (const auto& it : shapes) {
          tensor_shapes.emplace(
              it.first, CreateTensorShape(it.second, TensorProto::FLOAT));
        }
        auto results =
            onnx::OnnxExporter(dummy).Caffe2OpToOnnxNodes(op, tensor_shapes);
        std::pair<std::vector<py::bytes>, std::vector<py::bytes>> ret;
        auto& nodes_str = ret.first;
        auto& tensors_str = ret.second;
        for (const auto& node : results.first) {
          std::string out;
          node.SerializeToString(&out);
          nodes_str.emplace_back(py::bytes(out));
        }
        for (const auto& tensor : results.second) {
          std::string out;
          tensor.SerializeToString(&out);
          tensors_str.emplace_back(py::bytes(out));
        }
        return ret;
      });

#define CAFFE2_CPU_FEATURE_SUPPORT(feature) \
  m.def("builtin_cpu_supports_" #feature, []() { return GetCpuId().feature(); })

  CAFFE2_CPU_FEATURE_SUPPORT(avx2);

#undef CAFFE2_CPU_FEATURE_SUPPORT
  m.def("transform_exists", [](const std::string& transform_name) {
    return OptimizationPassRegistry()->Has(transform_name);
  });
  m.def("workspace_transform_exists", [](const std::string& transform_name) {
    return WorkspaceOptimizationPassRegistry()->Has(transform_name);
  });
  m.def("run_transform", [](const std::string& transform_name, py::bytes def) {
    caffe2::NetDef proto;
    CAFFE_ENFORCE(ParseProtoFromLargeString(def.cast<std::string>(), &proto));
    auto nn = caffe2::convertToNNModule(proto);
    auto pass = OptimizationPassRegistry()->Create(transform_name, &nn);

    CAFFE_ENFORCE(pass, "Pass doesn't exist: ", transform_name);
    pass->run();

    auto new_proto = caffe2::convertToCaffe2Proto(nn, proto);
    std::string out;
    new_proto.SerializeToString(&out);
    return py::bytes(out);
  });
  m.def(
      "create_offline_tensor",
      [](const std::string& name,
         const std::vector<int>& dims,
         int datatype) -> bool {
        Workspace* curr_ws = GetCurrentWorkspace();
        auto* b = curr_ws->CreateBlob(name);
        auto* offline = b->GetMutable<OfflineTensor>();
        CAFFE_ENFORCE(offline);
        offline->setShapeAndType(
            dims,
            CPU,
            DataTypeToTypeMeta(static_cast<TensorProto::DataType>(datatype)));
        return true;
      });
  m.def(
      "onnxifi_set_option",
      [](const std::string& optionName,
         const std::string& optionValue) -> bool {
        OnnxifiOptionHelper ts;
        return ts.setOnnxifiOption(optionName, optionValue);
      });
  m.def("onnxifi_get_option", [](const std::string& optionName) -> std::string {
    OnnxifiOptionHelper ts;
    return ts.getOnnxifiOption(optionName);
  });
  m.def(
      "onnxifi",
      [](const py::bytes& pred_net_str,
         const py::bytes& shapes_str,
         const std::vector<int>& block_list,
         const std::vector<std::string>& weight_names,
         int max_batch_size,
         int max_seq_size,
         int timeout,
         bool adjust_batch,
         bool debug_builder,
         bool merge_fp32_inputs_into_fp16,
         bool net_ssa_rewritten,
         bool use_onnx) -> py::bytes {
        caffe2::NetDef pred_net;
        CAFFE_ENFORCE(
            ParseProtoFromLargeString(
                pred_net_str.cast<std::string>(), &pred_net),
            "broken pred_net protobuf");
        Workspace* curr_ws = GetCurrentWorkspace();
        CAFFE_ENFORCE(curr_ws);
        splitSparseLengthsSumSparse(&pred_net, *curr_ws);
        caffe2::TensorBoundShapes tbs;
        CAFFE_ENFORCE(
            ParseProtoFromLargeString(shapes_str.cast<std::string>(), &tbs),
            "broken TensorBoundShapes protobuf");
        ShapeInfoMap shape_map = caffe2::extractShapeInfoFromTensorBoundShapes(
            tbs, max_batch_size, max_seq_size);
        OnnxifiTransformerOptions opts;
        opts.bound_shape_spec.max_batch_size = max_batch_size;
        opts.bound_shape_spec.max_seq_size = max_seq_size;
        opts.timeout = timeout;
        opts.adjust_batch = adjust_batch;
        opts.debug = debug_builder;
        opts.merge_fp32_inputs_into_fp16 = merge_fp32_inputs_into_fp16;
        opts.predictor_net_ssa_rewritten = net_ssa_rewritten;
        opts.use_onnx = use_onnx;
        OnnxifiTransformer ts(opts);
        std::unordered_set<int> blocklist_set(
            block_list.begin(), block_list.end());
        std::vector<std::string> weight_names_overwrite{};
        if (weight_names.size() == 0) {
          weight_names_overwrite = curr_ws->Blobs();
        } else {
          weight_names_overwrite = weight_names;
        }
        ts.transform(
            curr_ws,
            &pred_net,
            weight_names_overwrite,
            shape_map,
            blocklist_set);
        std::string pred_net_str2;
        pred_net.SerializeToString(&pred_net_str2);
        return py::bytes(pred_net_str2);
      });
  m.def(
      "run_workspace_transform",
      [](const std::string& transform_name, py::bytes def) {
        Workspace* workspace = caffe2::python::GetCurrentWorkspace();
        CAFFE_ENFORCE(workspace);
        caffe2::NetDef proto;
        CAFFE_ENFORCE(
            ParseProtoFromLargeString(def.cast<std::string>(), &proto));
        auto nn = caffe2::convertToNNModule(proto);
        auto pass = WorkspaceOptimizationPassRegistry()->Create(
            transform_name, &nn, workspace);

        CAFFE_ENFORCE(pass, "Pass doesn't exist: ", transform_name);
        pass->run();

        auto new_proto = caffe2::convertToCaffe2Proto(nn, proto);
        std::string out;
        new_proto.SerializeToString(&out);
        return py::bytes(out);
      });
  m.def("fakeFp16FuseOps", [](const py::bytes& net_str) {
    caffe2::NetDef netDef;
    CAFFE_ENFORCE(
        ParseProtoFromLargeString(net_str.cast<std::string>(), &netDef),
        "broken pred_net protobuf");
    opt::fakeFp16FuseOps(&netDef);
    std::string out_net;
    netDef.SerializeToString(&out_net);
    return py::bytes(out_net);
  });

  // Transformations are exposed as functions here and wrapped
  // into a python interface in transformations.py
  // Prefix the transformation with transform_ to avoid clobbering the
  // function namespace.
  m.def("transform_optimizeForMKLDNN", [](py::bytes def, bool training_mode) {
    caffe2::NetDef proto;
    CAFFE_ENFORCE(ParseProtoFromLargeString(def.cast<std::string>(), &proto));

    auto nn = caffe2::convertToNNModule(proto);
    Workspace* workspace = caffe2::python::GetCurrentWorkspace();
    opt::OptimizeForMkldnn(&nn, workspace, training_mode);
    auto new_proto = caffe2::convertToCaffe2Proto(nn, proto);

    std::string out;
    new_proto.SerializeToString(&out);
    return py::bytes(out);
  });

  m.def("transform_addNNPACK", [](py::bytes def) {
    caffe2::NetDef proto;
    CAFFE_ENFORCE(ParseProtoFromLargeString(def.cast<std::string>(), &proto));

    auto nn = caffe2::convertToNNModule(proto);
    opt::addNNPACK(&nn);
    auto new_proto = caffe2::convertToCaffe2Proto(nn, proto);

    std::string out;
    new_proto.SerializeToString(&out);
    return py::bytes(out);
  });

  m.def("transform_fuseConvBN", [](py::bytes def) {
    Workspace* workspace = caffe2::python::GetCurrentWorkspace();
    CAFFE_ENFORCE(workspace);
    caffe2::NetDef proto;
    CAFFE_ENFORCE(ParseProtoFromLargeString(def.cast<std::string>(), &proto));

    auto nn = caffe2::convertToNNModule(proto);
    opt::fuseConvBN(&nn, workspace);
    auto new_proto = caffe2::convertToCaffe2Proto(nn);

    std::string out;
    new_proto.SerializeToString(&out);
    return py::bytes(out);
  });

  m.def("transform_fuseNNPACKConvRelu", [](py::bytes def) {
    caffe2::NetDef proto;
    CAFFE_ENFORCE(ParseProtoFromLargeString(def.cast<std::string>(), &proto));

    auto nn = caffe2::convertToNNModule(proto);
    opt::fuseNNPACKConvRelu(&nn);
    auto new_proto = caffe2::convertToCaffe2Proto(nn, proto);

    std::string out;
    new_proto.SerializeToString(&out);
    return py::bytes(out);
  });

  auto initialize = [&]() {
  // Initialization of the module
#ifdef USE_NUMPY
    ([]() -> void {
      // import_array1() forces a void return value.
      import_array1();
    })();
#endif // USE_NUMPY
    // Single threaded, so safe
    static bool initialized = false;
    if (initialized) {
      return;
    }
    // We will create a default workspace for us to run stuff.
    caffe2::python::SwitchWorkspaceInternal("default", true);
    initialized = true;
  };

  initialize();
};

PYBIND11_MODULE(caffe2_pybind11_state, m) {
  m.doc() = "pybind11 stateful interface to Caffe2 workspaces";

  C10_LOG_API_USAGE_ONCE("caffe2.python.import");

  addGlobalMethods(m);
  addObjectMethods(m);
  for (const auto& addition : PybindAdditionRegistry()->Keys()) {
    PybindAdditionRegistry()->Create(addition, m);
  }
}

} // namespace python
} // namespace caffe2
