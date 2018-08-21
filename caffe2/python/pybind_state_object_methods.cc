#include "caffe2/contrib/script/compiler.h"
#include "caffe2/core/db.h"
#include "caffe2/observers/time_observer.h"
#include "caffe2/onnx/backend.h"
#include "caffe2/onnx/helper.h"
#include "caffe2/onnx/onnx_exporter.h"
#include "caffe2/python/pybind_state_background_plan.h"
#include "caffe2/python/pybind_state_detail.h"
#include "caffe2/python/pybind_state_dlpack.h"
#include "caffe2/python/pybind_state_fetcher_feeder.h"
#include "caffe2/python/pybind_state_global_workspace.h"
#include "caffe2/python/pybind_state_registry.h"

namespace caffe2 {
namespace python {

namespace py = pybind11;

// A dummy variable to overcome the pybind11 py::arg::operator= ambiguity
// for some earlier versions of pybind11.
constexpr bool kPyBindFalse = false;

template <typename Registry>
std::function<const char*(const string&)> DefinitionGetter(
    const Registry* registry) {
  return [registry](const string& name) { return registry->HelpMessage(name); };
}

void addObjectMethods(py::module& m) {
  py::class_<NetBase>(m, "Net").def("run", [](NetBase* net) {
    py::gil_scoped_release g;
    CAFFE_ENFORCE(net->Run());
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
          [](Blob* blob) { return py::cast(blob->GetMutableTensor(CPU)); },
          py::return_value_policy::reference_internal)
      .def(
          "_feed",
          [](Blob* blob,
             const py::object& arg,
             const py::object device_option) {
            DeviceOption option;
            if (!device_option.is(py::none())) {
              // If we have a device option passed in, read it.
              CAFFE_ENFORCE(ParseProtoFromLargeString(
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

  py::class_<DLPackWrapper<CPUContext>>(m, "DLPackTensorCPU")
      .def_property_readonly(
          "data",
          [](DLPackWrapper<CPUContext>* t) -> py::object {
            CAFFE_ENFORCE_EQ(
                t->device_option.device_type(),
                CPU,
                "Expected CPU device option for CPU tensor");
            return t->data();
          },
          "Return DLPack tensor with tensor's data.")
      .def(
          "feed",
          [](DLPackWrapper<CPUContext>* t, py::object obj) {
            CAFFE_ENFORCE_EQ(
                t->device_option.device_type(),
                CPU,
                "Expected CPU device option for CPU tensor");
            t->feed(obj);
          },
          "Copy data from given DLPack tensor into this tensor.")
      .def_property_readonly(
          "_shape",
          [](const DLPackWrapper<CPUContext>& t) {
            auto* tensor = t.tensor;
            return tensor->dims();
          })
      .def(
          "_reshape",
          [](DLPackWrapper<CPUContext>* t, std::vector<TIndex> dims) {
            auto* tensor = t->tensor;
            tensor->Resize(dims);
          });

  py::class_<TensorCPU>(m, "TensorCPU")
      .def_property_readonly(
          "data",
          [](TensorCPU* t) -> py::object {
            if (t->meta() == TypeMeta{}) {
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
            auto res = TensorFetcher().FetchTensor(*t, true);
            return res.obj;
          },
          "Copy data from this tensor into a new numpy array.")
      .def(
          "init",
          [](Tensor* t, std::vector<TIndex> dims, int caffe_type) {
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
              nets[name] = py::cast(self->GetNet(name));
            }
            return nets;
          },
          py::return_value_policy::reference_internal)
      .def_property_readonly(
          "blobs",
          [](Workspace* self) {
            CHECK_NOTNULL(self);
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
        auto ws = PythonWorkspaces::Get().find(
            PythonWorkspaces::GetCurrentWorkspaceName());
        CAFFE_ENFORCE(ws != PythonWorkspaces::Get().end());
        CAFFE_ENFORCE(ws->second.get());
        return py::cast(ws->second.get(), py::return_value_policy::reference);
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
          grad_ops.push_back(op.SerializeAsString());
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
            if (args.is(py::none())) {
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
            Predictor::TensorMap tensors;
            std::map<std::string, TensorCPU> tensors_data{};
            for (const auto pair : inputs) {
              const auto& name = pair.first;
              const auto& input = pair.second;
              tensors_data.emplace(name, Tensor(CPU));
              CAFFE_ENFORCE(
                  PyArray_Check(input.ptr()),
                  "Input must be of type numpy array.");
              PyArrayObject* array =
                  reinterpret_cast<PyArrayObject*>(input.ptr());
              TensorFeeder<CPUContext>().FeedTensor(
                  DeviceOption(), array, &tensors_data.at(name));
              tensors.insert(std::make_pair(name, &tensors_data.at(name)));
            }

            std::vector<TensorCPU*> out;
            instance.RunMap(tensors, &out);
            std::vector<py::object> pyout;
            for (auto t : out) {
              pyout.push_back(TensorFetcher().FetchTensor(*t, true).obj);
            }
            return pyout;
          })
      .def(
          "run",
          [](caffe2::onnx::Caffe2BackendRep& instance,
             std::vector<py::object> inputs) -> std::vector<py::object> {
            Predictor::TensorVector tensors;
            std::vector<TensorCPU> tensors_data;
            for (auto i = 0; i < inputs.size(); ++i) {
              tensors_data.emplace_back(caffe2::CPU);
            }
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
            instance.Run(tensors, &out);
            std::vector<py::object> pyout;
            for (auto t : out) {
              pyout.push_back(TensorFetcher().FetchTensor(*t, true).obj);
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
        CAFFE_ENFORCE(PythonWorkspaces::GetCurrent());
        NetDef init_net_, predict_net_;
        CAFFE_ENFORCE(ParseProtoFromLargeString(
            init_net.cast<std::string>(), &init_net_));
        CAFFE_ENFORCE(ParseProtoFromLargeString(
            predict_net.cast<std::string>(), &predict_net_));
        return new Predictor(
            init_net_, predict_net_, PythonWorkspaces::GetCurrent());
      }))
      .def(
          "run",
          [](Predictor& instance,
             std::vector<py::object> inputs) -> std::vector<py::object> {
            Predictor::TensorVector tensors;
            std::vector<Tensor> tensors_data;
            for (auto i = 0; i < inputs.size(); ++i) {
              tensors_data.emplace_back(CPU);
            }
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
              pyout.push_back(TensorFetcher().FetchTensor(*t, true).obj);
            }
            return pyout;
          })
      .def(
          "run",
          [](Predictor& instance, std::map<std::string, py::object> inputs)
              -> std::vector<py::object> {
            Predictor::TensorMap tensors;
            std::map<std::string, TensorCPU> tensors_data{};
            for (const auto pair : inputs) {
              const auto& name = pair.first;
              const auto& input = pair.second;
              tensors_data.emplace(name, Tensor(CPU));
              CAFFE_ENFORCE(
                  PyArray_Check(input.ptr()),
                  "Input must be of type numpy array.");
              PyArrayObject* array =
                  reinterpret_cast<PyArrayObject*>(input.ptr());
              TensorFeeder<CPUContext>().FeedTensor(
                  DeviceOption(), array, &tensors_data.at(name));
              tensors.insert(std::make_pair(name, &tensors_data.at(name)));
            }
            std::vector<TensorCPU*> out;
            instance.run_map(tensors, &out);
            std::vector<py::object> pyout;
            for (auto t : out) {
              pyout.push_back(TensorFetcher().FetchTensor(*t, true).obj);
            }
            return pyout;
          });

  py::class_<script::CompilationUnit>(m, "CompilationUnit")
      .def(py::init<>())
      .def("define", &script::CompilationUnit::define)
      .def("get_proto", &script::CompilationUnit::getProto)
      .def(
          "create_net",
          [](script::CompilationUnit* self, const std::string& name) {
            auto net = self->createNet(PythonWorkspaces::GetCurrent(), name);
            CAFFE_ENFORCE(net);
            return net;
          })
      .def(
          "extern",
          [](script::CompilationUnit* self,
             const std::string& name,
             py::object py_proto) {
            py::bytes bytes = py_proto.attr("SerializeToString")();
            std::unique_ptr<caffe2::NetDef> proto(new NetDef());
            CAFFE_ENFORCE(ParseProtoFromLargeString(
                bytes.cast<std::string>(), proto.get()));
            self->defineExtern(name, std::move(proto));
          });
}

REGISTER_PYBIND_ADDITION(addObjectMethods);

} // namespace python
} // namespace caffe2
