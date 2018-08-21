#include "caffe2/python/pybind_state_registry.h"

#include "caffe2/core/asan.h"
#include "caffe2/core/blob_stats.h"
#include "caffe2/core/init.h"
#include "caffe2/core/memonger.h"
#include "caffe2/core/operator.h"
#include "caffe2/core/stats.h"
#include "caffe2/core/transform.h"
#include "caffe2/core/workspace.h"
#include "caffe2/observers/runcnt_observer.h"
#include "caffe2/observers/time_observer.h"
#include "caffe2/onnx/backend.h"
#include "caffe2/onnx/helper.h"
#include "caffe2/onnx/onnx_exporter.h"
#include "caffe2/opt/converter.h"
#include "caffe2/opt/fusion.h"
#include "caffe2/opt/mobile.h"
#include "caffe2/opt/onnxifi_transformer.h"
#include "caffe2/opt/optimize_ideep.h"
#include "caffe2/opt/passes.h"
#include "caffe2/opt/sink.h"
#include "caffe2/python/pybind_state_background_plan.h"
#include "caffe2/python/pybind_state_detail.h"
#include "caffe2/python/pybind_state_fetcher_feeder.h"
#include "caffe2/python/pybind_state_global_workspace.h"
#include "caffe2/utils/cpuid.h"
#include "caffe2/utils/string_utils.h"

namespace caffe2 {
namespace python {

namespace py = pybind11;

// A dummy variable to overcome the pybind11 py::arg::operator= ambiguity
// for some earlier versions of pybind11.
constexpr bool kPyBindFalse = false;

void addGlobalMethods(py::module& m) {
  m.attr("is_asan") = py::bool_(CAFFE2_ASAN_ENABLED);
  m.def("get_build_options", []() { return GetBuildOptions(); });

  m.attr("has_mkldnn") = py::bool_(
#ifdef CAFFE2_HAS_MKL_DNN
      true
#else // CAFFE2_HAS_MKL_DNN
      false
#endif // CAFFE2_HAS_MKL_DNN
  );

  m.attr("use_ideep") = py::bool_(
#ifdef CAFFE2_USE_IDEEP
      true
#else // CAFFE2_USE_IDEEP
      false
#endif // CAFFE2_USE_IDEEP
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
         const CaffeMap<int, EnginePrefType>& op_pref) -> void {
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
      keys.push_back(key);
    }
    return keys;
  });
  m.def("on_module_exit", []() { PythonWorkspaces::Get().clear(); });
  // create_if_missing not used by necessary for pybind to do
  // properly do function overloading.
  m.def(
      "switch_workspace",
      [](const std::string& name, const py::object create_if_missing) {
        if (create_if_missing.is(py::none())) {
          return PythonWorkspaces::SwitchWorkspaceInternal(name, false);
        }
        return PythonWorkspaces::SwitchWorkspaceInternal(
            name, create_if_missing.cast<bool>());
      },
      "Switch to the specified workspace, creating if necessary",
      py::arg("name"),
      py::arg("create_if_missing") = py::none());
  m.def(
      "reset_workspace",
      [](const py::object& root_folder) {
        VLOG(1) << "Resetting workspace.";
        if (root_folder.is(py::none())) {
          PythonWorkspaces::Get()[PythonWorkspaces::GetCurrentWorkspaceName()]
              .reset(new Workspace());
        } else {
          PythonWorkspaces::Get()[PythonWorkspaces::GetCurrentWorkspaceName()]
              .reset(new Workspace(root_folder.cast<std::string>()));
        }
        PythonWorkspaces::GetCurrent() =
            PythonWorkspaces::Get()[PythonWorkspaces::GetCurrentWorkspaceName()]
                .get();
        return true;
      },
      "Reset the workspace",
      py::arg("root_folder") = py::none());

  m.def("root_folder", []() {
    CAFFE_ENFORCE(PythonWorkspaces::GetCurrent());
    return PythonWorkspaces::GetCurrent()->RootFolder();
  });
  m.def("current_workspace", []() {
    return PythonWorkspaces::GetCurrentWorkspaceName();
  });
  m.def("workspaces", []() {
    std::vector<std::string> names;
    for (const auto& kv : PythonWorkspaces::Get()) {
      names.push_back(kv.first);
    }
    return names;
  });
  m.def("nearby_opnames", [](const std::string& name) {
    std::vector<std::string> alternatives;
    int editTolerance = 3;
    for (auto it : caffe2::CPUOperatorRegistry()->Keys()) {
      if (editDistance(it, name, editTolerance) < editTolerance + 1) {
        alternatives.push_back(it);
      }
    }
    return alternatives;
  });
  m.def("local_blobs", []() {
    CAFFE_ENFORCE(PythonWorkspaces::GetCurrent());
    return PythonWorkspaces::GetCurrent()->LocalBlobs();
  });
  m.def("blobs", []() {
    CAFFE_ENFORCE(PythonWorkspaces::GetCurrent());
    return PythonWorkspaces::GetCurrent()->Blobs();
  });
  m.def("has_blob", [](const std::string& name) {
    CAFFE_ENFORCE(PythonWorkspaces::GetCurrent());
    return PythonWorkspaces::GetCurrent()->HasBlob(name);
  });
  m.def(
      "create_net",
      [](py::bytes net_def, bool overwrite) {
        CAFFE_ENFORCE(PythonWorkspaces::GetCurrent());
        caffe2::NetDef proto;
        CAFFE_ENFORCE(
            ParseProtoFromLargeString(net_def.cast<std::string>(), &proto),
            "Can't parse net proto: ",
            net_def.cast<std::string>());
        CAFFE_ENFORCE(
            PythonWorkspaces::GetCurrent()->CreateNet(proto, overwrite),
            "Error creating net with proto: ",
            net_def.cast<std::string>());
        return true;
      },
      py::arg("net_def"),
      py::arg("overwrite") = kPyBindFalse);
  m.def("run_net", [](const std::string& name, int num_iter, bool allow_fail) {
    CAFFE_ENFORCE(PythonWorkspaces::GetCurrent());
    CAFFE_ENFORCE(
        PythonWorkspaces::GetCurrent()->GetNet(name), "Can't find net ", name);
    py::gil_scoped_release g;
    for (int i = 0; i < num_iter; i++) {
      bool success = PythonWorkspaces::GetCurrent()->RunNet(name);
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
        CAFFE_ENFORCE(PythonWorkspaces::GetCurrent());
        CAFFE_ENFORCE(
            PythonWorkspaces::GetCurrent()->GetNet(net_name),
            "Can't find net ",
            net_name);
        py::gil_scoped_release g;

        NetBase* net = PythonWorkspaces::GetCurrent()->GetNet(net_name);
        const Observable<NetBase>::Observer* observer = nullptr;

#define REGISTER_PYTHON_EXPOSED_OBSERVER(ob_type)             \
  {                                                           \
    if (observer_type.compare(#ob_type) == 0) {               \
      unique_ptr<ob_type> net_ob = make_unique<ob_type>(net); \
      observer = net->AttachObserver(std::move(net_ob));      \
    }                                                         \
  }

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
        CAFFE_ENFORCE(PythonWorkspaces::GetCurrent());
        CAFFE_ENFORCE(
            PythonWorkspaces::GetCurrent()->GetNet(net_name),
            "Can't find net ",
            net_name);
        py::gil_scoped_release g;

        NetBase* net = PythonWorkspaces::GetCurrent()->GetNet(net_name);
        net->DetachObserver(observer);
      });
  m.def("num_observers_on_net", [](const std::string& net_name) {
    CAFFE_ENFORCE(PythonWorkspaces::GetCurrent());
    CAFFE_ENFORCE(
        PythonWorkspaces::GetCurrent()->GetNet(net_name),
        "Can't find net ",
        net_name);
    py::gil_scoped_release g;

    NetBase* net = PythonWorkspaces::GetCurrent()->GetNet(net_name);
    return net->NumObservers();
  });
  m.def(
      "benchmark_net",
      [](const std::string& name,
         size_t warmup_runs,
         size_t main_runs,
         bool run_individual) {
        CAFFE_ENFORCE(PythonWorkspaces::GetCurrent());
        auto* net = PythonWorkspaces::GetCurrent()->GetNet(name);
        CAFFE_ENFORCE(net, "Didn't find net: ", name);
        py::gil_scoped_release g;
        vector<float> stat =
            net->TEST_Benchmark(warmup_runs, main_runs, run_individual);
        return stat;
      });

  m.def("delete_net", [](const std::string& name) {
    CAFFE_ENFORCE(PythonWorkspaces::GetCurrent());
    PythonWorkspaces::GetCurrent()->DeleteNet(name);
    return true;
  });
  m.def("nets", []() { return PythonWorkspaces::GetCurrent()->Nets(); });
  m.def("run_operator_once", [](const py::bytes& op_def) {
    CAFFE_ENFORCE(PythonWorkspaces::GetCurrent());
    OperatorDef def;
    CAFFE_ENFORCE(ParseProtoFromLargeString(op_def.cast<std::string>(), &def));
    py::gil_scoped_release g;
    CAFFE_ENFORCE(PythonWorkspaces::GetCurrent()->RunOperatorOnce(def));
    return true;
  });
  m.def(
      "get_operator_cost",
      [](const py::bytes& op_def, const std::vector<string>& input_blobs) {
        CAFFE_ENFORCE(PythonWorkspaces::GetCurrent());
        OperatorDef def;
        CAFFE_ENFORCE(
            ParseProtoFromLargeString(op_def.cast<std::string>(), &def),
            "Couldn't parse operator proto.");
        const auto op_type = def.type();
        auto* schema = OpSchemaRegistry::Schema(op_type);
        CAFFE_ENFORCE(schema);
        vector<TensorShape> shapes;
        for (const auto& blob_name : input_blobs) {
          auto* blob = PythonWorkspaces::GetCurrent()->GetBlob(blob_name);
          shapes.emplace_back(GetTensorShapeOfBlob(blob));
        }
        const auto c = schema->InferCost(def, shapes);
        return std::make_tuple(c.flops, c.bytes_written);
      });
  m.def("run_net_once", [](const py::bytes& net_def) {
    CAFFE_ENFORCE(PythonWorkspaces::GetCurrent());
    NetDef def;
    CAFFE_ENFORCE(ParseProtoFromLargeString(net_def.cast<std::string>(), &def));
    py::gil_scoped_release g;
    CAFFE_ENFORCE(PythonWorkspaces::GetCurrent()->RunNetOnce(def));
    return true;
  });
  m.def("run_plan", [](const py::bytes& plan_def) {
    CAFFE_ENFORCE(PythonWorkspaces::GetCurrent());
    PlanDef def;
    CAFFE_ENFORCE(
        ParseProtoFromLargeString(plan_def.cast<std::string>(), &def));
    py::gil_scoped_release g;
    CAFFE_ENFORCE(PythonWorkspaces::GetCurrent()->RunPlan(def));
    return true;
  });
  m.def("run_plan_in_background", [](const py::bytes& plan_def) {
    CAFFE_ENFORCE(PythonWorkspaces::GetCurrent());
    PlanDef def;
    CAFFE_ENFORCE(
        ParseProtoFromLargeString(plan_def.cast<std::string>(), &def));
    py::gil_scoped_release g;

    auto background_plan =
        std::make_shared<BackgroundPlan>(PythonWorkspaces::GetCurrent(), def);
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
        CAFFE_ENFORCE(PythonWorkspaces::GetCurrent());

        // Parse protobuffers to NetDefs
        std::vector<std::unique_ptr<caffe2::NetDef>> nets;
        std::vector<caffe2::NetDef*> nets_ptr;
        for (auto proto : net_protos) {
          std::unique_ptr<NetDef> def(new NetDef());
          CAFFE_ENFORCE(def->ParseFromString(proto));
          nets_ptr.push_back(def.get());
          nets.push_back(std::move(def));
        }

        auto blob_info = InferBlobShapesAndTypesFromWorkspace(
            PythonWorkspaces::GetCurrent(), nets_ptr);

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
        std::vector<caffe2::NetDef*> nets_ptr;
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
         const std::map<std::string, std::vector<TIndex>> blob_dimensions,
         const std::map<std::string, int> int_blob_types) {
        // Parse protobuffers to NetDefs
        std::vector<std::unique_ptr<caffe2::NetDef>> nets;
        std::vector<caffe2::NetDef*> nets_ptr;
        for (auto proto : net_protos) {
          std::unique_ptr<NetDef> def(new NetDef());
          CAFFE_ENFORCE(def->ParseFromString(proto));
          nets_ptr.push_back(def.get());
          nets.push_back(std::move(def));
        }
        std::map<std::string, TensorProto_DataType> blob_types;
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
  m.def("create_blob", [](const std::string& name) {
    CAFFE_ENFORCE(PythonWorkspaces::GetCurrent());
    CAFFE_ENFORCE(PythonWorkspaces::GetCurrent()->CreateBlob(name));
    return true;
  });
  m.def("fetch_blob", [](const std::string& name) -> py::object {
    return python_detail::fetchBlob(PythonWorkspaces::GetCurrent(), name);
  });
  m.def(
      "feed_blob",
      [](const std::string& name, py::object arg, py::object device_option) {
        DeviceOption option;
        if (!device_option.is(py::none())) {
          // If we have a device option passed in, read it.
          CAFFE_ENFORCE(ParseProtoFromLargeString(
              py::bytes(device_option).cast<std::string>(), &option));
        }
        auto* blob = PythonWorkspaces::GetCurrent()->CreateBlob(name);
        if (PyArray_Check(arg.ptr())) { // numpy array
          PyArrayObject* array = reinterpret_cast<PyArrayObject*>(arg.ptr());
          auto feeder = CreateFeeder(option.device_type());
          CAFFE_ENFORCE(
              feeder,
              "Unknown device type encountered in FeedBlob: ",
              option.device_type());
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
    CAFFE_ENFORCE(PythonWorkspaces::GetCurrent());
    auto* blob = PythonWorkspaces::GetCurrent()->GetBlob(name);
    CAFFE_ENFORCE(blob);
    return py::bytes(blob->Serialize(name));
  });
  m.def(
      "deserialize_blob",
      [](const std::string& name, const py::bytes& serialized) {
        CAFFE_ENFORCE(PythonWorkspaces::GetCurrent());
        auto* blob = PythonWorkspaces::GetCurrent()->CreateBlob(name);
        blob->Deserialize(serialized.cast<std::string>());
      });

  // we support 2 possible signatures of python op: (inputs, outputs) or
  // (inputs, outputs, workspace)
  m.def(
      "register_python_op",
      [](py::object func, bool pass_workspace, std::string name) {
        using namespace python_detail;
        CAFFE_ENFORCE(!func.is(py::none()));
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
        CAFFE_ENFORCE(!func.is(py::none()));
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
      in_res.push_back(py::bytes(protob));
    }
    for (auto& out_dev : device_info.second) {
      std::string protob;
      CAFFE_ENFORCE(out_dev.SerializeToString(&protob));
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
    CAFFE_ENFORCE(PythonWorkspaces::GetCurrent());
    auto* blob = PythonWorkspaces::GetCurrent()->GetBlob(blob_name);
    CAFFE_ENFORCE(blob);
    const TensorCPU& tensor = blob->Get<TensorCPU>();
    const void* raw_data = tensor.raw_data();
    CAFFE_ENFORCE(raw_data);
    return GetNUMANode(raw_data);
  });
  m.def("get_blob_size_bytes", [](const std::string& blob_name) {
    CAFFE_ENFORCE(PythonWorkspaces::GetCurrent());
    auto* blob = PythonWorkspaces::GetCurrent()->GetBlob(blob_name);
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
      "onnxifi",
      [](const py::bytes& pred_net_str,
         const std::unordered_map<std::string, std::vector<int>>& shapes,
         bool debug_builder) -> py::bytes {
        caffe2::NetDef pred_net;
        CAFFE_ENFORCE(
            ParseProtoFromLargeString(
                pred_net_str.cast<std::string>(), &pred_net),
            "broken pred_net protobuf");
        std::unordered_map<std::string, TensorShape> tensor_shapes;
        for (const auto& it : shapes) {
          tensor_shapes.emplace(
              it.first, CreateTensorShape(it.second, TensorProto::FLOAT));
        }
        OnnxifiTransformer ts(debug_builder);
        ts.Transform(PythonWorkspaces::GetCurrent(), &pred_net, tensor_shapes);
        std::string pred_net_str2;
        pred_net.SerializeToString(&pred_net_str2);
        return py::bytes(pred_net_str2);
      });
  m.def(
      "run_workspace_transform",
      [](const std::string& transform_name, py::bytes def) {
        CAFFE_ENFORCE(PythonWorkspaces::GetCurrent());
        caffe2::NetDef proto;
        CAFFE_ENFORCE(
            ParseProtoFromLargeString(def.cast<std::string>(), &proto));
        auto nn = caffe2::convertToNNModule(proto);
        auto pass = WorkspaceOptimizationPassRegistry()->Create(
            transform_name, &nn, PythonWorkspaces::GetCurrent());

        CAFFE_ENFORCE(pass, "Pass doesn't exist: ", transform_name);
        pass->run();

        auto new_proto = caffe2::convertToCaffe2Proto(nn, proto);
        std::string out;
        new_proto.SerializeToString(&out);
        return py::bytes(out);
      });

  // Transformations are exposed as functions here and wrapped
  // into a python interface in transformations.py
  // Prefix the transformation with transform_ to avoid clobbering the
  // function namespace.
  m.def("transform_optimizeForIDEEP", [](py::bytes def, bool training_mode) {
    caffe2::NetDef proto;
    CAFFE_ENFORCE(ParseProtoFromLargeString(def.cast<std::string>(), &proto));

    auto nn = caffe2::convertToNNModule(proto);
    opt::OptimizeForIdeep(&nn, PythonWorkspaces::GetCurrent(), training_mode);
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
    CAFFE_ENFORCE(PythonWorkspaces::GetCurrent());
    caffe2::NetDef proto;
    CAFFE_ENFORCE(ParseProtoFromLargeString(def.cast<std::string>(), &proto));

    auto nn = caffe2::convertToNNModule(proto);
    opt::fuseConvBN(&nn, PythonWorkspaces::GetCurrent());
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

  m.def("transform_sinkMaxPool", [](py::bytes def) {
    caffe2::NetDef proto;
    CAFFE_ENFORCE(ParseProtoFromLargeString(def.cast<std::string>(), &proto));

    auto nn = caffe2::convertToNNModule(proto);
    opt::sinkMaxPool(&nn);
    auto new_proto = caffe2::convertToCaffe2Proto(nn, proto);

    std::string out;
    new_proto.SerializeToString(&out);
    return py::bytes(out);
  });

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
    PythonWorkspaces::SwitchWorkspaceInternal("default", true);
    PythonWorkspaces::GetCurrentWorkspaceName() = "default";
    initialized = true;
  };

  initialize();
};

REGISTER_PYBIND_ADDITION(addGlobalMethods);

} // namespace python
} // namespace caffe2
