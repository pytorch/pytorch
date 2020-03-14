#include "caffe2/core/operator.h"

#include <algorithm>

#include "caffe2/core/init.h"
#include "caffe2/core/logging.h"
#include "caffe2/core/net.h"
#include "caffe2/core/operator_gradient.h"
#include "caffe2/core/tensor.h"
#include "caffe2/core/tensor_int8.h"
#include "caffe2/core/types.h"
#include "caffe2/core/workspace.h"

#include "caffe2/proto/caffe2_pb.h"
#include "caffe2/utils/proto_utils.h"
#include "caffe2/utils/string_utils.h"
#if !defined(CAFFE2_IS_XPLAT_BUILD) && !defined(C10_MOBILE)
#include <ATen/core/List.h>
#endif

#include "caffe2/core/export_c10_op_to_caffe2.h"

C10_DEFINE_int(
    caffe2_operator_max_engine_name_length,
    10,
    "Maximum engine name length to be stored");
C10_DEFINE_bool(
    caffe2_disable_implicit_engine_preference,
    false,
    "If set, disable implicit engine preferences. This is useful for unit "
    "testing and debugging cases.");
C10_DEFINE_bool(
    caffe2_operator_throw_if_fp_exceptions,
    false,
    "If set, throws if floating point exceptions (FE_DIVBYZERO, FE_INVALID) "
    "are detected when running any operator. FE_OVERFLOW is handled separately "
    "by caffe2_operator_throw_if_fp_overflow_exceptions option.");
C10_DEFINE_bool(
    caffe2_operator_throw_if_fp_overflow_exceptions,
    false,
    "If set, throws if floating point exception FE_OVERFLOW is detected when "
    "running any operator.");
#ifdef __GNU_LIBRARY__
C10_DEFINE_bool(
    caffe2_operator_throw_on_first_occurrence_if_fp_exceptions,
    false,
    "If set with caffe2_operator_throw_if_fp_exceptions or "
    "caffe2_operator_throw_if_fp_overflow_exceptions, throw on the first "
    "occurrence of corresponding floating point exceptions that is detected when "
    "running any operator.");
#endif

namespace caffe2 {

OperatorBase::OperatorBase(const OperatorDef& operator_def, Workspace* ws)
    : operator_ws_(ws),
      operator_def_(std::make_shared<OperatorDef>(operator_def)),
      device_option_(
          operator_def.has_device_option() ? operator_def.device_option()
                                           : DeviceOption()),
#if defined(EXPOSE_C2_OPS) || \
    !defined(CAFFE2_IS_XPLAT_BUILD) && !defined(C10_MOBILE)
      newstyle_outputs_(),
#endif
      input_size_(operator_def.input_size()),
      event_(std::make_unique<Event>(device_option_)) {
  static GlobalInitIsCalledGuard guard;
  inputs_.reserve(operator_def.input_size());
  for (const string& input_str : operator_def.input()) {
    auto* blob = ws->GetBlob(input_str);
    CAFFE_ENFORCE(
        blob != nullptr,
        "op ",
        operator_def.type(),
        ": Encountered a non-existing input blob: ",
        input_str);
    inputs_.push_back(blob);
  }

  GetOperatorLogger()(operator_def);

  outputs_.reserve(operator_def.output_size());
  for (const string& output_str : operator_def.output()) {
    outputs_.push_back(CHECK_NOTNULL(ws->CreateBlob(output_str)));
  }

  type_ = operator_def.type();
}

#if defined(EXPOSE_C2_OPS) || \
    !defined(CAFFE2_IS_XPLAT_BUILD) && !defined(C10_MOBILE)
namespace {
int
C10_UNUSED  // Suppress unused function warning on mobile.
compute_input_size_(const std::vector<c10::IValue>& inputs) {
  if (inputs.empty()) {
    return 0;
  }
  if (inputs[0].isTensorList()) {
    // if the first input is a tensor list, we get input tensors by indexing
    // into that list. currently, this means that only tensors from that list
    // are accessible as inputs. any hypothetical input tensors that come after
    // the list are not accessible.
    return inputs[0].toTensorVector().size();
  }
  // it's not a tensor list. Count the number of tensor inputs and return them.
  size_t num_tensor_inputs = 0;
  bool found_nontensor = false;
  for (const auto& input : inputs) {
    if (input.isTensor()) {
      AT_ASSERTM(
          !found_nontensor,
          "All tensor arguments must come before non-tensor arguments");
      ++num_tensor_inputs;
    } else {
      found_nontensor = true;
    }
  }
  return num_tensor_inputs;
}
} // namespace

OperatorBase::OperatorBase(
    const c10::FunctionSchema& fn_schema,
    std::vector<c10::IValue> inputs,
    c10::List<at::Tensor> outputs)
    : fn_schema_(make_unique<c10::FunctionSchema>(std::move(fn_schema))),
      newstyle_inputs_(std::move(inputs)),
      newstyle_outputs_(std::move(outputs)),
      input_size_(compute_input_size_(newstyle_inputs_)) {
  input_tensors_.resize(input_size_);
  output_tensors_.resize(newstyle_outputs_.size());
}
#endif

vector<TensorShape> OperatorBase::InputTensorShapes() const {
  CAFFE_ENFORCE(
      isLegacyOperator(),
      "InputTensorShapes() not supported for operators exported to c10.");
  vector<TensorShape> tps;
  for (const auto& blob : inputs_) {
    tps.push_back(GetTensorShapeOfBlob(blob));
  }
  return tps;
}

namespace {

PerOpEnginePrefType& g_per_op_engine_pref() {
  static auto* g_per_op_engine_pref_ = new PerOpEnginePrefType();
  return *g_per_op_engine_pref_;
}

GlobalEnginePrefType& g_global_engine_pref() {
  static auto* g_global_engine_pref_ =
      new GlobalEnginePrefType{{CUDA, {"CUDNN"}}, {HIP, {"MIOPEN"}}};
  return *g_global_engine_pref_;
}

unique_ptr<OperatorBase> TryCreateOperator(
    const string& key,
    const OperatorDef& operator_def,
    Workspace* ws) {
  const auto& type_proto = operator_def.device_option().device_type();
  const auto& type = ProtoToType(static_cast<DeviceTypeProto>(type_proto));
  CAFFE_ENFORCE(
      gDeviceTypeRegistry()->count(type),
      "Device type ",
      type,
      " not registered.");
  OperatorRegistry* registry = gDeviceTypeRegistry()->at(type);
  VLOG(1) << "Creating operator with device type " << type;
  try {
    return registry->Create(key, operator_def, ws);
  } catch (const UnsupportedOperatorFeature& err) {
    LOG(WARNING) << "Operator " << operator_def.type()
                 << " does not support the requested feature. Msg: "
                 << err.what()
                 << ". Proto is: " << ProtoDebugString(operator_def);
    return nullptr;
  }
}

unique_ptr<OperatorBase> _CreateOperator(
    const OperatorDef& operator_def,
    Workspace* ws) {
  static StaticLinkingProtector g_protector;
  const auto& op_type = operator_def.type();
  const auto& device_type_proto = operator_def.device_option().device_type();
  const auto& device_type =
      ProtoToType(static_cast<DeviceTypeProto>(device_type_proto));

#ifndef CAFFE2_NO_OPERATOR_SCHEMA
  // first, check with OpSchema if the operator is legal.
  auto* schema = OpSchemaRegistry::Schema(op_type);
  if (schema) {
    CAFFE_ENFORCE(
        schema->Verify(operator_def),
        "Operator def did not pass schema checking: ",
        ProtoDebugString(operator_def));
  } else {
    // We would like to recommend every op to register its schema, so if there
    // is not one, we print a LOG_ERROR. But we will still allow the operator
    // to be constructed.
    LOG(ERROR) << "Cannot find operator schema for " << op_type
               << ". Will skip schema checking.";
  }
#endif

  // second try engines specified in the operator_def and preferred engines
  std::vector<std::string> engines{};
  if (operator_def.engine().size()) {
    const auto op_def_engines = split(',', operator_def.engine());
    engines.insert(engines.end(), op_def_engines.begin(), op_def_engines.end());
  }
  if (!FLAGS_caffe2_disable_implicit_engine_preference &&
      g_per_op_engine_pref().count(device_type) &&
      g_per_op_engine_pref()[device_type].count(op_type)) {
    const auto& preferred_engines =
        g_per_op_engine_pref()[device_type][op_type];
    VLOG(2) << "Inserting per-op engine preference: " << preferred_engines;
    engines.insert(
        engines.end(), preferred_engines.begin(), preferred_engines.end());
  }
  if (!FLAGS_caffe2_disable_implicit_engine_preference &&
      g_global_engine_pref().count(device_type)) {
    const auto& preferred_engines = g_global_engine_pref()[device_type];
    VLOG(2) << "Inserting global engine preference: " << preferred_engines;
    engines.insert(
        engines.end(), preferred_engines.begin(), preferred_engines.end());
  }
  for (const auto& engine : engines) {
    const std::string key = OpRegistryKey(op_type, engine);
    VLOG(1) << "Trying to create operator " << op_type << " with engine "
            << engine;
    auto op = TryCreateOperator(key, operator_def, ws);
    if (op) {
      if (engine.size() <=
          (unsigned)FLAGS_caffe2_operator_max_engine_name_length) {
        op->annotate_engine(engine);
      } else {
        op->annotate_engine(
            engine.substr(0, FLAGS_caffe2_operator_max_engine_name_length));
      }
      return op;
    } else {
      // If the above fails, we will just return the normal case with the
      // default implementation.
      VLOG(1) << "Engine " << engine
              << " is not available for operator " << op_type << ".";
    }
  }
  if (operator_def.engine().size() && !VLOG_IS_ON(1)) {
    static int log_occurrences = 0;
    if (log_occurrences <= 64) {
      ++log_occurrences;
      LOG(INFO) << "Engine " << operator_def.engine()
                << " is not available for operator " << op_type << ".";
    }
  }
  VLOG(1) << "Using default implementation.";

  // Lastly, if the engine does not work here, try using the default engine.
  auto op = TryCreateOperator(op_type, operator_def, ws);
  CAFFE_ENFORCE(
      op,
      "Cannot create operator of type '",
      op_type,
      "' on the device '",
      DeviceTypeName(device_type),
      "'. Verify that implementation for the corresponding device exist. It "
      "might also happen if the binary is not linked with the operator "
      "implementation code. If Python frontend is used it might happen if "
      "dyndep.InitOpsLibrary call is missing. Operator def: ",
      ProtoDebugString(operator_def));
  return op;
}

} // namespace

const std::string OpRegistryKey(
    const std::string& op_type,
    const std::string& engine) {
  if (engine == "" || engine == "DEFAULT") {
    return op_type;
  } else {
    return op_type + "_ENGINE_" + engine;
  }
}

void SetPerOpEnginePref(const PerOpEnginePrefType& per_op_engine_pref) {
  for (const auto& device_pref_pair : per_op_engine_pref) {
    const auto& device_type = device_pref_pair.first;
    CAFFE_ENFORCE(
        gDeviceTypeRegistry()->count(device_type),
        "Device type ",
        device_type,
        " not registered.");
    auto* registry = gDeviceTypeRegistry()->at(device_type);

    for (const auto& op_pref_pair : device_pref_pair.second) {
      const auto& op_type = op_pref_pair.first;
      CAFFE_ENFORCE(
          registry->Has(op_type),
          "Operator type ",
          op_type,
          " not registered in ",
          device_type,
          " registry.");
    }
  }
  g_per_op_engine_pref() = per_op_engine_pref;
}

void SetGlobalEnginePref(const GlobalEnginePrefType& global_engine_pref) {
  for (const auto& device_pref_pair : global_engine_pref) {
    const auto& device_type = device_pref_pair.first;
    CAFFE_ENFORCE(
        gDeviceTypeRegistry()->count(device_type),
        "Device type ",
        device_type,
        " not registered.");
  }
  g_global_engine_pref() = global_engine_pref;
}

void SetEnginePref(
    const PerOpEnginePrefType& per_op_engine_pref,
    const GlobalEnginePrefType& global_engine_pref) {
  SetPerOpEnginePref(per_op_engine_pref);
  SetGlobalEnginePref(global_engine_pref);
}

void SetOpEnginePref(
    const std::string& op_type,
    const CaffeMap<DeviceType, EnginePrefType>& op_pref) {
  for (const auto& device_pref_pair : op_pref) {
    const auto& device_type_proto = device_pref_pair.first;
    const auto& device_type =
        ProtoToType(static_cast<DeviceTypeProto>(device_type_proto));
    CAFFE_ENFORCE(
        gDeviceTypeRegistry()->count(device_type),
        "Device type ",
        device_type,
        " not registered.");
    CAFFE_ENFORCE(
        gDeviceTypeRegistry()->at(device_type)->Has(op_type),
        "Operator type ",
        op_type,
        " not registered in ",
        device_type,
        " registry.");
    g_per_op_engine_pref()[device_type][op_type] = device_pref_pair.second;
  }
}

unique_ptr<OperatorBase> CreateOperator(
    const OperatorDef& operator_def,
    Workspace* ws,
    int net_position) {
  try {
    auto op = _CreateOperator(operator_def, ws);
    op->set_net_position(net_position);
    return op;
  } catch (...) {
    if (net_position != 0) {
      VLOG(1) << "Operator constructor with net position " << net_position
              << " failed";
      ws->last_failed_op_net_position = net_position;
    } else {
      VLOG(1) << "Failed operator constructor doesn't have an id set";
    }
    throw;
  }
}

std::map<DeviceType, OperatorRegistry*>* gDeviceTypeRegistry() {
  static std::map<DeviceType, OperatorRegistry*> g_device_type_registry;
  return &g_device_type_registry;
}

C10_DEFINE_REGISTRY(
    CPUOperatorRegistry,
    OperatorBase,
    const OperatorDef&,
    Workspace*);
CAFFE_REGISTER_DEVICE_TYPE(CPU, CPUOperatorRegistry);

C10_DEFINE_REGISTRY(
    CUDAOperatorRegistry,
    OperatorBase,
    const OperatorDef&,
    Workspace*);
CAFFE_REGISTER_DEVICE_TYPE(CUDA, CUDAOperatorRegistry);

C10_DEFINE_REGISTRY(
    HIPOperatorRegistry,
    OperatorBase,
    const OperatorDef&,
    Workspace*);
CAFFE_REGISTER_DEVICE_TYPE(HIP, HIPOperatorRegistry);

C10_DEFINE_REGISTRY(
    GradientRegistry,
    GradientMakerBase,
    const OperatorDef&,
    const vector<GradientWrapper>&);

GradientOpsMeta GetGradientForOp(
    const OperatorDef& def, const vector<GradientWrapper>& g_output) {
  C10_LOG_API_USAGE_ONCE("caffe2.gradient_maker");
  std::unique_ptr<GradientMakerBase> maker(
      GradientRegistry()->Create(def.type(), def, g_output));
  CAFFE_ENFORCE(maker,
      "Gradient maker for operator ", def.type(), " not implemented.");
  GradientOpsMeta meta = maker->Get();
  // Copy device option, engine, and arguments if needed.
  if (maker->CopyDeviceOption() && def.has_device_option()) {
    for (OperatorDef& grad_def : meta.ops_) {
      grad_def.mutable_device_option()->CopyFrom(def.device_option());
    }
  }
  // Copy engine if needed.
  if (maker->CopyEngine() && def.has_engine()) {
    for (OperatorDef& grad_def : meta.ops_) {
      grad_def.set_engine(def.engine());
    }
  }
  // Copy arguments if needed.
  if (maker->CopyArguments() && def.arg_size()) {
    for (OperatorDef& grad_def : meta.ops_) {
      for (auto& arg : def.arg()) {
        grad_def.add_arg()->CopyFrom(arg);
      }
    }
  }
  // VLOG for debugging purposes.
  for (const OperatorDef& grad_def : meta.ops_) {
    VLOG(1) << "Gradient ops: " << ProtoDebugString(grad_def);
  }
  // Check if the gradient computation has returned the right size for the
  // gradient vector.
  CAFFE_ENFORCE_EQ(meta.g_input_.size(), def.input_size());
  VLOG(1) << "Gradients:";
  for (const GradientWrapper& grad : meta.g_input_) {
    // The gradient should either be (1) not set, or (2) dense, or (3) sparse,
    // but cannot be both dense and sparse.
    if (!grad.IsDense() && !grad.IsSparse()) {
      VLOG(1) << "\t [no gradient]";
    } else if (grad.IsDense()) {
      VLOG(1) << "\t [dense]" << grad.dense_;
    } else {
      CAFFE_ENFORCE(
          grad.indices_.size() && grad.values_.size(),
          "For sparse gradient, one should set both indices and values. "
          "Currently we have: (" +
              grad.indices_ + ", " + grad.values_ + ").");
      VLOG(1) << "\t [sparse] " << grad.indices_ << ", " << grad.values_;
    }
  }
  return meta;
}

TensorShapes InferBlobShapesAndTypes(
    CaffeMap<string, TensorShape>& blob_desc,
    const vector<NetDef*>& nets) {
  for (auto& defptr : nets) {
    // Hack to work with auto split gradients
    CaffeMap<string, string> unmatched_sum_blobs;
    CaffeMap<string, TensorShape> reshape_cache;

    for (const OperatorDef& op : defptr->op()) {
      // Hack to ignore queues
      if (op.type().find("Dequeue") != std::string::npos ||
          op.type().find("Enqueue") != std::string::npos) {
        continue;
      }

      vector<TensorShape> input_desc;
      bool found_all = true;
      for (const string& in : op.input()) {
        auto inp_desc = blob_desc.find(in);
        if (inp_desc == blob_desc.end()) {
          LOG(WARNING) << "Shape and type inference failed for input: " << in
                       << " for op " << op.type() << ", skipping.";
          found_all = false;
          break;
        }
        input_desc.push_back(inp_desc->second);
      }
      if (!found_all) {
        continue;
      }
      auto op_schema = OpSchemaRegistry::Schema(op.type());
      if (op_schema == nullptr) {
        LOG(WARNING) << "Shape inference failed, no schema for: " << op.type();
        continue;
      }

      // Special handling for Sum as it used with the autosplits, which have
      // different naming convention. Assuming that all sum inputs must be of
      // same size, we can infer their shapes.
      if (op.type() == "Sum") {
        TensorShape sum_shape;
        for (auto inp : op.input()) {
          auto it = blob_desc.find(inp);
          if (it != blob_desc.end() && !it->second.unknown_shape()) {
            if (it->second.dims_size() > 0) {
              sum_shape = blob_desc[inp];
              break;
            }
          }
        }
        for (auto inp : op.input()) {
          auto it = blob_desc.find(inp);
          if (it == blob_desc.end() || it->second.unknown_shape()) {
            blob_desc[inp] = sum_shape;
            if (sum_shape.dims_size() == 0) {
              // Match later with the output
              unmatched_sum_blobs[inp] = op.output(0);
            }
          }
        }
      }

      if (op.type() == "Reshape" && op.is_gradient_op()) {
        CAFFE_ENFORCE(reshape_cache.find(op.input(1)) != reshape_cache.end());
        TensorShape cached = reshape_cache[op.input(1)];
        blob_desc[op.output(0)] = cached;
        continue;
      }

      std::vector<TensorShape> out;
      try {
        out = op_schema->InferTensor(op, input_desc);
        if (op.is_gradient_op() && out.size()) {
          // Special handling for gradient ops. We can assume gradients
          // are of same size as the corresponding variables. This is bit
          // ugly to base on string matching, but we don't have the connection
          // between variable and its gradient specified

          CaffeMap<string, string> grads_to_params =
              GradientMakerBase::MatchGradsToParams(op);

          for (size_t i = 0; i < out.size(); i++) {
            if (out[i].unknown_shape()) {
              std::string gradout = op.output(i);

              if (grads_to_params.find(gradout) != grads_to_params.end()) {
                std::string var = grads_to_params[gradout];
                if (blob_desc.find(var) != blob_desc.end()) {
                  out[i] = blob_desc[var];
                }
              }
            }
          }
        }

        if (op.type() == "Reshape") {
          // Reshape stores the original input shape to its second output
          // blob. We need this for gradient reshape.
          reshape_cache[op.output(1)] = input_desc[0];
        }

      } catch (::caffe2::EnforceNotMet& enf) {
        LOG(ERROR) << "Shape inference error: " << enf.msg();
        LOG(ERROR) << "Operator: " << ProtoDebugString(op) << std::endl;
        LOG(ERROR) << "Returning empty results.";

        TensorShapes tps;
        return tps;
      }

      if (out.size() != (unsigned)op.output_size()) {
        if (op.type() == "Slice") {
          CAFFE_ENFORCE(
              out.size() == 0,
              "For Slice operator, either shape of all output blobs are "
              "inferred or shape of none can be inferred.");
        } else {
          CAFFE_THROW(
              "Invalid shape inference for operator ",
              op.type(),
              " Expected ",
              op.output_size(),
              " outputs, but got ",
              out.size());
        }
      } else {
        for (size_t i = 0; i < out.size(); i++) {
          blob_desc[op.output(i)] = out[i];
        }
      }
    } // net.ops

    for (auto& unmatched : unmatched_sum_blobs) {
      if (blob_desc.find(unmatched.second) != blob_desc.end()) {
        blob_desc[unmatched.first] = blob_desc[unmatched.second];
      }
    }

  } // nets
  TensorShapes tps;
  for (auto kv : blob_desc) {
    TensorShape& tp = kv.second;
    TensorShape* tpnew = tps.add_shapes();
    tpnew->CopyFrom(tp);
    tpnew->set_name(kv.first);
  }
  return tps;
}

void LoadInt8TensorInfoOfBlob(
    std::vector<float>* scale,
    std::vector<float>* offset,
    uint32_t* axis,
    const Blob* b) {
  const int8::Int8TensorCPU* int8_tensor =
      static_cast<const int8::Int8TensorCPU*>(b->GetRaw());
  scale->clear();
  offset->clear();
  scale->push_back(int8_tensor->scale);
  offset->push_back(int8_tensor->zero_point);
  *axis = 1;
}

TensorShape GetTensorShapeOfBlob(const Blob* b) {
  TensorShape tp;
#ifndef C10_MOBILE
  auto function_ptr =
      ExternalTensorFunctionsBaseRegistry()->Create(b->meta().id());
  if (function_ptr != nullptr) {
    // This is dnnlowp tensor and we cant deal with it using regular path
    auto dtype = function_ptr->GetExternalTensorType(b->GetRaw());
    tp.set_data_type(TypeMetaToDataType(dtype));

    size_t _capacity;
    DeviceOption _device;
    auto dshape =
        function_ptr->GetExternalTensorInfo(b->GetRaw(), &_capacity, &_device);
    for (auto d : dshape) {
      tp.add_dims(d);
    }
    return tp;
  }
#endif

  TypeCall type_fun = GetTypeCallFunction(b->meta().id());
  TensorInfoCall tensor_info_fun = GetTensorInfoFunction(b->meta().id());
  if (type_fun) {
    tp.set_data_type(TypeMetaToDataType(type_fun(b->GetRaw())));
  }
  if (tensor_info_fun) {
    size_t _capacity;
    DeviceOption _device;
    auto shape = tensor_info_fun(b->GetRaw(), &_capacity, &_device);
    for (auto d : shape) {
      tp.add_dims(d);
    }
  } else {
    tp.set_unknown_shape(true);
  }
  return tp;
}

TensorShapes InferBlobShapesAndTypesFromWorkspace(
    Workspace* ws,
    const vector<NetDef*>& nets) {
  CaffeMap<string, TensorShape> blob_desc;
  // Populate shapes from workplace
  const std::vector<string>& ws_blobs = ws->Blobs();
  for (const auto& s : ws_blobs) {
    Blob* b = ws->GetBlob(s);
    TensorShape tp = GetTensorShapeOfBlob(b);
    blob_desc[s] = tp;
  }
  return InferBlobShapesAndTypes(blob_desc, nets);
}

TensorShapes InferBlobShapesAndTypesFromMap(
    const CaffeMap<std::string, std::vector<int64_t>>& blob_dimensions,
    const vector<NetDef*>& nets) {
  CaffeMap<string, TensorShape> blob_desc;
  // Populate shapes from known blobs
  for (const auto& blob : blob_dimensions) {
    TensorShape tp;
    for (auto d : blob.second) {
      CAFFE_ENFORCE_GE(d, 0, blob.first);
      tp.add_dims(d);
    }
    blob_desc[blob.first] = tp;
  }
  return InferBlobShapesAndTypes(blob_desc, nets);
}

TensorShapes InferBlobShapesAndTypesFromMap(
    const CaffeMap<std::string, std::vector<int64_t>>& blob_dimensions,
    const CaffeMap<std::string, TensorProto_DataType>& blob_types,
    const vector<NetDef*>& nets) {
  CaffeMap<string, TensorShape> blob_desc;
  // Populate shapes from known blobs
  for (const auto& blob : blob_dimensions) {
    TensorShape tp;
    for (auto d : blob.second) {
      CAFFE_ENFORCE_GE(d, 0, blob.first);
      tp.add_dims(d);
    }
    auto blob_type = blob_types.find(blob.first);
    if (blob_type == blob_types.end()) {
      LOG(WARNING) << "Missing type of " << blob.first
                   << "; assuming to be UNDEFINED";
      tp.set_data_type(TensorProto_DataType_UNDEFINED);
    } else {
      tp.set_data_type(blob_type->second);
    }
    blob_desc[blob.first] = tp;
  }
  return InferBlobShapesAndTypes(blob_desc, nets);
}

std::map<string, std::pair<DeviceOption, DeviceOption>> ValidateTensorDevices(
    OperatorBase& op,
    const OperatorDef& op_def) {
  std::map<string, std::pair<DeviceOption, DeviceOption>> mismatches;
  DeviceOption op_device = op_def.device_option();

#ifndef CAFFE2_NO_OPERATOR_SCHEMA
  // Check from op schema if this op is used for crossing devices
  auto op_schema = OpSchemaRegistry::Schema(op_def.type());
  if (op_schema != nullptr) {
    if (op_schema->inputs_can_cross_devices()) {
      return mismatches;
    }
  }
#endif // CAFFE2_NO_OPERATOR_SCHEMA

  auto Check = [&](const Blob& blob, std::string blob_name) {
    TensorInfoCall tensor_info_fun = GetTensorInfoFunction(blob.meta().id());
    if (tensor_info_fun) {
      size_t _capacity;
      DeviceOption blob_device;
      tensor_info_fun(
          const_cast<Blob&>(blob).GetRaw(),
          &_capacity,
          &blob_device);

      if ((blob_device.device_type() == PROTO_CUDA ||
           blob_device.device_type() == PROTO_HIP) &&
          blob_device.device_id() != op_device.device_id()) {
        mismatches[blob_name] = std::make_pair(op_device, blob_device);
      }
    }
  };

  // Check that inputs have same device type as the op
  for (int i = 0; i < op.InputSize(); i++) {
    Check(op.InputBlob(i), op_def.input(i));
  }
  for (int i = 0; i < op.OutputSize(); i++) {
    Check(*op.OutputBlob(i), op_def.output(i));
  }
  return mismatches;
}

std::set<std::string> GetRegisteredOperators() {
  std::set<std::string> all_keys;

  // CPU operators
  for (const auto& name : CPUOperatorRegistry()->Keys()) {
    all_keys.emplace(name);
  }
  // CUDA operators
  for (const auto& name : CUDAOperatorRegistry()->Keys()) {
    all_keys.emplace(name);
  }

  // HIP operators
  for (const auto& name : HIPOperatorRegistry()->Keys()) {
    all_keys.emplace(name);
  }

  return all_keys;
}

static std::function<void(const OperatorDef&)> OperatorLogger =
    [](const OperatorDef&) { return; };

void SetOperatorLogger(std::function<void(const OperatorDef&)> tracer) {
  OperatorLogger = tracer;
}

std::function<void(const OperatorDef&)> GetOperatorLogger() {
  return OperatorLogger;
}

c10::optional<int> OperatorBase::argumentIndexWithName(
    const std::string& name) const {
#if defined(EXPOSE_C2_OPS) || \
    !defined(CAFFE2_IS_XPLAT_BUILD) && !defined(C10_MOBILE)
  return getFunctionSchema().argumentIndexWithName(name);
#else
  CAFFE_THROW("Non-legacy operators are not legal in xplat/caffe2");
#endif
}

OperatorBase::~OperatorBase() noexcept = default;

#ifndef C10_MOBILE
C10_DEFINE_TYPED_REGISTRY(
    ExternalTensorFunctionsBaseRegistry,
    TypeIdentifier,
    ExternalTensorFunctionsBase,
    std::unique_ptr);
#endif

}  // namespace caffe2
