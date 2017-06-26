#include "caffe2/core/operator.h"

#include <algorithm>

#include "caffe2/core/logging.h"
#include "caffe2/core/net.h"
#include "caffe2/core/operator_gradient.h"
#include "caffe2/core/tensor.h"
#include "caffe2/core/workspace.h"

#include "caffe2/proto/caffe2.pb.h"
#include "caffe2/utils/proto_utils.h"
#include "caffe2/utils/string_utils.h"

namespace caffe2 {

OperatorBase::OperatorBase(const OperatorDef& operator_def, Workspace* ws)
    : operator_ws_(ws),
      operator_def_(operator_def),
      arg_helper_(operator_def_) {
  for (const string& input_str : operator_def_.input()) {
    auto* blob = ws->GetBlob(input_str);
    CAFFE_ENFORCE(
        blob != nullptr,
        "op ",
        operator_def_.type(),
        ": Encountered a non-existing input blob: ",
        input_str);
    inputs_.push_back(blob);
  }

  GetOperatorLogger()(operator_def_);

  for (const string& output_str : operator_def_.output()) {
    outputs_.push_back(CHECK_NOTNULL(ws->CreateBlob(output_str)));
  }
}

namespace {
unique_ptr<OperatorBase> TryCreateOperator(
    const string& key, const OperatorDef& operator_def, Workspace* ws) {
  auto type = operator_def.device_option().device_type();
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
  // first, check with OpSchema if the operator is legal.
  auto* schema = OpSchemaRegistry::Schema(operator_def.type());
  if (schema) {
    CAFFE_ENFORCE(
        schema->Verify(operator_def),
        "Operator def did not pass schema checking: ",
        ProtoDebugString(operator_def));
  } else {
    // We would like to recommend every op to register its schema, so if there
    // is not one, we print a LOG_ERROR. But we will still allow the operator
    // to be constructed.
    LOG(ERROR) << "Cannot find operator schema for "
               << operator_def.type()
               << ". Will skip schema checking.";
  }

  // Second, if the user has provided an engine, try create that engine
  if (operator_def.engine().size()) {
    vector<string> engine_choices = split(',', operator_def.engine());
    for (const string& engine : engine_choices) {
      string key = operator_def.type() + "_ENGINE_" + engine;
      VLOG(1) << "Trying to create operator " << operator_def.type()
              << " with engine " << engine;
      auto op = TryCreateOperator(key, operator_def, ws);
      if (op) {
        return op;
      } else {
        // If the above fails, we will just return the normal case with the
        // default implementation.
        VLOG(1) << "Operator with engine " << engine
                << " is not available. Using default implementation.";
      }
    }
  }

  // Lastly, if the engine does not work here, try using the default engine.
  auto op = TryCreateOperator(operator_def.type(), operator_def, ws);
  CAFFE_ENFORCE(
      op,
      "Cannot create operator of type '",
      operator_def.type(),
      "' on the device '",
      DeviceTypeName(operator_def.device_option().device_type()),
      "'. Verify that implementation for the corresponding device exist. It "
      "might also happen if the binary is not linked with the operator "
      "implementation code. If Python frontend is used it might happen if "
      "dyndep.InitOpsLibrary call is missing. Operator def: ",
      ProtoDebugString(operator_def));
  return op;
}

} // namespace

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

std::map<int32_t, OperatorRegistry*>* gDeviceTypeRegistry() {
  static std::map<int32_t, OperatorRegistry*> g_device_type_registry;
  return &g_device_type_registry;
}

CAFFE_DEFINE_REGISTRY(
    CPUOperatorRegistry,
    OperatorBase,
    const OperatorDef&,
    Workspace*);
CAFFE_REGISTER_DEVICE_TYPE(DeviceType::CPU, CPUOperatorRegistry);

CAFFE_DEFINE_REGISTRY(
    CUDAOperatorRegistry,
    OperatorBase,
    const OperatorDef&,
    Workspace*);
CAFFE_REGISTER_DEVICE_TYPE(DeviceType::CUDA, CUDAOperatorRegistry);

CAFFE_DEFINE_REGISTRY(
    GradientRegistry,
    GradientMakerBase,
    const OperatorDef&, const vector<GradientWrapper>&);

GradientOpsMeta GetGradientForOp(
    const OperatorDef& def, const vector<GradientWrapper>& g_output) {
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

static TensorShapes InferBlobShapesAndTypes(
    CaffeMap<string, TensorShape>& blob_desc,
    const vector<std::unique_ptr<NetDef>>& nets) {
  for (auto& defptr : nets) {
    // Hack to work with auto split gradients
    CaffeMap<string, string> unmatched_sum_blobs;

    for (const OperatorDef& op : defptr.get()->op()) {
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

          for (int i = 0; i < out.size(); i++) {
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
      } catch (::caffe2::EnforceNotMet& enf) {
        LOG(ERROR) << "Shape inference error: " << enf.msg();
        LOG(ERROR) << "Operator: " << ProtoDebugString(op) << std::endl;
        LOG(ERROR) << "Returning empty results.";
        TensorShapes tps;
        return tps;
      }

      if (out.size() != op.output_size()) {
        CAFFE_THROW(
            "Invalid shape inference for operator ",
            op.type(),
            " Expected ",
            op.output_size(),
            " outputs, but got ",
            out.size());
      } else {
        for (int i = 0; i < out.size(); i++) {
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

TensorShapes InferBlobShapesAndTypesFromWorkspace(
    Workspace* ws,
    const vector<std::unique_ptr<NetDef>>& nets) {
  CaffeMap<string, TensorShape> blob_desc;
  // Populate shapes from workplace
  const std::vector<string>& ws_blobs = ws->Blobs();
  for (const auto& s : ws_blobs) {
    Blob* b = ws->GetBlob(s);
    TypeCall type_fun = GetTypeCallFunction(b->meta().id());
    ShapeCall shape_fun = GetShapeCallFunction(b->meta().id());
    TensorShape tp;

    if (type_fun) {
        tp.set_data_type(TypeMetaToDataType(type_fun(b->GetRaw())));
    }
    if (shape_fun) {
      bool _shares_data;
      size_t _capacity;
      auto shape = shape_fun(b->GetRaw(), _shares_data, _capacity);
      for (auto d : shape) {
        tp.add_dims(d);
      }
    } else {
      tp.set_unknown_shape(true);
    }
    blob_desc[s] = tp;
  }
  return InferBlobShapesAndTypes(blob_desc, nets);
}

TensorShapes InferBlobShapesAndTypesFromMap(
    const CaffeMap<std::string, std::vector<TIndex>>& blob_dimensions,
    const vector<std::unique_ptr<NetDef>>& nets) {
  CaffeMap<string, TensorShape> blob_desc;
  // Populate shapes from known blobs
  for (const auto& blob : blob_dimensions) {
    TensorShape tp;
    for (auto d : blob.second) {
      CAFFE_ENFORCE_GT(d, 0);
      tp.add_dims(d);
    }
    blob_desc[blob.first] = tp;
  }
  return InferBlobShapesAndTypes(blob_desc, nets);
}

}  // namespace caffe2
