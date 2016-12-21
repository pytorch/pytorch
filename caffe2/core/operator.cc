#include "caffe2/core/operator.h"

#include <algorithm>

#include "caffe2/core/logging.h"
#include "caffe2/core/net.h"
#include "caffe2/core/operator_gradient.h"
#include "caffe2/core/workspace.h"
#include "caffe2/proto/caffe2.pb.h"
#include "caffe2/utils/proto_utils.h"
#include "caffe2/utils/string_utils.h"

namespace caffe2 {

// TODO(Yangqing): move all the checks to a less fatal check mechanism.
OperatorBase::OperatorBase(const OperatorDef& operator_def, Workspace* ws)
    : operator_def_(operator_def), arg_helper_(operator_def_) {
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
    VLOG(1) << "Operator " << operator_def.type()
            << " does not support the requested feature. Msg: " << err.what()
            << ". Proto is: " << ProtoDebugString(operator_def);
    return nullptr;
  }
}
}  // namespace

unique_ptr<OperatorBase> CreateOperator(
    const OperatorDef& operator_def, Workspace* ws) {
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
      "'. Verify that implementation for the corresponding device exist. It "
      "might also happen if the binary is not linked with the operator "
      "implementation code. If Python frontend is used it might happen if "
      "dyndep.InitOpsLibrary call is missing. Operator def: ",
      ProtoDebugString(operator_def));
  return op;
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

}  // namespace caffe2
