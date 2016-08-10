#include "caffe2/core/operator.h"

#include <algorithm>
#include <ctime>

#include "caffe2/core/net.h"
#include "caffe2/core/operator_gradient.h"
#include "caffe2/core/workspace.h"
#include "caffe2/proto/caffe2.pb.h"
#include "caffe2/utils/proto_utils.h"

namespace caffe2 {

// TODO(Yangqing): move all the checks to a less fatal check mechanism.
OperatorBase::OperatorBase(const OperatorDef& operator_def, Workspace* ws)
    : operator_def_(operator_def) {
  for (auto& arg : operator_def_.arg()) {
    CHECK_GT(arg.name().size(), 0) << "Argument must have a name.";
    CHECK_EQ(arg_map_.count(arg.name()), 0) << "Duplicated argument name.";
    arg_map_[arg.name()] = &arg;
  }
  for (const string& input_str : operator_def_.input()) {
    auto* blob = ws->GetBlob(input_str);
    CAFFE_ENFORCE(blob != nullptr,
                  "Encountered a non-existing input blob: ", input_str);
    inputs_.push_back(blob);
  }
  for (const string& output_str : operator_def_.output()) {
    outputs_.push_back(CHECK_NOTNULL(ws->CreateBlob(output_str)));
  }
}

// Parameter getters. You can use these to get the arguments that you want.
// We need to deal with the fact that we cannot really template into
// protocol buffers... yuck.
#define INSTANTIATE_GET_SINGLE_ARGUMENT(T, fieldname)                         \
  template <>                                                                 \
  T OperatorBase::GetSingleArgument<T>(                                       \
      const string& name, const T& default_value) {                           \
    if (arg_map_.count(name) == 0) {                                          \
      VLOG(1) << "Using default parameter value " << default_value            \
              << " for parameter " << name;                                   \
      return default_value;                                                   \
    }                                                                         \
    CAFFE_ENFORCE(arg_map_[name]->has_##fieldname(),                          \
        "Argument ", name, " does not have the right field: expected field "  \
        #fieldname);                                                          \
    return arg_map_[name]->fieldname();                                       \
  }                                                                           \
  template <>                                                                 \
  bool OperatorBase::HasSingleArgumentOfType<T>(const string& name) {         \
    if (arg_map_.count(name) == 0) {                                          \
      return false;                                                           \
    }                                                                         \
    return arg_map_[name]->has_##fieldname();                                 \
  }

INSTANTIATE_GET_SINGLE_ARGUMENT(float, f)
INSTANTIATE_GET_SINGLE_ARGUMENT(int, i)
INSTANTIATE_GET_SINGLE_ARGUMENT(bool, i)
INSTANTIATE_GET_SINGLE_ARGUMENT(int64_t, i)
INSTANTIATE_GET_SINGLE_ARGUMENT(size_t, i)
INSTANTIATE_GET_SINGLE_ARGUMENT(string, s)
// Undefine the argument just to be safe.
#undef INSTANTIATE_GET_SINGLE_ARGUMENT

#define INSTANTIATE_GET_REPEATED_ARGUMENT(T, fieldname)                        \
template <>                                                                    \
vector<T> OperatorBase::GetRepeatedArgument<T>(                                \
    const string& name) {                                                      \
  if (arg_map_.count(name) == 0) {                                             \
    return vector<T>();                                                        \
  }                                                                            \
  vector<T> values;                                                            \
  for (const auto& v : arg_map_[name]->fieldname()) values.push_back(v);       \
  return values;                                                               \
}

INSTANTIATE_GET_REPEATED_ARGUMENT(float, floats)
INSTANTIATE_GET_REPEATED_ARGUMENT(int, ints)
INSTANTIATE_GET_REPEATED_ARGUMENT(bool, ints)
INSTANTIATE_GET_REPEATED_ARGUMENT(int64_t, ints)
INSTANTIATE_GET_REPEATED_ARGUMENT(size_t, ints)
INSTANTIATE_GET_REPEATED_ARGUMENT(string, strings)
#undef INSTANTIATE_GET_REPEATED_ARGUMENT

namespace {
unique_ptr<OperatorBase> TryCreateOperator(
    const string& key, const OperatorDef& operator_def, Workspace* ws) {
  switch (operator_def.device_option().device_type()) {
  case CPU:
    VLOG(1) << "Creating CPU operator " << key;
    return CPUOperatorRegistry()->Create(key, operator_def, ws);
  case CUDA:
    VLOG(1) << "Creating CUDA operator " << key;
    return CUDAOperatorRegistry()->Create(key, operator_def, ws);
  default:
    LOG(FATAL) << "Unknown device type: "
                << operator_def.device_option().device_type();
    return nullptr;
  }
}
}  // namespace

unique_ptr<OperatorBase> CreateOperator(
    const OperatorDef& operator_def, Workspace* ws) {
  // first, check with OpSchema if the operator is legal.
  auto* schema = OpSchemaRegistry::Schema(operator_def.type());
  if (schema) {
    if (!schema->Verify(operator_def)) {
      LOG(ERROR) << "Operator def did not pass schema checking: "
                 << ProtoDebugString(operator_def);
      return nullptr;
    }
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
    string key = operator_def.type() +  "_ENGINE_" + operator_def.engine();
    VLOG(1) << "Trying to create operator " << operator_def.type()
            << " with engine " << operator_def.engine();
    auto op = TryCreateOperator(key, operator_def, ws);
    if (op) {
      return op;
    }
    // If the above fails, we will just return the normal case with the default
    // implementation.
    VLOG(1) << "Operator with engine " << operator_def.engine()
            << " is not available. Using default implementation.";
  }

  // Lastly, if the engine does not work here, try using the default engine.
  auto op = TryCreateOperator(operator_def.type(), operator_def, ws);
  if (!op) {
    LOG(ERROR) << "Cannot create op from def: "
               << ProtoDebugString(operator_def);
  }
  return op;
}

CAFFE_DEFINE_REGISTRY(
    CPUOperatorRegistry,
    OperatorBase,
    const OperatorDef&,
    Workspace*);
CAFFE_DEFINE_REGISTRY(
    CUDAOperatorRegistry,
    OperatorBase,
    const OperatorDef&,
    Workspace*);

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
      grad_def.mutable_arg()->CopyFrom(def.arg());
    }
  }
  // VLOG for debugging purposes.
  for (const OperatorDef& grad_def : meta.ops_) {
    VLOG(1) << "Gradient ops: " << ProtoDebugString(grad_def);
  }
  // Check if the gradient computation has returned the right size for the
  // gradient vector.
  CHECK_EQ(meta.g_input_.size(), def.input_size());
  VLOG(1) << "Gradients:";
  for (const GradientWrapper& grad : meta.g_input_) {
    // The gradient should either be (1) not set, or (2) dense, or (3) sparse,
    // but cannot be both dense and sparse.
    if (!grad.IsDense() && !grad.IsSparse()) {
      VLOG(1) << "\t [no gradient]";
    } else if (grad.IsDense()) {
      VLOG(1) << "\t [dense]" << grad.dense_;
    } else {
      CHECK(grad.indices_.size() && grad.values_.size())
          << "For sparse gradient, one should set both indices and values. "
          << "Currently we have: (" << grad.indices_ << ", " << grad.values_
          << ").";
      VLOG(1) << "\t [sparse] " << grad.indices_ << ", " << grad.values_;
    }
  }
  return meta;
}

}  // namespace caffe2
