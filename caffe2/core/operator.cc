#include <algorithm>
#include <ctime>

#include "caffe2/core/net.h"
#include "caffe2/core/operator.h"
#include "caffe2/core/workspace.h"
#include "caffe2/proto/caffe2.pb.h"

namespace caffe2 {

// TODO(Yangqing): move all the checks to a less fatal check mechanism.
OperatorBase::OperatorBase(const OperatorDef& operator_def, Workspace* ws)
    : operator_def_(operator_def) {
  for (auto& arg : operator_def.arg()) {
    CAFFE_CHECK_GT(arg.name().size(), 0) << "Argument must have a name.";
    CAFFE_CHECK_EQ(arg_map_.count(arg.name()), 0) << "Duplicated argument name.";
    arg_map_[arg.name()] = &arg;
  }
  for (const string& input_str : operator_def_.input()) {
    auto* blob = ws->GetBlob(input_str);
    CAFFE_CHECK(blob) << "Non-existing blob: " << input_str;
    inputs_.push_back(blob);
  }
  for (const string& output_str : operator_def_.output()) {
    outputs_.push_back(CAFFE_CHECK_NOTNULL(ws->CreateBlob(output_str)));
  }
}

// Parameter getters. You can use these to get the arguments that you want.
// We need to deal with the fact that we cannot really template into
// protocol buffers... yuck.
#define INSTANTIATE_GET_SINGLE_ARGUMENT(T, fieldname)                          \
template <>                                                                    \
T OperatorBase::GetSingleArgument<T>(                                          \
    const string& name, const T& default_value) {                              \
  if (arg_map_.count(name) == 0) {                                             \
    CAFFE_VLOG(1) << "Using default parameter value " << default_value;        \
    return default_value;                                                      \
  }                                                                            \
  CAFFE_CHECK(arg_map_[name]->has_##fieldname())                               \
      << "Argument does not have the right field: expected "                   \
      << #fieldname;                                                           \
  return arg_map_[name]->fieldname();                                          \
}

INSTANTIATE_GET_SINGLE_ARGUMENT(float, f)
INSTANTIATE_GET_SINGLE_ARGUMENT(int, i)
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
INSTANTIATE_GET_REPEATED_ARGUMENT(string, strings)
#undef INSTANTIATE_GET_REPEATED_ARGUMENT

bool OperatorBase::Verify() {
  // (1) Check Blob counts: the input sizes and the output sizes should match.
  if (operator_def_.input_size() < MinInput() ||
      operator_def_.input_size() > MaxInput()) {
    CAFFE_LOG_ERROR << "Input size " << operator_def_.input_size()
               << " not in range [min=" << MinInput() << ", max="
               << MaxInput() << "].";
    CAFFE_LOG_ERROR << "Error at operator " << operator_def_.name() << ":"
               << operator_def_.type();
    return false;
  }
  if (operator_def_.output_size() < MinOutput() ||
      operator_def_.output_size() > MaxOutput()) {
    CAFFE_LOG_ERROR << "Output size " << operator_def_.output_size()
        << " not in range [min=" << MinOutput() << ", max="
        << MaxOutput() << "].";
    CAFFE_LOG_ERROR << "Error at operator " << operator_def_.name() << ":"
        << operator_def_.type();
    return false;
  }
  // (2) Check if input and output in-place computation are correct. Currently,
  // this runs O(num_input * num_output), so it's relatively slow. However,
  // verification only runs once so it's probably fine.
  for (int in_idx = 0; in_idx < operator_def_.input_size(); ++in_idx) {
    for (int out_idx = 0; out_idx < operator_def_.output_size(); ++out_idx) {
      // If an input is the same as an output but in-place is not opt-in for
      // this pair, we will fail the verification.
      if (operator_def_.input(in_idx) == operator_def_.output(out_idx) &&
          !InplaceAllowed(in_idx, out_idx)) {
        CAFFE_LOG_ERROR << "Input index " << in_idx << " and output idx "
            << out_idx << " have the same symbol, but the operator "
            << operator_def_.type()
            << " has not opted-in inplace computation for this pair.";
        return false;
      }
    }
  }
  return true;
}

namespace {
OperatorBase* TryCreateOperator(
    const string& key, const OperatorDef& operator_def, Workspace* ws) {
  switch (operator_def.device_option().device_type()) {
  case CPU:
    CAFFE_VLOG(1) << "Creating CPU operator " << key;
    return CPUOperatorRegistry()->Create(key, operator_def, ws);
  case CUDA:
    CAFFE_VLOG(1) << "Creating CUDA operator " << key;
    return CUDAOperatorRegistry()->Create(key, operator_def, ws);
  }
}
}  // namespace

OperatorBase* CreateOperator(const OperatorDef& operator_def, Workspace* ws) {
  string key = operator_def.type();
  // First, if the user has provided an engine, try create that engine
  if (operator_def.engine().size()) {
    key += "_ENGINE_" + operator_def.engine();
    OperatorBase* op = TryCreateOperator(key, operator_def, ws);
    if (op != nullptr) {
      return op;
    }
  }
  // If the above fails, we will just return the normal case with the default
  // implementation.
  CAFFE_VLOG(1) << "Operator with engine " << operator_def.engine()
                << " is not available. Using default implementation.";
  return TryCreateOperator(operator_def.type(), operator_def, ws);
}

DEFINE_REGISTRY(CPUOperatorRegistry, OperatorBase,
                const OperatorDef&, Workspace*);
DEFINE_REGISTRY(CUDAOperatorRegistry, OperatorBase,
                const OperatorDef&, Workspace*);

DEFINE_REGISTRY(GradientRegistry, vector<OperatorDef>, const OperatorDef&);

}  // namespace caffe2
