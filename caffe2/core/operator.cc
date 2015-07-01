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
    CHECK_GT(arg.name().size(), 0) << "Argument must have a name.";
    CHECK_EQ(arg_map_.count(arg.name()), 0) << "Duplicated argument name.";
    arg_map_[arg.name()] = &arg;
  }
  for (const string& input_str : operator_def_.input()) {
    inputs_.push_back(CHECK_NOTNULL(ws->GetBlob(input_str)));
  }
  for (const string& output_str : operator_def_.output()) {
    outputs_.push_back(CHECK_NOTNULL(ws->CreateBlob(output_str)));
  }
}

// Parameter getters. You can use these to get the arguments that you want.
// We need to deal with the fact that we cannot really template into
// protocol buffers... yuck.
#define INSTANTIATE_GET_SINGLE_ARGUMENT(dtype, fieldname)                      \
template <>                                                                    \
dtype OperatorBase::GetSingleArgument<dtype>(                                  \
    const string& name, const dtype& default_value) {                          \
  if (arg_map_.count(name) == 0) {                                             \
    DVLOG(1) << "Using default parameter value " << default_value;             \
    return default_value;                                                      \
  }                                                                            \
  CHECK(arg_map_[name]->has_##fieldname())                                     \
      << "Argument does not have the right field: expected "                   \
      << #fieldname;                                                           \
  return arg_map_[name]->fieldname();                                          \
}

INSTANTIATE_GET_SINGLE_ARGUMENT(float, f)
INSTANTIATE_GET_SINGLE_ARGUMENT(int, i)
INSTANTIATE_GET_SINGLE_ARGUMENT(string, s)
// Undefine the argument just to be safe.
#undef INSTANTIATE_GET_SINGLE_ARGUMENT

#define INSTANTIATE_GET_REPEATED_ARGUMENT(dtype, fieldname)                    \
template <>                                                                    \
vector<dtype> OperatorBase::GetRepeatedArgument<dtype>(                        \
    const string& name) {                                                      \
  if (arg_map_.count(name) == 0) {                                             \
    return vector<dtype>();                                                    \
  }                                                                            \
  vector<dtype> values;                                                        \
  CHECK(arg_map_[name]->fieldname##_size())                                    \
      << "Argument does not have the right field: expected "                   \
      << #fieldname;                                                           \
  for (const auto& v : arg_map_[name]->fieldname()) values.push_back(v);       \
  return values;                                                               \
}

INSTANTIATE_GET_REPEATED_ARGUMENT(float, floats)
INSTANTIATE_GET_REPEATED_ARGUMENT(int, ints)
INSTANTIATE_GET_REPEATED_ARGUMENT(string, strings)
#undef INSTANTIATE_GET_REPEATED_ARGUMENT

bool OperatorBase::Verify() {
  // Check Blob counts.
  if (operator_def_.input_size() < MinInput() ||
      operator_def_.input_size() > MaxInput()) {
    LOG(ERROR) << "Input size " << operator_def_.input_size()
               << " not in range [min=" << MinInput() << ", max="
               << MaxInput() << "].";
    LOG(ERROR) << "Error at operator " << operator_def_.name() << ":"
               << operator_def_.type();
    return false;
  }
  if (operator_def_.output_size() < MinOutput() ||
      operator_def_.output_size() > MaxOutput()) {
    LOG(ERROR) << "Output size " << operator_def_.output_size()
               << " not in range [min=" << MinOutput() << ", max="
               << MaxOutput() << "].";
    LOG(ERROR) << "Error at operator " << operator_def_.name() << ":"
               << operator_def_.type();
    return false;
  }
  return true;
}

OperatorBase* CreateOperator(const OperatorDef& operator_def, Workspace* ws) {
  const string& key = operator_def.type();
  switch (operator_def.device_option().device_type()) {
  case CPU:
    VLOG(1) << "Creating CPU operator " << key;
    return CPUOperatorRegistry()->Create(key, operator_def, ws);
  case CUDA:
    VLOG(1) << "Creating CUDA operator " << key;
    // In Cuda, if we have cudnn, we will prefer to use cudnn first.
    if (CUDNNOperatorRegistry()->Has(key)) {
      VLOG(1) << "Using CuDNN implementation.";
      return CUDNNOperatorRegistry()->Create(key, operator_def, ws);
    }
    return CUDAOperatorRegistry()->Create(key, operator_def, ws);
  }
  // Just to suppress some compiler error
  return nullptr;
}

DEFINE_REGISTRY(CPUOperatorRegistry, OperatorBase,
                const OperatorDef&, Workspace*);
DEFINE_REGISTRY(CUDAOperatorRegistry, OperatorBase,
                const OperatorDef&, Workspace*);
DEFINE_REGISTRY(CUDNNOperatorRegistry, OperatorBase,
                const OperatorDef&, Workspace*);

}  // namespace caffe2
