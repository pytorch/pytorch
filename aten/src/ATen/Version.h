#include <ATen/Context.h>

namespace at {

/// Returns a detailed string describing the configuration PyTorch.
CAFFE2_API std::string show_config();

}  // namespace at
