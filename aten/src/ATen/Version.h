#include <ATen/Context.h>

namespace at {

/// Returns a detailed string describing the version of PyTorch, as well
/// versions of its constituent libraries.
CAFFE2_API std::string detailed_version();

}  // namespace at
