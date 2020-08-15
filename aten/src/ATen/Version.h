#include <ATen/Context.h>

namespace at {

/// Returns a detailed string describing the configuration PyTorch.
CAFFE2_API std::string show_config();

CAFFE2_API std::string get_mkl_version();

CAFFE2_API std::string get_mkldnn_version();

CAFFE2_API std::string get_openmp_version();

}  // namespace at
