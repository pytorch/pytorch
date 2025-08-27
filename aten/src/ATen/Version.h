#include <ATen/Context.h>

namespace at {

/// Returns a detailed string describing the configuration PyTorch.
TORCH_API std::string show_config();

TORCH_API std::string get_mkl_version();

TORCH_API std::string get_mkldnn_version();

TORCH_API std::string get_openmp_version();

TORCH_API std::string get_cxx_flags();

TORCH_API std::string get_cpu_capability();

TORCH_API int get_sve_len();

} // namespace at
