#pragma once
#include <c10/macros/Export.h>
#include <c10/util/Flags.h>

TORCH_DECLARE_bool(torch_lazy_ir_debug);
TORCH_DECLARE_bool(torch_lazy_handle_special_scalars);
TORCH_DECLARE_bool(torch_lazy_all_numbers_special_scalars);
TORCH_DECLARE_bool(torch_lazy_param_aliasing);
TORCH_DECLARE_bool(torch_lazy_reuse_ir);
TORCH_DECLARE_bool(torch_lazy_use_thread_pool);
TORCH_DECLARE_bool(torch_lazy_enable_device_data_cache);

TORCH_DECLARE_int(torch_lazy_compilation_cache_size);
TORCH_DECLARE_int(torch_lazy_device_data_cache_size);
TORCH_DECLARE_int(torch_lazy_io_thread_pool_size);
TORCH_DECLARE_int(torch_lazy_metrics_samples);
TORCH_DECLARE_int(torch_lazy_trim_graph_check_frequency);
TORCH_DECLARE_int(torch_lazy_trim_graph_size);

TORCH_DECLARE_string(torch_lazy_metrics_percentiles);

TORCH_DECLARE_int(torch_lazy_shape_cache_size);

namespace torch::lazy {
TORCH_API std::string& getLTCForceFallback();
}
