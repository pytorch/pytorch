#pragma once
#include <c10/macros/Export.h>
#include <c10/util/Flags.h>

C10_DECLARE_bool(torch_lazy_ir_debug);
C10_DECLARE_bool(torch_lazy_handle_special_scalars);
C10_DECLARE_bool(torch_lazy_all_numbers_special_scalars);
C10_DECLARE_bool(torch_lazy_param_aliasing);
C10_DECLARE_bool(torch_lazy_reuse_ir);
C10_DECLARE_bool(torch_lazy_use_thread_pool);
C10_DECLARE_bool(torch_lazy_enable_compilation_cache);
C10_DECLARE_bool(torch_lazy_enable_device_data_cache);

C10_DECLARE_int(torch_lazy_compilation_cache_size);
C10_DECLARE_int(torch_lazy_device_data_cache_size);
C10_DECLARE_int(torch_lazy_io_thread_pool_size);
C10_DECLARE_int(torch_lazy_metrics_samples);
C10_DECLARE_int(torch_lazy_trim_graph_check_frequency);
C10_DECLARE_int(torch_lazy_trim_graph_size);

C10_DECLARE_string(torch_lazy_metrics_percentiles);

C10_DECLARE_int(torch_lazy_shape_cache_size);

namespace torch {
namespace lazy {
TORCH_API std::string& getLTCForceFallback();
}
} // namespace torch
