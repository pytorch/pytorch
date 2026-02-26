#include <ATen/ATen.h>
#include <ATen/autocast_mode.h>
#include <ATen/cuda/CUDAConfig.h>
#include <torch/library.h>

namespace at {
namespace autocast {

/**********************************************************************
Autocast wrapper for MIOpen RNNs
**********************************************************************/
std::tuple<Tensor, Tensor, Tensor, Tensor, Tensor>
miopen_rnn(const Tensor & input_r,
           TensorList weight,
           int64_t weight_stride0,
           const Tensor & hx,
           const std::optional<Tensor>& cx_opt,
           int64_t fn_mode,
           int64_t fn_hidden_size,
           int64_t fn_num_layers,
           bool batch_first,
           double fn_dropout,
           bool fn_train,
           bool fn_bidirectional,
           IntArrayRef fn_batch_sizes,
           const std::optional<Tensor>& fn_dropout_state_opt) {

#if AT_ROCM_ENABLED()

    c10::impl::ExcludeDispatchKeyGuard no_autocast(DispatchKey::Autocast);

    return at::miopen_rnn(
                cached_cast(at::kHalf, input_r),
                cached_cast(at::kHalf, weight),
                weight_stride0,
                cached_cast(at::kHalf, hx),
                cached_cast(at::kHalf, cx_opt),
                fn_mode,
                fn_hidden_size,
                fn_num_layers,
                batch_first,
                fn_dropout,
                fn_train,
                fn_bidirectional,
                fn_batch_sizes,
                fn_dropout_state_opt);

#else
    TORCH_CHECK(false, "autocast::miopen_rnn: ATen not compiled with ROCm enabled");
    return {Tensor{}, Tensor{}, Tensor{}, Tensor{}, Tensor{}}; // placate the compiler
#endif

}

// Register Autocast dispatch
namespace {
TORCH_LIBRARY_IMPL(aten, Autocast, m) {
  m.impl("miopen_rnn",
         TORCH_FN((&at::autocast::miopen_rnn)));
}
} // anonymous namespace

} // namespace autocast
} // namespace at
