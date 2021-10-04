#include <ATen/ATen.h>
#include <ATen/autocast_mode.h>
#include <torch/library.h>

// pulls in AT_CUDNN_ENABLED() as defined by cmake
#include <ATen/cuda/CUDAConfig.h>

#if AT_CUDNN_ENABLED()
#include <ATen/native/cudnn/RNNUtils.h>
#endif

namespace at {
namespace autocast {

/********************************************************************************
Autocast wrapper for CuDNN RNNs (the weight reflattening needs special attention)
********************************************************************************/

// To be registered for the "_cudnn_rnn(...)" schema.
// _cudnn_rnn is autograd-exposed (test_autocast_cudnn_rnn in test_cuda.py includes a test to confirm)
std::tuple<Tensor,Tensor,Tensor,Tensor,Tensor>
_cudnn_rnn_cast_reflatten(const Tensor & input,
                          TensorList weight,
                          int64_t weight_stride0,
                          const c10::optional<Tensor>& weight_buf_opt,
                          const Tensor& hx,
                          const c10::optional<Tensor>& cx,
                          int64_t mode,
                          int64_t hidden_size,
                          int64_t proj_size,
                          int64_t num_layers,
                          bool batch_first,
                          double dropout,
                          bool train,
                          bool bidirectional,
                          IntArrayRef batch_sizes,
                          const c10::optional<Tensor>& dropout_state) {
#if AT_CUDNN_ENABLED()
  c10::impl::ExcludeDispatchKeyGuard no_autocast(DispatchKey::Autocast);

  for (const auto& t : weight) {
    TORCH_CHECK(weight[0].scalar_type() == t.scalar_type(), "Weight scalar types do not match.");
  }
  // weight_stride0 is the number of weight tensors per layer and direction, as seen by model.parameters().
  // If bias is enabled, there are 4 such tensors (ih and hh weights, ih and hh biases).
  // If bias is not enabled, there are 2 (ih and hh weights).
  // This organization holds for all rnn types (RNN, GRU, and LSTM). If LSTM with projections is
  // used, additional hr weight is added.
  if (proj_size > 0) {
    TORCH_INTERNAL_ASSERT((weight_stride0 == 3) || (weight_stride0 == 5),
                          "weight_stride0 must be 3 (if no bias) or 5 (if bias) for LSTM with projections.  Received ",
                          weight_stride0);
  } else {
    TORCH_INTERNAL_ASSERT((weight_stride0 == 2) || (weight_stride0 == 4),
                          "weight_stride0 must be 2 (if no bias) or 4 (if bias).  Received ",
                          weight_stride0);
  }


  Tensor weight_buf, redispatch_weight_buf;
  std::vector<Tensor> redispatch_weight;
  // There's an implicit contract here with native/cudnn/RNN.cpp:_cudnn_impl, which calls at:_cudnn_rnn.
  // Code here assumes if _cudnn_impl passes weight_buf_opt containing a defined tensor, that tensor
  // is valid flat storage of the weights in their incoming dtype.
  if (weight_buf_opt.has_value()) {
    weight_buf = *weight_buf_opt;
  }
  bool needs_cast_and_flatten = (weight_buf.defined() ?
                                 // weight_buf is valid.  Only change it if it's eligible and not already FP16.
                                 is_eligible(weight_buf) && (weight_buf.scalar_type() != at::kHalf) :
                                 // weight_buf is not valid.  Only create it if other weights are eligible and not already FP16.
                                 is_eligible(weight[0]) && (weight[0].scalar_type() != at::kHalf));
  if (needs_cast_and_flatten) {
    // Casts weight tensors to FP16 and ensures all weights for all layers are views into a large flat buffer,
    // with the right locations and layouts expected by cudnn.
    // This is (and should be) autograd-exposed.
    bool include_bias = true;
    if (weight_stride0 == 2 || (weight_stride0 == 3 && proj_size > 0)) {
      include_bias = false;
    }
    std::tie(redispatch_weight_buf, redispatch_weight) =
        at::native::cudnn_rnn::copy_weights_to_flat_buf_views(
            weight,
            weight_stride0,
            input.size(-1),
            mode,
            hidden_size,
            proj_size,
            num_layers,
            batch_first,
            bidirectional,
            /*flat_buf_datatype=*/at::native::getCudnnDataTypeFromScalarType(at::kHalf), // could just hardcode CUDNN_DATA_HALF
            /*flat_buf_options=*/weight[0].options().dtype(at::kHalf),
            /*set_orig_weights_to_flat_buf=*/false,
            /*allow_type_change=*/true,
            /*include_bias=*/include_bias);
  }
  return at::_cudnn_rnn(
      cached_cast(at::kHalf, input),
      needs_cast_and_flatten ? TensorList(redispatch_weight) : weight,
      weight_stride0,
      needs_cast_and_flatten ? redispatch_weight_buf : weight_buf,
      cached_cast(at::kHalf, hx),
      cached_cast(at::kHalf, cx),
      mode,
      hidden_size,
      proj_size,
      num_layers,
      batch_first,
      dropout,
      train,
      bidirectional,
      batch_sizes,
      dropout_state);
#else // AT_CUDNN_ENABLED()
  AT_ERROR("autocast::_cudnn_rnn_cast_reflatten: ATen not compiled with cuDNN support");
  return {Tensor{}, Tensor{}, Tensor{}, Tensor{}, Tensor{}}; // never reached, placates the compiler
#endif // AT_CUDNN_ENABLED()
}

namespace {
TORCH_LIBRARY_IMPL(aten, Autocast, m) {
  m.impl("_cudnn_rnn",
         TORCH_FN((&at::autocast::_cudnn_rnn_cast_reflatten)));
}
} // anonymous namespace

} // namespace autocast
} // namespace at
