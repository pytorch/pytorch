#include <pytorch_qnnpack.h>
#include <qnnpack/pack.h>
#include <qnnpack_func.h>
#include <cstring>
#include <cstdlib>

namespace qnnpack {
// For runtime quantization packing.
PackBMatrix::PackBMatrix(
    const size_t input_channels,
    const size_t output_channels,
    const uint8_t* kernel_zero_points,
    const float* requantization_scales,
    const uint8_t* kernel,
    const int32_t* bias) {
  for (size_t i = 0; i < output_channels; ++i) {
    if (requantization_scales[i] <= 0.0f ||
        !std::isnormal(requantization_scales[i])) {
      pytorch_qnnp_log_error(
          "failed to create fully connected operator with requant scale of "
          "%.7g for output channel %d."
          "Scale must be finite and positive",
          requantization_scales[i], (int)i);
      assert("QNNPACK Runtime Error.");
    }
  }

  const uint32_t nr = pytorch_qnnp_params.q8conv.nr;
  const uint32_t kr = pytorch_qnnp_params.q8conv.kr;

  const uint32_t n_stride = (output_channels + (nr - 1)) & -nr;
  const uint32_t k_stride = (input_channels + (kr - 1)) & -kr;

  input_channels_ = input_channels;
  output_channels_ = output_channels;
  packed_weights_ =
      malloc(n_stride * (k_stride * sizeof(uint8_t) + sizeof(int32_t)));
  if (packed_weights_ == NULL) {
    pytorch_qnnp_log_error(
        "failed to allocate %zu bytes for packed weights",
        n_stride * (k_stride * sizeof(uint8_t) + sizeof(int32_t)));
    assert("QNNPACK Runtime Error.");
  }

  pytorch_pack_q8gemm_wrq(
      output_channels,
      input_channels,
      nr,
      nr,
      kr,
      kernel,
      bias,
      kernel_zero_points,
      packed_weights_);
}

} // namespace qnnpack
