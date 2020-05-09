#include <cinttypes>

namespace caffe2 {
namespace fake_nnpi {

constexpr int kNUM_OF_BITS_FOR_Q_APPROX = 15;
constexpr int kELTWISE_ULIMIT_SHORT = 0x7FFF;

int8_t nnpiQuantize(
    int32_t input_val,
    float multiplier,
    int32_t outputOffset,
    bool round_bit_en,
    bool is_signed,
    bool round_half_to_nearest_up);

void matmul_u8i8u8acc32_ref(
    int M,
    int N,
    int K,
    int lda,
    int ldb,
    int ldc,
    const uint8_t* A,
    int32_t A_zero_point,
    const int8_t* B,
    int32_t B_zero_point,
    const int32_t* bias,
    uint8_t* C,
    float C_multiplier, // A_scale * B_scale / C_scale
    int32_t C_zero_point,
    bool fuse_relu = false);

} // namespace fake_nnpi
} // namespace caffe2
