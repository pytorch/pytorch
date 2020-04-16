
#include <qnnpack/indirection.h>
#include <qnnpack/pack.h>
#include <qnnpack_func.h>

namespace qnnpack {

// TODO: WIP
enum pytorch_qnnp_status qnnpackDeConv(
    const deconv_param_t& deconv_p,
    void* packed_weights,
    const size_t batch_size,
    const size_t input_height,
    const size_t input_width,
    const float input_scale,
    const uint8_t input_zero_point,
    const uint8_t* input,
    const float output_scale,
    const uint8_t output_zero_point,
    const bool transpose,
    uint8_t* output,
    pthreadpool_t threadpool);


}  // namespace qnnpack
