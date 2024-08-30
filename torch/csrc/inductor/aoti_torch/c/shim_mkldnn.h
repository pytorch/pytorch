#ifndef AOTI_TORCH_SHIM_MKLDNN
#define AOTI_TORCH_SHIM_MKLDNN

#include <ATen/Config.h>
#include <torch/csrc/inductor/aoti_torch/c/shim.h>

#if AT_MKLDNN_ENABLED()
#ifdef __cplusplus
extern "C" {
#endif

AOTI_TORCH_EXPORT AOTITorchError aoti_torch_cpu_mkldnn_rnn_layer(
    AtenTensorHandle input,
    AtenTensorHandle weight0,
    AtenTensorHandle weight1,
    AtenTensorHandle weight2,
    AtenTensorHandle weight3,
    AtenTensorHandle hx_,
    AtenTensorHandle cx_,
    int32_t reverse,
    const int64_t* batch_sizes,
    int64_t batch_sizes_len_,
    int64_t mode,
    int64_t hidden_size,
    int64_t num_layers,
    int32_t has_biases,
    int32_t bidirectional,
    int32_t batch_first,
    int32_t train,
    AtenTensorHandle* ret0,
    AtenTensorHandle* ret1,
    AtenTensorHandle* ret2,
    AtenTensorHandle* ret3);

#ifdef __cplusplus
} // extern "C"
#endif
#endif // AT_MKLDNN_ENABLED()
#endif // AOTI_TORCH_SHIM_MKLDNN
