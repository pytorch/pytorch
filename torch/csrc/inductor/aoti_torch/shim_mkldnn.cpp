
#include <ATen/ops/mkldnn_rnn_layer_cpu_dispatch.h>
#include <torch/csrc/inductor/aoti_torch/c/shim_mkldnn.h>
#include <torch/csrc/inductor/aoti_torch/utils.h>

using namespace torch::aot_inductor;

#if AT_MKLDNN_ENABLED()

AOTITorchError aoti_torch_cpu_mkldnn_rnn_layer(
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
    AtenTensorHandle* ret3) {
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE({
    auto tmp_result = at::cpu::mkldnn_rnn_layer(
        *tensor_handle_to_tensor_pointer(input),
        *tensor_handle_to_tensor_pointer(weight0),
        *tensor_handle_to_tensor_pointer(weight1),
        *tensor_handle_to_tensor_pointer(weight2),
        *tensor_handle_to_tensor_pointer(weight3),
        *tensor_handle_to_tensor_pointer(hx_),
        *tensor_handle_to_tensor_pointer(cx_),
        reverse,
        pointer_to_list<int64_t>(batch_sizes, batch_sizes_len_),
        mode,
        hidden_size,
        num_layers,
        has_biases,
        bidirectional,
        batch_first,
        train);
    *ret0 = new_tensor_handle(std::move(std::get<0>(tmp_result)));
    *ret1 = new_tensor_handle(std::move(std::get<1>(tmp_result)));
    *ret2 = new_tensor_handle(std::move(std::get<2>(tmp_result)));
    *ret3 = new_tensor_handle(std::move(std::get<3>(tmp_result)));
  });
}

#endif // AT_MKLDNN_ENABLED()
