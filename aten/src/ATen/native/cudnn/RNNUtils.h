#include <ATen/cudnn/cudnn-wrapper.h>
#include <ATen/cudnn/Descriptors.h>
#include <ATen/cudnn/Types.h>
#include <ATen/cudnn/Utils.h>

// Declares utilities used by RNN.cpp and also needed by external consumers
namespace at {
namespace native {
namespace cudnn_rnn {

TORCH_CUDA_CPP_API std::tuple<Tensor, std::vector<Tensor>>
copy_weights_to_flat_buf_views(
    TensorList weight_arr,
    int64_t weight_stride0,
    int64_t input_size,
    int64_t mode,
    c10::SymInt hidden_size,
    c10::SymInt proj_size,
    int64_t num_layers,
    bool batch_first,
    bool bidirectional,
    const cudnnDataType_t flat_buf_datatype,
    const TensorOptions& flat_buf_options,
    bool set_orig_weights_to_flat_buf,
    bool allow_type_change = false,
    bool include_bias = true);

} // namespace cudnn_rnn
} // namespace native
} // namespace at
