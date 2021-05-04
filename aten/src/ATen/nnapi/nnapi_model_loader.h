#ifndef NNAPI_MODEL_LOADER_H_
#define NNAPI_MODEL_LOADER_H_

#include <stdint.h>

#include <ATen/nnapi/NeuralNetworks.h>
#include <ATen/nnapi/nnapi_wrapper.h>

namespace caffe2 {
namespace nnapi {

int load_nnapi_model(
    struct nnapi_wrapper* nnapi,
    ANeuralNetworksModel* model,
    const void* serialized_model,
    int64_t model_length,
    size_t num_buffers,
    const void** buffer_ptrs,
    int32_t* buffer_sizes,
    size_t num_memories,
    ANeuralNetworksMemory** memories,
    int32_t* memory_sizes,
    int32_t* out_input_count,
    int32_t* out_output_count,
    size_t* out_bytes_consumed);

}} // namespace caffe2::nnapi

#endif // NNAPI_MODEL_LOADER_H_
