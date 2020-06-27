#include <ATen/ATen.h>
#include <torch/library.h>

#include "caffe2/caffe2/perfkernels/fused_nbit_rowwise_conversion.h"

namespace at {
namespace native {
namespace {

Tensor embedding_bag_byte_prepack(
    const Tensor& weight) {
    // TODO(radkris) Any other options to add as parameter?
    // TODO(radkris) Add validation to the incoming params.
    int64_t embedding_rows = weight.size(0);
    int64_t embedding_dims = weight.size(1);

    const auto weight_data = weight.data_ptr<float>();
    std::vector<int64_t> output_shape = {embedding_rows, embedding_dims + 8};
    auto output = at::empty(output_shape, weight.options().dtype(at::kByte));
    auto* output_data = output.data_ptr<uint8_t>();
    constexpr float kEpsilon = 1e-8f;

    // int output_columns = embedding_dims + 2 * sizeof(float);
    caffe2::FloatToFused8BitRowwiseQuantized(
        weight_data,
        embedding_rows,
        embedding_dims,
        output_data);

    // for (std::size_t row = 0; row < embedding_rows; ++row) {
    //     const float* input_row = weight_data + row * embedding_dims;
    //     std::uint8_t* output_row = output_data + row * output_columns;
    //     float* output_row_scale_bias =
    //         reinterpret_cast<float*>(output_row + embedding_dims);

    //     float minimum_element =
    //         *std::min_element(input_row, input_row + embedding_dims);
    //     float maximum_element =
    //         *std::max_element(input_row, input_row + embedding_dims);
    //     float range = maximum_element - minimum_element;

    //     output_row_scale_bias[0] = range / 255.0f;
    //     output_row_scale_bias[1] = minimum_element;
    //     const auto inverse_scale = 255.0f / (range + kEpsilon);
    //     for (std::size_t col = 0; col < embedding_dims; ++col) {
    //         output_row[col] = std::lrintf((input_row[col] - minimum_element) * inverse_scale);
    //     }
    // }

    return output;
}

Tensor embedding_bag_4bit_prepack(const Tensor& weight) {
    int64_t embedding_rows = weight.size(0);
    int64_t embedding_dims = weight.size(1);

    const auto weight_data = weight.data_ptr<float>();
    std::vector<int64_t> output_shape = {
        embedding_rows,
        static_cast<std::int64_t>((embedding_dims + 2 - 1) / 2 + 2 * sizeof(at::Half))
    };
    auto output = at::empty(output_shape, weight.options().dtype(at::kByte));
    auto* output_data = output.data_ptr<uint8_t>();
    constexpr int BIT_RATE = 4;
    constexpr int NUM_ELEM_PER_BYTE = 8 / BIT_RATE;
    int output_columns = static_cast<std::int64_t>(
        (embedding_dims + NUM_ELEM_PER_BYTE - 1) / NUM_ELEM_PER_BYTE +
        2 * sizeof(at::Half));
    for (int row = 0; row < embedding_rows; ++row) {
            const float* input_row = weight_data + row * embedding_dims;
            std::uint8_t* output_row = output_data + row * output_columns;
            at::Half* output_row_scale = reinterpret_cast<at::Half*>(
                output_row + (embedding_dims + NUM_ELEM_PER_BYTE - 1) / NUM_ELEM_PER_BYTE);

            at::Half* output_row_bias = reinterpret_cast<at::Half*>(
                output_row + (embedding_dims + NUM_ELEM_PER_BYTE - 1) / NUM_ELEM_PER_BYTE +
                sizeof(at::Half));

            float Xmin = *std::min_element(input_row, input_row + embedding_dims);
            float Xmax = *std::max_element(input_row, input_row + embedding_dims);

            // Round Xmin to fp16 to match with dequantization that will use fp16
            // for Xmin.
            Xmin = static_cast<at::Half>(Xmin);
            const float range = Xmax - Xmin;
            // Round scale to fp16 to match with dequantization that will use fp16
            // for scale.
            // Set scale to 1.0f for the corner case of Xmax == Xmin .
            // Any non-zero scale would work because during quantization
            // (X - Xmin) / scale will be 0 for all X unless scale is 0.
            at::Half scale = range == 0 ? 1.0f : range / ((1 << BIT_RATE) - 1);
            if (scale == 0) {
                // Corner case handling when Xmax == Xmin
                // Any scale would work because X - Xmin will be 0 for all X
                scale = 1.0f;
            }

            *output_row_scale = scale;
            *output_row_bias = Xmin;

            for (int col = 0; col < embedding_dims; ++col) {
                float X = input_row[col];
                std::uint8_t quantized = std::max(
                    0, std::min<int>(lrintf((X - Xmin) / scale), (1 << BIT_RATE) - 1));
                if (col % NUM_ELEM_PER_BYTE == 0) {
                // LSB
                output_row[col / NUM_ELEM_PER_BYTE] = quantized;
                } else {
                output_row[col / NUM_ELEM_PER_BYTE] |=
                    (quantized << ((col % NUM_ELEM_PER_BYTE) * BIT_RATE));
                }
            }
    }
    return output;
}

TORCH_LIBRARY_IMPL(quantized, CPU, m) {
  m.impl("embedding_bag_byte_prepack", embedding_bag_byte_prepack);
  m.impl("embedding_bag_4bit_prepack", embedding_bag_4bit_prepack);
}

} // namespace
} // namespace native
} // namespace at
