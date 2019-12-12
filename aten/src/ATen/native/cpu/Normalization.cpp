#include <ATen/ATen.h>
#include <ATen/cpu/vec256/vec256.h>

namespace at { namespace native {

namespace {

template <typename T>
inline void apply(int64_t pos, const T& alpha, const T& beta, const T* input, T* output) {
    using Vec = vec256::Vec256<T>;

    Vec alpha_vec(alpha);
    Vec beta_vec(beta);
    Vec input_vec = Vec::loadu(&input[pos]);
    Vec output_vec = vec256::fmadd<Vec>(input_vec, alpha_vec, beta_vec);

    output_vec.store(&output[pos]);
}

static void batch_norm_cpu_inference_contiguous_kernel(
    Tensor& output,
    const Tensor& input, const Tensor& alpha, const Tensor& beta) {

    AT_DISPATCH_ALL_TYPES(output.scalar_type(), "batch_norm_cpu_contiguous_kernel", [&] {

        using Vec = vec256::Vec256<scalar_t>;

        int64_t n_batch = input.size(0);
        int64_t n_channel = input.size(1);
        int64_t image_size = input.numel() / n_batch / n_channel;

        const scalar_t* input_data = input.data_ptr<scalar_t>();
        scalar_t* output_data = output.data_ptr<scalar_t>();

        scalar_t* alpha_data = alpha.data_ptr<scalar_t>();
        scalar_t* beta_data = beta.data_ptr<scalar_t>();

        // Apply the linear terms to the input,
        // output(n, c, h, w) = input(n, c, h, w) * alpha(c) + beta(c)
        // No need to use parallel_for as this function is supposed to be memory-limited.
        // Keep the loop struture simple to make sure compiler vetorization kicks in.
        if (image_size != 1) {
            for (int64_t n = 0; n < n_batch; ++n) {
                for (int64_t c = 0; c < n_channel; ++c) {
                    int64_t offset = n * n_channel * image_size + c * image_size;
                    int64_t i = 0;
                    for (; i < image_size - Vec::size(); i += Vec::size()) {
                        apply<scalar_t>(offset+i, alpha_data[c], beta_data[c], input_data, output_data);
                    }

                    if (i < image_size) {
                        for (; i < image_size; ++i) {
                            output_data[offset+i] = input_data[offset+i] * alpha_data[c] + beta_data[c];
                        }
                    }
                }
            }
        }
        else {
            for (int64_t n = 0; n < n_batch; ++n) {
                int64_t offset = n * n_channel;
                int64_t c = 0;
                for (; c < n_channel - Vec::size(); c += Vec::size()) {
                    apply<scalar_t>(offset+c, alpha_data[c], beta_data[c], input_data, output_data);
                }

                if (c < n_channel) {
                    for (; c < n_channel; ++c) {
                        output_data[offset+c] = input_data[offset+c] * alpha_data[c] + beta_data[c];
                    }
                }
            }
        }
    });
}

static void batch_norm_cpu_inference_channels_last_kernel(
    Tensor& output,
    const Tensor& input, const Tensor& alpha, const Tensor& beta) {

    AT_DISPATCH_ALL_TYPES(output.scalar_type(), "batch_norm_cpu_channels_last_kernel", [&] {

        using Vec = vec256::Vec256<scalar_t>;

        int64_t n_batch = input.size(0);
        int64_t n_channel = input.size(1);
        int64_t image_size = input.numel() / n_batch / n_channel;

        const scalar_t* input_data = input.data_ptr<scalar_t>();
        scalar_t* output_data = output.data_ptr<scalar_t>();

        scalar_t* alpha_data = alpha.data_ptr<scalar_t>();
        scalar_t* beta_data = beta.data_ptr<scalar_t>();

        // Apply the linear terms to the input,
        // output(n, c, h, w) = input(n, c, h, w) * alpha(c) + beta(c)
        // No need to use parallel_for as this function is supposed to be memory-limited.
        // Keep the loop struture simple to make sure compiler vetorization kicks in.
        if (n_channel != 1) {
            for (int64_t n = 0; n < n_batch; ++n) {
                for (int64_t i = 0; i < image_size; ++i) {
                    int64_t offset = n * image_size * n_channel + i * n_channel;
                    int c = 0;
                    for (; c < n_channel - Vec::size(); c += Vec::size()) {
                        apply<scalar_t>(offset + c, alpha_data[c], beta_data[c], input_data, output_data);
                    }

                    if (c < n_channel) {
                        for (; c < n_channel; ++c) {
                            output_data[offset+c] = input_data[offset+c] * alpha_data[c] + beta_data[c];
                        }
                    }
                }
            }
        }
        else {
            for (int64_t n = 0; n < n_batch; ++n) {
                int64_t offset = n * image_size;
                int64_t i = 0;
                for (; i < image_size - Vec::size(); i += Vec::size()) {
                    apply<scalar_t>(offset+i, alpha_data[0], beta_data[0], input_data, output_data);
                }

                if (i < image_size) {
                    for (; i < image_size; ++i) {
                        output_data[offset+i] = input_data[offset+i] * alpha_data[0] + beta_data[0];
                    }
                }
            }
        }
    });
}

} // namespace

REGISTER_DISPATCH(batch_norm_cpu_inference_contiguous_stub, &batch_norm_cpu_inference_contiguous_kernel)
REGISTER_DISPATCH(batch_norm_cpu_inference_channels_last_stub, &batch_norm_cpu_inference_channels_last_kernel)

} // namespace native
} // namespace at
