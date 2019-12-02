#include <ATen/ATen.h>
#include <ATen/cpu/vec256/vec256.h>

namespace at { namespace native {

namespace {

void static batch_norm_cpu_inference_contiguous_fast_kernel(
    Tensor& output,
    const Tensor& input, const Tensor& alpha, const Tensor& beta) {

    AT_DISPATCH_ALL_TYPES(output.scalar_type(), "batch_norm_cpu_fast_kernel", [&] {
  
        using Vec = vec256::Vec256<scalar_t>;

        int64_t n_batch = input.size(0);
        int64_t n_channel = input.size(1);
        int64_t image_size = input.numel() / n_batch / n_channel;

        const scalar_t* input_data = input.data_ptr<scalar_t>();
        scalar_t* output_data = output.data_ptr<scalar_t>();

        scalar_t* alpha_data = alpha.data_ptr<scalar_t>();
        scalar_t* beta_data = beta.data_ptr<scalar_t>();

        int64_t vec_size = 2 * Vec::size();

        std::vector<Vec> alpha_vecs(n_channel);
        std::vector<Vec> beta_vecs(n_channel);
        for (int64_t c = 0; c < n_channel; ++c) {
            alpha_vecs[c] = Vec(alpha_data[c]);
            beta_vecs[c] = Vec(beta_data[c]);
        }

        if (image_size != 1) {
            for (int64_t n = 0; n < n_batch; ++n) {
                for (int64_t c = 0; c < n_channel; ++c) {
                    int64_t offset = n * n_channel * image_size + c * image_size;
                    int64_t i = 0;
                    for (; i < image_size - vec_size; i += vec_size) {
                        int64_t pos = offset + i;
                        Vec vec1 = Vec::loadu(&input_data[pos]) * alpha_vecs[c] + beta_vecs[c];
                        Vec vec2 = Vec::loadu(&input_data[pos] + Vec::size()) * alpha_vecs[c] + beta_vecs[c];
                        vec1.store(&output_data[pos]);
                        vec2.store(&output_data[pos] + Vec::size());
                    }

                    if (i < image_size) {
                        for (; i < image_size; ++i) {
                            output_data[offset + i] = input_data[offset + i] * alpha_data[c] + beta_data[c];
                        }
                    }
                }
            }
        }
        else {
            for (int64_t n = 0; n < n_batch; ++n) {
                for (int64_t c = 0; c < n_channel; ++c) {
                    int64_t offset = n * n_channel + c;
                    output_data[offset] = input_data[offset] * alpha_data[c] + beta_data[c];
                }
            }
        }
       
        // Apply the linear terms to the input,
        // output(n, c, h, w) = input(n, c, h, w) * alpha(c) + beta(c)
        // No need to use parallel_for as this function is supposed to be
        // memory-limited.
        // Keep the loop struture simple to make sure compiler vetorization kicks in.
        /*if (image_size != 1) {
            for (int64_t n = 0; n < n_batch; ++n) {
                for (int64_t c = 0; c < n_channel; ++c) {
                    for (int64_t i = 0; i < image_size; ++i) {
                        // Keep all the offset calculation within the inner loop for
                        // simplicity. Compilers are very good at hoisting the common part
                        // outside.
                        int64_t offset = n * n_channel * image_size + c * image_size + i;
                        output_data[offset] = input_data[offset] * alpha_data[c] + beta_data[c];
                    }
                }
            }
        } 
        else {
            // image_size == 1
            for (int64_t n = 0; n < n_batch; ++n) {
                for (int64_t c = 0; c < n_channel; ++c) {
                    int64_t offset = n * n_channel + c;
                     output_data[offset] = input_data[offset] * alpha_data[c] + beta_data[c];
                }
            }
        }*/
    });
}

} // namespace

REGISTER_DISPATCH(batch_norm_cpu_inference_contiguous_fast_stub, &batch_norm_cpu_inference_contiguous_fast_kernel)

} // namespace native
} // namespace at
