// Copyright (c) 2022, Tri Dao.

// Splitting the different head dimensions to different files to speed up compilation.

#include <ATen/native/transformers/cuda/flash_attn/fmha_bwd_launch_template.h>

namespace pytorch_fmha {

void run_fmha_bwd_hdim32(FMHA_dgrad_params &params, cudaStream_t stream, const bool configure) {
    FP16_SWITCH(params.is_bf16, ([&] {
        if (params.seqlen_k == 128) {
            using Kernel_traits = FMHA_kernel_traits<128, 32, 16, 1, 8, 0x08u, elem_type>;
            run_fmha_bwd_loop<Kernel_traits>(params, stream, configure);
        } else if (params.seqlen_k >= 256) {
            using Kernel_traits = FMHA_kernel_traits<256, 32, 16, 1, 8, 0x08u, elem_type>;
            run_fmha_bwd_loop<Kernel_traits>(params, stream, configure);
        }
    }));
}

}; // namespace pytorch_fmha
