// Copyright (c) 2022, Tri Dao.

// Splitting the different head dimensions to different files to speed up compilation.

#include <ATen/native/transformers/cuda/flash_attn/fmha_bwd_launch_template.h>

namespace pytorch_fmha {

void run_fmha_bwd_hdim128(FMHA_dgrad_params &params, cudaStream_t stream, const bool configure) {
    FP16_SWITCH(params.is_bf16, ([&] {
        using Kernel_traits = FMHA_kernel_traits<128, 128, 16, 1, 8, 0x100u, elem_type>;
        run_fmha_bwd_loop<Kernel_traits>(params, stream, configure);
    }));
}

}; // namespace pytorch_fmha
