// Copyright (c) 2022, Tri Dao.

// Splitting the different head dimensions to different files to speed up compilation.

#include <ATen/native/transformers/cuda/flash_attn/fmha_fwd_launch_template.h>

namespace pytorch_fmha {

void run_fmha_fwd_hdim64(Launch_params<FMHA_fprop_params> &launch_params) {
    FP16_SWITCH(launch_params.params.is_bf16, ([&] {
        if (launch_params.params.seqlen_k == 128) {
            using Kernel_traits = FMHA_kernel_traits<128, 64, 16, 1, 4, 0x08u, elem_type>;
            run_fmha_fwd_loop<Kernel_traits>(launch_params);
        } else if (launch_params.params.seqlen_k >= 256) {
            using Kernel_traits = FMHA_kernel_traits<256, 64, 16, 1, 4, 0x08u, elem_type>;
            run_fmha_fwd_loop<Kernel_traits>(launch_params);
        }
    }));
}

}; // namespace pytorch_fmha
