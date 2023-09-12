// Copyright (c) 2022, Tri Dao.

// Splitting the different head dimensions to different files to speed up compilation.

#include <ATen/native/transformers/cuda/flash_attn/fmha_fwd_launch_template.h>

namespace pytorch_fmha {

void run_fmha_fwd_hdim128(Launch_params<FMHA_fprop_params> &launch_params) {
    FP16_SWITCH(launch_params.params.is_bf16, ([&] {
        using Kernel_traits = FMHA_kernel_traits<128, 128, 16, 1, 4, 0x08u, elem_type>;
        run_fmha_fwd_loop<Kernel_traits>(launch_params);
    }));
}

}; // namespace pytorch_fmha
