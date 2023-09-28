// Copyright (c) 2022, Tri Dao.

// Splitting the different head dimensions to different files to speed up compilation.

#include <ATen/native/transformers/cuda/flash_attn/fmha_bwd_launch_template.h>

namespace pytorch_fmha {

void run_fmha_bwd_hdim64(FMHA_dgrad_params &params, cudaStream_t stream, const bool configure) {
    FP16_SWITCH(params.is_bf16, ([&] {
        auto dprops = at::cuda::getCurrentDeviceProperties();
        if (params.seqlen_k == 128) {
            using Kernel_traits = FMHA_kernel_traits<128, 64, 16, 1, 8, 0x08u, elem_type>;
            run_fmha_bwd_loop<Kernel_traits>(params, stream, configure);
        } else if (params.seqlen_k >= 256) {
          if ((dprops->major == 8 && dprops->minor == 0) ||
              (dprops->major == 9 && dprops->minor == 0)) {
            // Don't share smem for K & V, and don't keep V in registers
            // This speeds things up by 2-3% by avoiding register spills, but it
            // uses more shared memory, which is fine on A100 and H100 but not other
            // GPUs. For other GPUs, we keep V in registers.
            using Kernel_traits =
                FMHA_kernel_traits<256, 64, 16, 1, 8, 0x100u, elem_type>;
            run_fmha_bwd_loop<Kernel_traits>(params, stream, configure);
          } else if (dprops->major == 8 && dprops->minor > 0) {
            using Kernel_traits =
                FMHA_kernel_traits<256, 64, 16, 1, 8, 0x08u, elem_type>;
            run_fmha_bwd_loop<Kernel_traits>(params, stream, configure);
          } else if (dprops->major == 7 && dprops->minor == 5) {
            using Kernel_traits =
                FMHA_kernel_traits<128, 64, 16, 1, 8, 0x08u, elem_type>;
            run_fmha_bwd_loop<Kernel_traits>(params, stream, configure);
          }
        }
    }));
}

}; // namespace pytorch_fmha
