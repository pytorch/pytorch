// This file is auto-generated. See "generate_kernels.sh"
#include <ATen/native/transformers/cuda/mem_eff_attention/kernel_backward.h>
INSTANTIATE_ATTENTION_KERNEL_BACKWARD_SM50(cutlass::bfloat16_t, false, 64);
INSTANTIATE_ATTENTION_KERNEL_BACKWARD_SM70(cutlass::bfloat16_t, false, 64);
INSTANTIATE_ATTENTION_KERNEL_BACKWARD_SM75(cutlass::bfloat16_t, false, 64);
INSTANTIATE_ATTENTION_KERNEL_BACKWARD_SM80(cutlass::bfloat16_t, false, 64);
