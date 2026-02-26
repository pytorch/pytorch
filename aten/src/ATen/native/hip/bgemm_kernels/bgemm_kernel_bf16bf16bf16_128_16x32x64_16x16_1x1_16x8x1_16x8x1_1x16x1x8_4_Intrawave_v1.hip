#undef __HIP_NO_HALF_CONVERSIONS__

#include <ATen/native/hip/bgemm_kernels/bgemm_kernel_template.h>

namespace at::native {

void bgemm_kernel_bf16bf16bf16_128_16x32x64_16x16_1x1_16x8x1_16x8x1_1x16x1x8_4_Intrawave_v1(CUDABLAS_BGEMM_ARGTYPES(at::BFloat16)) {
    bool transa_ = std::tolower(transa) != 'n';
    bool transb_ = std::tolower(transb) != 'n';
    if (transa_ && transb_) {
        bgemm_kernel_impl<
            ck::bhalf_t, // A_DATA_TYPE
            ck::bhalf_t, // B_DATA_TYPE
            128, // BLOCK_SIZE
            16, // M_BLOCK
            32, // N_BLOCK
            64, // K_BLOCK
            4, // AK1
            4, // BK1
            16, // WAVE_TILE_M
            16, // WAVE_TILE_N
            1, // WAVE_MAP_M
            1, // WAVE_MAP_N
            S<16, 8, 1>, // ABLOCK_TRANSFER
            4, // ABLOCK_TRANSFER_SSPV
            4, // ABLOCK_TRANSFER_DSPV_K1
            S<16, 8, 1>, // BBLOCK_TRANSFER
            4, // BBLOCK_TRANSFER_SSPV
            4, // BBLOCK_TRANSFER_SSPV_K1
            1, // CSHUFFLE_MXDL_PWPS
            1,// CSHUFFLE_NXDL_PWPS
            S<1, 16, 1, 8>, // CSHUFFLEBLOCK_TRANSFER
            S<4>, // CDESHUFFLEBLOCK_TRANSFER
            ck::BlockGemmPipelineScheduler::Intrawave, // LOOP_SCHED
            ck::BlockGemmPipelineVersion::v1, // PIPELINE_VERSION
            ck::tensor_operation::device::GemmSpecialization::Default,
            true, // TRANS_A
            true>(CUDABLAS_BGEMM_ARGS(at::BFloat16));
    } else if (transa_ && !transb_) {
        bgemm_kernel_impl<
            ck::bhalf_t, // A_DATA_TYPE
            ck::bhalf_t, // B_DATA_TYPE
            128, // BLOCK_SIZE
            16, // M_BLOCK
            32, // N_BLOCK
            64, // K_BLOCK
            4, // AK1
            4, // BK1
            16, // WAVE_TILE_M
            16, // WAVE_TILE_N
            1, // WAVE_MAP_M
            1, // WAVE_MAP_N
            S<16, 8, 1>, // ABLOCK_TRANSFER
            4, // ABLOCK_TRANSFER_SSPV
            4, // ABLOCK_TRANSFER_DSPV_K1
            S<16, 8, 1>, // BBLOCK_TRANSFER
            4, // BBLOCK_TRANSFER_SSPV
            4, // BBLOCK_TRANSFER_SSPV_K1
            1, // CSHUFFLE_MXDL_PWPS
            1,// CSHUFFLE_NXDL_PWPS
            S<1, 16, 1, 8>, // CSHUFFLEBLOCK_TRANSFER
            S<4>, // CDESHUFFLEBLOCK_TRANSFER
            ck::BlockGemmPipelineScheduler::Intrawave, // LOOP_SCHED
            ck::BlockGemmPipelineVersion::v1, // PIPELINE_VERSION
            ck::tensor_operation::device::GemmSpecialization::Default,
            true, // TRANS_A
            false>(CUDABLAS_BGEMM_ARGS(at::BFloat16));
    } else if (!transa_ && transb_) {
        bgemm_kernel_impl<
            ck::bhalf_t, // A_DATA_TYPE
            ck::bhalf_t, // B_DATA_TYPE
            128, // BLOCK_SIZE
            16, // M_BLOCK
            32, // N_BLOCK
            64, // K_BLOCK
            4, // AK1
            4, // BK1
            16, // WAVE_TILE_M
            16, // WAVE_TILE_N
            1, // WAVE_MAP_M
            1, // WAVE_MAP_N
            S<16, 8, 1>, // ABLOCK_TRANSFER
            4, // ABLOCK_TRANSFER_SSPV
            4, // ABLOCK_TRANSFER_DSPV_K1
            S<16, 8, 1>, // BBLOCK_TRANSFER
            4, // BBLOCK_TRANSFER_SSPV
            4, // BBLOCK_TRANSFER_SSPV_K1
            1, // CSHUFFLE_MXDL_PWPS
            1,// CSHUFFLE_NXDL_PWPS
            S<1, 16, 1, 8>, // CSHUFFLEBLOCK_TRANSFER
            S<4>, // CDESHUFFLEBLOCK_TRANSFER
            ck::BlockGemmPipelineScheduler::Intrawave, // LOOP_SCHED
            ck::BlockGemmPipelineVersion::v1, // PIPELINE_VERSION
            ck::tensor_operation::device::GemmSpecialization::Default,
            false, // TRANS_A
            true>(CUDABLAS_BGEMM_ARGS(at::BFloat16));
    } else {
        bgemm_kernel_impl<
            ck::bhalf_t, // A_DATA_TYPE
            ck::bhalf_t, // B_DATA_TYPE
            128, // BLOCK_SIZE
            16, // M_BLOCK
            32, // N_BLOCK
            64, // K_BLOCK
            4, // AK1
            4, // BK1
            16, // WAVE_TILE_M
            16, // WAVE_TILE_N
            1, // WAVE_MAP_M
            1, // WAVE_MAP_N
            S<16, 8, 1>, // ABLOCK_TRANSFER
            4, // ABLOCK_TRANSFER_SSPV
            4, // ABLOCK_TRANSFER_DSPV_K1
            S<16, 8, 1>, // BBLOCK_TRANSFER
            4, // BBLOCK_TRANSFER_SSPV
            4, // BBLOCK_TRANSFER_SSPV_K1
            1, // CSHUFFLE_MXDL_PWPS
            1,// CSHUFFLE_NXDL_PWPS
            S<1, 16, 1, 8>, // CSHUFFLEBLOCK_TRANSFER
            S<4>, // CDESHUFFLEBLOCK_TRANSFER
            ck::BlockGemmPipelineScheduler::Intrawave, // LOOP_SCHED
            ck::BlockGemmPipelineVersion::v1, // PIPELINE_VERSION
            ck::tensor_operation::device::GemmSpecialization::Default,
            false, // TRANS_A
            false>(CUDABLAS_BGEMM_ARGS(at::BFloat16));
    }
}
};