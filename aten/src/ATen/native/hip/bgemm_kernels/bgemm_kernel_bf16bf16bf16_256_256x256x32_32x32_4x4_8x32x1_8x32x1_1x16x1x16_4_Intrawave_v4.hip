#undef __HIP_NO_HALF_CONVERSIONS__

#include <ATen/native/hip/bgemm_kernels/bgemm_kernel_template.h>

namespace at::native {

void bgemm_kernel_bf16bf16bf16_256_256x256x32_32x32_4x4_8x32x1_8x32x1_1x16x1x16_4_Intrawave_v4(CUDABLAS_BGEMM_ARGTYPES(at::BFloat16)) {
    bool transa_ = std::tolower(transa) != 'n';
    bool transb_ = std::tolower(transb) != 'n';
    if (transa_ && transb_) {
        bgemm_kernel_impl<
            ck::bhalf_t, // A_DATA_TYPE
            ck::bhalf_t, // B_DATA_TYPE
            256, // BLOCK_SIZE
            256, // M_BLOCK
            256, // N_BLOCK
            32, // K_BLOCK
            4, // AK1
            4, // BK1
            32, // WAVE_TILE_M
            32, // WAVE_TILE_N
            4, // WAVE_MAP_M
            4, // WAVE_MAP_N
            S<8, 32, 1>, // ABLOCK_TRANSFER
            4, // ABLOCK_TRANSFER_SSPV
            4, // ABLOCK_TRANSFER_DSPV_K1
            S<8, 32, 1>, // BBLOCK_TRANSFER
            4, // BBLOCK_TRANSFER_SSPV
            4, // BBLOCK_TRANSFER_SSPV_K1
            1, // CSHUFFLE_MXDL_PWPS
            1,// CSHUFFLE_NXDL_PWPS
            S<1, 16, 1, 16>, // CSHUFFLEBLOCK_TRANSFER
            S<4>, // CDESHUFFLEBLOCK_TRANSFER
            ck::BlockGemmPipelineScheduler::Intrawave, // LOOP_SCHED
            ck::BlockGemmPipelineVersion::v4, // PIPELINE_VERSION
            ck::tensor_operation::device::GemmSpecialization::Default,
            true, // TRANS_A
            true>(CUDABLAS_BGEMM_ARGS(at::BFloat16));
    } else if (transa_ && !transb_) {
        bgemm_kernel_impl<
            ck::bhalf_t, // A_DATA_TYPE
            ck::bhalf_t, // B_DATA_TYPE
            256, // BLOCK_SIZE
            256, // M_BLOCK
            256, // N_BLOCK
            32, // K_BLOCK
            4, // AK1
            4, // BK1
            32, // WAVE_TILE_M
            32, // WAVE_TILE_N
            4, // WAVE_MAP_M
            4, // WAVE_MAP_N
            S<8, 32, 1>, // ABLOCK_TRANSFER
            4, // ABLOCK_TRANSFER_SSPV
            4, // ABLOCK_TRANSFER_DSPV_K1
            S<8, 32, 1>, // BBLOCK_TRANSFER
            4, // BBLOCK_TRANSFER_SSPV
            4, // BBLOCK_TRANSFER_SSPV_K1
            1, // CSHUFFLE_MXDL_PWPS
            1,// CSHUFFLE_NXDL_PWPS
            S<1, 16, 1, 16>, // CSHUFFLEBLOCK_TRANSFER
            S<4>, // CDESHUFFLEBLOCK_TRANSFER
            ck::BlockGemmPipelineScheduler::Intrawave, // LOOP_SCHED
            ck::BlockGemmPipelineVersion::v4, // PIPELINE_VERSION
            ck::tensor_operation::device::GemmSpecialization::Default,
            true, // TRANS_A
            false>(CUDABLAS_BGEMM_ARGS(at::BFloat16));
    } else if (!transa_ && transb_) {
        bgemm_kernel_impl<
            ck::bhalf_t, // A_DATA_TYPE
            ck::bhalf_t, // B_DATA_TYPE
            256, // BLOCK_SIZE
            256, // M_BLOCK
            256, // N_BLOCK
            32, // K_BLOCK
            4, // AK1
            4, // BK1
            32, // WAVE_TILE_M
            32, // WAVE_TILE_N
            4, // WAVE_MAP_M
            4, // WAVE_MAP_N
            S<8, 32, 1>, // ABLOCK_TRANSFER
            4, // ABLOCK_TRANSFER_SSPV
            4, // ABLOCK_TRANSFER_DSPV_K1
            S<8, 32, 1>, // BBLOCK_TRANSFER
            4, // BBLOCK_TRANSFER_SSPV
            4, // BBLOCK_TRANSFER_SSPV_K1
            1, // CSHUFFLE_MXDL_PWPS
            1,// CSHUFFLE_NXDL_PWPS
            S<1, 16, 1, 16>, // CSHUFFLEBLOCK_TRANSFER
            S<4>, // CDESHUFFLEBLOCK_TRANSFER
            ck::BlockGemmPipelineScheduler::Intrawave, // LOOP_SCHED
            ck::BlockGemmPipelineVersion::v4, // PIPELINE_VERSION
            ck::tensor_operation::device::GemmSpecialization::Default,
            false, // TRANS_A
            true>(CUDABLAS_BGEMM_ARGS(at::BFloat16));
    } else {
        bgemm_kernel_impl<
            ck::bhalf_t, // A_DATA_TYPE
            ck::bhalf_t, // B_DATA_TYPE
            256, // BLOCK_SIZE
            256, // M_BLOCK
            256, // N_BLOCK
            32, // K_BLOCK
            4, // AK1
            4, // BK1
            32, // WAVE_TILE_M
            32, // WAVE_TILE_N
            4, // WAVE_MAP_M
            4, // WAVE_MAP_N
            S<8, 32, 1>, // ABLOCK_TRANSFER
            4, // ABLOCK_TRANSFER_SSPV
            4, // ABLOCK_TRANSFER_DSPV_K1
            S<8, 32, 1>, // BBLOCK_TRANSFER
            4, // BBLOCK_TRANSFER_SSPV
            4, // BBLOCK_TRANSFER_SSPV_K1
            1, // CSHUFFLE_MXDL_PWPS
            1,// CSHUFFLE_NXDL_PWPS
            S<1, 16, 1, 16>, // CSHUFFLEBLOCK_TRANSFER
            S<4>, // CDESHUFFLEBLOCK_TRANSFER
            ck::BlockGemmPipelineScheduler::Intrawave, // LOOP_SCHED
            ck::BlockGemmPipelineVersion::v4, // PIPELINE_VERSION
            ck::tensor_operation::device::GemmSpecialization::Default,
            false, // TRANS_A
            false>(CUDABLAS_BGEMM_ARGS(at::BFloat16));
    }
}
};