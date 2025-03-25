#undef __HIP_NO_HALF_CONVERSIONS__

#include <ATen/native/hip/bgemm_kernels/bgemm_kernel_template.h>

namespace at::native {

void bgemm_kernel_bf16bf16bf16_256_16x256x64_16x16_1x4_8x16x1_8x16x1_1x16x1x16_4_Intrawave_v2(CUDABLAS_BGEMM_ARGTYPES(at::BFloat16)) {
    bool transa_ = std::tolower(transa) != 'n';
    bool transb_ = std::tolower(transb) != 'n';
    if (transa_ && transb_) {
        bgemm_kernel_impl<
            ck::bhalf_t, // A_DATA_TYPE
            ck::bhalf_t, // B_DATA_TYPE
            256, // BLOCK_SIZE
            16, // M_BLOCK
            256, // N_BLOCK
            64, // K_BLOCK
            8, // AK1
            8, // BK1
            16, // WAVE_TILE_M
            16, // WAVE_TILE_N
            1, // WAVE_MAP_M
            4, // WAVE_MAP_N
            S<8, 16, 1>, // ABLOCK_TRANSFER
            8, // ABLOCK_TRANSFER_SSPV
            8, // ABLOCK_TRANSFER_DSPV_K1
            S<8, 16, 1>, // BBLOCK_TRANSFER
            8, // BBLOCK_TRANSFER_SSPV
            8, // BBLOCK_TRANSFER_SSPV_K1
            1, // CSHUFFLE_MXDL_PWPS
            1,// CSHUFFLE_NXDL_PWPS
            S<1, 16, 1, 16>, // CSHUFFLEBLOCK_TRANSFER
            S<4>, // CDESHUFFLEBLOCK_TRANSFER
            ck::BlockGemmPipelineScheduler::Intrawave, // LOOP_SCHED
            ck::BlockGemmPipelineVersion::v2, // PIPELINE_VERSION
            ck::tensor_operation::device::GemmSpecialization::Default,
            true, // TRANS_A
            true>(CUDABLAS_BGEMM_ARGS(at::BFloat16));
    } else if (transa_ && !transb_) {
        bgemm_kernel_impl<
            ck::bhalf_t, // A_DATA_TYPE
            ck::bhalf_t, // B_DATA_TYPE
            256, // BLOCK_SIZE
            16, // M_BLOCK
            256, // N_BLOCK
            64, // K_BLOCK
            8, // AK1
            8, // BK1
            16, // WAVE_TILE_M
            16, // WAVE_TILE_N
            1, // WAVE_MAP_M
            4, // WAVE_MAP_N
            S<8, 16, 1>, // ABLOCK_TRANSFER
            8, // ABLOCK_TRANSFER_SSPV
            8, // ABLOCK_TRANSFER_DSPV_K1
            S<8, 16, 1>, // BBLOCK_TRANSFER
            8, // BBLOCK_TRANSFER_SSPV
            8, // BBLOCK_TRANSFER_SSPV_K1
            1, // CSHUFFLE_MXDL_PWPS
            1,// CSHUFFLE_NXDL_PWPS
            S<1, 16, 1, 16>, // CSHUFFLEBLOCK_TRANSFER
            S<4>, // CDESHUFFLEBLOCK_TRANSFER
            ck::BlockGemmPipelineScheduler::Intrawave, // LOOP_SCHED
            ck::BlockGemmPipelineVersion::v2, // PIPELINE_VERSION
            ck::tensor_operation::device::GemmSpecialization::Default,
            true, // TRANS_A
            false>(CUDABLAS_BGEMM_ARGS(at::BFloat16));
    } else if (!transa_ && transb_) {
        bgemm_kernel_impl<
            ck::bhalf_t, // A_DATA_TYPE
            ck::bhalf_t, // B_DATA_TYPE
            256, // BLOCK_SIZE
            16, // M_BLOCK
            256, // N_BLOCK
            64, // K_BLOCK
            8, // AK1
            8, // BK1
            16, // WAVE_TILE_M
            16, // WAVE_TILE_N
            1, // WAVE_MAP_M
            4, // WAVE_MAP_N
            S<8, 16, 1>, // ABLOCK_TRANSFER
            8, // ABLOCK_TRANSFER_SSPV
            8, // ABLOCK_TRANSFER_DSPV_K1
            S<8, 16, 1>, // BBLOCK_TRANSFER
            8, // BBLOCK_TRANSFER_SSPV
            8, // BBLOCK_TRANSFER_SSPV_K1
            1, // CSHUFFLE_MXDL_PWPS
            1,// CSHUFFLE_NXDL_PWPS
            S<1, 16, 1, 16>, // CSHUFFLEBLOCK_TRANSFER
            S<4>, // CDESHUFFLEBLOCK_TRANSFER
            ck::BlockGemmPipelineScheduler::Intrawave, // LOOP_SCHED
            ck::BlockGemmPipelineVersion::v2, // PIPELINE_VERSION
            ck::tensor_operation::device::GemmSpecialization::Default,
            false, // TRANS_A
            true>(CUDABLAS_BGEMM_ARGS(at::BFloat16));
    } else {
        bgemm_kernel_impl<
            ck::bhalf_t, // A_DATA_TYPE
            ck::bhalf_t, // B_DATA_TYPE
            256, // BLOCK_SIZE
            16, // M_BLOCK
            256, // N_BLOCK
            64, // K_BLOCK
            8, // AK1
            8, // BK1
            16, // WAVE_TILE_M
            16, // WAVE_TILE_N
            1, // WAVE_MAP_M
            4, // WAVE_MAP_N
            S<8, 16, 1>, // ABLOCK_TRANSFER
            8, // ABLOCK_TRANSFER_SSPV
            8, // ABLOCK_TRANSFER_DSPV_K1
            S<8, 16, 1>, // BBLOCK_TRANSFER
            8, // BBLOCK_TRANSFER_SSPV
            8, // BBLOCK_TRANSFER_SSPV_K1
            1, // CSHUFFLE_MXDL_PWPS
            1,// CSHUFFLE_NXDL_PWPS
            S<1, 16, 1, 16>, // CSHUFFLEBLOCK_TRANSFER
            S<4>, // CDESHUFFLEBLOCK_TRANSFER
            ck::BlockGemmPipelineScheduler::Intrawave, // LOOP_SCHED
            ck::BlockGemmPipelineVersion::v2, // PIPELINE_VERSION
            ck::tensor_operation::device::GemmSpecialization::Default,
            false, // TRANS_A
            false>(CUDABLAS_BGEMM_ARGS(at::BFloat16));
    }
}
};