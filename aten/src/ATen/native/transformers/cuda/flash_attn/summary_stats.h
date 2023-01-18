/******************************************************************************
 * Copyright (c) 2022, Tri Dao.
 ******************************************************************************/

#pragma once

namespace fmha {

////////////////////////////////////////////////////////////////////////////////////////////////////

template<int kRows, int kRowsPerMma, int kWarpCountM>
struct Smem_tile_softmax_lse {

    static constexpr int kMmaM = (kRows / kWarpCountM) / kRowsPerMma;
    static_assert(kMmaM * kRowsPerMma * kWarpCountM == kRows, "");
    // static_assert(kWarpCountM == 1);
    // Otherwise we might need to check warp_idx / kWarpCountM == 0 instead of just warp_idx == 0

    // The size of one buffer in bytes in shared memory.
    static constexpr size_t BYTES_PER_TILE = kRows * sizeof(float);

    inline __device__ Smem_tile_softmax_lse(float *smem) : smem_(smem) {
    }

    inline __device__ void store_pair(const float (&sum)[kMmaM * 2]) {
        // Broadcast the warp_id computed by lane 0 to ensure dependent code
        // is compiled as warp-uniform.
        // This makes a difference of 50us for BERT.
        // const int warp_idx = threadIdx.x / 32;
        const int warp_idx = __shfl_sync(0xffffffff, threadIdx.x / 32, 0);
        const int lane_idx =  threadIdx.x % 32;
        const int warp_n = warp_idx / kWarpCountM;
        // Extract the position in the warp.
        const int row = lane_idx / 4;
        if ((lane_idx % 4 == 0) && (warp_n == 0)) {
            #pragma unroll
            for (int mi = 0; mi < kMmaM; ++mi) {
                smem_[mi * kRowsPerMma + row + 0] = sum[mi * 2 + 0];
                smem_[mi * kRowsPerMma + row + 8] = sum[mi * 2 + 1];
            }
        }
    }

    template<int N>
    inline __device__ void load(float (&sum)[N], const int (&row)[N]) {
        #pragma unroll
        for( int ni = 0; ni < N; ni++ ) {
            sum[ni] = smem_[row[ni]];
        }
    }

    float * const smem_;
};

}  // namespace fmha
