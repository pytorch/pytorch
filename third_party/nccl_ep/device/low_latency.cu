/*
 * Portions of this file are adapted from DeepEP (https://github.com/deepseek-ai/DeepEP).
 * Copyright (c) 2025 DeepSeek. Licensed under the MIT License.
 * SPDX-License-Identifier: MIT
 */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 * See LICENSE.txt for more license information.
 */

#include <cooperative_groups.h>
#include "nccl_device.h"
#include "device_primitives.cuh"

namespace cg = cooperative_groups;

namespace nccl_ep {

namespace internode_ll {
// Mask convention: 1 = active, 0 = masked/failed. nullptr means masking disabled.
template <bool useWarpSync = false>
__forceinline__ __device__ bool isRankMasked(int* rankMask, int rank) {
    if (rankMask == nullptr) {
        return false;
    }
    if constexpr (useWarpSync) {
        return __shfl_sync(0xffffffff, ld_acquire_global(rankMask + rank), 0) == 0;
    } else {
        return ld_acquire_global(rankMask + rank) == 0;
    }
}
__device__ __constant__ bool dP2pDisabled = false;

__device__ __forceinline__ int getCommId(int hashKey) {
    return hashKey / MAX_NCCL_GIN_CTX_PER_COMM;
}

__device__ __forceinline__ int getCtxId(int hashKey) {
    return hashKey % MAX_NCCL_GIN_CTX_PER_COMM;
}

__device__ __forceinline__ int getLocalExpertIdx(int expertIdx, int numLocalExperts) {
    return (expertIdx >= 0) ? expertIdx % numLocalExperts : -1;
}

__device__ __forceinline__ int getExpertRankIdx(int expertIdx, int numLocalExperts) {
    return (expertIdx >= 0) ? expertIdx / numLocalExperts : -1;
}

// Warp-cooperative: all 32 lanes must call together.
// Each lane inspects its own topk index and matches it with others via __match_any_sync.
// If this lane is not the first occurrence of its source rank, topkIdxByLane is dropped to -1.
__device__ __forceinline__
int warpMarkFirstOccurrence(int topkIdxByLane, int numLocalExperts, int laneId) {
    int rank = getExpertRankIdx(topkIdxByLane, numLocalExperts);
    uint32_t mask = __match_any_sync(0xffffffff, rank);
    bool isFirst = (laneId == (__ffs(mask) - 1));
    return topkIdxByLane * (int)isFirst - (int)(!isFirst);
}

__device__ __forceinline__ void syncSmGroup(int groupIdx, int nThreads) {
    asm volatile("bar.sync %0, %1;" :: "r"(groupIdx), "r"(nThreads));
}

#define SYNC_DISP_SEND_COPY 1
#define SYNC_DISP_SEND_COMM(idx) (SYNC_DISP_SEND_COPY + idx)
#define SYNC_DISP_RECV_COPY(topK, idx) (SYNC_DISP_SEND_COMM(topK) + idx)

__forceinline__ __device__ uint64_t ncclGetP2pPtr(const uint64_t& dstPtr,
    const size_t& offset,
    const int& rank,
    const int& dstRank,
    const ncclWindow_t* ncclWindows,
    ncclDevComm* devComms) {
    // Local rank, no need for peer mapping
    if (rank == dstRank) {
        return dstPtr;
    }

    // If P2P is globally disabled, always use RDMA path
    if (dP2pDisabled) {
        return 0;
    }

    // P2P/NVLink only works between ranks on the same node (LSA team)
    // Use NCCL team APIs to check if dstRank is in the same LSA team.
    // Always use commId=0: single devComm with all GIN contexts (1-comm N-context design).
    constexpr int commId = 0;
    ncclTeam lsa = ncclTeamLsa(devComms[commId]);
    ncclTeam world = ncclTeamWorld(devComms[commId]);
    if (!ncclTeamRankIsMember(lsa, world, dstRank))
        return 0;  // Different nodes (not in same LSA team), must use RDMA

    auto const p2pPtr = reinterpret_cast<uint64_t>(ncclGetPeerPointer(ncclWindows[commId],
                                                                       offset, dstRank));

    return p2pPtr ? p2pPtr : 0;
}

// ============================================================================
// Helper functions for dispatch kernel modularization
// ============================================================================

// Cast BF16 data to FP8 and write to send buffer, or copy BF16 data directly
template<bool kUseFP8>
__forceinline__ __device__ void castAndWriteToSendBuf(
    const int4* srcDataInt4,
    typename std::conditional<kUseFP8, int2, int4>::type* sendBufVec,
    float* sendBufScales,
    int threadId,
    int numThreads,
    int laneId,
    int hiddenBf16Int4,
    bool roundScale) {
    constexpr int kNumElemsPerRead = sizeof(int4) / sizeof(nv_bfloat16);
    constexpr int kNumPerChannels = 128;

    EP_DEVICE_ASSERT(hiddenBf16Int4 % 32 == 0);
    #pragma unroll
    for (int i = threadId; i < hiddenBf16Int4; i += numThreads) {
        auto dataInt4 = __ldg(srcDataInt4 + i);

        if constexpr (kUseFP8) {
            // Calculate local amax
            auto bf16Data = reinterpret_cast<nv_bfloat16*>(&dataInt4);
            float fp32Data[kNumElemsPerRead];
            float amax = kFP8Margin, scale, scaleInv;
            #pragma unroll
            for (int j = 0; j < kNumElemsPerRead; ++j) {
                fp32Data[j] = static_cast<float>(bf16Data[j]);
                amax = fmaxf(amax, fabsf(fp32Data[j]));
            }

            // Reduce amax and scale
            EP_STATIC_ASSERT(kNumElemsPerRead * 32 / kNumPerChannels == 2, "Invalid vectorization");
            amax = warp_reduce_max<16>(amax);
            calculate_fp8_scales(amax, scale, scaleInv, roundScale);
            if (laneId == 0 or laneId == 16)
                sendBufScales[i * kNumElemsPerRead / 128] = scaleInv;

            // Cast into send buffer
            typename std::conditional<kUseFP8, int2, int4>::type dataInt2;
            auto fp8x2Data = reinterpret_cast<__nv_fp8x2_storage_t*>(&dataInt2);
            #pragma unroll
            for (int j = 0; j < kNumElemsPerRead; j += 2) {
                float2 fp32x2 = {fp32Data[j] * scale, fp32Data[j + 1] * scale};
                fp8x2Data[j / 2] = __nv_cvt_float2_to_fp8x2(fp32x2, __NV_SATFINITE, __NV_E4M3);
            }
            sendBufVec[i] = dataInt2;
        } else {
            // Reinterpret-cast is for C++14 compatibility
            sendBufVec[i] = *reinterpret_cast<typename std::conditional<kUseFP8, int2, int4>::type*>(&dataInt4);
        }
    }
}

// Send a token to one peer.
//
// Picks the transport per dstRank:
//
//   - Intra-LSA peer (P2P reachable): write directly into the peer's recvBuf
//     using the NVLink split layout — the per-slot header is copied from this
//     rank's local staging slot into the peer's per-srcRank header section,
//     and the payload is cast from `inData` directly into the per-srcRank
//     payload section. The staging slot's data half is not read on this path.
//
//   - Cross-LSA peer (RDMA): gin.put the full per-slot interleaved
//     [hdr | data | scales] message from the local staging slot into the
//     peer's recvBuf at the corresponding per-slot offset.
//
// Templated on `kUseFP8` so the NVLink direct path can reuse
// `castAndWriteToSendBuf` for the quantized-vs-bf16 split.
template <bool kUseFP8>
__forceinline__ __device__ void sendToken(
    // Local sources.
    const int4* sendDataInt4,        // local staging slot base (header + RDMA-path payload).
    const int4* srcDataInt4,         // input bf16 data for this token (NVLink-path payload source).
    // Peer destination addressing: per-srcRank region base + slot index.
    uint64_t srcRankRegionLocalPtr,  // sender-view pointer at peer's per-srcRank region.
    size_t srcRankRegionOffset,      // window offset of that per-srcRank region.
    int slotIdx,                     // slot within the per-srcRank region.
    // Sender-side window addressing (RDMA source).
    int tokenIdx,
    size_t sendOff,
    // Layout / sizing.
    size_t numBytesPerMsg,
    size_t dispatch_hdr_sz,
    size_t hiddenBytes,
    size_t hiddenBf16Int4,
    int maxTokensPerRank,
    // Misc.
    int dstRank,
    int hashKey,
    int currRank,
    bool roundScale,
    int* rankMask,
    const ncclWindow_t* windows,
    ncclDevComm* devComms,
    int laneId) {
    using vec_t = std::conditional_t<kUseFP8, int2, int4>;

    const auto dstSrcRankP2pPtr = ncclGetP2pPtr(
        srcRankRegionLocalPtr, srcRankRegionOffset, currRank, dstRank, windows, devComms);

    if (isRankMasked<true>(rankMask, dstRank)) return;

    if (dstSrcRankP2pPtr != 0) {
        // ---- NVLink direct path (split layout) ----
        const size_t hdrSectionBytes =
            static_cast<size_t>(maxTokensPerRank) * dispatch_hdr_sz;
        const size_t payloadBytes = numBytesPerMsg - dispatch_hdr_sz;
        const int numHdrInt4 = static_cast<int>(dispatch_hdr_sz / sizeof(int4));

        auto* dstSrcRankBase = reinterpret_cast<uint8_t*>(dstSrcRankP2pPtr);
        auto* dstHdrSlot = dstSrcRankBase + slotIdx * dispatch_hdr_sz;
        auto* dstPayloadSlot = dstSrcRankBase + hdrSectionBytes + slotIdx * payloadBytes;

        int4* dstHdrInt4 = reinterpret_cast<int4*>(dstHdrSlot);
        for (int i = laneId; i < numHdrInt4; i += 32) {
            st_na_global(dstHdrInt4 + i, sendDataInt4[i]);
        }

        auto* dstDataVec = reinterpret_cast<vec_t*>(dstPayloadSlot);
        auto* dstScales = reinterpret_cast<float*>(dstPayloadSlot + hiddenBytes);
        castAndWriteToSendBuf<kUseFP8>(
            srcDataInt4, dstDataVec, dstScales,
            /*threadId=*/laneId, /*numThreads=*/32,
            laneId, hiddenBf16Int4, roundScale);
    } else {
        // ---- RDMA path (interleaved layout) ----
        if (laneId == 0) {
            const size_t perSlotDstOffset = srcRankRegionOffset + slotIdx * numBytesPerMsg;
            const size_t expectedSrcOffset = sendOff + tokenIdx * numBytesPerMsg;
            constexpr int commId = 0;
            auto ctxId = getCtxId(hashKey);
            ncclGin net(devComms[commId], ctxId);
            ncclTeam world = ncclTeamWorld(devComms[commId]);
            auto ncclWindow = windows[commId];
            net.put(world,
                    dstRank,
                    ncclWindow,
                    perSlotDstOffset,
                    ncclWindow,
                    expectedSrcOffset,
                    numBytesPerMsg,
                    ncclGin_None{},  // no signal
                    ncclGin_None{},  // no counter
                    ncclCoopThread());
        }
    }
}

// Clean next receive count buffer
__forceinline__ __device__ void cleanNextRecvCntBuf(
    int* nextRecvCntBuf,
    int nextRecvCntBufSize,
    int laneId) {
    if (!dP2pDisabled) {
        #pragma unroll
        for (int i = laneId; i < nextRecvCntBufSize; i += 32)
            nextRecvCntBuf[i] = 0;
    }
}

// This function is very efficient and outperforms orignal expert counting
// one (that it is replacing) even though it is more complex.
__forceinline__ __device__ void
countTokensPerRank_mask(
    const int64_t* inTopkIdx,
    int numTokens,
    int numTopk,
    int numLocalExperts,
    int rankBeginIdx,
    int rankEndIdx,
    int* rankCount,
    uint64_t* rankMap,
    int* sharedRankCount,
    int laneId) {

    const int batchSize = 32;
    const int rankRangeSize = rankEndIdx - rankBeginIdx;
    assert(rankRangeSize <= 8 * sizeof(uint64_t));

    // Per lane count

    int batchIdx = 0;

    for (int i = laneId; i < numTokens; i += 32) {
      // Scan all topK indices and store in rankMap
      #pragma unroll 8
      for(int k = 0; k < numTopk; k++){
        auto idx = static_cast<int>(__ldg(inTopkIdx + i * numTopk + k));
        auto rankIdx = getExpertRankIdx(idx, numLocalExperts);
        if(rankIdx < 0) {
          continue;
        }
        bool rankInRange = rankIdx >= rankBeginIdx and rankIdx < rankEndIdx;
        rankMap[batchIdx] |= rankInRange * (1 << (rankIdx - rankBeginIdx));
      }
      batchIdx++;
      if (batchIdx == batchSize) {
        for(int j = 0; j < batchSize; ++j) {
          #pragma unroll 8
          for(int k = 0; k < rankRangeSize; k++){
            if (rankMap[j] & (1 << k)) {
              rankCount[k]++;
            }
          }
          // Clear the rankMap for the next batch
          rankMap[j] = 0;
        }
        batchIdx = 0;
      }
    }

    for(int j = 0; j < batchIdx; ++j) {
      #pragma unroll 8
      for(int k = 0; k < rankRangeSize; k++){
        if (rankMap[j] & (1 << k)) {
          rankCount[k]++;
        }
        // no need to clear the rankMap for the last batch
      }
    }

    // Warp reduce
    #pragma unroll
    for (int i = rankBeginIdx; i < rankEndIdx; ++i) {
        auto sum = warp_reduce_sum(rankCount[i - rankBeginIdx]);
        if (laneId == 0) {
          sharedRankCount[i - rankBeginIdx] = sum;
        }
    }
}

// Send expert count to destination rank
__forceinline__ __device__ void sendExpertCount(
    int numTokensSent,
    int dstRank,
    int dstExpertLocalIdx,
    int currRank,
    int numRanks,
    uint64_t recvCntPtr,
    size_t recvCntOffset,
    int* recvCntBuf,
    int* rankMask,
    unsigned signalsBase,
    const ncclWindow_t* windows,
    ncclDevComm* devComms) {
    const auto dstP2pPtr = ncclGetP2pPtr(recvCntPtr, recvCntOffset, currRank, dstRank,
                                         windows, devComms);

    if (not isRankMasked(rankMask, dstRank)) {
        if (dstP2pPtr == 0) {
            constexpr int commId = 0;
            auto ctxId = getCtxId(dstExpertLocalIdx);
            auto signalId = signalsBase + dstExpertLocalIdx * numRanks + currRank;
            ncclGin net(devComms[commId], ctxId);
            ncclTeam world = ncclTeamWorld(devComms[commId]);
            auto ncclWindow = windows[commId];
            net.put(world,
                    dstRank,
                    ncclWindow,
                    recvCntOffset,
                    ncclWindow,
                    0,
                    0,  // 0 bytes transfer
                    ncclGin_SignalAdd{signalId, static_cast<uint64_t>(numTokensSent) + 1},
                    ncclGin_None{},  // no counter
                    ncclCoopThread());
        } else {
            st_release_sys_global(reinterpret_cast<int*>(dstP2pPtr), -numTokensSent - 1);
        }
    }
}

// Wait for receive tokens to arrive and return count
// Note that this function doesn't guarantee acquire semantics
__forceinline__ __device__ int waitForRecvTokensRelaxed(
    int srcRank,
    int rankLaneIdx,
    int currRank,
    int numRanks,
    size_t recvCntOff,
    int* recvCntBuf,
    int* rankMask,
    int* asyncErrorFlag,
    unsigned signalsBase,
    const ncclWindow_t* windows,
    ncclDevComm* devComms,
    int* recvStats,
    int64_t* waitStats,
    uint64_t timeoutCycles) {
    auto startTime = clock64();
    uint64_t waitRecvCost = 0;
    int numRecvTokens = 0;

    if (not isRankMasked(rankMask, srcRank)) {
        size_t srcOffset = recvCntOff + (rankLaneIdx * numRanks + srcRank) * sizeof(int);
        auto srcP2pPtr = ncclGetP2pPtr(0x01, srcOffset, currRank, srcRank, windows, devComms);

        if (srcP2pPtr == 0) {
            constexpr int commId = 0;
            auto ctxId = getCtxId(rankLaneIdx);
            ncclGin net(devComms[commId], ctxId);

            uint64_t curValue;
            ncclGinSignal_t signalId = signalsBase + rankLaneIdx * numRanks + srcRank;
            do {
                curValue = net.readSignal(signalsBase + rankLaneIdx * numRanks + srcRank);
            } while (curValue < 1                                                       // data not arrived
                     && (waitRecvCost = clock64() - startTime) <= timeoutCycles  // not timeout
            );
            net.resetSignal(signalId);
            numRecvTokens = -(int)curValue;
        } else {
            // TODO: Double check that we can rely on this + __threadfence_system() in dispatch?
            // to ensure consistency on "another" SM that's going to access the data buffer protected by this atomic
            while ((numRecvTokens = ld_acquire_sys_global((recvCntBuf + rankLaneIdx * numRanks + srcRank))) ==
                       0                                                               // data not arrived
                   && (waitRecvCost = clock64() - startTime) <= timeoutCycles  // not timeout
                  );
        }
    }

    // Do not receive tokens if rank timeout or masked
    if (numRecvTokens == 0)
        numRecvTokens = -1;

    // Mask rank if timeout
    if (waitRecvCost > timeoutCycles) {
        printf("Warning: NCCL EP timeout for dispatch receive, rank %d, local_expert_idx %d, src_rank %d\n",
               currRank, rankLaneIdx, srcRank);
        if (rankMask == nullptr)
            trap();
        atomicExch(rankMask + srcRank, 0);
        if (asyncErrorFlag != nullptr)
            atomicExch_system(asyncErrorFlag, 1);
    }

    numRecvTokens = -numRecvTokens - 1;

    // Add stats for diagnosis
    if (recvStats != nullptr)
        atomicAdd(recvStats + rankLaneIdx, numRecvTokens);
    if (waitStats != nullptr)
        atomicAdd(reinterpret_cast<unsigned long long*>(waitStats + srcRank), waitRecvCost);

    return numRecvTokens;
}

// Copy received token data and scales.
// `isNvlinkSrc` selects the wire layout. When true, the sender (an intra-LSA
// peer) used the NVLink split layout: all per-slot headers at the head of the
// per-srcRank region followed by all per-slot payloads. When false, the sender
// (a cross-LSA RDMA peer) used the legacy interleaved layout where each
// per-slot message is [hdr | data | scales].
template<bool kUseFP8, bool kUseUE8M0>
__forceinline__ __device__ void copyRecvTokenData(
    const uint8_t* recvBufUint8,
    int recvIdx,
    int tokenIdx,
    int4* outDataInt4,
    typename std::conditional<kUseUE8M0, uint8_t, float>::type* outScales,
    int hiddenInt4,
    int hiddenBytes,
    int numScales,
    int numBytesPerMsg,
    int dispatch_hdr_sz,
    int maxTokensPerRank,
    int numRanks,
    int laneId,
    bool isNvlinkSrc) {
    using scale_t = std::conditional_t<kUseUE8M0, uint8_t, float>;
    using packed_t = std::conditional_t<kUseUE8M0, uint32_t, float>;

    // Locate the payload (data + optional FP8 scales) for this slot.
    const int payloadBytes = numBytesPerMsg - dispatch_hdr_sz;
    const uint8_t* recvPayloadPtr;
    if (isNvlinkSrc) {
        // Split layout: payload section follows the per-srcRank header section.
        recvPayloadPtr = recvBufUint8 + maxTokensPerRank * dispatch_hdr_sz +
                         recvIdx * payloadBytes;
    } else {
        // Interleaved layout: payload sits right after the per-slot header.
        recvPayloadPtr = recvBufUint8 + recvIdx * numBytesPerMsg + dispatch_hdr_sz;
    }

    // Copy data
    // NOTES: only 2 load iterations for 7K hidden with 7 unrolls
    const auto recvDataInt4 = reinterpret_cast<const int4*>(recvPayloadPtr);

    const auto outDataInt4Ptr = outDataInt4 + tokenIdx * hiddenInt4;
    UNROLLED_WARP_COPY(7, laneId, hiddenInt4, outDataInt4Ptr, recvDataInt4, ld_nc_global, st_na_global);

    // Copy scales
    if constexpr (kUseFP8) {
        // Equivalent CuTe layout:
        //   (num_tokens, (num_packed, num_elems_per_pack)):(num_elems_per_pack, (num_tokens * num_elems_per_pack, 1))
        const auto recvScales = reinterpret_cast<const float*>(reinterpret_cast<const uint8_t*>(recvDataInt4) + hiddenBytes);
        const auto numElemsPerPack = static_cast<int>(sizeof(packed_t) / sizeof(scale_t));
        const auto tokenStride = numElemsPerPack;
        const auto packStride = numRanks * maxTokensPerRank * numElemsPerPack;
        if (laneId < numScales) {
            const auto packIdx = laneId / numElemsPerPack;
            const auto elemIdx = laneId % numElemsPerPack;
            auto scale = extract_required_scale_format<kUseUE8M0>(ld_nc_global(recvScales + laneId));
            outScales[tokenIdx * tokenStride + packIdx * packStride + elemIdx] = scale;
        }
        if (laneId + 32 < numScales) {
            const auto packIdx = (laneId + 32) / numElemsPerPack;
            const auto elemIdx = (laneId + 32) % numElemsPerPack;
            auto scale = extract_required_scale_format<kUseUE8M0>(ld_nc_global(recvScales + laneId + 32));
            outScales[tokenIdx * tokenStride + packIdx * packStride + elemIdx] = scale;
        }
    }
}

template <bool kUseFP8, bool kUseUE8M0, int kHidden, ncclEpLayout_t kLayout, bool kNvlinkOnly>
__global__ __launch_bounds__(1024, 1) void dispatch(// INPUT
                                                    const void* inData,
                                                    const int64_t* inTopkIdx,
                                                    const float* inTopkWeights,
                                                    int* rankMask,
                                                    int* asyncErrorFlag,
                                                    // OUTPUT
                                                    void* outDataBuf,
                                                    void* outScalesBuf,
                                                    int* outSrcInfo,
                                                    int* outRecvRankCounter,
                                                    int64_t* outLayout,
                                                    int* outCnt,
                                                    float* outRecvTopkWeights,
                                                    int32_t* outRecvTopkIdx,
                                                    // INTERMEDIATE
                                                    void* sendBuf,
                                                    void* recvBuf,
                                                    int* recvCntBuf,
                                                    size_t sendOff,
                                                    size_t recvOff,
                                                    size_t recvCntOff,
                                                    int* rankCountersBase,
                                                    int* rankDone,
                                                    int* nextRecvCntBuf,
                                                    int nextRecvCntBufSize,
                                                    int* recvStats,
                                                    int64_t* waitStats,
                                                    // CONFIG
                                                    int numTokens,
                                                    int maxTokensPerRank,
                                                    int numTopk,
                                                    int numExperts,
                                                    int currRank,
                                                    int numRanks,
                                                    int numWarpGroups,
                                                    int numWarpsPerGroup,
                                                    bool roundScale,
                                                    int phases,
                                                    int numComms,
                                                    ncclDevComm* devComms,
                                                    const ncclWindow_t* windows,
                                                    unsigned signalsBase,
                                                    uint64_t timeoutCycles) {
    const auto smId = static_cast<int>(blockIdx.x);
    const auto threadId = static_cast<int>(threadIdx.x);
    const auto warpId = threadId / 32, laneId = get_lane_id();
    const auto numSms = static_cast<int>(gridDim.x);
    const auto numWarps = numWarpGroups * numWarpsPerGroup;
    const auto numLocalExperts = numExperts / numRanks;
    const auto warpGroupId = warpId / numWarpsPerGroup;
    const auto subWarpId = warpId % numWarpsPerGroup;
    const auto responsibleExpertIdx = smId * numWarpGroups + warpGroupId;

    auto rankSentCnt = rankCountersBase;
    auto rankArrivedCnt = rankCountersBase + numRanks;

    // May extract UE8M0 from the scales
    using scale_t = std::conditional_t<kUseUE8M0, uint8_t, float>;
    using packed_t = std::conditional_t<kUseUE8M0, uint32_t, float>;
    EP_STATIC_ASSERT(sizeof(packed_t) % sizeof(scale_t) == 0, "Invalid vector length");

    // FP8 staffs
    constexpr int kNumPerChannels = 128;
    const int numScales = kHidden / kNumPerChannels;
    const size_t hiddenBytes = kHidden * (kUseFP8 ? sizeof(__nv_fp8_storage_t) : sizeof(nv_bfloat16));
    const size_t hiddenInt4 = hiddenBytes / sizeof(int4);

    // Message package: header, hidden data, FP8 scales
    // NOTES: header contains token_id + expert_id per topk
    using vec_t = std::conditional_t<kUseFP8, int2, int4>;
    const size_t dispatch_hdr_sz = get_dispatch_hdr_sz<kLayout>(numTopk);
    const size_t numBytesPerMsg = dispatch_hdr_sz + (kUseFP8 ?
                                                     (kHidden + numScales * sizeof(float)) :
                                                     (kHidden * sizeof(nv_bfloat16)));

    EP_DEVICE_ASSERT(numBytesPerMsg % sizeof(int4) == 0);

    // Rank counts
    constexpr int kNumMaxWarpGroups = 32;
    __shared__ int sharedNumTokensSentPerRank[kNumMaxWarpGroups];

    // Sending phase
    if ((phases & LOW_LATENCY_SEND_PHASE) == 0)
        goto LOW_LATENCY_DISPATCH_RECV;



    // There are 2 kinds of warps in this part:
    // 1. The first-kind warps for FP8 cast and sending top-k tokens
    // 2. The last warp for reading `topk_idx` and count for per-expert information
    if (warpId < numWarps - 1) {
        constexpr int kNumElemsPerRead = sizeof(int4) / sizeof(nv_bfloat16);
        EP_STATIC_ASSERT(kHidden % (32 * kNumElemsPerRead) == 0, "Invalid hidden");
        EP_STATIC_ASSERT(kNumElemsPerRead * 32 % kNumPerChannels == 0, "Invalid vectorization");
        const auto numWritingThreads = (numWarps - 1) * 32;
        const size_t hiddenBf16Int4 = kHidden / kNumElemsPerRead;

        // Peer's recvBuf is laid out as numRanks back-to-back regions of size
        // `maxTokensPerRank * numBytesPerMsg`. sendToken indexes within a
        // region using the slot index and picks NVLink (split layout) or
        // RDMA (interleaved layout) addressing internally.
        const size_t srcRankRegionBytes =
            static_cast<size_t>(maxTokensPerRank) * numBytesPerMsg;

        // Split token processing across SMs
        for (int tokenIdx = smId; tokenIdx < numTokens; tokenIdx += numSms) {
            const auto srcDataInt4 = static_cast<const int4*>(inData) + tokenIdx * hiddenBf16Int4;

            // Local staging slot. Header is always written here; data is
            // written here for the RDMA path only (NVLink writes data
            // directly to the peer without round-tripping through sendBuf).
            auto* sendBufBase = static_cast<uint8_t*>(sendBuf) + tokenIdx * numBytesPerMsg;
            const auto sendBufVec = reinterpret_cast<vec_t*>(sendBufBase + dispatch_hdr_sz);
            const auto sendBufScales = reinterpret_cast<float*>(reinterpret_cast<uint8_t*>(sendBufVec) + hiddenBytes);


            // Each expert is handled by a different warp in the SM.
            auto dstExpertIdx = warpId < numTopk ? static_cast<int>(__ldg(inTopkIdx + tokenIdx * numTopk + warpId)) : -1;
            auto dstRank = getExpertRankIdx(dstExpertIdx, numLocalExperts);

            // Write token_id and routing information into the local staging
            // slot. Lane 0 of warps 0..numTopk-1 contribute one rtr entry each.
            auto* sendBufHdr = reinterpret_cast<DispatchHdr<kLayout>*>(sendBufBase);
            if (warpId < numTopk and laneId == 0) {
                if (warpId == 0) {
                    sendBufHdr->token_id = tokenIdx;
                }
                sendBufHdr->rtr[warpId].expert_id = static_cast<uint16_t>(dstExpertIdx);
                if constexpr (kLayout == NCCL_EP_LAYOUT_RANK_MAJOR) {
                    sendBufHdr->rtr[warpId].topk_weight = __ldg(inTopkWeights + tokenIdx * numTopk + warpId);
                }
            }

            // Cast and write data to send buffer (consumed by the RDMA path).
            // NVLink dsts cast directly from inData instead, in the per-warp
            // send branch below. When kNvlinkOnly is set, every dst is
            // guaranteed to take the NVLink direct path so the staging-buffer
            // payload is never read and the cast can be skipped entirely.
            if constexpr (!kNvlinkOnly) {
                castAndWriteToSendBuf<kUseFP8>(
                    srcDataInt4, sendBufVec, sendBufScales,
                    threadId, numWritingThreads, laneId, hiddenBf16Int4, roundScale);
            }


            // Make sure that all working warps in the SM have completed the header writing.
            syncSmGroup(SYNC_DISP_SEND_COPY, numWritingThreads);

            // Do filtering to avoid duplicate sending of tokens to the same rank.
            if (dstExpertIdx >= 0) {
                // Optimized: rank-level slot allocation (aggregates across experts)
                int minTopkIdx = numTopk;
                for (int i = laneId; i < numTopk; i += 32) {
                    const auto otherExpertIdx = sendBufHdr->rtr[i].expert_id;
                    const auto otherRank = getExpertRankIdx(otherExpertIdx, numLocalExperts);
                    // if another expert is located on the same rank - disqualify this warp
                    // from sending the token to avoid duplication
                    if (otherRank == dstRank) {
                        minTopkIdx = min(minTopkIdx, i);
                    }
                }
                minTopkIdx = warp_reduce_min(minTopkIdx);

                // If this warp is the first in topK for the dstRank, send the token.
                if (minTopkIdx == warpId) {
                    int slotIdx = laneId == 0 ? atomicAdd(rankSentCnt + dstRank, 1) : 0;
                    slotIdx = __shfl_sync(0xffffffff, slotIdx, 0);

                    // evenly distribute sends by available channels
                    const auto ctxHash = slotIdx % numLocalExperts;
                    // Per-srcRank region base on the peer; sendToken picks the
                    // NVLink (split layout, direct from `inData`) or RDMA
                    // (interleaved, gin.put from staging) path internally.
                    const size_t srcRankOffset = currRank * srcRankRegionBytes;
                    const auto srcRankLocalPtr =
                        reinterpret_cast<uint64_t>(recvBuf) + srcRankOffset;
                    const auto sendBufInt4 = reinterpret_cast<const int4*>(sendBufBase);
                    sendToken<kUseFP8>(
                        sendBufInt4, srcDataInt4,
                        srcRankLocalPtr, recvOff + srcRankOffset, slotIdx,
                        tokenIdx, sendOff,
                        numBytesPerMsg, dispatch_hdr_sz, hiddenBytes, hiddenBf16Int4, maxTokensPerRank,
                        dstRank, ctxHash, currRank, roundScale,
                        rankMask, windows, devComms, laneId);
                    if (laneId == 0) {
                        // Mark that one more token is sent to the rank
                        atomic_add_release_global(rankDone + dstRank, 1);
                    }
                }
            }
        }
    } else if (warpId == numWarps - 1) {
        EP_DEVICE_ASSERT(numSms > 1);
        if (smId == 0) {
            cleanNextRecvCntBuf(nextRecvCntBuf, nextRecvCntBufSize, laneId);
            // Notify before executing `int_p`
            __syncwarp();
            #pragma unroll
            for (int i = laneId; i < numRanks; i += 32) {
                atomic_add_release_global(rankDone + i, FINISHED_SUM_TAG);
            }
        }



        // Count tokens per expert from topk indices
        constexpr int kNumMaxRanks = 32;
        int rankCount[kNumMaxRanks] = {0};
        uint64_t rankMap[kNumMaxRanks] = {0};

        const auto expertBeginIdx = smId * numWarpGroups;
        const auto expertEndIdx = min(expertBeginIdx + numWarpGroups, numExperts);

        // Note: it's possible that experts from the same rank are handled by multiple SMs
        // In this case, the counters will be calculated on each involved SM
        const int rankBeginIdx = expertBeginIdx / numLocalExperts;
        const int rankEndIdx = expertEndIdx / numLocalExperts + 1;

        countTokensPerRank_mask(
            inTopkIdx, numTokens, numTopk, numLocalExperts, rankBeginIdx, rankEndIdx,
            rankCount, rankMap, sharedNumTokensSentPerRank, laneId);

        // Note: however, only one warp updates the counter for each rank
        #pragma unroll
        for (int i = rankBeginIdx + laneId; i < rankEndIdx; i += 32) {
            int firstExpertIdx = i * numLocalExperts;
            if(firstExpertIdx >= expertBeginIdx and firstExpertIdx < expertEndIdx) {

                atomic_add_release_global(rankDone + i, FINISHED_SUM_TAG - sharedNumTokensSentPerRank[i - rankBeginIdx]);
            }
        }
    }

    __syncthreads();

    // Issue count sends
    if (responsibleExpertIdx < numExperts and subWarpId == 0 and laneId == 0) {
        const auto expertBeginIdx = smId * numWarpGroups;
        const int rankBeginIdx = expertBeginIdx / numLocalExperts;
        const auto dstRank = responsibleExpertIdx / numLocalExperts;
        const auto dstExpertLocalIdx = responsibleExpertIdx % numLocalExperts;
        const auto numTokensSent = sharedNumTokensSentPerRank[dstRank - rankBeginIdx];

        // Wait local sends issued and send expert counts
        while (ld_acquire_global(rankDone + dstRank) != FINISHED_SUM_TAG * 2);
        auto recvCntPtr = reinterpret_cast<uint64_t>(recvCntBuf + dstExpertLocalIdx * numRanks + currRank);
        size_t recvCntOffset = recvCntOff + (dstExpertLocalIdx * numRanks + currRank) * sizeof(int);
        // We are writing the rank counter (same value) via multiple expert-specific channels rank count to flush all previous tokens
        sendExpertCount(numTokensSent, dstRank, dstExpertLocalIdx, currRank, numRanks,
            recvCntPtr, recvCntOffset, recvCntBuf, rankMask, signalsBase,
            windows, devComms);

        // Clean `packed_recv_count` (expert-major only: outCnt is the per-expert slot allocator)
        if constexpr (kLayout == NCCL_EP_LAYOUT_EXPERT_MAJOR) {
            if (dstRank == 0)
                outCnt[dstExpertLocalIdx] = 0;
        }
    }
    __syncwarp();

    // Resent dispatch/recvcounters for the next dispatch
    if (responsibleExpertIdx < numRanks) {
        rankArrivedCnt[responsibleExpertIdx] = 0;
    }

// Receiving phase
LOW_LATENCY_DISPATCH_RECV:
    if ((phases & LOW_LATENCY_RECV_PHASE) == 0)
        return;

    // For send-and-recv kernels, we need a grid sync for making `packed_recv_count` visible
    if (phases & LOW_LATENCY_SEND_PHASE)
        cg::this_grid().sync();

    // Resent counters for the next dispatch
    if (responsibleExpertIdx < numRanks) {
        rankSentCnt[responsibleExpertIdx] = 0;
        rankDone[responsibleExpertIdx] = 0;
    }

    // Receiving and packing
    if (responsibleExpertIdx < numExperts) {
        // responsibleExpertIdx defines 2 things:
        // 1. the source rank of tokens
        // 2. the local expert index within the target (this) rank
        const auto srcRank = responsibleExpertIdx / numLocalExperts;
        // Each pair of rank establish numLocalExperts channels for parallelization
        const auto rankLaneIdx = responsibleExpertIdx % numLocalExperts;
        // Point to the beginning of the per-token table (the first numRanks entries hold per-rank token counts)
        const auto recvSrcInfoBase = outSrcInfo + numRanks;
        // First global expert id that is hosted by this rank
        const auto globalExpertStartIdx = currRank * numLocalExperts;
        // Align scales to the nearest multiple of float
        const auto numAlignedScales = align<int>(numScales, sizeof(float) / sizeof(scale_t));

        // Shared between sub-warps in warp groups
        __shared__ int sharedNumRecvTokens[kNumMaxWarpGroups];

        // Wait tokens to arrive
        int numRecvTokens;
        EP_DEVICE_ASSERT(numWarpsPerGroup > 1 and numWarpGroups < 15);
        if (subWarpId == 1 and laneId == 0) {
            numRecvTokens = waitForRecvTokensRelaxed(
                srcRank, rankLaneIdx, currRank, numRanks, recvCntOff,
                recvCntBuf, rankMask, asyncErrorFlag, signalsBase, windows, devComms,
                recvStats, waitStats, timeoutCycles);
            atomic_add_release_global(rankArrivedCnt + srcRank, 1);
            sharedNumRecvTokens[warpGroupId] = numRecvTokens;
        }

        // Wait for all counts from a specific rank to arrive
        // TODO: use relaxed semantics instead of acquire
        while (ld_acquire_sys_global(rankArrivedCnt + srcRank) != numLocalExperts);
        //(void)ld_acquire_sys_global(rankArrivedCnt + srcRank);

        numRecvTokens = sharedNumRecvTokens[warpGroupId];
        if (laneId == 0 and rankLaneIdx == 0) {
            // The first lane of the rank also stores the total number of tokens received from this rank
            outSrcInfo[srcRank] = numRecvTokens;
            if (outRecvRankCounter) outRecvRankCounter[srcRank] = numRecvTokens;
        }

        // Pick per srcRank wire layout: NVLink (split) if the sender is an
        // intra-LSA peer, RDMA (interleaved) otherwise. The mapping mirrors
        // the sender-side ncclGetP2pPtr check above.
        bool isNvlinkSrc;
        {
            constexpr int kCommId = 0;
            ncclTeam lsa = ncclTeamLsa(devComms[kCommId]);
            ncclTeam world = ncclTeamWorld(devComms[kCommId]);
            isNvlinkSrc = ncclTeamRankIsMember(lsa, world, srcRank);
        }

        if constexpr (kLayout == NCCL_EP_LAYOUT_EXPERT_MAJOR) {
            for (int i = rankLaneIdx * numWarpsPerGroup + subWarpId; i < numRecvTokens * numTopk; i += numWarpsPerGroup * numLocalExperts) {
                int tokenIdx = i / numTopk;
                int topkIdx = i % numTopk;

                const auto recvBufUint8 = reinterpret_cast<uint8_t*>(recvBuf) + srcRank * maxTokensPerRank * numBytesPerMsg;
                // NVLink split layout puts the per-slot header at the head of
                // the per-srcRank region; the legacy RDMA layout has each
                // header inline at the start of its [hdr|data|scales] message.
                const auto recvHdrPtr = isNvlinkSrc
                    ? (recvBufUint8 + tokenIdx * dispatch_hdr_sz)
                    : (recvBufUint8 + tokenIdx * numBytesPerMsg);
                const auto recvBufHdr = reinterpret_cast<const DispatchHdr<kLayout>*>(recvHdrPtr);
                const auto slotsPerToken = numTopk + 1; // token_id + topk_idx's
                const auto recvSrcInfo = recvSrcInfoBase + (srcRank * maxTokensPerRank + tokenIdx) * slotsPerToken;
                const auto recvSrcTopkInfo = recvSrcInfo + 1;

                // Initialize the token_id in the source tracking table
                if(laneId == 0 && topkIdx == 0) {
                    recvSrcInfo[0] = recvBufHdr->token_id;
                }

                // Locate local experts
                int localExpertIdx = recvBufHdr->rtr[topkIdx].expert_id - globalExpertStartIdx;
                if (localExpertIdx < 0 || localExpertIdx >= numLocalExperts) {
                    // skip this topK, mark as invalid
                    if (laneId == 0) {
                        recvSrcTopkInfo[topkIdx] = -1;
                    }
                    continue;
                }

                // Locate the output base for the local expert
                int outDataOffset = localExpertIdx * numRanks * maxTokensPerRank;
                const auto outDataInt4 = reinterpret_cast<int4*>(static_cast<uint8_t*>(outDataBuf) +
                                                                  static_cast<size_t>(outDataOffset) * hiddenBytes);
                const auto outScales = static_cast<scale_t*>(outScalesBuf) + outDataOffset * numAlignedScales;

                // Locate the next available slot
                int recvTokenBeginIdx = 0;
                if (laneId == 0) {
                    recvTokenBeginIdx = atomicAdd(outCnt + localExpertIdx, 1);
                    // Save the slot used for this topK
                    recvSrcTopkInfo[topkIdx] = outDataOffset + recvTokenBeginIdx;
                }
                recvTokenBeginIdx = __shfl_sync(0xFFFFFFFF, recvTokenBeginIdx, 0);


                // MEMOPT: Possibly we need to fix it
                copyRecvTokenData<kUseFP8, kUseUE8M0>(
                                recvBufUint8, tokenIdx,         // Location in the receive buffer
                                recvTokenBeginIdx,       // Location in the output data
                                outDataInt4,             // Output data
                                outScales,               // Output scales
                                hiddenInt4, hiddenBytes, numScales, numBytesPerMsg,
                                dispatch_hdr_sz,
                                maxTokensPerRank,
                                numRanks,
                                laneId,
                                isNvlinkSrc);
            }
        } else if constexpr (kLayout == NCCL_EP_LAYOUT_RANK_MAJOR) {
            // Rank-major: one output slot per received token.
            // outRecvTopkIdx/Weights are written so the user can route and reduce.
            for (int i = rankLaneIdx * numWarpsPerGroup + subWarpId; i < numRecvTokens; i += numWarpsPerGroup * numLocalExperts) {
                const auto recvBufUint8 = reinterpret_cast<uint8_t*>(recvBuf) + srcRank * maxTokensPerRank * numBytesPerMsg;
                const auto recvHdrPtr = isNvlinkSrc
                    ? (recvBufUint8 + i * dispatch_hdr_sz)
                    : (recvBufUint8 + i * numBytesPerMsg);
                const auto recvBufHdr = reinterpret_cast<const DispatchHdr<kLayout>*>(recvHdrPtr);
                const auto slotsPerToken = numTopk + 1; // token_id + topk_idx's
                const auto recvSrcInfo = recvSrcInfoBase + (srcRank * maxTokensPerRank + i) * slotsPerToken;
                const auto recvSrcTopkInfo = recvSrcInfo + 1;

                auto* outDataInt4 = static_cast<int4*>(outDataBuf);
                auto* outScales   = static_cast<scale_t*>(outScalesBuf);
                const int slot = srcRank * maxTokensPerRank + i;

                // Lane 0: write token_id and linear slot index.
                // srcInfo: j=0 is the linear slot index into inData (for processAndSendToken);
                // j=1 is the first topk position on this rank (j_eff), used by combine send to
                //   compute the receive slot: (tokenIdx * numTopk + j_eff) * numBytesPerSlot.
                // j>1 = -1 (no additional sends needed).
                if (laneId == 0) {
                    recvSrcInfo[0] = recvBufHdr->token_id;
                    recvSrcTopkInfo[0] = slot;
                }

                // Each lane writes its own topk entry in parallel.
                EP_DEVICE_ASSERT(outRecvTopkIdx != nullptr && outRecvTopkWeights != nullptr);
                if (laneId < numTopk) {
                    int localExpertIdx = (int)recvBufHdr->rtr[laneId].expert_id - globalExpertStartIdx;
                    bool valid = (localExpertIdx >= 0 && localExpertIdx < numLocalExperts);
                    outRecvTopkIdx[slot * numTopk + laneId] = valid ? (int32_t)localExpertIdx : (int32_t)-1;
                    outRecvTopkWeights[slot * numTopk + laneId] = recvBufHdr->rtr[laneId].topk_weight;
                }

                // Compute j_eff via __ballot_sync: first topk position mapping to currRank.
                if (numTopk > 1) {
                    bool matchesCurrRank = (laneId < numTopk) &&
                        (getExpertRankIdx((int)recvBufHdr->rtr[laneId].expert_id, numLocalExperts) == currRank);
                    uint32_t currRankMask = __ballot_sync(0xffffffff, matchesCurrRank);
                    int j_eff = currRankMask ? (__ffs(currRankMask) - 1) : -1;
                    if (laneId == 0) {
                        recvSrcTopkInfo[1] = j_eff;
                    }
                }
                copyRecvTokenData<kUseFP8, kUseUE8M0>(
                                recvBufUint8, i,         // Location in the receive buffer
                                slot,                    // Location in the output data
                                outDataInt4,             // Output data
                                outScales,               // Output scales
                                hiddenInt4, hiddenBytes, numScales, numBytesPerMsg,
                                dispatch_hdr_sz,
                                maxTokensPerRank,
                                numRanks,
                                laneId,
                                isNvlinkSrc);
            }
        }
    }
}

void dispatch(const void* inData,
              const int64_t* inTopkIdx,
              const float* inTopkWeights,
              void* outDataBuf,
              void* outScalesBuf,
              int* outSrcInfo,
              int* outRecvRankCounter,
              int64_t* outLayout,
              int* outCnt,
              float* outRecvTopkWeights,
              int32_t* outRecvTopkIdx,
              void* sendBuf,
              void* recvBuf,
              int* recvCntBuf,
              size_t sendOff,
              size_t recvOff,
              size_t recvCntOff,
              int* nextRecvCntBuf,
              int nextRecvCntBufSize,
              int* recvStats,
              int64_t* waitStats,
              int numTokens,
              int hidden,
              int maxTokensPerRank,
              int numTopk,
              int numExperts,
              int currRank,
              int numRanks,
              bool useFp8,
              bool roundScale,
              bool useUe8m0,
              ncclEpLayout_t layout,
              int phases,
              int numComms,
              ncclDevComm* devComms,
              const ncclWindow_t* windows,
              unsigned signalsBase,
              void* workspace,
              int numDeviceSms,
              int* rankMask,
              int* asyncErrorFlag,
              uint64_t timeoutCycles,
              bool nvlinkOnly,
              cudaStream_t stream) {
    constexpr int kNumMaxTopK = 9;
    const int numWarpGroups = ceil_div(numExperts, numDeviceSms);
    const int numWarpsPerGroup = 32 / numWarpGroups;
    EP_HOST_ASSERT(numWarpGroups > 0 and numWarpsPerGroup > 0);
    EP_HOST_ASSERT(kNumMaxTopK + 1 <= numWarpGroups * numWarpsPerGroup);

    const auto numWarps = numWarpGroups * numWarpsPerGroup;
    const auto numSms = ceil_div(numExperts, numWarpGroups);
    EP_HOST_ASSERT(numTopk <= kNumMaxTopK);

    // Workspace checks
    // rankCountersBase is used to track the number of tokens sent & received by each rank.
    // expertDone is used to track the number of tokens sent to each expert.
    auto rankCountersBase = static_cast<int*>(workspace);
    auto expertDone = rankCountersBase + 2 * numRanks /* Using 2 arrays of per-rank flags */;
    EP_HOST_ASSERT((2 * numRanks + numExperts) * sizeof(int) <= NUM_WORKSPACE_BYTES);

    // FP8 checks
    if (useUe8m0)
        EP_HOST_ASSERT(roundScale and "UE8M0 SF requires `round_scale=True`");

    SETUP_LAUNCH_CONFIG(numSms, numWarps * 32, stream);
#define DISPATCH_DO_LAUNCH(fp8, ue8m0, nvlinkOnlyV, hidden, kLayout) \
LAUNCH_KERNEL(&cfg, (dispatch<fp8, ue8m0, hidden, kLayout, nvlinkOnlyV>), \
              inData, \
              inTopkIdx, \
              inTopkWeights, \
              rankMask, \
              asyncErrorFlag, \
              outDataBuf, \
              outScalesBuf, \
              outSrcInfo, \
              outRecvRankCounter, \
              outLayout, \
              outCnt, \
              outRecvTopkWeights, \
              outRecvTopkIdx, \
              sendBuf, \
              recvBuf, \
              recvCntBuf, \
              sendOff, \
              recvOff, \
              recvCntOff, \
              rankCountersBase, \
              expertDone, \
              nextRecvCntBuf, \
              nextRecvCntBufSize, \
              recvStats, \
              waitStats, \
              numTokens, \
              maxTokensPerRank, \
              numTopk, \
              numExperts, \
              currRank, \
              numRanks, \
              numWarpGroups, \
              numWarpsPerGroup, \
              roundScale, \
              phases, \
              numComms, \
              devComms, \
              windows, \
              signalsBase, \
              timeoutCycles)
#define DISPATCH_LAUNCH_CASE_IMPL(hidden, kLayout) { \
    if (nvlinkOnly) { \
        if (useFp8 and useUe8m0)      { DISPATCH_DO_LAUNCH(true,  true,  true,  hidden, kLayout); } \
        else if (useFp8)              { DISPATCH_DO_LAUNCH(true,  false, true,  hidden, kLayout); } \
        else                          { DISPATCH_DO_LAUNCH(false, false, true,  hidden, kLayout); } \
    } else { \
        if (useFp8 and useUe8m0)      { DISPATCH_DO_LAUNCH(true,  true,  false, hidden, kLayout); } \
        else if (useFp8)              { DISPATCH_DO_LAUNCH(true,  false, false, hidden, kLayout); } \
        else                          { DISPATCH_DO_LAUNCH(false, false, false, hidden, kLayout); } \
    } \
} break
#define DISPATCH_LAUNCH_CASE_RM(hidden) DISPATCH_LAUNCH_CASE_IMPL(hidden, NCCL_EP_LAYOUT_RANK_MAJOR)
#define DISPATCH_LAUNCH_CASE_EM(hidden) DISPATCH_LAUNCH_CASE_IMPL(hidden, NCCL_EP_LAYOUT_EXPERT_MAJOR)
    if (layout == NCCL_EP_LAYOUT_RANK_MAJOR) {
        SWITCH_HIDDEN(DISPATCH_LAUNCH_CASE_RM);
    } else {
        SWITCH_HIDDEN(DISPATCH_LAUNCH_CASE_EM);
    }
#undef DISPATCH_DO_LAUNCH
#undef DISPATCH_LAUNCH_CASE_IMPL
#undef DISPATCH_LAUNCH_CASE_RM
#undef DISPATCH_LAUNCH_CASE_EM
}

template <int kNumSendUnrolls>
__forceinline__ __device__ int logfmtEncode(void* buffer, nv_bfloat162 *sharedAmaxmin, const int& laneId) {
    constexpr int kNumElemsPerInt4 = sizeof(int4) / sizeof(nv_bfloat16);
    constexpr float kLogThreshold = 0;
    constexpr float kMinClip = 32; // `== log_2(2 ^ (2 ^ 5))`
    constexpr int kNumBits = 10;
    constexpr int kNumValues = 1 << (kNumBits - 1);

    int4 int4Data[kNumSendUnrolls];
    const auto& uint32Data = reinterpret_cast<uint32_t*>(int4Data);
    const auto& bf162Data = reinterpret_cast<nv_bfloat162*>(int4Data);

    // Calculate lane offset
    const auto& loadBuf = reinterpret_cast<uint32_t*>(static_cast<uint8_t*>(buffer) + laneId * (kNumSendUnrolls * sizeof(int4)));
    const auto& storeBuf = reinterpret_cast<uint32_t*>(static_cast<uint8_t*>(buffer) + laneId * (kNumSendUnrolls * sizeof(int4) * 10 / 16));

    // Local log amax
    auto bf162Amax = __nv_bfloat162(CUDART_ZERO_BF16, CUDART_ZERO_BF16);
    auto bf162Amin = __nv_bfloat162(CUDART_INF_BF16, CUDART_INF_BF16);
    uint32_t localSigns = 0;
    #pragma unroll
    for (int k = 0; k < kNumSendUnrolls * kNumElemsPerInt4 / 2; ++ k) {
        uint32Data[k] = loadBuf[k];
        localSigns |= ((uint32Data[k] >> 15) & 1) << (k * 2);
        localSigns |= ((uint32Data[k] >> 31) & 1) << (k * 2 + 1);
        uint32Data[k] &= 0x7fff7fff;

        bf162Amax = __hmax2(bf162Amax, bf162Data[k]);
        bf162Amin = __hmin2(bf162Amin, bf162Data[k]);
    }

    // Reduce per 128 channels
    auto amax = std::max(static_cast<float>(bf162Amax.x), static_cast<float>(bf162Amax.y));
    auto amin = std::min(static_cast<float>(bf162Amin.x), static_cast<float>(bf162Amin.y));
    constexpr static int kNumLanesToReduce = 128 * sizeof(nv_bfloat16) / (kNumSendUnrolls * sizeof(int4));
    amax = warp_reduce_max<kNumLanesToReduce>(amax);
    amin = warp_reduce_min<kNumLanesToReduce>(amin);

    // Write min/max into the shared memory
    if (sharedAmaxmin != nullptr)
        *sharedAmaxmin = __nv_bfloat162(amax, amin);
    __syncwarp();

    // Calculate log amin/amax float
    const auto& logAmax = log2f_approx(amax);
    const auto& logAmin = fmaxf(log2f_approx(amin), logAmax - kMinClip);
    const bool& enableCast = warp_reduce_and<kNumLanesToReduce, true>(logAmax < kLogThreshold and logAmin < logAmax);

    // Case into LogFMT-10 if satisfied
    if (enableCast) {
        const auto step = (logAmax - logAmin) / static_cast<float>(kNumValues - 2);
        const auto stepInv = 1.0f / step;
        const auto rounding = 2.0f - log2f_approx((1.0f + exp2f_approx(step)) * 0.5f) * stepInv;
        const auto fusedRounding = rounding - logAmin * stepInv;

        // Pack every 256 bits into 160 bits
        EP_STATIC_ASSERT(kNumSendUnrolls == 2 or kNumSendUnrolls == 4, "kNumSendUnrolls == 2 or 4 only");
        uint32_t encodedData[kNumElemsPerInt4 * 2];
        #pragma unroll 1
        for (int i = 0; i < kNumSendUnrolls / 2; ++ i) {
            #pragma unroll
            for (int k = 0; k < kNumElemsPerInt4; ++ k) {
                const auto& [x, y] = __bfloat1622float2(bf162Data[i * kNumElemsPerInt4 + k]);
                encodedData[k * 2 + 0] = __float2uint_rd(fmaxf(log2f_approx(x) * stepInv + fusedRounding, 0));
                encodedData[k * 2 + 1] = __float2uint_rd(fmaxf(log2f_approx(y) * stepInv + fusedRounding, 0));
            }
            storeBuf[i * 5 + 0] = (encodedData[ 0] >> 0) | (encodedData[ 1] << 9) | (encodedData[ 2] << 18) | (encodedData[ 3] << 27);
            storeBuf[i * 5 + 1] = (encodedData[ 3] >> 5) | (encodedData[ 4] << 4) | (encodedData[ 5] << 13) | (encodedData[ 6] << 22) | (encodedData[7]  << 31);
            storeBuf[i * 5 + 2] = (encodedData[ 7] >> 1) | (encodedData[ 8] << 8) | (encodedData[ 9] << 17) | (encodedData[10] << 26);
            storeBuf[i * 5 + 3] = (encodedData[10] >> 6) | (encodedData[11] << 3) | (encodedData[12] << 12) | (encodedData[13] << 21) | (encodedData[14] << 30);
            storeBuf[i * 5 + 4] = (encodedData[14] >> 2) | (encodedData[15] << 7) | ((i == 0) ? (localSigns << 16) : (localSigns & 0xffff0000u));
        }
        tma_store_fence();
        __syncwarp();
    }

    // Return TMA copy bytes
    return enableCast ? (32 * (kNumSendUnrolls * sizeof(int4) * 8 * 10 / 16 / 8)):
                         (32 * (kNumSendUnrolls * sizeof(int4)));
}

template <int kNumLanes, int kNumSendUnrolls, int kNumRecvUnrolls>
__forceinline__ __device__ void logfmtCheckAmaxmin(uint8_t* metaBuffer, float2* sharedLogAmax,
                                                   float2* sharedLogAmin, int* sharedCastInfo,
                                                   const int laneId) {
    constexpr float kLogThreshold = 0;
    constexpr float kMinClip = 32; // `== log_2(2 ^ (2 ^ 5))`

    bool enableCast = true;
    if (laneId < kNumLanes) {
        // Calculate log amin/amax float
        auto amaxminData = reinterpret_cast<uint64_t*>(metaBuffer)[laneId];
        const auto& bf162Amaxmin = reinterpret_cast<__nv_bfloat162*>(&amaxminData);
        float logAmax[2], logAmin[2];
        #pragma unroll
        for (int i = 0; i < 2; ++ i) {
            auto amax = static_cast<float>(bf162Amaxmin[i].x);
            auto amin = static_cast<float>(bf162Amaxmin[i].y);
            logAmax[i] = log2f_approx(amax);
            logAmin[i] = amin == 0 ? logAmax[i] - kMinClip : fmaxf(log2f_approx(amin), logAmax[i] - kMinClip);
            enableCast = enableCast and logAmax[i] < kLogThreshold and logAmin[i] < logAmax[i];
        }
        sharedLogAmax[laneId] = make_float2(logAmax[0], logAmax[1]);
        sharedLogAmin[laneId] = make_float2(logAmin[0], logAmin[1]);
    }

    const auto& casted = warp_reduce_and<kNumSendUnrolls>(enableCast) ? 1u << (laneId / kNumRecvUnrolls): 0u;
    const auto& numCastedPrefix = __popc(warp_reduce_or<kNumRecvUnrolls, true>(casted) & ((1u << (laneId / kNumRecvUnrolls)) - 1));

    if (laneId < kNumLanes and laneId % kNumRecvUnrolls == 0)
        sharedCastInfo[laneId / kNumRecvUnrolls] = (numCastedPrefix << 1) | (casted ? 1u : 0u);
    __syncwarp();
}

template <int kNumRecvUnrolls>
__forceinline__ __device__ void decodeAndAccumulate(uint32_t* ldBuffer, float* accum,
                                                    const float& logAmax, const float& logAmin,
                                                    const bool& enableCast, const float& weight) {
    if (enableCast) {
        constexpr int kNumBits = 10;
        constexpr int kNumValues = 1 << (kNumBits - 1);

        const auto& step = (logAmax - logAmin) / static_cast<float>(kNumValues - 2);
        auto decode = [=](const uint32_t &encoded, const uint32_t &sign) {
            const auto decoded = encoded == 0 ? .0f : exp2f_approx((encoded - 1) * step + logAmin);
            return sign ? -decoded : decoded;
        };

        EP_STATIC_ASSERT(kNumRecvUnrolls == 2 or kNumRecvUnrolls == 4, "kNumRecvUnrolls == 2 or 4 only");
        #pragma unroll
        for (int i = 0; i < kNumRecvUnrolls / 2; ++ i) {
            uint32_t concatData[6];
            concatData[0] = ldBuffer[i * 5];
            #pragma unroll
            for (int k = 1; k < 5; ++ k)
                concatData[k] = (ldBuffer[i * 5 + k - 1] >> (32 - k * 5)) | (ldBuffer[i * 5 + k] << (k * 5));
            concatData[5] = ldBuffer[i * 5 + 4] >> 7;

            const uint32_t& localSigns = ldBuffer[i * 5 + 4] >> 16;
            #pragma unroll
            for (int k = 0; k < 5; ++ k) {
                accum[i * 16 + k * 3 + 0] += decode((concatData[k] >>  0) & 0x1ff, (localSigns >> (k * 3 + 0)) & 1) * weight;
                accum[i * 16 + k * 3 + 1] += decode((concatData[k] >>  9) & 0x1ff, (localSigns >> (k * 3 + 1)) & 1) * weight;
                accum[i * 16 + k * 3 + 2] += decode((concatData[k] >> 18) & 0x1ff, (localSigns >> (k * 3 + 2)) & 1) * weight;
            }
            accum[i * 16 + 15] += decode(concatData[5] & 0x1ff, (localSigns >> 15) & 1) * weight;
        }
    } else {
        #pragma unroll
        for (int k = 0; k < kNumRecvUnrolls * 4; ++ k) {
            auto bf16Pack = *reinterpret_cast<__nv_bfloat162*>(ldBuffer + k);
            accum[k * 2 + 0] += static_cast<float>(bf16Pack.x) * weight;
            accum[k * 2 + 1] += static_cast<float>(bf16Pack.y) * weight;
        }
    }
}

// Clean next receive count buffer and notify atomic clean flag
__forceinline__ __device__ void cleanNextRecvCntBufAndNotify(
    int* nextRecvCntBuf,
    int nextRecvCntBufSize,
    int* atomicCleanFlag,
    int numExperts,
    int laneId) {
    if (!dP2pDisabled) {
        #pragma unroll
        for (int i = laneId; i < nextRecvCntBufSize; i += 32)
            nextRecvCntBuf[i] = 0;
    }
    // Notify before executing `int_p`
    __syncwarp();
    if (laneId == 0)
        atomic_add_release_global(atomicCleanFlag, numExperts);
}

// Send finish flag to destination rank
__forceinline__ __device__ void sendFinishFlag(
    int globalExpertIdx,
    int dstRank,
    int localExpertIdx,
    int currRank,
    int numLocalExperts,
    uint64_t recvFlagPtr,
    size_t recvFlagOffset,
    int* recvFlagBuf,
    int* rankMask,
    unsigned signalsBase,
    const ncclWindow_t* windows,
    ncclDevComm* devComms) {
    auto dstP2pPtr = ncclGetP2pPtr(recvFlagPtr, recvFlagOffset,
                                   currRank, dstRank, windows, devComms);

    if (not isRankMasked(rankMask, dstRank)) {
        if (dstP2pPtr == 0) {
            auto signalId = signalsBase + globalExpertIdx;
            constexpr int commId = 0;
            auto ctxId = localExpertIdx % MAX_NCCL_GIN_CTX_PER_COMM;

            ncclGin net(devComms[commId], ctxId);
            ncclTeam world = ncclTeamWorld(devComms[commId]);
            auto ncclWindow = windows[commId];

            net.put(world,
                    dstRank,
                    ncclWindow,
                    recvFlagOffset,
                    ncclWindow,
                    0,
                    0,  // 0 bytes transfer
                    ncclGin_SignalAdd{signalId, 1},
                    ncclGin_None{},  // no counter
                    ncclCoopThread());
        } else {
            st_release_sys_global(reinterpret_cast<int*>(dstP2pPtr), 1);
        }
    }
}

// Wait for receive flag to arrive
__forceinline__ __device__ void waitForRecvFlag(
    int responsibleExpertIdx,
    int srcRank,
    int currRank,
    int numLocalExperts,
    size_t recvFlagOff,
    int* recvFlagBuf,
    int* rankMask,
    int* asyncErrorFlag,
    unsigned signalsBase,
    const ncclWindow_t* windows,
    ncclDevComm* devComms,
    int64_t* waitStats,
    uint64_t timeoutCycles) {
    auto startTime = clock64();
    uint64_t waitRecvCost = 0;

    size_t srcOffset = recvFlagOff + responsibleExpertIdx * sizeof(int);
    auto srcP2pPtr = ncclGetP2pPtr(0x01, srcOffset, currRank, srcRank, windows, devComms);
    if (not isRankMasked(rankMask, srcRank)) {
        if (srcP2pPtr == 0) {
            uint64_t curValue;
            auto localExpertIdxWait = responsibleExpertIdx % numLocalExperts;
            constexpr int commIdWait = 0;
            auto ctxIdWait = localExpertIdxWait % MAX_NCCL_GIN_CTX_PER_COMM;
            ncclGin net(devComms[commIdWait], ctxIdWait);
            do {
                curValue = net.readSignal(signalsBase + responsibleExpertIdx);
            } while (curValue < 1                                                       // signal not arrived
                     && (waitRecvCost = clock64() - startTime) <= timeoutCycles  // not timeout
            );
            net.resetSignal(signalsBase + responsibleExpertIdx);
        } else {
            while (ld_acquire_sys_global(recvFlagBuf + responsibleExpertIdx) == 0  // recv not ready
                   && (waitRecvCost = clock64() - startTime) <= timeoutCycles   // not timeout
            );
        }
    }
    // Mask rank if timeout
    if (waitRecvCost > timeoutCycles) {
        printf("Warning: NCCL EP timeout for combine receive, rank %d, local_expert_idx %d, src_rank %d\n",
               currRank,
               responsibleExpertIdx % numLocalExperts,
               srcRank);
        if (rankMask == nullptr)
            trap();
        atomicExch(rankMask + srcRank, 0);
        if (asyncErrorFlag != nullptr)
            atomicExch_system(asyncErrorFlag, 1);
    }

    if (waitStats != nullptr) {
        atomicAdd(reinterpret_cast<unsigned long long*>(waitStats + srcRank), waitRecvCost);
    }
}

// Send token via RDMA
__forceinline__ __device__ void sendTokenViaRdma(
    int dstRank,
    int rankLaneIdx,
    int currRank,
    int numRanks,
    int maxTokensPerRank,
    int tokenIdx,
    size_t sendOff,
    size_t recvOff,
    size_t numBytesPerSlot,
    int hidden,
    ncclDevComm* devComms,
    const ncclWindow_t* windows) {

    const auto expectedDstOffset = recvOff;
    const auto expectedBufOffset = sendOff;

    constexpr int commId = 0;
    auto ctxId = getCtxId(rankLaneIdx);
    ncclGin net(devComms[commId], ctxId);
    ncclTeam world = ncclTeamWorld(devComms[commId]);
    auto ncclWindow = windows[commId];
    net.put(world,
            dstRank,
            ncclWindow,
            expectedDstOffset,
            ncclWindow,
            expectedBufOffset,
            hidden * sizeof(nv_bfloat16),
            ncclGin_None{},  // no signal
            ncclGin_None{},  // no counter
            ncclCoopThread());
}

// Process and send a single token with TMA copy and optional RDMA
template<bool kUseLogFMT, int kHidden, int kNumSendUnrolls, int kNumStages, int kNumPrefetch,
         typename TmaBuffersT, typename FullBarriersT, typename TmaLoadAndArriveT, typename GetNumTmaBytesT>
__forceinline__ __device__ void processAndSendToken(
    int tokenIdx,
    const int4* srcDataInt4Ptr,
    int4* sendBufInt4Ptr,
    uint64_t recvPtr,
    size_t recvOff,
    size_t sendOff,
    int dstRank,
    int rankLaneIdx,
    int currRank,
    int numRanks,
    int maxTokensPerRank,
    size_t numBytesPerSlot,
    int hidden,
    int hiddenBf16Int4,
    int hiddenBf16Int4Pad,
    int kNumMetaBytes,
    int kNumTMABufferBytes,
    bool zeroCopy,
    uint64_t dstP2pPtr,
    TmaBuffersT tmaBuffers,
    FullBarriersT fullBarriers,
    nv_bfloat162* metaBuffers,
    uint32_t& tmaPhase,
    TmaLoadAndArriveT& tmaLoadAndArrive,
    GetNumTmaBytesT& getNumTmaBytes,
    int laneId,
    ncclDevComm* devComms,
    const ncclWindow_t* windows) {
    constexpr int kNumElemsPerInt4 = sizeof(int4) / sizeof(nv_bfloat16);
    const int kNumIters = hiddenBf16Int4Pad / (32 * kNumSendUnrolls);

    if (not zeroCopy or dstP2pPtr != 0) {
        // Read from `copySrcPtr` and copy into `copyDstPtr`
        const auto copySrcPtr = zeroCopy ? sendBufInt4Ptr : srcDataInt4Ptr;
        const auto copyDstPtr =
            dstP2pPtr == 0 ? sendBufInt4Ptr : reinterpret_cast<int4*>(dstP2pPtr);

        // Prefetch
        if (elect_one_sync())
            tmaLoadAndArrive(0, copySrcPtr, getNumTmaBytes(0));
        __syncwarp();

        int tmaOffsetBytes = kNumMetaBytes;
        #pragma unroll
        for (int i = laneId * kNumSendUnrolls, iterIdx = 0; i < hiddenBf16Int4Pad; i += 32 * kNumSendUnrolls, ++iterIdx) {
            // Load the next iteration
            const int& stageIdx = iterIdx % kNumStages;
            const int& nextStageIdx = (iterIdx + 1) % kNumStages;
            if (iterIdx + 1 < kNumIters and elect_one_sync()) {
                tma_store_wait<kNumStages - kNumPrefetch - 1>();
                const auto& offsetInt4 = i + 32 * kNumSendUnrolls;
                tmaLoadAndArrive(nextStageIdx, copySrcPtr + offsetInt4, getNumTmaBytes(offsetInt4));
            }
            __syncwarp();

            // Wait the current TMA arrival
            EP_STATIC_ASSERT(kNumStages < 32, "Too many stages");
            mbarrier_wait<true>(fullBarriers[stageIdx], tmaPhase, stageIdx);
            if constexpr (kUseLogFMT) {
                // Cast if possible
                constexpr int kNumInt4PerDivision = 128 / kNumElemsPerInt4;
                int numTmaBytes = logfmtEncode<kNumSendUnrolls>(
                    tmaBuffers[stageIdx],
                    // NOTES: only the leader lane will write the result
                    (i % kNumInt4PerDivision == 0) ? metaBuffers + i / kNumInt4PerDivision : nullptr,
                    laneId);
                if (elect_one_sync())
                    tma_store_1d(
                        tmaBuffers[stageIdx], reinterpret_cast<uint8_t*>(copyDstPtr) + tmaOffsetBytes, numTmaBytes);
                tmaOffsetBytes += numTmaBytes;
            } else {
                // BF16 original values
                if (elect_one_sync())
                    tma_store_1d(tmaBuffers[stageIdx], copyDstPtr + i, getNumTmaBytes(i));
            }
            __syncwarp();
        }

        // Store metadata (min/max values) for LogFMT
        if constexpr (kUseLogFMT) {
            if (elect_one_sync())
                tma_store_1d(metaBuffers, copyDstPtr, kNumMetaBytes);
        }

        // Flush all stores
        tma_store_wait<0>();
        __syncwarp();
    }

    // Issue RDMA
    // NOTES: for zero-copy mode, we assume the data is already in the send buffer
    if (dstP2pPtr == 0) {
        if (laneId == 0) {
            sendTokenViaRdma(
                dstRank, rankLaneIdx, currRank, numRanks,
                maxTokensPerRank, tokenIdx,
                sendOff, recvOff,
                numBytesPerSlot, hidden, devComms, windows);
        }
    }
}

template <bool kUseLogFMT, int kHidden, int kNumMaxTopk, int kNumMaxUnrolls, ncclEpLayout_t kLayout>
__global__ __launch_bounds__(1024, 1) void combine(// INPUT
                                                   const void* inData,
                                                   const int* srcInfo,
                                                   const int64_t* layoutRange,
                                                   const int64_t* inTopkIdx,
                                                   const float* topkWeights,
                                                   int* rankMask,
                                                   int* asyncErrorFlag,
                                                   // OUTPUT
                                                   void* outData,
                                                   // INTERMEDIATE
                                                   void* sendBuf,
                                                   void* recvBuf,
                                                   int* recvFlagBuf,
                                                   size_t sendOff,
                                                   size_t recvOff,
                                                   size_t recvFlagOff,
                                                   int* atomicCleanFlag,
                                                   int* nextRecvCntBuf,
                                                   int nextRecvCntBufSize,
                                                   int64_t* waitStats,
                                                   // CONFIG
                                                   int numCombinedTokens,
                                                   int hidden,
                                                   int numTopk,
                                                   int maxTokensPerRank,
                                                   int numExperts,
                                                   int currRank,
                                                   int numRanks,
                                                   int numWarpGroups,
                                                   int numWarpsPerGroup,
                                                   int phases,
                                                   bool zeroCopy,
                                                   int numComms,
                                                   ncclDevComm* devComms,
                                                   const ncclWindow_t* windows,
                                                   unsigned signalsBase,
                                                   uint64_t timeoutCycles) {
    const auto smId = __shfl_sync(0xffffffff, static_cast<int>(blockIdx.x), 0);
    const auto numSms = __shfl_sync(0xffffffff, static_cast<int>(gridDim.x), 0);
    const auto threadId = static_cast<int>(threadIdx.x);
    const auto numThreads = __shfl_sync(0xffffffff, static_cast<int>(blockDim.x), 0);
    const auto warpId = __shfl_sync(0xffffffff, threadId / 32, 0), laneId = get_lane_id();
    const auto numLocalExperts = numExperts / numRanks;
    const auto warpGroupId = warpId / numWarpsPerGroup;
    const auto subWarpId = warpId % numWarpsPerGroup;
    const auto responsibleExpertIdx = smId * numWarpGroups + warpGroupId;

    extern __shared__ __align__(1024) uint8_t smemBuffer[];

    // Data type staffs
    constexpr int kNumElemsPerInt4 = sizeof(int4) / sizeof(nv_bfloat16);
    constexpr int64_t hiddenBf16Int4 = kHidden / kNumElemsPerInt4;

    // Use different unroll factors for send and recv phases
    constexpr int kNumSendUnrolls = kHidden % (32 * 4 * sizeof(int4) / sizeof(nv_bfloat16)) == 0 ? 4 : 2;
    constexpr int kNumRecvUnrolls = 2;
    constexpr int hiddenBf16Int4Pad = align(static_cast<int>(hiddenBf16Int4), 32 * kNumSendUnrolls);
    EP_STATIC_ASSERT(kHidden % (32 * 2 * sizeof(int4) / sizeof(nv_bfloat16)) == 0, "Invalid hidden");
    EP_STATIC_ASSERT(kNumSendUnrolls <= kNumMaxUnrolls and kNumRecvUnrolls <= kNumMaxUnrolls, "Invalid unrolls");
    EP_STATIC_ASSERT(hiddenBf16Int4 % kNumSendUnrolls == 0, "Invalid hidden");
    EP_STATIC_ASSERT(kNumSendUnrolls >= kNumRecvUnrolls, "Invalid unroll factors");

    // Message package
    EP_STATIC_ASSERT(kHidden % 128 == 0, "Invalid hidden");
    constexpr int kNumDivisions = kHidden / 128;
    constexpr int kNumMetaBytes = kNumDivisions * sizeof(nv_bfloat162);
    constexpr size_t numBytesPerSlot = kHidden * sizeof(nv_bfloat16) + kNumMetaBytes;
    EP_STATIC_ASSERT(numBytesPerSlot % sizeof(int4) == 0, "Invalid vectorization");

    // Sending phase
    if ((phases & LOW_LATENCY_SEND_PHASE) == 0)
        goto LOW_LATENCY_COMBINE_RECV;

    // Clean up next buffer
    if (smId == 0 and warpGroupId == 0 and subWarpId == 0) {
        cleanNextRecvCntBufAndNotify(
            nextRecvCntBuf, nextRecvCntBufSize, atomicCleanFlag, numExperts, laneId);
    }

    // Issue tokens sending
    if (responsibleExpertIdx < numExperts) {
        const auto dstRank = responsibleExpertIdx / numLocalExperts;
        // Each pair of rank establish numLocalExperts channels for parallelization
        const auto rankLaneIdx = responsibleExpertIdx % numLocalExperts;

        // Read # of tokens received from this rank and set it's source info
        const auto numRecvTokens = __shfl_sync(0xffffffff, __ldg(srcInfo + dstRank), 0);

        auto slotsPerToken = numTopk + 1; // token_id + topk_idx's
        // We have numRanks entries with per-rank count first, then the rest is (tokenId + numTopk) entries for each token
        const auto localSrcInfo = srcInfo + numRanks + dstRank * maxTokensPerRank * slotsPerToken;

        // TMA stuffs
        constexpr int kNumTMABufferBytes = sizeof(int4) * 32 * kNumSendUnrolls;
        constexpr int kNumStages = 3;
        constexpr int kNumPrefetch = 1;
        EP_STATIC_ASSERT(kNumStages == 3 and kNumPrefetch == 1, "Invalid stages");

        auto smemPtr = smemBuffer + warpId * (kNumStages * (kNumTMABufferBytes + 16) + kNumMetaBytes);
        uint32_t tmaPhase = 0;
        auto tmaBuffers   = PatternVisitor([=](const int& i) { return reinterpret_cast<int4*>(smemPtr + i * (kNumTMABufferBytes + 16)); });
        auto fullBarriers = PatternVisitor([=](const int& i) { return reinterpret_cast<uint64_t*>(smemPtr + i * (kNumTMABufferBytes + 16) + kNumTMABufferBytes); });
        auto metaBuffers  = kUseLogFMT ? reinterpret_cast<nv_bfloat162*>(smemPtr + kNumStages * (kNumTMABufferBytes + 16)) : nullptr;
        EP_STATIC_ASSERT(kNumSendUnrolls * kNumStages <= 12, "TMA buffer size exceed limit");

        // Initialize m-barriers
        if (laneId < kNumStages) {
            mbarrier_init(fullBarriers[laneId], 1);
            fence_barrier_init();
        }
        __syncwarp();

        auto tmaLoadAndArrive = [&](const int& stageIdx, const int4* gmemPtr, const int& numBytes) {
            tma_load_1d(tmaBuffers[stageIdx], gmemPtr, fullBarriers[stageIdx], numBytes);
            mbarrier_arrive_and_expect_tx(fullBarriers[stageIdx], numBytes);
        };
        auto getNumTmaBytes = [&](const int& offsetInt4) {
            return min(kNumTMABufferBytes, static_cast<int>((hiddenBf16Int4 - offsetInt4) * sizeof(int4)));
        };

        // Issue sends for each token from the responsible dstRank
        if (not isRankMasked<true>(rankMask, dstRank)) {

            // Use all lanes to send tokens
            if constexpr (kLayout == NCCL_EP_LAYOUT_EXPERT_MAJOR) {
                for (int i = subWarpId * numLocalExperts + rankLaneIdx; i < numRecvTokens * numTopk; i += numWarpsPerGroup * numLocalExperts) {
                    int tokenIdx = i / numTopk;
                    int topkIdx = i % numTopk;
                    auto localSrcTokenInfo = localSrcInfo + tokenIdx * slotsPerToken;
                    auto localSrcTopkInfo = localSrcTokenInfo + 1;

                    // Load the global expert index for this topk (skipping token index via "+1")
                    int cachedOffset = __shfl_sync(0xffffffff, __ldg(localSrcTopkInfo + topkIdx), 0);
                    if (cachedOffset < 0) {
                        continue;
                    }
                    size_t offset = static_cast<size_t>(cachedOffset);
                    int remoteTokenIdx = __shfl_sync(0xffffffff, __ldg(localSrcTokenInfo), 0);

                    const auto srcDataInt4Ptr = static_cast<const int4*>(inData) + offset * hiddenBf16Int4;

                    // Currently, a staging RDMA buffer is used for copying
                    // TODO: Fix this code path to use less memory
                    // offset can reach (numLocalExperts-1)*numRanks*maxTPR + maxTPR-1 which exceeds INT_MAX
                    // when multiplied by numBytesPerSlot — use size_t to avoid overflow.
                    size_t sndTokenOffset = offset * numBytesPerSlot;
                    const auto sendBufUint8 = static_cast<uint8_t*>(sendBuf) + sndTokenOffset;
                    const auto sendBufInt4 = reinterpret_cast<int4*>(sendBufUint8);
                    auto expectedSendOffset = sendOff + sndTokenOffset;

                    // Receive location calculation
                    size_t rcvTokenOffset = static_cast<size_t>(remoteTokenIdx * numTopk + topkIdx) * numBytesPerSlot;
                    const auto recvPtr = reinterpret_cast<uint64_t>(recvBuf) + rcvTokenOffset;
                    const auto expectedDstOffset = recvOff + rcvTokenOffset;
                    const auto dstP2pPtr =
                        ncclGetP2pPtr(recvPtr, expectedDstOffset, currRank, dstRank, windows, devComms);

                    processAndSendToken<kUseLogFMT, kHidden, kNumSendUnrolls, kNumStages, kNumPrefetch>(
                        tokenIdx, srcDataInt4Ptr, sendBufInt4, recvPtr,
                        expectedDstOffset, expectedSendOffset, dstRank, rankLaneIdx,
                        currRank, numRanks, maxTokensPerRank, numBytesPerSlot,
                        hidden, hiddenBf16Int4, hiddenBf16Int4Pad, kNumMetaBytes,
                        kNumTMABufferBytes, zeroCopy,
                        dstP2pPtr, tmaBuffers, fullBarriers, metaBuffers, tmaPhase,
                        tmaLoadAndArrive, getNumTmaBytes, laneId, devComms, windows);
                }
            } else if constexpr (kLayout == NCCL_EP_LAYOUT_RANK_MAJOR) {
                // Rank-major: one send per received slot.
                // srcInfo[0] = linear slot into inData; srcInfo[1] = j_eff (topk return index).
                // The receive slot is (tokenIdx * numTopk + j_eff) * numBytesPerSlot, placing
                // each expert rank's contribution into a distinct position in the recv buffer.
                for (int i = subWarpId * numLocalExperts + rankLaneIdx; i < numRecvTokens; i += numWarpsPerGroup * numLocalExperts) {
                    auto localSrcTokenInfo = localSrcInfo + i * slotsPerToken;
                    auto localSrcTopkInfo = localSrcTokenInfo + 1;
                    int tokenIdx = __shfl_sync(0xffffffff, __ldg(localSrcTokenInfo), 0);

                    int slot  = __shfl_sync(0xffffffff, __ldg(localSrcTopkInfo + 0), 0);
                    int j_eff = (numTopk > 1)
                                ? __shfl_sync(0xffffffff, __ldg(localSrcTopkInfo + 1), 0)
                                : 0;
                    if (j_eff < 0) continue; // no local expert found (shouldn't happen)

                    int sndTokenOffset = slot * numBytesPerSlot;
                    int rcvTokenOffset = (tokenIdx * numTopk + j_eff) * numBytesPerSlot;

                    const auto srcDataInt4Ptr = static_cast<const int4*>(inData) + (int64_t)slot * hiddenBf16Int4;
                    const auto sendBufUint8   = static_cast<uint8_t*>(sendBuf) + sndTokenOffset;
                    const auto sendBufPtr     = reinterpret_cast<int4*>(sendBufUint8);
                    const auto recvPtr        = reinterpret_cast<uint64_t>(recvBuf) + rcvTokenOffset;
                    const auto expectedDstOffset = recvOff + rcvTokenOffset;
                    const auto dstP2pPtr =
                        ncclGetP2pPtr(recvPtr, expectedDstOffset, currRank, dstRank, windows, devComms);

                    processAndSendToken<kUseLogFMT, kHidden, kNumSendUnrolls, kNumStages, kNumPrefetch>(
                        tokenIdx, srcDataInt4Ptr, sendBufPtr, recvPtr,
                        recvOff + rcvTokenOffset, sendOff + sndTokenOffset, dstRank, rankLaneIdx,
                        currRank, numRanks, maxTokensPerRank, numBytesPerSlot,
                        hidden, hiddenBf16Int4, hiddenBf16Int4Pad, kNumMetaBytes,
                        kNumTMABufferBytes, zeroCopy,
                        dstP2pPtr, tmaBuffers, fullBarriers, metaBuffers, tmaPhase,
                        tmaLoadAndArrive, getNumTmaBytes, laneId, devComms, windows);
                }
            }
        }

        // Put the finishing flag
        EP_DEVICE_ASSERT(numWarpsPerGroup > 1 and numWarpGroups < 16);

        // TODO use the new sync primitive
        asm volatile("bar.sync %0, %1;" :: "r"(warpGroupId + 1), "r"(numWarpsPerGroup * 32));
        if (subWarpId == 1 and laneId == 0) {
            while (ld_acquire_global(atomicCleanFlag) == 0);

            int globalExpertIdx = currRank * numLocalExperts + rankLaneIdx;
            auto recvFlagPtr = reinterpret_cast<uint64_t>(recvFlagBuf + globalExpertIdx);
            size_t recvFlagOffset = recvFlagOff + globalExpertIdx * sizeof(int);

            sendFinishFlag(
                globalExpertIdx, dstRank, rankLaneIdx, currRank, numLocalExperts,
                recvFlagPtr, recvFlagOffset, recvFlagBuf, rankMask, signalsBase,
                windows, devComms);
            atomic_add_release_global(atomicCleanFlag, -1);
        }
        __syncwarp();

        // Destroy m-barriers
        if (laneId < kNumStages) {
            mbarrier_inval(fullBarriers[laneId]);
            fence_barrier_init();
        }
        __syncwarp();
    }

// Receiving phase
LOW_LATENCY_COMBINE_RECV:
    if ((phases & LOW_LATENCY_RECV_PHASE) == 0)
        return;

    // Wait all ranks to arrive
    if (responsibleExpertIdx < numExperts) {
        EP_DEVICE_ASSERT(numWarpsPerGroup > 1);
        if (subWarpId == 0 and laneId == 0) {
            const auto srcRank = responsibleExpertIdx / numLocalExperts;
            waitForRecvFlag(
                responsibleExpertIdx, srcRank, currRank, numLocalExperts,
                recvFlagOff, recvFlagBuf, rankMask, asyncErrorFlag, signalsBase,
                windows, devComms, waitStats, timeoutCycles);
        }
    }
    cg::this_grid().sync();

    // Reassign warp groups
    constexpr int kMaxNumGroups = 2;
    const int numDecodeWarps = hiddenBf16Int4Pad / (kNumRecvUnrolls * 32);
    const int numGroups = min(kMaxNumGroups, (numThreads / 32) / (numDecodeWarps + 1));
    const int decodeWarpIdx = __shfl_sync(0xffffffff, warpId % (numDecodeWarps + 1), 0);
    const int groupIdx = __shfl_sync(0xffffffff, warpId / (numDecodeWarps + 1), 0);
    EP_STATIC_ASSERT(kHidden % (32 * kNumElemsPerInt4) == 0, "Invalid vectorization");
    EP_DEVICE_ASSERT(numTopk <= 32);
    EP_DEVICE_ASSERT(numGroups > 0);


    if (groupIdx < numGroups) {
        constexpr int kNumStages = 3;
        constexpr int kNumTMABufferBytes = 16 * 2 + kHidden * 2;
        constexpr int kNumBF16PerWarpBytes = 32 * kNumRecvUnrolls * kNumElemsPerInt4 * 2;
        constexpr int kNumLogFMTPerWarpBytes = kNumBF16PerWarpBytes / 16 * 10;
        constexpr int kNumDivisionBytes = kNumDivisions * sizeof(uint32_t);
        constexpr int kNumBytesPerGroup = kNumStages * kNumTMABufferBytes + kHidden * 2 + kNumStages * kNumDivisionBytes * 3;

        // Reallocate shared memory
        const auto smemGroupBuffer = smemBuffer + kNumBytesPerGroup * groupIdx;
        auto fullBarriers  = PatternVisitor([=](const int& i) { return reinterpret_cast<uint64_t*>(smemGroupBuffer + i * kNumTMABufferBytes); });
        auto emptyBarriers = PatternVisitor([=](const int& i) { return reinterpret_cast<uint64_t*>(smemGroupBuffer + i * kNumTMABufferBytes + 8); });
        auto tmaLdBuffers = PatternVisitor([=](const int& i) { return reinterpret_cast<uint8_t* >(smemGroupBuffer + i * kNumTMABufferBytes + 16); });
        auto tmaStBuffers = PatternVisitor([=](const int& i) { return reinterpret_cast<uint32_t*>(smemGroupBuffer + kNumStages * kNumTMABufferBytes + i * kNumBF16PerWarpBytes); });

        // Redundant when logfmt is disabled
        const auto smemGroupPtr = smemGroupBuffer + kNumStages * kNumTMABufferBytes + kHidden * 2;
        auto logAmaxBuffers  = PatternVisitor([=](const int& i) { return reinterpret_cast<float*>(smemGroupPtr + i * kNumDivisionBytes); });
        auto logAminBuffers  = PatternVisitor([=](const int& i) { return reinterpret_cast<float*>(smemGroupPtr + kNumStages * kNumDivisionBytes + i * kNumDivisionBytes); });
        auto castInfoBuffers = PatternVisitor([=](const int& i) { return reinterpret_cast<int*>  (smemGroupPtr + kNumStages * kNumDivisionBytes * 2 + i * kNumDivisionBytes); });

        uint32_t tmaPhase = 0;
        EP_STATIC_ASSERT(kNumStages < 32, "Too many stages");
        if (decodeWarpIdx == numDecodeWarps)
            tmaPhase = (1 << kNumStages) - 1;

        // Initialize m-barriers
        if (decodeWarpIdx == numDecodeWarps and laneId < kNumStages) {
            mbarrier_init(fullBarriers[laneId], 1);
            mbarrier_init(emptyBarriers[laneId], numDecodeWarps);
        }
        asm volatile("bar.sync %0, %1;" :: "r"(groupIdx + 1), "r"((numDecodeWarps + 1) * 32));

        int stageIdx = 0, topkIdxByLane = 0;
        EP_STATIC_ASSERT(kNumMaxTopk <= 32, "numTopk must not exceed warp size: warpMarkFirstOccurrence relies on active lanes (0..numTopk-1) having lower IDs than inactive lanes (numTopk..31)");

        if (decodeWarpIdx == numDecodeWarps) {
            // TMA load warp
            for (int tokenIdx = smId + numSms * groupIdx; tokenIdx < numCombinedTokens; tokenIdx += numSms * numGroups) {
                if (laneId < numTopk)
                    topkIdxByLane = static_cast<int>(__ldg(inTopkIdx + tokenIdx * numTopk + laneId));

                if constexpr (kLayout == NCCL_EP_LAYOUT_RANK_MAJOR) {
                    topkIdxByLane = warpMarkFirstOccurrence(topkIdxByLane, numLocalExperts, laneId);
                }

                for (int i = 0; i < numTopk; ++ i) {
                    int topkIdxReg = __shfl_sync(0xffffffff, topkIdxByLane, i);
                    if (topkIdxReg < 0)
                        continue;
                    if (isRankMasked(rankMask, topkIdxReg / numLocalExperts))
                        continue;

                    uint64_t start_time = clock64();
                    mbarrier_wait<true>(emptyBarriers[stageIdx], tmaPhase, stageIdx);
                    uint64_t end_time = clock64();
                    auto recvBufData = static_cast<uint8_t*>(recvBuf) + (tokenIdx * numTopk + i) * numBytesPerSlot;

                    if constexpr (kUseLogFMT) {
                        logfmtCheckAmaxmin<kNumDivisions / 2, kNumSendUnrolls, kNumRecvUnrolls>(
                            recvBufData,
                            reinterpret_cast<float2*>(logAmaxBuffers[stageIdx]),
                            reinterpret_cast<float2*>(logAminBuffers[stageIdx]),
                            castInfoBuffers[stageIdx],
                            laneId);
                    }
                    if (elect_one_sync()) {
                        int numCasted = 0;
                        if constexpr (kUseLogFMT) {
                            const auto& info = castInfoBuffers[stageIdx][numDecodeWarps - 1];
                            numCasted = (info >> 1) + (info & 1);
                        }
                        int numTmaBytes = numCasted * kNumLogFMTPerWarpBytes + (numDecodeWarps - numCasted) * kNumBF16PerWarpBytes;
                        tma_load_1d(tmaLdBuffers[stageIdx], recvBufData + (kUseLogFMT ? kNumMetaBytes : 0), fullBarriers[stageIdx], numTmaBytes);
                        mbarrier_arrive_and_expect_tx(fullBarriers[stageIdx], numTmaBytes);
                    }
                    __syncwarp();
                    stageIdx = (stageIdx + 1) % kNumStages;
                }
            }
        } else {
            // Reduction warps
            float topkWeightsByLane;
            for (int tokenIdx = smId + numSms * groupIdx; tokenIdx < numCombinedTokens; tokenIdx += numSms * numGroups) {
                if (laneId < numTopk) {
                    topkIdxByLane = static_cast<int>(__ldg(inTopkIdx + tokenIdx * numTopk + laneId));
                    // Rank-major: weights applied by caller before combine; kernel uses weight=1
                    if constexpr (kLayout != NCCL_EP_LAYOUT_RANK_MAJOR)
                        topkWeightsByLane = __ldg(topkWeights + tokenIdx * numTopk + laneId);
                }
                __syncwarp();

                if constexpr (kLayout == NCCL_EP_LAYOUT_RANK_MAJOR) {
                    topkIdxByLane = warpMarkFirstOccurrence(topkIdxByLane, numLocalExperts, laneId);
                }

                float combinedData[kNumElemsPerInt4 * kNumRecvUnrolls] = {0.0f};
                for (int i = 0; i < numTopk; ++ i) {
                    int topkIdxReg = __shfl_sync(0xffffffff, topkIdxByLane, i);
                    if (topkIdxReg < 0)
                        continue;
                    if (isRankMasked(rankMask, topkIdxReg / numLocalExperts))
                        continue;
                    float topkWeight;
                    if constexpr (kLayout == NCCL_EP_LAYOUT_EXPERT_MAJOR) {
                        topkWeight = __shfl_sync(0xffffffff, topkWeightsByLane, i);
                    } else if constexpr (kLayout == NCCL_EP_LAYOUT_RANK_MAJOR) {
                        // Per-rank weight sum is applied by the caller before ncclEpCombine.
                        topkWeight = 1.0f;
                    }

                    mbarrier_wait<true>(fullBarriers[stageIdx], tmaPhase, stageIdx);
                    if constexpr (kUseLogFMT) {
                        const auto& info = castInfoBuffers[stageIdx][decodeWarpIdx];
                        bool enableCast = info & 1;
                        int numCastedPrefix = info >> 1;
                        int tmaOffset = kNumLogFMTPerWarpBytes * numCastedPrefix + kNumBF16PerWarpBytes * (decodeWarpIdx - numCastedPrefix);
                        int divisionIdx = decodeWarpIdx * (kNumRecvUnrolls * 2) + laneId * kNumRecvUnrolls / 16;
                        decodeAndAccumulate<kNumRecvUnrolls>(
                            reinterpret_cast<uint32_t*>(tmaLdBuffers[stageIdx] + tmaOffset +
                                                        (enableCast ? kNumLogFMTPerWarpBytes : kNumBF16PerWarpBytes) / 32 * laneId),
                            combinedData,
                            logAmaxBuffers[stageIdx][divisionIdx],
                            logAminBuffers[stageIdx][divisionIdx],
                            enableCast,
                            topkWeight);
                    } else {
                        int tmaOffset = kNumBF16PerWarpBytes * decodeWarpIdx;
                        decodeAndAccumulate<kNumRecvUnrolls>(
                            reinterpret_cast<uint32_t*>(tmaLdBuffers[stageIdx] + tmaOffset + kNumBF16PerWarpBytes / 32 * laneId),
                            combinedData,
                            0,
                            0,
                            false,
                            topkWeight);
                    }

                    if (elect_one_sync())
                        mbarrier_arrive(emptyBarriers[stageIdx]);
                    stageIdx = (stageIdx + 1) % kNumStages;
                }
                tma_store_wait<0>();

                #pragma unroll
                for (int k = 0; k < kNumRecvUnrolls * 4; ++ k) {
                    auto combinedPack = __nv_bfloat162(combinedData[k * 2], combinedData[k * 2 + 1]);
                    tmaStBuffers[decodeWarpIdx][kNumRecvUnrolls * 4 * laneId + k] = *reinterpret_cast<uint32_t*>(&combinedPack);
                }
                tma_store_fence();
                if (elect_one_sync()) {
                    tma_store_1d(tmaStBuffers[decodeWarpIdx],
                                 static_cast<int4*>(outData) + tokenIdx * hiddenBf16Int4 + decodeWarpIdx * kNumRecvUnrolls * 32,
                                 kNumBF16PerWarpBytes);
                }
                __syncwarp();
            }
        }
    }
}

// Constants for combine kernel template parameters
constexpr int kCombineMaxTopk = 9;
constexpr int kCombineMaxUnrolls = 4;

void combine(const void* inData,
             const int* srcInfo,
             const int64_t* layoutRange,
             const int64_t* inTopkIdx,
             const float* topkWeights,
             void* outData,
             void* sendBuf,
             void* recvBuf,
             int* recvFlagBuf,
             size_t sendOff,
             size_t recvOff,
             size_t recvFlagOff,
             int* nextRecvCntBuf,
             int nextRecvCntBufSize,
             int64_t* waitStats,
             int numCombinedTokens,
             int hidden,
             int maxTokensPerRank,
             int numTopk,
             int numExperts,
             int currRank,
             int numRanks,
             bool useLogfmt,
             ncclEpLayout_t layout,
             int phases,
             bool zeroCopy,
             int numComms,
             ncclDevComm* devComms,
             const ncclWindow_t* windows,
             unsigned signalsBase,
             void* workspace,
             int numDeviceSms,
             int* rankMask,
             int* asyncErrorFlag,
             uint64_t timeoutCycles,
             cudaStream_t stream) {
    const int numWarpGroups = ceil_div(numExperts, numDeviceSms);
    const int numWarpsPerGroup = 32 / numWarpGroups;
    const int numRecvPerSm = ceil_div(numCombinedTokens, numDeviceSms);
    EP_HOST_ASSERT(numWarpGroups > 0 and numWarpsPerGroup > 0 and numRecvPerSm >= 0);

    const auto numWarps = numWarpGroups * numWarpsPerGroup;
    const auto numSms = max(ceil_div(numExperts, numWarpGroups),
                             numRecvPerSm == 0 ? 1 : ceil_div(numCombinedTokens, numRecvPerSm));

    // Check workspace
    auto atomicCleanFlag = static_cast<int*>(workspace);
    EP_HOST_ASSERT(sizeof(int) <= NUM_WORKSPACE_BYTES);
    EP_HOST_ASSERT(numTopk <= kCombineMaxTopk);

    // Online cast cannot use zero-copy
    EP_HOST_ASSERT(not (zeroCopy and useLogfmt));

    constexpr int kNumStages = 3;
    constexpr int kMaxNumGroups = 2;

    // Send buffer size
    const int numMetaBytes = hidden / 128 * 4;
    const int numSendTmaBytes = 32 * sizeof(int4) * kCombineMaxUnrolls + 16;
    const int smemSendSize = numWarps * (kNumStages * numSendTmaBytes + numMetaBytes);

    // Receive buffer size
    const int numRecvTmaBytes = 16 + hidden * 2;
    const int smemRecvSize = kMaxNumGroups * (kNumStages * numRecvTmaBytes + hidden * 2 + kNumStages * numMetaBytes * 3);

    // Total requirement
    const int smem_size = max(smemSendSize, smemRecvSize);

    SETUP_LAUNCH_CONFIG(numSms, numWarps * 32, stream);
#define COMBINE_LAUNCH_CASE_IMPL(hidden, kLayout) { \
if (useLogfmt) { \
    SET_SHARED_MEMORY_FOR_TMA((combine<true, hidden, kCombineMaxTopk, kCombineMaxUnrolls, kLayout>)); \
    LAUNCH_KERNEL(&cfg, (combine<true, hidden, kCombineMaxTopk, kCombineMaxUnrolls, kLayout>), \
                  inData, \
                  srcInfo, \
                  layoutRange, \
                  inTopkIdx, \
                  topkWeights, \
                  rankMask, \
                  asyncErrorFlag, \
                  outData, \
                  sendBuf, \
                  recvBuf, \
                  recvFlagBuf, \
                  sendOff, \
                  recvOff, \
                  recvFlagOff, \
                  atomicCleanFlag, \
                  nextRecvCntBuf, \
                  nextRecvCntBufSize, \
                  waitStats, \
                  numCombinedTokens, \
                  hidden, \
                  numTopk, \
                  maxTokensPerRank, \
                  numExperts, \
                  currRank, \
                  numRanks, \
                  numWarpGroups, \
                  numWarpsPerGroup, \
                  phases, \
                  zeroCopy, \
                  numComms, \
                  devComms, \
                  windows, \
                  signalsBase, \
                  timeoutCycles); \
} else { \
    SET_SHARED_MEMORY_FOR_TMA((combine<false, hidden, kCombineMaxTopk, kCombineMaxUnrolls, kLayout>)); \
    LAUNCH_KERNEL(&cfg, (combine<false, hidden, kCombineMaxTopk, kCombineMaxUnrolls, kLayout>), \
                  inData, \
                  srcInfo, \
                  layoutRange, \
                  inTopkIdx, \
                  topkWeights, \
                  rankMask, \
                  asyncErrorFlag, \
                  outData, \
                  sendBuf, \
                  recvBuf, \
                  recvFlagBuf, \
                  sendOff, \
                  recvOff, \
                  recvFlagOff, \
                  atomicCleanFlag, \
                  nextRecvCntBuf, \
                  nextRecvCntBufSize, \
                  waitStats, \
                  numCombinedTokens, \
                  hidden, \
                  numTopk, \
                  maxTokensPerRank, \
                  numExperts, \
                  currRank, \
                  numRanks, \
                  numWarpGroups, \
                  numWarpsPerGroup, \
                  phases, \
                  zeroCopy, \
                  numComms, \
                  devComms, \
                  windows, \
                  signalsBase, \
                  timeoutCycles); \
} } break
#define COMBINE_LAUNCH_CASE_RM(hidden) COMBINE_LAUNCH_CASE_IMPL(hidden, NCCL_EP_LAYOUT_RANK_MAJOR)
#define COMBINE_LAUNCH_CASE_EM(hidden) COMBINE_LAUNCH_CASE_IMPL(hidden, NCCL_EP_LAYOUT_EXPERT_MAJOR)
    if (layout == NCCL_EP_LAYOUT_RANK_MAJOR) {
        SWITCH_HIDDEN(COMBINE_LAUNCH_CASE_RM);
    } else {
        SWITCH_HIDDEN(COMBINE_LAUNCH_CASE_EM);
    }
#undef COMBINE_LAUNCH_CASE_IMPL
#undef COMBINE_LAUNCH_CASE_RM
#undef COMBINE_LAUNCH_CASE_EM
}

// ============================================================================
// clean_low_latency_buffer: barrier → zero RDMA buffers → barrier
// Hybrid barrier: NVLink peers use P2P stores, RDMA peers use GIN signals.
// ============================================================================

template <int kNumThreads>
__forceinline__ __device__ void maskAwareBarrier(
    int threadId, int myRank, ncclDevComm& dcomm,
    int* rankMask, int* syncBuffer, const ncclWindow_t* syncWindow,
    unsigned barrierSignalBase,
    ncclDevComm* devComms,
    uint64_t timeoutCycles) {

    if (isRankMasked(rankMask, myRank)) return;

    int nRanks = dcomm.nRanks;
    EP_DEVICE_ASSERT(kNumThreads >= nRanks);

    // Decrement local sync counter (monotonically decreasing)
    if (threadId == 0)
        atomicAdd(syncBuffer + myRank, -1);
    __syncthreads();

    int cnt = syncBuffer[myRank];

    // syncBuffer is the entirety of its window — peer-slot offsets are
    // relative to the window base, not nested inside a larger region.
    auto syncWin = syncWindow[0];

    // Publish our counter to each active peer and wait for theirs.
    // NVLink (P2P) peers: direct store + load on the shared syncBuffer.
    // RDMA peers: GIN signal (0-byte put + SignalAdd) instead of putValue.
    if (threadId < nRanks && threadId != myRank) {
        int peer = threadId;
        if (!isRankMasked(rankMask, peer)) {
            size_t peerSlotOffset = myRank * sizeof(int);
            auto p2pPtr = ncclGetP2pPtr(
                reinterpret_cast<uint64_t>(syncBuffer), peerSlotOffset,
                myRank, peer, syncWindow, devComms);

            if (p2pPtr == 0) {
                // RDMA peer: use GIN signal
                constexpr int commId = 0;
                auto ctxId = peer % MAX_NCCL_GIN_CTX_PER_COMM;
                ncclGin net(devComms[commId], ctxId);
                ncclTeam world = ncclTeamWorld(devComms[commId]);
                net.put(world, peer,
                        syncWin, peerSlotOffset,
                        syncWin, 0,
                        0,
                        ncclGin_SignalAdd{barrierSignalBase, 1},
                        ncclGin_None{},
                        ncclCoopThread());
            } else {
                // NVLink peer: direct P2P store
                st_release_sys_global(reinterpret_cast<int*>(p2pPtr), cnt);
            }
        }
    }
    __syncthreads();

    // Wait for all active peers
    if (threadId < nRanks && threadId != myRank) {
        int peer = threadId;
        if (!isRankMasked(rankMask, peer)) {
            size_t peerSlotOffset = peer * sizeof(int);
            auto p2pPtr = ncclGetP2pPtr(
                reinterpret_cast<uint64_t>(syncBuffer), peerSlotOffset,
                myRank, peer, syncWindow, devComms);

            if (p2pPtr != 0) {
                // NVLink peer: poll syncBuffer directly
                auto startTime = clock64();
                uint64_t elapsed = 0;
                while (ld_acquire_sys_global(syncBuffer + peer) != cnt
                       && (elapsed = clock64() - startTime) <= timeoutCycles)
                    ;
                if (elapsed > timeoutCycles) {
                    printf("Warning: NCCL EP clean barrier timeout (P2P), myRank %d, peer %d\n",
                           myRank, peer);
                    atomicExch(rankMask + peer, 0);
                }
            }
        }
    }

    // RDMA peers: wait on GIN signal (thread 0 handles aggregate)
    if (threadId == 0) {
        constexpr int commId = 0;
        auto ctxId = myRank % MAX_NCCL_GIN_CTX_PER_COMM;
        ncclGin net(devComms[commId], ctxId);

        int numExpectedSignals = 0;
        for (int r = 0; r < nRanks; r++) {
            if (r == myRank || isRankMasked(rankMask, r)) continue;
            auto p2p = ncclGetP2pPtr(0x01, 0, myRank, r, syncWindow, devComms);
            if (p2p == 0)
                numExpectedSignals++;
        }

        if (numExpectedSignals > 0) {
            auto startTime = clock64();
            uint64_t elapsed = 0;
            while (net.readSignal(barrierSignalBase) < static_cast<uint64_t>(numExpectedSignals)
                   && (elapsed = clock64() - startTime) <= timeoutCycles)
                ;
            net.resetSignal(barrierSignalBase);

            if (elapsed > timeoutCycles) {
                printf("Warning: NCCL EP clean barrier timeout (GIN), myRank %d\n", myRank);
                for (int r = 0; r < nRanks; r++) {
                    if (r == myRank || isRankMasked(rankMask, r)) continue;
                    auto p2p = ncclGetP2pPtr(0x01, 0, myRank, r, syncWindow, devComms);
                    if (p2p == 0)
                        atomicExch(rankMask + r, 0);
                }
            }
        }
    }
    __syncthreads();
}

template <int kNumThreads>
__launch_bounds__(kNumThreads, 1)
__global__ void cleanLowLatencyBufferKernel(
    int* clean_0, int num_clean_int_0,
    int* clean_1, int num_clean_int_1,
    int* rankMask,
    int* syncBuffer, ncclWindow_t* syncWindow,
    ncclDevComm* devComms,
    unsigned barrierSignalBase,
    uint64_t timeoutCycles) {

    int threadId = static_cast<int>(threadIdx.x);
    auto dcomm = devComms[0];

    // Pre-clean barrier
    if (rankMask == nullptr) {
        ncclGin net(dcomm, 0);
        ncclGinBarrierSession<ncclCoopCta> bar(ncclCoopCta(), net,
            ncclTeamTagWorld(), blockIdx.x);
        bar.sync(ncclCoopCta(), cuda::memory_order_relaxed,
                 ncclGinFenceLevel::Relaxed);
    } else {
        maskAwareBarrier<kNumThreads>(threadId, dcomm.rank, dcomm,
                                       rankMask, syncBuffer, syncWindow,
                                       barrierSignalBase,
                                       devComms, timeoutCycles);
    }

    // Zero out RDMA buffers
    for (int i = threadId; i < num_clean_int_0; i += kNumThreads)
        clean_0[i] = 0;
    for (int i = threadId; i < num_clean_int_1; i += kNumThreads)
        clean_1[i] = 0;
    __threadfence_system();

    // Post-clean barrier
    if (rankMask == nullptr) {
        ncclGin net(dcomm, 0);
        ncclGinBarrierSession<ncclCoopCta> bar(ncclCoopCta(), net,
            ncclTeamTagWorld(), blockIdx.x);
        bar.sync(ncclCoopCta(), cuda::memory_order_relaxed,
                 ncclGinFenceLevel::Relaxed);
    } else {
        maskAwareBarrier<kNumThreads>(threadId, dcomm.rank, dcomm,
                                       rankMask, syncBuffer, syncWindow,
                                       barrierSignalBase,
                                       devComms, timeoutCycles);
    }
}

void clean_low_latency_buffer(int* clean_0, int num_clean_int_0,
                              int* clean_1, int num_clean_int_1,
                              int* rankMask,
                              int* syncBuffer, ncclWindow_t* syncWindow,
                              ncclDevComm* devComms,
                              unsigned barrierSignalBase,
                              uint64_t timeoutCycles,
                              cudaStream_t stream) {
    constexpr int kNumThreads = 256;
    SETUP_LAUNCH_CONFIG(1, kNumThreads, stream);
    LAUNCH_KERNEL(&cfg, cleanLowLatencyBufferKernel<kNumThreads>,
                  clean_0, num_clean_int_0, clean_1, num_clean_int_1,
                  rankMask, syncBuffer, syncWindow, devComms,
                  barrierSignalBase, timeoutCycles);
}

} // namespace internode_ll

} // namespace nccl_ep
