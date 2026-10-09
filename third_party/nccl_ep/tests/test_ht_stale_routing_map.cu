/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Regression test for the HT global_routing_map stale-tail bug.
 *
 *   Introduced by commit a22f699b7 ("HT handle memory optimization") which moved
 *   global_routing_map to a group-scoped allocation and removed the per-iteration
 *   cudaMemsetAsync in ncclEpUpdateHandle. The convert_topk_to_routing_map kernel
 *   only writes rows 0..num_tokens-1 of the local routing map, but the subsequent
 *   ncclAllGather (in ncclEpUpdateHandle) ships the full max_dispatch_tokens_per_rank
 *   rows per rank to peers. Tail rows num_tokens..max_dispatch_tokens_per_rank-1
 *   retain stale bits from the previous iteration. Peers interpret those stale bits
 *   as live routing and the HT dispatch kernel follows them to out-of-range token
 *   slots, producing CUDA_ERROR_ILLEGAL_ADDRESS on the second iteration's dispatch.
 *
 *   Fixed by commit 12b2607b4 — a single cudaMemsetAsync on the local routing
 *   send pointer in ncclEpUpdateHandle before the convert kernel, clearing the
 *   tail rows so AllGather doesn't ship stale data.
 *
 * Test recipe:
 *   1. ncclEpInitHandle (HT FLAT, kTopK=1, kNumExperts=8, max_tokens=kNumTokens=4).
 *   2. Iteration 1: ncclEpUpdateHandle with num_tokens=kNumTokens=4 and a topk_idx
 *      that touches every expert across ranks. Run forward dispatch + combine once
 *      so the rest of the handle state is consistent; this also forces the routing
 *      map for rows 0..3 to be fully populated on every rank's local send-region.
 *   3. Iteration 2: ncclEpUpdateHandle with num_tokens=1 and a topk_idx that routes
 *      the single token only to expert 0. The convert kernel writes row 0 only;
 *      rows 1..3 retain iter-1 routing. AllGather ships those stale rows to peers.
 *   4. ncclEpDispatch on iter-2 inputs. Without the fix, the HT dispatch kernel
 *      follows the stale tail bits and writes/reads past the recv-buffer end —
 *      cudaStreamSynchronize returns a non-success CUDA error (typically
 *      cudaErrorIllegalAddress / cudaErrorMisalignedAddress). With the fix, the
 *      tail rows are zeroed, dispatch completes cleanly, and the synchronize
 *      returns cudaSuccess.
 *
 * The test asserts that the second-iteration dispatch + sync succeed. Running this
 * test on origin/master is expected to fail (illegal address). Running it with the
 * one-liner fix applied is expected to pass.
 */

#include "test_common.h"
#include "../nccl_ep_test_internal.h"

// ── Fixture ───────────────────────────────────────────────────────────────────
//
// We do not inherit EpTestBase here because the base fixture pre-creates a
// kNumTokens-sized topk_idx tensor. The second iteration needs a different,
// smaller topk_idx tensor (num_tokens = 1), so we manage tensors locally.

class HtStaleRoutingMapTest : public ::testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {}
};

// One token's worth of expert assignment in iter-2. Routing the lone token only
// to a low-index expert ensures the kernel's iter-2 forward routing itself is
// trivial; the crash comes purely from the *stale tail* shipped via AllGather.
static constexpr int64_t kIter2Expert = 0;

TEST_F(HtStaleRoutingMapTest, StaleTailTriggersIllegalAddress) {
    // ── iter-1 topk_idx [kNumTokens, kTopK] ──────────────────────────────────
    // Use the same per-rank/per-token expert assignment as test_common.h's
    // expert_for_token() so iter-1 routing exercises every expert at least once
    // across the 4 ranks (kNumExperts=8 = nranks * kNumTokens).
    int64_t* d_topk_iter1 = nullptr;
    CUDA_ASSERT(cudaMalloc(&d_topk_iter1, kNumTokens * kTopK * sizeof(int64_t)));
    int64_t h_iter1[kNumTokens];
    for (int i = 0; i < kNumTokens; ++i) h_iter1[i] = expert_for_token(i);
    CUDA_ASSERT(cudaMemcpy(d_topk_iter1, h_iter1, sizeof(h_iter1), cudaMemcpyHostToDevice));

    ncclEpTensor_t* t_topk_iter1 = nullptr;
    NCCL_ASSERT(epTensorCreate(&t_topk_iter1, 2, ncclInt64,
                               d_topk_iter1, kNumTokens, kTopK));

    // ── iter-2 topk_idx [1, kTopK] ───────────────────────────────────────────
    // Single token; routed to expert 0 only. Forward-pass routing for this
    // iteration is trivial; the bug is triggered by stale tail rows.
    constexpr int kIter2NumTokens = 1;
    int64_t* d_topk_iter2 = nullptr;
    CUDA_ASSERT(cudaMalloc(&d_topk_iter2, kIter2NumTokens * kTopK * sizeof(int64_t)));
    int64_t h_iter2[kIter2NumTokens * kTopK] = { kIter2Expert };
    CUDA_ASSERT(cudaMemcpy(d_topk_iter2, h_iter2, sizeof(h_iter2), cudaMemcpyHostToDevice));

    ncclEpTensor_t* t_topk_iter2 = nullptr;
    NCCL_ASSERT(epTensorCreate(&t_topk_iter2, 2, ncclInt64,
                               d_topk_iter2, kIter2NumTokens, kTopK));

    // ── init the HT handle (FLAT layout) ─────────────────────────────────────
    ncclEpHandle_t h = nullptr;
    NCCL_ASSERT(ncclEpInitHandle(&h, g_ep_group, NCCL_EP_LAYOUT_FLAT,
                                 /*config=*/nullptr, kTopK,
                                 /*handle_mem=*/nullptr));
    ASSERT_NE(h, nullptr);

    // ── Iteration 1: full num_tokens, dirty the routing map ──────────────────
    NCCL_ASSERT(ncclEpUpdateHandle(h, t_topk_iter1, nullptr, g_stream));
    CUDA_ASSERT(cudaStreamSynchronize(g_stream));

    // Run a full forward dispatch + combine on iter-1 to ensure the handle is
    // in a quiesced, valid state — and to make the stale-tail behaviour purely
    // a function of ncclEpUpdateHandle's (missing) memset.
    {
        std::vector<nv_bfloat16> h_tok(kNumTokens * kHidden);
        for (int i = 0; i < kNumTokens; ++i) {
            float v = static_cast<float>(g_rank * kNumTokens + i + 1);
            for (int hh = 0; hh < kHidden; ++hh)
                h_tok[i * kHidden + hh] = __float2bfloat16(v);
        }
        std::vector<float> h_w(kNumTokens * kTopK, 1.0f);

        nv_bfloat16 *d_tok = nullptr, *d_recv = nullptr;
        float       *d_w   = nullptr, *d_recv_w = nullptr;
        int64_t     *d_recv_idx = nullptr;
        CUDA_ASSERT(cudaMalloc(&d_tok,      kNumTokens    * kHidden * sizeof(nv_bfloat16)));
        CUDA_ASSERT(cudaMalloc(&d_recv,     kMaxRecvSlots * kHidden * sizeof(nv_bfloat16)));
        CUDA_ASSERT(cudaMemset(d_recv, 0,   kMaxRecvSlots * kHidden * sizeof(nv_bfloat16)));
        CUDA_ASSERT(cudaMalloc(&d_w,        kNumTokens    * kTopK   * sizeof(float)));
        CUDA_ASSERT(cudaMalloc(&d_recv_w,   kMaxRecvSlots * kTopK   * sizeof(float)));
        CUDA_ASSERT(cudaMemset(d_recv_w, 0, kMaxRecvSlots * kTopK   * sizeof(float)));
        CUDA_ASSERT(cudaMalloc(&d_recv_idx, kMaxRecvSlots * kTopK   * sizeof(int64_t)));
        CUDA_ASSERT(cudaMemcpy(d_tok, h_tok.data(),
                               kNumTokens * kHidden * sizeof(nv_bfloat16),
                               cudaMemcpyHostToDevice));
        CUDA_ASSERT(cudaMemcpy(d_w, h_w.data(),
                               kNumTokens * kTopK * sizeof(float),
                               cudaMemcpyHostToDevice));

        ncclEpTensor_t *t_tok = nullptr, *t_recv = nullptr,
                       *t_w   = nullptr, *t_recv_w = nullptr,
                       *t_recv_idx = nullptr;
        NCCL_ASSERT(epTensorCreate(&t_tok,      2, ncclBfloat16, d_tok,      kNumTokens,    kHidden));
        NCCL_ASSERT(epTensorCreate(&t_recv,     2, ncclBfloat16, d_recv,     kMaxRecvSlots, kHidden));
        NCCL_ASSERT(epTensorCreate(&t_w,        2, ncclFloat32,  d_w,        kNumTokens,    kTopK));
        NCCL_ASSERT(epTensorCreate(&t_recv_w,   2, ncclFloat32,  d_recv_w,   kMaxRecvSlots, kTopK));
        NCCL_ASSERT(epTensorCreate(&t_recv_idx, 2, ncclInt64,    d_recv_idx, kMaxRecvSlots, kTopK));

        ncclEpDispatchInputs_t  d_in  = NCCL_EP_DISPATCH_INPUTS_INIT;
        ncclEpDispatchOutputs_t d_out = NCCL_EP_DISPATCH_OUTPUTS_INIT;
        d_in.tokens        = t_tok;
        d_in.topk_weights  = t_w;
        d_out.tokens       = t_recv;
        d_out.topk_weights = t_recv_w;
        d_out.topk_idx     = t_recv_idx;
        ncclEpDispatchConfig_t dcfg = NCCL_EP_DISPATCH_CONFIG_INIT;
        NCCL_ASSERT(ncclEpDispatch(h, &d_in, &d_out, nullptr, &dcfg, g_stream));
        NCCL_ASSERT(ncclEpComplete(h, nullptr, g_stream));
        CUDA_ASSERT(cudaStreamSynchronize(g_stream));

        ncclEpTensorDestroy(t_recv_idx);
        ncclEpTensorDestroy(t_recv_w);
        ncclEpTensorDestroy(t_w);
        ncclEpTensorDestroy(t_recv);
        ncclEpTensorDestroy(t_tok);
        cudaFree(d_recv_idx);
        cudaFree(d_recv_w);
        cudaFree(d_w);
        cudaFree(d_recv);
        cudaFree(d_tok);
    }

    // ── Iteration 2: tiny num_tokens, route only to expert 0 ─────────────────
    // ncclEpUpdateHandle's convert_topk_to_routing_map writes row 0 only.
    // Without the fix, rows 1..(max_tokens-1) keep the iter-1 bits, and
    // AllGather ships those stale rows to peers.
    NCCL_ASSERT(ncclEpUpdateHandle(h, t_topk_iter2, nullptr, g_stream));
    CUDA_ASSERT(cudaStreamSynchronize(g_stream));

    // Read post-allgather routing-derived recv count. With the fix in place,
    // iter-2's lone token (routed to expert 0) means only rank 0 receives
    // anything (4 copies — one per sending rank). Without the fix, stale
    // iter-1 tail rows shipped via AllGather inflate per-rank counts on
    // every rank — see project_ncclep_ht_alloc_callback_segv.md.
    static constexpr unsigned int kExpectedRecvWithFix[4] = {4, 0, 0, 0};
    ASSERT_LT(g_rank, 4);
    unsigned int num_recv = 0;
    NCCL_ASSERT(ncclEpHandle_test_getNumRecvTokens(h, &num_recv));
    EXPECT_EQ(num_recv, kExpectedRecvWithFix[g_rank])
        << "Rank " << g_rank
        << ": iter-2 num_recv_tokens=" << num_recv
        << ", expected " << kExpectedRecvWithFix[g_rank]
        << ". A mismatch on rank g indicates stale routing-map tail bits "
           "from iter-1 were shipped to peers via AllGather — the fix at "
           "nccl_ep.cc:2082 (cudaMemsetAsync of the local routing send slot) "
           "is missing.";

    // Iter-2 dispatch buffers. The token buffer is sized to the iter-2
    // num_tokens (1), so any stale-tail-driven sender-side read past row 0 is
    // an OOB device access; on the receive side, stale bits inflate the slot
    // count past kMaxRecvSlots * kHidden, also OOB.
    std::vector<nv_bfloat16> h_tok2(kIter2NumTokens * kHidden);
    for (int hh = 0; hh < kHidden; ++hh)
        h_tok2[hh] = __float2bfloat16(static_cast<float>(g_rank + 1));
    std::vector<float> h_w2(kIter2NumTokens * kTopK, 1.0f);

    nv_bfloat16 *d_tok2 = nullptr, *d_recv2 = nullptr;
    float       *d_w2   = nullptr, *d_recv_w2 = nullptr;
    int64_t     *d_recv_idx2 = nullptr;
    CUDA_ASSERT(cudaMalloc(&d_tok2,      kIter2NumTokens * kHidden * sizeof(nv_bfloat16)));
    CUDA_ASSERT(cudaMalloc(&d_recv2,     kMaxRecvSlots   * kHidden * sizeof(nv_bfloat16)));
    CUDA_ASSERT(cudaMemset(d_recv2, 0,   kMaxRecvSlots   * kHidden * sizeof(nv_bfloat16)));
    CUDA_ASSERT(cudaMalloc(&d_w2,        kIter2NumTokens * kTopK   * sizeof(float)));
    CUDA_ASSERT(cudaMalloc(&d_recv_w2,   kMaxRecvSlots   * kTopK   * sizeof(float)));
    CUDA_ASSERT(cudaMemset(d_recv_w2, 0, kMaxRecvSlots   * kTopK   * sizeof(float)));
    CUDA_ASSERT(cudaMalloc(&d_recv_idx2, kMaxRecvSlots   * kTopK   * sizeof(int64_t)));
    CUDA_ASSERT(cudaMemcpy(d_tok2, h_tok2.data(),
                           kIter2NumTokens * kHidden * sizeof(nv_bfloat16),
                           cudaMemcpyHostToDevice));
    CUDA_ASSERT(cudaMemcpy(d_w2, h_w2.data(),
                           kIter2NumTokens * kTopK * sizeof(float),
                           cudaMemcpyHostToDevice));

    ncclEpTensor_t *t_tok2 = nullptr, *t_recv2 = nullptr,
                   *t_w2   = nullptr, *t_recv_w2 = nullptr,
                   *t_recv_idx2 = nullptr;
    NCCL_ASSERT(epTensorCreate(&t_tok2,      2, ncclBfloat16, d_tok2,      kIter2NumTokens, kHidden));
    NCCL_ASSERT(epTensorCreate(&t_recv2,     2, ncclBfloat16, d_recv2,     kMaxRecvSlots,   kHidden));
    NCCL_ASSERT(epTensorCreate(&t_w2,        2, ncclFloat32,  d_w2,        kIter2NumTokens, kTopK));
    NCCL_ASSERT(epTensorCreate(&t_recv_w2,   2, ncclFloat32,  d_recv_w2,   kMaxRecvSlots,   kTopK));
    NCCL_ASSERT(epTensorCreate(&t_recv_idx2, 2, ncclInt64,    d_recv_idx2, kMaxRecvSlots,   kTopK));

    ncclEpDispatchInputs_t  d_in2  = NCCL_EP_DISPATCH_INPUTS_INIT;
    ncclEpDispatchOutputs_t d_out2 = NCCL_EP_DISPATCH_OUTPUTS_INIT;
    d_in2.tokens        = t_tok2;
    d_in2.topk_weights  = t_w2;
    d_out2.tokens       = t_recv2;
    d_out2.topk_weights = t_recv_w2;
    d_out2.topk_idx     = t_recv_idx2;
    ncclEpDispatchConfig_t dcfg2 = NCCL_EP_DISPATCH_CONFIG_INIT;

    // Capture both the API return code and the post-kernel CUDA error.
    // Without the fix: cudaStreamSynchronize is expected to return
    //   cudaErrorIllegalAddress (or similar OOB-class error).
    // With the fix: both return success.
    const ncclResult_t disp_ret = ncclEpDispatch(h, &d_in2, &d_out2, nullptr, &dcfg2, g_stream);
    const ncclResult_t comp_ret = (disp_ret == ncclSuccess)
                                    ? ncclEpComplete(h, nullptr, g_stream)
                                    : ncclSuccess;
    const cudaError_t  sync_err = cudaStreamSynchronize(g_stream);

    // Clear any sticky CUDA error so teardown of subsequent state doesn't
    // cascade-fail. (cudaGetLastError clears the last error.)
    const cudaError_t  last_err = (sync_err == cudaSuccess) ? cudaSuccess : cudaGetLastError();
    (void)last_err;

    EXPECT_EQ(disp_ret, ncclSuccess)
        << "Rank " << g_rank
        << ": iter-2 ncclEpDispatch returned error " << disp_ret
        << ". Stale tail rows of global_routing_map shipped via AllGather "
           "drove the HT dispatch kernel to an out-of-range slot. "
           "Apply the cudaMemsetAsync fix in ncclEpUpdateHandle.";
    EXPECT_EQ(comp_ret, ncclSuccess)
        << "Rank " << g_rank
        << ": iter-2 ncclEpComplete returned error " << comp_ret;
    EXPECT_EQ(sync_err, cudaSuccess)
        << "Rank " << g_rank
        << ": cudaStreamSynchronize after iter-2 dispatch returned CUDA error "
        << sync_err << " (" << cudaGetErrorName(sync_err) << "): "
        << cudaGetErrorString(sync_err)
        << ". Expected cudaSuccess — without the fix this is typically "
           "cudaErrorIllegalAddress, produced by the HT dispatch kernel "
           "following stale-tail routing bits to an OOB token slot.";

    // ── Cleanup ──────────────────────────────────────────────────────────────
    ncclEpTensorDestroy(t_recv_idx2);
    ncclEpTensorDestroy(t_recv_w2);
    ncclEpTensorDestroy(t_w2);
    ncclEpTensorDestroy(t_recv2);
    ncclEpTensorDestroy(t_tok2);
    cudaFree(d_recv_idx2);
    cudaFree(d_recv_w2);
    cudaFree(d_w2);
    cudaFree(d_recv2);
    cudaFree(d_tok2);

    ncclEpTensorDestroy(t_topk_iter2);
    ncclEpTensorDestroy(t_topk_iter1);
    cudaFree(d_topk_iter2);
    cudaFree(d_topk_iter1);

    // Best-effort handle destroy. If the kernel crashed earlier, the handle
    // teardown may itself report errors; we have already recorded the
    // diagnostic above.
    (void)ncclEpHandleDestroy(h);
}

// ── main ──────────────────────────────────────────────────────────────────────

int main(int argc, char* argv[]) {
    if (!ep_bootstrap(argc, argv, "te_ep_ht_stale_routing_map_uid")) return 0;
    int ret = RUN_ALL_TESTS();
    ep_teardown();
    return ret;
}
