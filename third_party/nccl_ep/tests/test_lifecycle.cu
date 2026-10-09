/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Unit tests for HT handle lifecycle:
 *   SeparatedInitUpdate — ncclEpInitHandle followed by ncclEpUpdateHandle
 *                         (no implicit Update inside Create), then dispatch+combine.
 *                         Same handle reused across two iterations with the same routing.
 *   CallerOwnedHandleMem — caller allocates handle_mem (>= ncclEpHandleMemSize) and
 *                          passes a 1D ncclUint8 tensor to ncclEpInitHandle.
 *                          ncclEpHandleDestroy must not free the caller buffer.
 *   MultiIterReuse        — single ncclEpCreateHandle followed by 4 back-to-back
 *                          dispatch+combine pairs (buffer ping-pong / signal counter
 *                          correctness across iterations).
 *
 * Setup: same as test_output_layout (4 ranks, 8 experts, top-k=1, FLAT/RM by default).
 */

#include "test_common.h"
#include "../nccl_ep_test_internal.h"

static float bf16_val(nv_bfloat16 v) { return __bfloat162float(v); }

class LifecycleTest : public EpTestBase {
protected:
    // Run one dispatch+combine round under FLAT (rank-major) layout on the given handle.
    // Returns the per-token first-hidden-element bf16 value vector of length kNumTokens.
    std::vector<float> run_dispatch_combine_flat(ncclEpHandle_t handle) {
        std::vector<nv_bfloat16> h_tok(kNumTokens * kHidden);
        for (int i = 0; i < kNumTokens; ++i) {
            float v = static_cast<float>(g_rank * kNumTokens + i + 1);
            for (int hh = 0; hh < kHidden; ++hh)
                h_tok[i * kHidden + hh] = __float2bfloat16(v);
        }

        nv_bfloat16 *d_tok, *d_recv, *d_out;
        float       *d_weights, *d_recv_w;
        int64_t     *d_recv_idx;
        EXPECT_EQ(cudaMalloc(&d_tok,      kNumTokens    * kHidden * sizeof(nv_bfloat16)), cudaSuccess);
        EXPECT_EQ(cudaMalloc(&d_recv,     kMaxRecvSlots * kHidden * sizeof(nv_bfloat16)), cudaSuccess);
        EXPECT_EQ(cudaMalloc(&d_out,      kNumTokens    * kHidden * sizeof(nv_bfloat16)), cudaSuccess);
        EXPECT_EQ(cudaMalloc(&d_weights,  kNumTokens    * kTopK   * sizeof(float)),       cudaSuccess);
        EXPECT_EQ(cudaMalloc(&d_recv_w,   kMaxRecvSlots * kTopK   * sizeof(float)),       cudaSuccess);
        EXPECT_EQ(cudaMalloc(&d_recv_idx, kMaxRecvSlots * kTopK   * sizeof(int64_t)),     cudaSuccess);
        EXPECT_EQ(cudaMemset(d_recv, 0,   kMaxRecvSlots * kHidden * sizeof(nv_bfloat16)), cudaSuccess);

        std::vector<float> h_w(kNumTokens * kTopK, 1.0f);
        EXPECT_EQ(cudaMemcpy(d_tok,     h_tok.data(), kNumTokens*kHidden*sizeof(nv_bfloat16), cudaMemcpyHostToDevice), cudaSuccess);
        EXPECT_EQ(cudaMemcpy(d_weights, h_w.data(),   kNumTokens*kTopK*sizeof(float),         cudaMemcpyHostToDevice), cudaSuccess);

        ncclEpTensor_t *t_tok, *t_recv, *t_out, *t_w, *t_recv_w, *t_recv_idx;
        EXPECT_EQ(epTensorCreate(&t_tok,      2, ncclBfloat16, d_tok,      kNumTokens,    kHidden), ncclSuccess);
        EXPECT_EQ(epTensorCreate(&t_recv,     2, ncclBfloat16, d_recv,     kMaxRecvSlots, kHidden), ncclSuccess);
        EXPECT_EQ(epTensorCreate(&t_out,      2, ncclBfloat16, d_out,      kNumTokens,    kHidden), ncclSuccess);
        EXPECT_EQ(epTensorCreate(&t_w,        2, ncclFloat32,  d_weights,  kNumTokens,    kTopK),   ncclSuccess);
        EXPECT_EQ(epTensorCreate(&t_recv_w,   2, ncclFloat32,  d_recv_w,   kMaxRecvSlots, kTopK),   ncclSuccess);
        EXPECT_EQ(epTensorCreate(&t_recv_idx, 2, ncclInt64,    d_recv_idx, kMaxRecvSlots, kTopK),   ncclSuccess);

        ncclEpDispatchInputs_t  d_in  = NCCL_EP_DISPATCH_INPUTS_INIT;
        ncclEpDispatchOutputs_t d_out_s = NCCL_EP_DISPATCH_OUTPUTS_INIT;
        d_in.tokens         = t_tok;
        d_in.topk_weights   = t_w;
        d_out_s.tokens       = t_recv;
        d_out_s.topk_weights = t_recv_w;
        d_out_s.topk_idx     = t_recv_idx;
        ncclEpDispatchConfig_t dcfg = NCCL_EP_DISPATCH_CONFIG_INIT;
        EXPECT_EQ(ncclEpDispatch(handle, &d_in, &d_out_s, nullptr, &dcfg, g_stream), ncclSuccess);
        EXPECT_EQ(ncclEpComplete(handle, nullptr, g_stream), ncclSuccess);
        EXPECT_EQ(cudaStreamSynchronize(g_stream), cudaSuccess);

        ncclEpCombineInputs_t  c_in  = NCCL_EP_COMBINE_INPUTS_INIT;
        ncclEpCombineOutputs_t c_out_s = NCCL_EP_COMBINE_OUTPUTS_INIT;
        c_in.tokens   = t_recv;
        c_out_s.tokens = t_out;
        EXPECT_EQ(ncclEpCombine(handle, &c_in, &c_out_s, nullptr, g_stream), ncclSuccess);
        EXPECT_EQ(cudaStreamSynchronize(g_stream), cudaSuccess);

        std::vector<nv_bfloat16> h_out(kNumTokens * kHidden);
        EXPECT_EQ(cudaMemcpy(h_out.data(), d_out, kNumTokens*kHidden*sizeof(nv_bfloat16), cudaMemcpyDeviceToHost), cudaSuccess);

        ncclEpTensorDestroy(t_tok);
        ncclEpTensorDestroy(t_recv);
        ncclEpTensorDestroy(t_out);
        ncclEpTensorDestroy(t_w);
        ncclEpTensorDestroy(t_recv_w);
        ncclEpTensorDestroy(t_recv_idx);
        cudaFree(d_tok); cudaFree(d_recv); cudaFree(d_out);
        cudaFree(d_weights); cudaFree(d_recv_w); cudaFree(d_recv_idx);

        std::vector<float> vals(kNumTokens);
        for (int i = 0; i < kNumTokens; ++i)
            vals[i] = bf16_val(h_out[i * kHidden]);
        return vals;
    }
};

// ── Test: SeparatedInitUpdate — Init then Update, dispatch+combine ────────────
// ncclEpCreateHandle bundles Init+Update; this exercises them separately to
// verify Update alone establishes the routing for subsequent dispatch.

TEST_F(LifecycleTest, SeparatedInitUpdate) {
    ncclEpHandle_t h = nullptr;
    NCCL_ASSERT(ncclEpInitHandle(&h, g_ep_group, NCCL_EP_LAYOUT_FLAT, nullptr, kTopK, nullptr));
    ASSERT_NE(h, nullptr);
    NCCL_ASSERT(ncclEpUpdateHandle(h, topk_idx_, nullptr, g_stream));
    CUDA_ASSERT(cudaStreamSynchronize(g_stream));

    unsigned int num_recv = 0;
    NCCL_ASSERT(ncclEpHandle_test_getNumRecvTokens(h, &num_recv));
    EXPECT_EQ(num_recv, 4u) << "FLAT recv count for top-k=1 fixed routing";

    auto combined = run_dispatch_combine_flat(h);
    for (int i = 0; i < kNumTokens; ++i) {
        float expected = static_cast<float>(g_rank * kNumTokens + i + 1);
        EXPECT_NEAR(combined[i], expected, 0.5f) << "rank " << g_rank << " token " << i;
    }

    NCCL_ASSERT(ncclEpHandleDestroy(h));
}

// ── Test: SeparatedInitUpdate_TwoUpdates — Init once, Update twice, reuse ─────
// Same handle, second Update with the same topk_idx (re-runs allgather + preprocess).
// Both iterations must yield identical correct results.

TEST_F(LifecycleTest, SeparatedInitUpdateTwoUpdates) {
    ncclEpHandle_t h = nullptr;
    NCCL_ASSERT(ncclEpInitHandle(&h, g_ep_group, NCCL_EP_LAYOUT_FLAT, nullptr, kTopK, nullptr));
    ASSERT_NE(h, nullptr);

    NCCL_ASSERT(ncclEpUpdateHandle(h, topk_idx_, nullptr, g_stream));
    CUDA_ASSERT(cudaStreamSynchronize(g_stream));
    auto combined_a = run_dispatch_combine_flat(h);

    NCCL_ASSERT(ncclEpUpdateHandle(h, topk_idx_, nullptr, g_stream));
    CUDA_ASSERT(cudaStreamSynchronize(g_stream));
    auto combined_b = run_dispatch_combine_flat(h);

    for (int i = 0; i < kNumTokens; ++i) {
        float expected = static_cast<float>(g_rank * kNumTokens + i + 1);
        EXPECT_NEAR(combined_a[i], expected, 0.5f) << "first iter token " << i;
        EXPECT_NEAR(combined_b[i], expected, 0.5f) << "second iter token " << i;
    }

    NCCL_ASSERT(ncclEpHandleDestroy(h));
}

// ── Test: CallerOwnedHandleMem — caller-owned handle_mem buffer ───────────────
// Caller queries the required size, allocates, wraps as ncclUint8 1D tensor,
// passes to ncclEpInitHandle. Verifies roundtrip correctness and that
// ncclEpHandleDestroy does not free the caller buffer (we cudaFree it ourselves).

TEST_F(LifecycleTest, CallerOwnedHandleMem) {
    size_t handle_mem_bytes = 0;
    NCCL_ASSERT(ncclEpHandleMemSize(g_ep_group, NCCL_EP_LAYOUT_FLAT, nullptr, &handle_mem_bytes, kTopK));
    ASSERT_GT(handle_mem_bytes, 0u);

    void* d_handle_mem = nullptr;
    CUDA_ASSERT(cudaMalloc(&d_handle_mem, handle_mem_bytes));

    ncclEpTensor_t* t_handle_mem = nullptr;
    NCCL_ASSERT(epTensorCreate(&t_handle_mem, 1, ncclUint8,
                                   d_handle_mem,
                                   static_cast<unsigned int>(handle_mem_bytes)));

    ncclEpHandle_t h = nullptr;
    NCCL_ASSERT(ncclEpInitHandle(&h, g_ep_group, NCCL_EP_LAYOUT_FLAT, nullptr, kTopK, t_handle_mem));
    ASSERT_NE(h, nullptr);
    NCCL_ASSERT(ncclEpUpdateHandle(h, topk_idx_, nullptr, g_stream));
    CUDA_ASSERT(cudaStreamSynchronize(g_stream));

    auto combined = run_dispatch_combine_flat(h);
    for (int i = 0; i < kNumTokens; ++i) {
        float expected = static_cast<float>(g_rank * kNumTokens + i + 1);
        EXPECT_NEAR(combined[i], expected, 0.5f) << "rank " << g_rank << " token " << i;
    }

    NCCL_ASSERT(ncclEpHandleDestroy(h));
    // The caller still owns d_handle_mem; free it ourselves.
    ncclEpTensorDestroy(t_handle_mem);
    CUDA_ASSERT(cudaFree(d_handle_mem));
}

// ── Test: MultiIterHandleReuse — 4 dispatch+combine pairs on one handle ───────
// Exercises buffer ping-pong and signal counter increments across multiple iterations.

TEST_F(LifecycleTest, MultiIterHandleReuse) {
    ncclEpHandle_t h = make_handle(nullptr);
    ASSERT_NE(h, nullptr);

    for (int iter = 0; iter < 4; ++iter) {
        auto combined = run_dispatch_combine_flat(h);
        for (int i = 0; i < kNumTokens; ++i) {
            float expected = static_cast<float>(g_rank * kNumTokens + i + 1);
            EXPECT_NEAR(combined[i], expected, 0.5f)
                << "iter " << iter << " rank " << g_rank << " token " << i;
        }
    }

    NCCL_ASSERT(ncclEpHandleDestroy(h));
}

// ── main ──────────────────────────────────────────────────────────────────────

int main(int argc, char* argv[]) {
    if (!ep_bootstrap(argc, argv, "te_ep_lifecycle_uid")) return 0;
    int ret = RUN_ALL_TESTS();
    ep_teardown();
    return ret;
}
