/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Unit tests for HT backward-flow paths:
 *
 *   HT_CombineWithGrad_* — ncclEpCombine with topk_weights input
 *                          (backward_combine=true). Reduces grad-recv tokens
 *                          and grad-recv topk_weights back to grad-tokens and
 *                          topk_weight_grad at the source.
 *                          Outputs: combined_x, combined_topk_weights.
 *
 *   HT_DispatchScatter_* — ncclEpDispatch with no topk_idx/topk_weights
 *                          (forward_dispatch=false). Pure token scatter to
 *                          per-expert positions; no weight handling.
 *
 * Both are covered for RM (FLAT, rank-major) and EM (expert-major) layouts.
 */

#include "test_common.h"
#include "../nccl_ep_test_internal.h"
#include <set>

static float bf16_val(nv_bfloat16 v) { return __bfloat162float(v); }

// ── Test fixture ──────────────────────────────────────────────────────────────

class HtBwdTest : public EpTestBase {
protected:
    // Pick a unique non-trivial per-(rank, token) topk weight so any routing
    // error in the BWD path produces a value mismatch (not a 1.0==1.0 false-pass).
    static float weight_for(int rank, int token, int k) {
        return 0.125f * static_cast<float>((rank + 1) * 100 + (token + 1) * 10 + k);
    }

    // Forward dispatch on the given handle. Inputs: tokens + topk_weights.
    // Outputs (RM): recv_x, recv_topk_weights, recv_topk_idx.
    // Outputs (EM): recv_x, recv_topk_weights (1D).
    // Returns the device pointers and tensor handles for downstream BWD calls.
    struct FwdState {
        nv_bfloat16* d_tok = nullptr;
        nv_bfloat16* d_recv = nullptr;
        float*       d_weights = nullptr;
        float*       d_recv_w = nullptr;
        int64_t*     d_recv_idx = nullptr;
        ncclEpTensor_t* t_tok = nullptr;
        ncclEpTensor_t* t_recv = nullptr;
        ncclEpTensor_t* t_w = nullptr;
        ncclEpTensor_t* t_recv_w = nullptr;
        ncclEpTensor_t* t_recv_idx = nullptr;
        bool expert_major = false;

        void free_all() {
            if (t_tok)      ncclEpTensorDestroy(t_tok);
            if (t_recv)     ncclEpTensorDestroy(t_recv);
            if (t_w)        ncclEpTensorDestroy(t_w);
            if (t_recv_w)   ncclEpTensorDestroy(t_recv_w);
            if (t_recv_idx) ncclEpTensorDestroy(t_recv_idx);
            if (d_tok)      cudaFree(d_tok);
            if (d_recv)     cudaFree(d_recv);
            if (d_weights)  cudaFree(d_weights);
            if (d_recv_w)   cudaFree(d_recv_w);
            if (d_recv_idx) cudaFree(d_recv_idx);
        }
    };

    void run_forward_dispatch(ncclEpHandle_t handle, bool expert_major, FwdState& st) {
        st.expert_major = expert_major;

        std::vector<nv_bfloat16> h_tok(kNumTokens * kHidden);
        for (int i = 0; i < kNumTokens; ++i) {
            float v = static_cast<float>(g_rank * kNumTokens + i + 1);
            for (int hh = 0; hh < kHidden; ++hh)
                h_tok[i * kHidden + hh] = __float2bfloat16(v);
        }
        std::vector<float> h_w(kNumTokens * kTopK);
        for (int i = 0; i < kNumTokens; ++i)
            for (int k = 0; k < kTopK; ++k)
                h_w[i * kTopK + k] = weight_for(g_rank, i, k);

        EXPECT_EQ(cudaMalloc(&st.d_tok,     kNumTokens     * kHidden * sizeof(nv_bfloat16)), cudaSuccess);
        EXPECT_EQ(cudaMalloc(&st.d_recv,    kMaxRecvSlots  * kHidden * sizeof(nv_bfloat16)), cudaSuccess);
        EXPECT_EQ(cudaMalloc(&st.d_weights, kNumTokens     * kTopK   * sizeof(float)),       cudaSuccess);
        EXPECT_EQ(cudaMalloc(&st.d_recv_w,  kMaxRecvSlots  * kTopK   * sizeof(float)),       cudaSuccess);
        EXPECT_EQ(cudaMalloc(&st.d_recv_idx,kMaxRecvSlots  * kTopK   * sizeof(int64_t)),     cudaSuccess);
        EXPECT_EQ(cudaMemset(st.d_recv, 0,  kMaxRecvSlots  * kHidden * sizeof(nv_bfloat16)), cudaSuccess);
        EXPECT_EQ(cudaMemset(st.d_recv_w, 0,kMaxRecvSlots  * kTopK   * sizeof(float)),       cudaSuccess);

        EXPECT_EQ(cudaMemcpy(st.d_tok,     h_tok.data(), kNumTokens*kHidden*sizeof(nv_bfloat16), cudaMemcpyHostToDevice), cudaSuccess);
        EXPECT_EQ(cudaMemcpy(st.d_weights, h_w.data(),   kNumTokens*kTopK*sizeof(float),         cudaMemcpyHostToDevice), cudaSuccess);

        EXPECT_EQ(epTensorCreate(&st.t_tok,  2, ncclBfloat16, st.d_tok,     kNumTokens,    kHidden), ncclSuccess);
        EXPECT_EQ(epTensorCreate(&st.t_recv, 2, ncclBfloat16, st.d_recv,    kMaxRecvSlots, kHidden), ncclSuccess);
        EXPECT_EQ(epTensorCreate(&st.t_w,    2, ncclFloat32,  st.d_weights, kNumTokens,    kTopK),   ncclSuccess);
        if (expert_major) {
            EXPECT_EQ(epTensorCreate(&st.t_recv_w, 1, ncclFloat32, st.d_recv_w, kMaxRecvSlots), ncclSuccess);
            st.t_recv_idx = nullptr;
        } else {
            EXPECT_EQ(epTensorCreate(&st.t_recv_w,   2, ncclFloat32, st.d_recv_w,   kMaxRecvSlots, kTopK), ncclSuccess);
            EXPECT_EQ(epTensorCreate(&st.t_recv_idx, 2, ncclInt64,   st.d_recv_idx, kMaxRecvSlots, kTopK), ncclSuccess);
        }

        ncclEpDispatchInputs_t  d_in_s  = NCCL_EP_DISPATCH_INPUTS_INIT;
        ncclEpDispatchOutputs_t d_out_s = NCCL_EP_DISPATCH_OUTPUTS_INIT;
        d_in_s.tokens        = st.t_tok;
        d_in_s.topk_weights  = st.t_w;
        d_out_s.tokens       = st.t_recv;
        d_out_s.topk_weights = st.t_recv_w;
        if (!expert_major) d_out_s.topk_idx = st.t_recv_idx;
        ncclEpDispatchConfig_t dcfg = NCCL_EP_DISPATCH_CONFIG_INIT;
        EXPECT_EQ(ncclEpDispatch(handle, &d_in_s, &d_out_s, nullptr, &dcfg, g_stream), ncclSuccess);
        EXPECT_EQ(ncclEpComplete(handle, nullptr, g_stream), ncclSuccess);
        EXPECT_EQ(cudaStreamSynchronize(g_stream), cudaSuccess);
    }
};

// ── HT CombineWithGrad = ncclEpCombine(backward_combine=true) ────────────────
// Inputs:  x = grad_recv_tokens [num_recv, hidden]
//          topk_weights = grad_recv_topk_weights — shape by layout:
//            RM: 2D [num_recv, top_k]   EM: 1D [num_recv]
// Outputs: combined_x [num_tokens, hidden]
//          combined_topk_weights [num_tokens, top_k]
//
// Test: echo the FWD recv tensors back as BWD combine input; verify the
// round-trip recovers the original source-side topk_weights.

TEST_F(HtBwdTest, CombineWithGradRankMajor) {
    ncclEpHandle_t h = make_handle(nullptr);
    ASSERT_NE(h, nullptr);

    FwdState fwd;
    run_forward_dispatch(h, /*expert_major=*/false, fwd);
    nv_bfloat16* d_combined_x = nullptr;
    float*       d_combined_w = nullptr;
    CUDA_ASSERT(cudaMalloc(&d_combined_x, kNumTokens * kHidden * sizeof(nv_bfloat16)));
    CUDA_ASSERT(cudaMalloc(&d_combined_w, kNumTokens * kTopK   * sizeof(float)));
    CUDA_ASSERT(cudaMemset(d_combined_w, 0, kNumTokens * kTopK * sizeof(float)));

    ncclEpTensor_t *t_combined_x, *t_combined_w;
    NCCL_ASSERT(epTensorCreate(&t_combined_x, 2, ncclBfloat16, d_combined_x, kNumTokens, kHidden));
    NCCL_ASSERT(epTensorCreate(&t_combined_w, 2, ncclFloat32,  d_combined_w, kNumTokens, kTopK));

    // BWD-combine input topk_weights must be 2D [num_recv, top_k]; reuse forward output.
    ncclEpCombineInputs_t  c_in_s  = NCCL_EP_COMBINE_INPUTS_INIT;
    ncclEpCombineOutputs_t c_out_s = NCCL_EP_COMBINE_OUTPUTS_INIT;
    c_in_s.tokens         = fwd.t_recv;
    c_in_s.topk_weights   = fwd.t_recv_w;
    c_out_s.tokens        = t_combined_x;
    c_out_s.topk_weights  = t_combined_w;
    ncclEpCombineConfig_t ccfg = NCCL_EP_COMBINE_CONFIG_INIT;
    ccfg.pass_direction = NCCL_EP_BWD_PASS;
    NCCL_ASSERT(ncclEpCombine(h, &c_in_s, &c_out_s, &ccfg, g_stream));
    CUDA_ASSERT(cudaStreamSynchronize(g_stream));

    std::vector<float> h_combined_w(kNumTokens * kTopK);
    CUDA_ASSERT(cudaMemcpy(h_combined_w.data(), d_combined_w, kNumTokens*kTopK*sizeof(float), cudaMemcpyDeviceToHost));

    for (int i = 0; i < kNumTokens; ++i)
        for (int k = 0; k < kTopK; ++k) {
            float expected = weight_for(g_rank, i, k);
            EXPECT_NEAR(h_combined_w[i*kTopK + k], expected, 1e-4f)
                << "rank " << g_rank << " token " << i << " k " << k
                << " expected=" << expected << " got=" << h_combined_w[i*kTopK + k];
        }

    ncclEpTensorDestroy(t_combined_x);
    ncclEpTensorDestroy(t_combined_w);
    cudaFree(d_combined_x); cudaFree(d_combined_w);
    fwd.free_all();
    NCCL_ASSERT(ncclEpHandleDestroy(h));
}

TEST_F(HtBwdTest, CombineWithGradExpertMajor) {
    ncclEpHandle_t h = make_handle_em(nullptr);
    ASSERT_NE(h, nullptr);

    FwdState fwd;
    run_forward_dispatch(h, /*expert_major=*/true, fwd);

    // EM: FWD recv_topk_weights is 1D [num_recv]; feed that 1D tensor DIRECTLY into
    // BWD combine input — no reshape needed. The output combined_topk_weights stays
    // 2D [num_combined_tokens, source_top_k].
    nv_bfloat16* d_combined_x = nullptr;
    float*       d_combined_w = nullptr;
    CUDA_ASSERT(cudaMalloc(&d_combined_x, kNumTokens * kHidden * sizeof(nv_bfloat16)));
    CUDA_ASSERT(cudaMalloc(&d_combined_w, kNumTokens * kTopK   * sizeof(float)));
    CUDA_ASSERT(cudaMemset(d_combined_w, 0, kNumTokens * kTopK * sizeof(float)));

    ncclEpTensor_t *t_combined_x, *t_combined_w;
    NCCL_ASSERT(epTensorCreate(&t_combined_x, 2, ncclBfloat16, d_combined_x, kNumTokens, kHidden));
    NCCL_ASSERT(epTensorCreate(&t_combined_w, 2, ncclFloat32,  d_combined_w, kNumTokens, kTopK));

    ncclEpCombineInputs_t  c_in_s  = NCCL_EP_COMBINE_INPUTS_INIT;
    ncclEpCombineOutputs_t c_out_s = NCCL_EP_COMBINE_OUTPUTS_INIT;
    c_in_s.tokens         = fwd.t_recv;
    c_in_s.topk_weights   = fwd.t_recv_w;   // 1D for EM
    c_out_s.tokens        = t_combined_x;
    c_out_s.topk_weights  = t_combined_w;
    ncclEpCombineConfig_t ccfg = NCCL_EP_COMBINE_CONFIG_INIT;
    ccfg.pass_direction = NCCL_EP_BWD_PASS;
    NCCL_ASSERT(ncclEpCombine(h, &c_in_s, &c_out_s, &ccfg, g_stream));
    CUDA_ASSERT(cudaStreamSynchronize(g_stream));

    std::vector<float> h_combined_w(kNumTokens * kTopK);
    CUDA_ASSERT(cudaMemcpy(h_combined_w.data(), d_combined_w, kNumTokens*kTopK*sizeof(float), cudaMemcpyDeviceToHost));

    for (int i = 0; i < kNumTokens; ++i)
        for (int k = 0; k < kTopK; ++k) {
            float expected = weight_for(g_rank, i, k);
            EXPECT_NEAR(h_combined_w[i*kTopK + k], expected, 1e-4f)
                << "EM rank " << g_rank << " token " << i << " k " << k
                << " expected=" << expected << " got=" << h_combined_w[i*kTopK + k];
        }

    ncclEpTensorDestroy(t_combined_x);
    ncclEpTensorDestroy(t_combined_w);
    cudaFree(d_combined_x); cudaFree(d_combined_w);
    fwd.free_all();
    NCCL_ASSERT(ncclEpHandleDestroy(h));
}

// ── HT DispatchScatter = ncclEpDispatch(forward_dispatch=false) ──────────────
// Inputs:  x [num_tokens, hidden]    (no topk_idx, no topk_weights)
// Outputs: recv_x [num_recv, hidden] (no recv_topk_*)
// Test: prime the handle with a forward dispatch, then re-dispatch new token
// values with no topk_idx; verify the recv set under the same routing.

TEST_F(HtBwdTest, DispatchScatterRankMajor) {
    ncclEpHandle_t h = make_handle(nullptr);
    ASSERT_NE(h, nullptr);

    // First forward dispatch primes the handle.
    FwdState fwd;
    run_forward_dispatch(h, /*expert_major=*/false, fwd);

    unsigned int num_recv = 0;
    NCCL_ASSERT(ncclEpHandle_test_getNumRecvTokens(h, &num_recv));

    // Re-dispatch with new token values; expect identical routing.
    std::vector<nv_bfloat16> h_tok2(kNumTokens * kHidden);
    for (int i = 0; i < kNumTokens; ++i) {
        float v = 100.0f + static_cast<float>(g_rank * kNumTokens + i);
        for (int hh = 0; hh < kHidden; ++hh)
            h_tok2[i * kHidden + hh] = __float2bfloat16(v);
    }
    nv_bfloat16 *d_tok2, *d_recv2;
    CUDA_ASSERT(cudaMalloc(&d_tok2,  kNumTokens    * kHidden * sizeof(nv_bfloat16)));
    CUDA_ASSERT(cudaMalloc(&d_recv2, kMaxRecvSlots * kHidden * sizeof(nv_bfloat16)));
    CUDA_ASSERT(cudaMemset(d_recv2, 0, kMaxRecvSlots * kHidden * sizeof(nv_bfloat16)));
    CUDA_ASSERT(cudaMemcpy(d_tok2, h_tok2.data(), kNumTokens*kHidden*sizeof(nv_bfloat16), cudaMemcpyHostToDevice));

    ncclEpTensor_t *t_tok2, *t_recv2;
    NCCL_ASSERT(epTensorCreate(&t_tok2,  2, ncclBfloat16, d_tok2,  kNumTokens,    kHidden));
    NCCL_ASSERT(epTensorCreate(&t_recv2, 2, ncclBfloat16, d_recv2, kMaxRecvSlots, kHidden));

    ncclEpDispatchInputs_t  d_in_s  = NCCL_EP_DISPATCH_INPUTS_INIT;
    ncclEpDispatchOutputs_t d_out_s = NCCL_EP_DISPATCH_OUTPUTS_INIT;
    d_in_s.tokens  = t_tok2;
    d_out_s.tokens = t_recv2;
    ncclEpDispatchConfig_t dcfg = NCCL_EP_DISPATCH_CONFIG_INIT;
    dcfg.pass_direction = NCCL_EP_BWD_PASS;
    NCCL_ASSERT(ncclEpDispatch(h, &d_in_s, &d_out_s, nullptr, &dcfg, g_stream));
    NCCL_ASSERT(ncclEpComplete(h, nullptr, g_stream));
    CUDA_ASSERT(cudaStreamSynchronize(g_stream));

    std::vector<nv_bfloat16> h_recv2(num_recv * kHidden);
    CUDA_ASSERT(cudaMemcpy(h_recv2.data(), d_recv2, num_recv*kHidden*sizeof(nv_bfloat16), cudaMemcpyDeviceToHost));

    // Expected: same routing as forward — the 4 recv slots come from the same 4 source
    // (rank, token) pairs that contributed in the forward pass. Each source token's
    // new value is 100 + (rank * kNumTokens + token).
    std::set<float> expected;
    if (g_rank == 0) {        // E0,E1 ← ranks 0,2
        expected = {100.f, 101.f, 108.f, 109.f};
    } else if (g_rank == 1) { // E2,E3 ← ranks 0,2
        expected = {102.f, 103.f, 110.f, 111.f};
    } else if (g_rank == 2) { // E4,E5 ← ranks 1,3
        expected = {104.f, 105.f, 112.f, 113.f};
    } else {                  // E6,E7 ← ranks 1,3
        expected = {106.f, 107.f, 114.f, 115.f};
    }
    std::set<float> got;
    for (unsigned s = 0; s < num_recv; ++s) got.insert(bf16_val(h_recv2[s * kHidden]));
    EXPECT_EQ(got, expected) << "rank " << g_rank << " dispatch-scatter recv set mismatch";

    ncclEpTensorDestroy(t_tok2);
    ncclEpTensorDestroy(t_recv2);
    cudaFree(d_tok2); cudaFree(d_recv2);
    fwd.free_all();
    NCCL_ASSERT(ncclEpHandleDestroy(h));
}

TEST_F(HtBwdTest, DispatchScatterExpertMajor) {
    ncclEpHandle_t h = make_handle_em(nullptr);
    ASSERT_NE(h, nullptr);

    FwdState fwd;
    run_forward_dispatch(h, /*expert_major=*/true, fwd);

    unsigned int num_recv = 0;
    NCCL_ASSERT(ncclEpHandle_test_getNumRecvTokens(h, &num_recv));

    std::vector<nv_bfloat16> h_tok2(kNumTokens * kHidden);
    for (int i = 0; i < kNumTokens; ++i) {
        float v = 100.0f + static_cast<float>(g_rank * kNumTokens + i);
        for (int hh = 0; hh < kHidden; ++hh)
            h_tok2[i * kHidden + hh] = __float2bfloat16(v);
    }
    nv_bfloat16 *d_tok2, *d_recv2;
    CUDA_ASSERT(cudaMalloc(&d_tok2,  kNumTokens    * kHidden * sizeof(nv_bfloat16)));
    CUDA_ASSERT(cudaMalloc(&d_recv2, kMaxRecvSlots * kHidden * sizeof(nv_bfloat16)));
    CUDA_ASSERT(cudaMemset(d_recv2, 0, kMaxRecvSlots * kHidden * sizeof(nv_bfloat16)));
    CUDA_ASSERT(cudaMemcpy(d_tok2, h_tok2.data(), kNumTokens*kHidden*sizeof(nv_bfloat16), cudaMemcpyHostToDevice));

    ncclEpTensor_t *t_tok2, *t_recv2;
    NCCL_ASSERT(epTensorCreate(&t_tok2,  2, ncclBfloat16, d_tok2,  kNumTokens,    kHidden));
    NCCL_ASSERT(epTensorCreate(&t_recv2, 2, ncclBfloat16, d_recv2, kMaxRecvSlots, kHidden));

    ncclEpDispatchInputs_t  d_in_s  = NCCL_EP_DISPATCH_INPUTS_INIT;
    ncclEpDispatchOutputs_t d_out_s = NCCL_EP_DISPATCH_OUTPUTS_INIT;
    d_in_s.tokens  = t_tok2;
    d_out_s.tokens = t_recv2;
    ncclEpDispatchConfig_t dcfg = NCCL_EP_DISPATCH_CONFIG_INIT;
    dcfg.pass_direction = NCCL_EP_BWD_PASS;
    NCCL_ASSERT(ncclEpDispatch(h, &d_in_s, &d_out_s, nullptr, &dcfg, g_stream));
    NCCL_ASSERT(ncclEpComplete(h, nullptr, g_stream));
    CUDA_ASSERT(cudaStreamSynchronize(g_stream));

    // EM with top-k=1: each local expert receives 2 tokens. Verify the same
    // expected set as RankMajor.
    std::vector<nv_bfloat16> h_recv2(num_recv * kHidden);
    CUDA_ASSERT(cudaMemcpy(h_recv2.data(), d_recv2, num_recv*kHidden*sizeof(nv_bfloat16), cudaMemcpyDeviceToHost));

    std::set<float> expected;
    if (g_rank == 0) {
        expected = {100.f, 101.f, 108.f, 109.f};
    } else if (g_rank == 1) {
        expected = {102.f, 103.f, 110.f, 111.f};
    } else if (g_rank == 2) {
        expected = {104.f, 105.f, 112.f, 113.f};
    } else {
        expected = {106.f, 107.f, 114.f, 115.f};
    }
    std::set<float> got;
    for (unsigned s = 0; s < num_recv; ++s) got.insert(bf16_val(h_recv2[s * kHidden]));
    EXPECT_EQ(got, expected) << "EM rank " << g_rank << " dispatch-scatter recv set mismatch";

    ncclEpTensorDestroy(t_tok2);
    ncclEpTensorDestroy(t_recv2);
    cudaFree(d_tok2); cudaFree(d_recv2);
    fwd.free_all();
    NCCL_ASSERT(ncclEpHandleDestroy(h));
}

// ── top-k=2 fixture (routing identical to TopK2MixedRoutingTest) ─────────────
// T0 → E0, E1   T1 → E2, E3   T2 → E4, E6   T3 → E5, E7
// Routing is sorted within each row so combined_topk_weights (packed in
// ascending expert-id order) matches the input k order.

static constexpr int kTopK2 = 2;

class HtBwdTopK2Test : public ::testing::Test {
protected:
    ncclEpTensor_t* topk_idx2_    = nullptr;
    ncclEpTensor_t* topk_idx2_em_ = nullptr;
    int64_t*       d_topk2_      = nullptr;

    static float w2(int rank, int token, int k) {
        return 0.0625f * static_cast<float>((rank + 1) * 100 + (token + 1) * 10 + k);
    }

    void SetUp() override {
        CUDA_ASSERT(cudaMalloc(&d_topk2_, kNumTokens * kTopK2 * sizeof(int64_t)));
        const int64_t routing[kNumTokens * kTopK2] = {
            0, 1,
            2, 3,
            4, 6,
            5, 7,
        };
        CUDA_ASSERT(cudaMemcpy(d_topk2_, routing, sizeof(routing), cudaMemcpyHostToDevice));
        NCCL_ASSERT(epTensorCreate(&topk_idx2_, 2, ncclInt64,
                                       d_topk2_, kNumTokens, kTopK2));
        NCCL_ASSERT(epTensorCreate(&topk_idx2_em_, 2, ncclInt64,
                                       d_topk2_, kNumTokens, kTopK2));
    }

    void TearDown() override {
        if (topk_idx2_em_) ncclEpTensorDestroy(topk_idx2_em_);
        if (topk_idx2_)    ncclEpTensorDestroy(topk_idx2_);
        if (d_topk2_)      cudaFree(d_topk2_);
    }

    ncclEpHandle_t mk(bool em) {
        ncclEpHandle_t h = nullptr;
        EXPECT_EQ(ncclEpCreateHandle(&h, em ? g_ep_group_em : g_ep_group,
                                     em ? NCCL_EP_LAYOUT_EXPERT_MAJOR : NCCL_EP_LAYOUT_FLAT,
                                     em ? topk_idx2_em_ : topk_idx2_,
                                     nullptr, nullptr, g_stream), ncclSuccess);
        EXPECT_EQ(cudaStreamSynchronize(g_stream), cudaSuccess);
        return h;
    }
};

// CombineWithGrad + RM, top-k=2.
// Expect: combined_topk_weights[t, k] == source-side weight w2(g_rank, t, k).
TEST_F(HtBwdTopK2Test, CombineWithGradRankMajor) {
    ncclEpHandle_t h = mk(false);
    ASSERT_NE(h, nullptr);

    // Forward dispatch with unique per-(rank, token, k) weights.
    std::vector<nv_bfloat16> h_tok(kNumTokens * kHidden);
    std::vector<float> h_w(kNumTokens * kTopK2);
    for (int i = 0; i < kNumTokens; ++i) {
        float v = static_cast<float>(g_rank * kNumTokens + i + 1);
        for (int hh = 0; hh < kHidden; ++hh)
            h_tok[i * kHidden + hh] = __float2bfloat16(v);
        for (int k = 0; k < kTopK2; ++k)
            h_w[i * kTopK2 + k] = w2(g_rank, i, k);
    }

    nv_bfloat16 *d_tok, *d_recv;
    float       *d_weights, *d_recv_w;
    int64_t     *d_recv_idx;
    CUDA_ASSERT(cudaMalloc(&d_tok,     kNumTokens    * kHidden * sizeof(nv_bfloat16)));
    CUDA_ASSERT(cudaMalloc(&d_recv,    kMaxRecvSlots * kHidden * sizeof(nv_bfloat16)));
    CUDA_ASSERT(cudaMemset(d_recv, 0,  kMaxRecvSlots * kHidden * sizeof(nv_bfloat16)));
    CUDA_ASSERT(cudaMalloc(&d_weights, kNumTokens    * kTopK2  * sizeof(float)));
    CUDA_ASSERT(cudaMalloc(&d_recv_w,  kMaxRecvSlots * kTopK2  * sizeof(float)));
    CUDA_ASSERT(cudaMalloc(&d_recv_idx,kMaxRecvSlots * kTopK2  * sizeof(int64_t)));
    CUDA_ASSERT(cudaMemcpy(d_tok, h_tok.data(), kNumTokens*kHidden*sizeof(nv_bfloat16), cudaMemcpyHostToDevice));
    CUDA_ASSERT(cudaMemcpy(d_weights, h_w.data(), kNumTokens*kTopK2*sizeof(float), cudaMemcpyHostToDevice));

    ncclEpTensor_t *t_tok, *t_recv, *t_w, *t_recv_w, *t_recv_idx;
    NCCL_ASSERT(epTensorCreate(&t_tok,      2, ncclBfloat16, d_tok,      kNumTokens,    kHidden));
    NCCL_ASSERT(epTensorCreate(&t_recv,     2, ncclBfloat16, d_recv,     kMaxRecvSlots, kHidden));
    NCCL_ASSERT(epTensorCreate(&t_w,        2, ncclFloat32,  d_weights,  kNumTokens,    kTopK2));
    NCCL_ASSERT(epTensorCreate(&t_recv_w,   2, ncclFloat32,  d_recv_w,   kMaxRecvSlots, kTopK2));
    NCCL_ASSERT(epTensorCreate(&t_recv_idx, 2, ncclInt64,    d_recv_idx, kMaxRecvSlots, kTopK2));

    ncclEpDispatchInputs_t  d_in_s  = NCCL_EP_DISPATCH_INPUTS_INIT;
    ncclEpDispatchOutputs_t d_out_s = NCCL_EP_DISPATCH_OUTPUTS_INIT;
    d_in_s.tokens        = t_tok;
    d_in_s.topk_weights  = t_w;
    d_out_s.tokens       = t_recv;
    d_out_s.topk_weights = t_recv_w;
    d_out_s.topk_idx     = t_recv_idx;
    ncclEpDispatchConfig_t dcfg = NCCL_EP_DISPATCH_CONFIG_INIT;
    NCCL_ASSERT(ncclEpDispatch(h, &d_in_s, &d_out_s, nullptr, &dcfg, g_stream));
    NCCL_ASSERT(ncclEpComplete(h, nullptr, g_stream));
    CUDA_ASSERT(cudaStreamSynchronize(g_stream));

    // Backward combine: input topk_weights = recv_topk_weights (forward output).
    nv_bfloat16* d_combined_x = nullptr;
    float*       d_combined_w = nullptr;
    CUDA_ASSERT(cudaMalloc(&d_combined_x, kNumTokens * kHidden * sizeof(nv_bfloat16)));
    CUDA_ASSERT(cudaMalloc(&d_combined_w, kNumTokens * kTopK2  * sizeof(float)));
    CUDA_ASSERT(cudaMemset(d_combined_w, 0, kNumTokens * kTopK2 * sizeof(float)));

    ncclEpTensor_t *t_combined_x, *t_combined_w;
    NCCL_ASSERT(epTensorCreate(&t_combined_x, 2, ncclBfloat16, d_combined_x, kNumTokens, kHidden));
    NCCL_ASSERT(epTensorCreate(&t_combined_w, 2, ncclFloat32,  d_combined_w, kNumTokens, kTopK2));

    ncclEpCombineInputs_t  c_in_s  = NCCL_EP_COMBINE_INPUTS_INIT;
    ncclEpCombineOutputs_t c_out_s = NCCL_EP_COMBINE_OUTPUTS_INIT;
    c_in_s.tokens        = t_recv;
    c_in_s.topk_weights  = t_recv_w;
    c_out_s.tokens       = t_combined_x;
    c_out_s.topk_weights = t_combined_w;
    ncclEpCombineConfig_t ccfg = NCCL_EP_COMBINE_CONFIG_INIT;
    ccfg.pass_direction = NCCL_EP_BWD_PASS;
    NCCL_ASSERT(ncclEpCombine(h, &c_in_s, &c_out_s, &ccfg, g_stream));
    CUDA_ASSERT(cudaStreamSynchronize(g_stream));

    std::vector<float> h_combined_w(kNumTokens * kTopK2);
    CUDA_ASSERT(cudaMemcpy(h_combined_w.data(), d_combined_w, kNumTokens*kTopK2*sizeof(float), cudaMemcpyDeviceToHost));

    for (int i = 0; i < kNumTokens; ++i)
        for (int k = 0; k < kTopK2; ++k) {
            float expected = w2(g_rank, i, k);
            EXPECT_NEAR(h_combined_w[i*kTopK2 + k], expected, 1e-4f)
                << "RM TopK2 rank " << g_rank << " token " << i << " k " << k
                << " expected=" << expected << " got=" << h_combined_w[i*kTopK2 + k];
        }

    ncclEpTensorDestroy(t_tok); ncclEpTensorDestroy(t_recv);
    ncclEpTensorDestroy(t_w);   ncclEpTensorDestroy(t_recv_w); ncclEpTensorDestroy(t_recv_idx);
    ncclEpTensorDestroy(t_combined_x); ncclEpTensorDestroy(t_combined_w);
    cudaFree(d_tok); cudaFree(d_recv); cudaFree(d_weights);
    cudaFree(d_recv_w); cudaFree(d_recv_idx);
    cudaFree(d_combined_x); cudaFree(d_combined_w);
    NCCL_ASSERT(ncclEpHandleDestroy(h));
}

// CombineWithGrad + EM, top-k=2.
// EM FWD recv_topk_weights is 1D [num_recv]; feed that 1D tensor directly as
// the BWD combine input. Output combined_topk_weights stays 2D
// [num_combined, top_k], packed in ascending expert-id order.
TEST_F(HtBwdTopK2Test, CombineWithGradExpertMajor) {
    ncclEpHandle_t h = mk(true);
    ASSERT_NE(h, nullptr);

    std::vector<nv_bfloat16> h_tok(kNumTokens * kHidden);
    std::vector<float> h_w(kNumTokens * kTopK2);
    for (int i = 0; i < kNumTokens; ++i) {
        float v = static_cast<float>(g_rank * kNumTokens + i + 1);
        for (int hh = 0; hh < kHidden; ++hh)
            h_tok[i * kHidden + hh] = __float2bfloat16(v);
        for (int k = 0; k < kTopK2; ++k)
            h_w[i * kTopK2 + k] = w2(g_rank, i, k);
    }

    nv_bfloat16 *d_tok, *d_recv;
    float       *d_weights, *d_recv_w_1d;
    CUDA_ASSERT(cudaMalloc(&d_tok,       kNumTokens    * kHidden * sizeof(nv_bfloat16)));
    CUDA_ASSERT(cudaMalloc(&d_recv,      kMaxRecvSlots * kHidden * sizeof(nv_bfloat16)));
    CUDA_ASSERT(cudaMemset(d_recv, 0,    kMaxRecvSlots * kHidden * sizeof(nv_bfloat16)));
    CUDA_ASSERT(cudaMalloc(&d_weights,   kNumTokens    * kTopK2  * sizeof(float)));
    CUDA_ASSERT(cudaMalloc(&d_recv_w_1d, kMaxRecvSlots           * sizeof(float)));
    CUDA_ASSERT(cudaMemset(d_recv_w_1d, 0, kMaxRecvSlots         * sizeof(float)));
    CUDA_ASSERT(cudaMemcpy(d_tok, h_tok.data(), kNumTokens*kHidden*sizeof(nv_bfloat16), cudaMemcpyHostToDevice));
    CUDA_ASSERT(cudaMemcpy(d_weights, h_w.data(), kNumTokens*kTopK2*sizeof(float), cudaMemcpyHostToDevice));

    ncclEpTensor_t *t_tok, *t_recv, *t_w, *t_recv_w_1d;
    NCCL_ASSERT(epTensorCreate(&t_tok,       2, ncclBfloat16, d_tok,       kNumTokens,    kHidden));
    NCCL_ASSERT(epTensorCreate(&t_recv,      2, ncclBfloat16, d_recv,      kMaxRecvSlots, kHidden));
    NCCL_ASSERT(epTensorCreate(&t_w,         2, ncclFloat32,  d_weights,   kNumTokens,    kTopK2));
    NCCL_ASSERT(epTensorCreate(&t_recv_w_1d, 1, ncclFloat32,  d_recv_w_1d, kMaxRecvSlots));

    ncclEpDispatchInputs_t  d_in_s  = NCCL_EP_DISPATCH_INPUTS_INIT;
    ncclEpDispatchOutputs_t d_out_s = NCCL_EP_DISPATCH_OUTPUTS_INIT;
    d_in_s.tokens        = t_tok;
    d_in_s.topk_weights  = t_w;
    d_out_s.tokens       = t_recv;
    d_out_s.topk_weights = t_recv_w_1d;
    ncclEpDispatchConfig_t dcfg = NCCL_EP_DISPATCH_CONFIG_INIT;
    NCCL_ASSERT(ncclEpDispatch(h, &d_in_s, &d_out_s, nullptr, &dcfg, g_stream));
    NCCL_ASSERT(ncclEpComplete(h, nullptr, g_stream));
    CUDA_ASSERT(cudaStreamSynchronize(g_stream));

    // Feed the 1D EM recv_topk_weights DIRECTLY into BWD combine as input topk_weights.
    nv_bfloat16* d_combined_x = nullptr;
    float*       d_combined_w = nullptr;
    CUDA_ASSERT(cudaMalloc(&d_combined_x, kNumTokens * kHidden * sizeof(nv_bfloat16)));
    CUDA_ASSERT(cudaMalloc(&d_combined_w, kNumTokens * kTopK2  * sizeof(float)));
    CUDA_ASSERT(cudaMemset(d_combined_w, 0, kNumTokens * kTopK2 * sizeof(float)));

    ncclEpTensor_t *t_combined_x, *t_combined_w;
    NCCL_ASSERT(epTensorCreate(&t_combined_x, 2, ncclBfloat16, d_combined_x, kNumTokens, kHidden));
    NCCL_ASSERT(epTensorCreate(&t_combined_w, 2, ncclFloat32,  d_combined_w, kNumTokens, kTopK2));

    ncclEpCombineInputs_t  c_in_s  = NCCL_EP_COMBINE_INPUTS_INIT;
    ncclEpCombineOutputs_t c_out_s = NCCL_EP_COMBINE_OUTPUTS_INIT;
    c_in_s.tokens        = t_recv;
    c_in_s.topk_weights  = t_recv_w_1d;
    c_out_s.tokens       = t_combined_x;
    c_out_s.topk_weights = t_combined_w;
    ncclEpCombineConfig_t ccfg = NCCL_EP_COMBINE_CONFIG_INIT;
    ccfg.pass_direction = NCCL_EP_BWD_PASS;
    NCCL_ASSERT(ncclEpCombine(h, &c_in_s, &c_out_s, &ccfg, g_stream));
    CUDA_ASSERT(cudaStreamSynchronize(g_stream));

    std::vector<float> h_combined_w(kNumTokens * kTopK2);
    CUDA_ASSERT(cudaMemcpy(h_combined_w.data(), d_combined_w, kNumTokens*kTopK2*sizeof(float), cudaMemcpyDeviceToHost));

    int errors = 0;
    for (int i = 0; i < kNumTokens; ++i)
        for (int k = 0; k < kTopK2; ++k) {
            float expected = w2(g_rank, i, k);
            float got = h_combined_w[i*kTopK2 + k];
            if (std::abs(got - expected) > 1e-4f) {
                if (errors < 8) {
                    printf("[Rank %d] EM TopK2 1D-in token %d k %d: expected %.6f got %.6f\n",
                           g_rank, i, k, expected, got);
                }
                errors++;
            }
        }
    EXPECT_EQ(errors, 0) << "EM TopK2 BWD combine with 1D input mismatches";

    ncclEpTensorDestroy(t_tok); ncclEpTensorDestroy(t_recv);
    ncclEpTensorDestroy(t_w);   ncclEpTensorDestroy(t_recv_w_1d);
    ncclEpTensorDestroy(t_combined_x); ncclEpTensorDestroy(t_combined_w);
    cudaFree(d_tok); cudaFree(d_recv); cudaFree(d_weights); cudaFree(d_recv_w_1d);
    cudaFree(d_combined_x); cudaFree(d_combined_w);
    NCCL_ASSERT(ncclEpHandleDestroy(h));
}

// ── main ──────────────────────────────────────────────────────────────────────

int main(int argc, char* argv[]) {
    if (!ep_bootstrap(argc, argv, "te_ep_ht_bwd_uid")) return 0;
    int ret = RUN_ALL_TESTS();
    ep_teardown();
    return ret;
}
