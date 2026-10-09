/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Unit tests for ncclEpCreateHandle output layout and combine round-trip:
 *   NCCL_EP_LAYOUT_RANK_MAJOR    — tokens grouped by source GPU rank
 *   NCCL_EP_LAYOUT_EXPERT_MAJOR — tokens grouped by local expert
 *
 * Tests (OutputLayoutTest fixture — top-k=1, each token targets one expert):
 *   RankMajorLayout           — dispatch output slot order (rank-major)
 *   ExpertMajorLayout        — dispatch output slot order (expert-major)
 *   ExpertMajorWithAlignment — dispatch output with per-expert zone alignment
 *   CombineRankMajor          — dispatch + identity expert + combine recovers original values (rank-major)
 *   CombineExpertMajor       — dispatch + identity expert + combine recovers original values (expert-major)
 *   DispatchMeta             — expert_token_counts_padded and expert_token_offsets
 *
 * Tests (TopK2MixedRoutingTest fixture — top-k=2, mixed same-rank and cross-rank routing):
 *   RankMajorLayout              — correct recv counts and no duplication for same-rank pairs
 *   ExpertMajorNoAlign          — E1 duplicated from E0 for same-rank pairs; both zones distinct for cross-rank
 *   ExpertMajorAlignZeroPadding — E1 duplicated+padded for same-rank pairs; E1 filled for cross-rank
 *   ExpertMajorDupTokens        — slot-by-slot equality of E0/E1 zones for same-rank pairs (expanded em_s2d entries)
 *
 * Setup: 4 ranks (LSA team size must be a multiple of 4), 8 experts total
 *        (experts_per_rank = 2).
 *   Rank 0 hosts E0 (local 0), E1 (local 1)
 *   Rank 1 hosts E2 (local 0), E3 (local 1)
 *   Rank 2 hosts E4 (local 0), E5 (local 1)
 *   Rank 3 hosts E6 (local 0), E7 (local 1)
 *
 * Token values: rank r, local token i → value = r*kNumTokens + i + 1
 *
 * OutputLayoutTest routing (top-k=1): expert_for_token(i) = (g_rank * kNumTokens + i) % kNumExperts
 *   → experts 0-3 receive tokens from ranks 0,2; experts 4-7 from ranks 1,3.
 *   Each local expert receives exactly 2 tokens (from 2 different source ranks).
 *
 *   Rank 0: T0=1, T1=2, T2=3, T3=4    routing: T0→E0, T1→E1, T2→E2, T3→E3
 *   Rank 1: T0=5, T1=6, T2=7, T3=8    routing: T0→E4, T1→E5, T2→E6, T3→E7
 *   Rank 2: T0=9, T1=10, T2=11, T3=12 routing: T0→E0, T1→E1, T2→E2, T3→E3
 *   Rank 3: T0=13,T1=14, T2=15, T3=16 routing: T0→E4, T1→E5, T2→E6, T3→E7
 *
 *   Rank 0 dispatch output (hosts E0,E1 ← ranks 0,2):
 *     rank-major:     slots [0,1]={1,2}   slots [2,3]={9,10}
 *     Expert-major:  slots [0,1]={1,9}   slots [2,3]={2,10}
 *     +alignment=4:  slots [0..3]={1,9,pad,pad}  slots [4..7]={2,10,pad,pad}
 *
 * TopK2MixedRoutingTest routing (top-k=2, fixed for all source ranks):
 *   T0 → E0 (rank 0) AND E1 (rank 0)  ← same-rank pair: both local experts of rank 0
 *   T1 → E2 (rank 1) AND E3 (rank 1)  ← same-rank pair: both local experts of rank 1
 *   T2 → E4 (rank 2) AND E6 (rank 3)  ← cross-rank pair: one expert on each of ranks 2,3
 *   T3 → E5 (rank 2) AND E7 (rank 3)  ← cross-rank pair: one expert on each of ranks 2,3
 *
 *   All 4 source ranks send T_i identically.
 *   Ranks 0,1 receive 4 tokens (T_{g_rank} from each source, rank-major only).
 *   Ranks 2,3 receive 8 tokens (T2 and T3 from each source, one token per local expert).
 *
 *   Same-rank pair behavior (ranks 0,1) — rank 0 shown:
 *     rank-major:             slots [0..3] = {1,5,9,13}  (no slot duplication; 4 recv)
 *     Expert-major no-align: E0 zone [0..3] = {1,5,9,13}, E1 zone [4..7] = {1,5,9,13} (2 entries per token in em_s2d; 8 recv)
 *     Expert-major align=4:  E0 zone [0..3] = {1,5,9,13}, E1 zone [4..7] = {1,5,9,13} (2 entries per token in em_s2d; 8 recv)
 *
 *   Cross-rank pair behavior (ranks 2,3) — rank 2 shown (E4=local-E0, E5=local-E1):
 *     rank-major:             slots [0..7] = {3,4,7,8,11,12,15,16} (T2 and T3, grouped by source)
 *     Expert-major no-align: E0 zone = {3,7,11,15} (T2→E4), E1 zone = {4,8,12,16} (T3→E5)
 *     Expert-major align=4:  same zones, each padded to 4 (already exactly 4 tokens)
 *
 * Build:  make -C contrib/nccl_ep/tests [BUILDDIR=...]
 * Run:    bash contrib/nccl_ep/tests/run_tests.sh 4
 */

#include "test_common.h"
#include "../nccl_ep_test_internal.h"
#include <set>

static float bf16_val(nv_bfloat16 v) { return __bfloat162float(v); }

// ── Test fixture ──────────────────────────────────────────────────────────────

class OutputLayoutTest : public EpTestBase {
protected:
    // All ranks run dispatch then combine (identity expert: dispatch output → combine input).
    // With topk=1 weight=1.0 and identity expert, combine output[i] == original token value.
    // Returns per-token first-hidden-element for the kNumTokens combined output tokens.
    std::vector<float> run_dispatch_combine(ncclEpHandle_t handle, bool expert_major) {
        std::vector<nv_bfloat16> h_tok(kNumTokens * kHidden);
        for (int i = 0; i < kNumTokens; ++i) {
            float v = static_cast<float>(g_rank * kNumTokens + i + 1);
            for (int hh = 0; hh < kHidden; ++hh)
                h_tok[i * kHidden + hh] = __float2bfloat16(v);
        }

        nv_bfloat16 *d_tok, *d_recv, *d_out;
        float*       d_weights;
        float*       d_recv_w;
        int64_t*     d_recv_idx;
        EXPECT_EQ(cudaMalloc(&d_tok,     kNumTokens     * kHidden * sizeof(nv_bfloat16)), cudaSuccess);
        EXPECT_EQ(cudaMalloc(&d_recv,    kMaxRecvSlots  * kHidden * sizeof(nv_bfloat16)), cudaSuccess);
        EXPECT_EQ(cudaMalloc(&d_out,     kNumTokens     * kHidden * sizeof(nv_bfloat16)), cudaSuccess);
        EXPECT_EQ(cudaMalloc(&d_weights, kNumTokens     * kTopK   * sizeof(float)),       cudaSuccess);
        EXPECT_EQ(cudaMalloc(&d_recv_w,  kMaxRecvSlots  * kTopK   * sizeof(float)),       cudaSuccess);
        EXPECT_EQ(cudaMalloc(&d_recv_idx,kMaxRecvSlots  * kTopK   * sizeof(int64_t)),     cudaSuccess);
        EXPECT_EQ(cudaMemset(d_recv, 0,  kMaxRecvSlots  * kHidden * sizeof(nv_bfloat16)), cudaSuccess);

        std::vector<float> h_w(kNumTokens * kTopK, 1.0f);
        EXPECT_EQ(cudaMemcpy(d_tok,     h_tok.data(), kNumTokens*kHidden*sizeof(nv_bfloat16), cudaMemcpyHostToDevice), cudaSuccess);
        EXPECT_EQ(cudaMemcpy(d_weights, h_w.data(),   kNumTokens*kTopK*sizeof(float),         cudaMemcpyHostToDevice), cudaSuccess);

        ncclEpTensor_t *t_tok, *t_recv, *t_out, *t_w, *t_recv_w, *t_recv_idx;
        EXPECT_EQ(epTensorCreate(&t_tok,  2, ncclBfloat16, d_tok,     kNumTokens,    kHidden), ncclSuccess);
        EXPECT_EQ(epTensorCreate(&t_recv, 2, ncclBfloat16, d_recv,    kMaxRecvSlots, kHidden), ncclSuccess);
        EXPECT_EQ(epTensorCreate(&t_out,  2, ncclBfloat16, d_out,     kNumTokens,    kHidden), ncclSuccess);
        EXPECT_EQ(epTensorCreate(&t_w,    2, ncclFloat32,  d_weights, kNumTokens,    kTopK),   ncclSuccess);
        // EM: recv_topk_weights is 1D [N]; FLAT: 2D [N, top_k] paired with recv_topk_idx.
        if (expert_major) {
            EXPECT_EQ(epTensorCreate(&t_recv_w, 1, ncclFloat32, d_recv_w, kMaxRecvSlots), ncclSuccess);
            t_recv_idx = nullptr;
        } else {
            EXPECT_EQ(epTensorCreate(&t_recv_w,   2, ncclFloat32, d_recv_w,   kMaxRecvSlots, kTopK), ncclSuccess);
            EXPECT_EQ(epTensorCreate(&t_recv_idx, 2, ncclInt64,   d_recv_idx, kMaxRecvSlots, kTopK), ncclSuccess);
        }

        ncclEpDispatchInputs_t  d_in_s  = NCCL_EP_DISPATCH_INPUTS_INIT;
        ncclEpDispatchOutputs_t d_out_s = NCCL_EP_DISPATCH_OUTPUTS_INIT;
        d_in_s.tokens         = t_tok;
        d_in_s.topk_weights   = t_w;
        d_out_s.tokens        = t_recv;
        d_out_s.topk_weights  = t_recv_w;
        if (!expert_major) d_out_s.topk_idx = t_recv_idx;
        ncclEpDispatchConfig_t dcfg = NCCL_EP_DISPATCH_CONFIG_INIT;
        EXPECT_EQ(ncclEpDispatch(handle, &d_in_s, &d_out_s, nullptr, &dcfg, g_stream), ncclSuccess);
        EXPECT_EQ(ncclEpComplete(handle, nullptr, g_stream), ncclSuccess);
        EXPECT_EQ(cudaStreamSynchronize(g_stream), cudaSuccess);

        ncclEpCombineInputs_t  c_in_s  = NCCL_EP_COMBINE_INPUTS_INIT;
        ncclEpCombineOutputs_t c_out_s = NCCL_EP_COMBINE_OUTPUTS_INIT;
        c_in_s.tokens  = t_recv;
        c_out_s.tokens = t_out;
        EXPECT_EQ(ncclEpCombine(handle, &c_in_s, &c_out_s, nullptr, g_stream), ncclSuccess);
        EXPECT_EQ(cudaStreamSynchronize(g_stream), cudaSuccess);

        std::vector<nv_bfloat16> h_out(kNumTokens * kHidden);
        EXPECT_EQ(cudaMemcpy(h_out.data(), d_out, kNumTokens*kHidden*sizeof(nv_bfloat16), cudaMemcpyDeviceToHost), cudaSuccess);

        ncclEpTensorDestroy(t_tok);
        ncclEpTensorDestroy(t_recv);
        ncclEpTensorDestroy(t_out);
        ncclEpTensorDestroy(t_w);
        ncclEpTensorDestroy(t_recv_w);
        if (t_recv_idx) ncclEpTensorDestroy(t_recv_idx);
        cudaFree(d_tok); cudaFree(d_recv); cudaFree(d_out);
        cudaFree(d_weights); cudaFree(d_recv_w); cudaFree(d_recv_idx);

        std::vector<float> vals(kNumTokens);
        for (int i = 0; i < kNumTokens; ++i)
            vals[i] = bf16_val(h_out[i * kHidden]);
        return vals;
    }

    // All ranks run dispatch; returns per-slot first-hidden-element value for this rank.
    std::vector<float> run_dispatch(ncclEpHandle_t handle, int num_recv, bool expert_major) {
        std::vector<nv_bfloat16> h_tok(kNumTokens * kHidden);
        for (int i = 0; i < kNumTokens; ++i) {
            float v = static_cast<float>(g_rank * kNumTokens + i + 1);
            for (int hh = 0; hh < kHidden; ++hh)
                h_tok[i * kHidden + hh] = __float2bfloat16(v);
        }

        nv_bfloat16 *d_tok, *d_recv;
        float*       d_weights;
        float*       d_recv_w;
        int64_t*     d_recv_idx;
        EXPECT_EQ(cudaMalloc(&d_tok,     kNumTokens    * kHidden * sizeof(nv_bfloat16)), cudaSuccess);
        EXPECT_EQ(cudaMalloc(&d_recv,    kMaxRecvSlots * kHidden * sizeof(nv_bfloat16)), cudaSuccess);
        EXPECT_EQ(cudaMemset(d_recv, 0,  kMaxRecvSlots * kHidden * sizeof(nv_bfloat16)), cudaSuccess);
        EXPECT_EQ(cudaMalloc(&d_weights, kNumTokens    * kTopK   * sizeof(float)),       cudaSuccess);
        EXPECT_EQ(cudaMalloc(&d_recv_w,  kMaxRecvSlots * kTopK   * sizeof(float)),       cudaSuccess);
        EXPECT_EQ(cudaMalloc(&d_recv_idx,kMaxRecvSlots * kTopK   * sizeof(int64_t)),     cudaSuccess);

        std::vector<float> h_w(kNumTokens * kTopK, 1.0f);
        EXPECT_EQ(cudaMemcpy(d_tok,     h_tok.data(), kNumTokens*kHidden*sizeof(nv_bfloat16), cudaMemcpyHostToDevice), cudaSuccess);
        EXPECT_EQ(cudaMemcpy(d_weights, h_w.data(),   kNumTokens*kTopK  *sizeof(float),       cudaMemcpyHostToDevice), cudaSuccess);

        ncclEpTensor_t *t_tok, *t_recv, *t_w, *t_recv_w, *t_recv_idx;
        EXPECT_EQ(epTensorCreate(&t_tok,  2, ncclBfloat16, d_tok,     kNumTokens,    kHidden), ncclSuccess);
        EXPECT_EQ(epTensorCreate(&t_recv, 2, ncclBfloat16, d_recv,    kMaxRecvSlots, kHidden), ncclSuccess);
        EXPECT_EQ(epTensorCreate(&t_w,    2, ncclFloat32,  d_weights, kNumTokens,    kTopK),   ncclSuccess);
        if (expert_major) {
            EXPECT_EQ(epTensorCreate(&t_recv_w, 1, ncclFloat32, d_recv_w, kMaxRecvSlots), ncclSuccess);
            t_recv_idx = nullptr;
        } else {
            EXPECT_EQ(epTensorCreate(&t_recv_w,   2, ncclFloat32, d_recv_w,   kMaxRecvSlots, kTopK), ncclSuccess);
            EXPECT_EQ(epTensorCreate(&t_recv_idx, 2, ncclInt64,   d_recv_idx, kMaxRecvSlots, kTopK), ncclSuccess);
        }

        ncclEpDispatchInputs_t  d_in_s  = NCCL_EP_DISPATCH_INPUTS_INIT;
        ncclEpDispatchOutputs_t d_out_s = NCCL_EP_DISPATCH_OUTPUTS_INIT;
        d_in_s.tokens        = t_tok;
        d_in_s.topk_weights  = t_w;
        d_out_s.tokens       = t_recv;
        d_out_s.topk_weights = t_recv_w;
        if (!expert_major) d_out_s.topk_idx = t_recv_idx;
        ncclEpDispatchConfig_t dcfg = NCCL_EP_DISPATCH_CONFIG_INIT;
        EXPECT_EQ(ncclEpDispatch(handle, &d_in_s, &d_out_s, nullptr, &dcfg, g_stream), ncclSuccess);
        EXPECT_EQ(ncclEpComplete(handle, nullptr, g_stream), ncclSuccess);
        EXPECT_EQ(cudaStreamSynchronize(g_stream), cudaSuccess);

        std::vector<nv_bfloat16> h_recv(num_recv * kHidden);
        EXPECT_EQ(cudaMemcpy(h_recv.data(), d_recv, num_recv*kHidden*sizeof(nv_bfloat16), cudaMemcpyDeviceToHost), cudaSuccess);

        ncclEpTensorDestroy(t_tok);
        ncclEpTensorDestroy(t_recv);
        ncclEpTensorDestroy(t_w);
        ncclEpTensorDestroy(t_recv_w);
        if (t_recv_idx) ncclEpTensorDestroy(t_recv_idx);
        cudaFree(d_tok); cudaFree(d_recv); cudaFree(d_weights);
        cudaFree(d_recv_w); cudaFree(d_recv_idx);

        std::vector<float> vals(num_recv);
        for (int s = 0; s < num_recv; ++s)
            vals[s] = bf16_val(h_recv[s * kHidden]);
        return vals;
    }
};

// ── Test: rank-major layout ────────────────────────────────────────────────────

TEST_F(OutputLayoutTest, RankMajorLayout) {
    ncclEpHandle_t h = make_handle(nullptr);
    ASSERT_NE(h, nullptr);

    // Each rank sends 2 tokens to this rank's experts (1 per expert × 2 experts).
    // Total received = 2 ranks × 2 tokens = 4.
    unsigned int num_recv = 0;
    NCCL_ASSERT(ncclEpHandle_test_getNumRecvTokens(h, &num_recv));
    EXPECT_EQ(num_recv, 4u);

    auto slots = run_dispatch(h, static_cast<int>(num_recv), /*expert_major=*/false);

    // rank-major: tokens from the lower-numbered contributing rank in [0,1],
    //            tokens from the higher-numbered rank in [2,3].
    // Ranks 0,2 contribute to experts 0-3; ranks 1,3 contribute to experts 4-7.
    std::set<float> first_half(slots.begin(), slots.begin() + 2);
    std::set<float> second_half(slots.begin() + 2, slots.end());
    if (g_rank == 0) {        // E0,E1 ← ranks 0,2
        EXPECT_EQ(first_half,  (std::set<float>{1.f,  2.f}));
        EXPECT_EQ(second_half, (std::set<float>{9.f, 10.f}));
    } else if (g_rank == 1) { // E2,E3 ← ranks 0,2
        EXPECT_EQ(first_half,  (std::set<float>{ 3.f,  4.f}));
        EXPECT_EQ(second_half, (std::set<float>{11.f, 12.f}));
    } else if (g_rank == 2) { // E4,E5 ← ranks 1,3
        EXPECT_EQ(first_half,  (std::set<float>{ 5.f,  6.f}));
        EXPECT_EQ(second_half, (std::set<float>{13.f, 14.f}));
    } else {                  // E6,E7 ← ranks 1,3
        EXPECT_EQ(first_half,  (std::set<float>{ 7.f,  8.f}));
        EXPECT_EQ(second_half, (std::set<float>{15.f, 16.f}));
    }
    NCCL_ASSERT(ncclEpHandleDestroy(h));
}

// ── Test: expert-major layout (no alignment) ──────────────────────────────────

TEST_F(OutputLayoutTest, ExpertMajorLayout) {
    ncclEpHandle_t h = make_handle_em(nullptr);
    ASSERT_NE(h, nullptr);

    unsigned int num_recv = 0;
    NCCL_ASSERT(ncclEpHandle_test_getNumRecvTokens(h, &num_recv));
    EXPECT_EQ(num_recv, 4u);

    auto slots = run_dispatch(h, static_cast<int>(num_recv), /*expert_major=*/true);

    // Expert-major: slots [0,1] = local expert 0 tokens, slots [2,3] = local expert 1 tokens.
    std::set<float> e_local0(slots.begin(), slots.begin() + 2);
    std::set<float> e_local1(slots.begin() + 2, slots.end());
    if (g_rank == 0) {        // E0={1,9},  E1={2,10}
        EXPECT_EQ(e_local0, (std::set<float>{ 1.f,  9.f}));
        EXPECT_EQ(e_local1, (std::set<float>{ 2.f, 10.f}));
    } else if (g_rank == 1) { // E2={3,11}, E3={4,12}
        EXPECT_EQ(e_local0, (std::set<float>{ 3.f, 11.f}));
        EXPECT_EQ(e_local1, (std::set<float>{ 4.f, 12.f}));
    } else if (g_rank == 2) { // E4={5,13}, E5={6,14}
        EXPECT_EQ(e_local0, (std::set<float>{ 5.f, 13.f}));
        EXPECT_EQ(e_local1, (std::set<float>{ 6.f, 14.f}));
    } else {                  // E6={7,15}, E7={8,16}
        EXPECT_EQ(e_local0, (std::set<float>{ 7.f, 15.f}));
        EXPECT_EQ(e_local1, (std::set<float>{ 8.f, 16.f}));
    }
    NCCL_ASSERT(ncclEpHandleDestroy(h));
}

// ── Test: expert-major + alignment ───────────────────────────────────────────

TEST_F(OutputLayoutTest, ExpertMajorWithAlignment) {
    constexpr size_t kAlign = 4;  // each expert zone padded to 4 tokens

    ncclEpHandleConfig_t cfg = NCCL_EP_HANDLE_CONFIG_INIT;
    cfg.dispatch_output_per_expert_alignment = kAlign;

    ncclEpHandle_t h = make_handle_em(&cfg);
    ASSERT_NE(h, nullptr);

    // GetNumRecvTokens must return the padded total:
    // 2 local experts × align(2 tokens, 4) = 2 × 4 = 8
    unsigned int num_recv = 0;
    NCCL_ASSERT(ncclEpHandle_test_getNumRecvTokens(h, &num_recv));
    EXPECT_EQ(num_recv, 2u * static_cast<unsigned>(kAlign))
        << "Padded total: 2 local experts × 4 = 8";

    auto slots = run_dispatch(h, static_cast<int>(num_recv), /*expert_major=*/true);

    // E_local0 zone = slots [0..3], E_local1 zone = slots [4..7]
    std::set<float> zone0, zone1;
    for (int s = 0; s < 4; ++s) if (slots[s] != 0.f) zone0.insert(slots[s]);
    for (int s = 4; s < 8; ++s) if (slots[s] != 0.f) zone1.insert(slots[s]);

    if (g_rank == 0) {
        EXPECT_EQ(zone0, (std::set<float>{ 1.f,  9.f}));
        EXPECT_EQ(zone1, (std::set<float>{ 2.f, 10.f}));
    } else if (g_rank == 1) {
        EXPECT_EQ(zone0, (std::set<float>{ 3.f, 11.f}));
        EXPECT_EQ(zone1, (std::set<float>{ 4.f, 12.f}));
    } else if (g_rank == 2) {
        EXPECT_EQ(zone0, (std::set<float>{ 5.f, 13.f}));
        EXPECT_EQ(zone1, (std::set<float>{ 6.f, 14.f}));
    } else {
        EXPECT_EQ(zone0, (std::set<float>{ 7.f, 15.f}));
        EXPECT_EQ(zone1, (std::set<float>{ 8.f, 16.f}));
    }
    NCCL_ASSERT(ncclEpHandleDestroy(h));
}

// ── Test: combine round-trip (rank-major) ─────────────────────────────────────
// dispatch + identity expert + combine must recover the original token values.

TEST_F(OutputLayoutTest, CombineRankMajor) {
    ncclEpHandle_t h = make_handle(nullptr);
    ASSERT_NE(h, nullptr);

    unsigned int num_recv = 0;
    NCCL_ASSERT(ncclEpHandle_test_getNumRecvTokens(h, &num_recv));
    EXPECT_EQ(num_recv, 4u);

    auto combined = run_dispatch_combine(h, /*expert_major=*/false);

    for (int i = 0; i < kNumTokens; ++i) {
        float expected = static_cast<float>(g_rank * kNumTokens + i + 1);
        EXPECT_NEAR(combined[i], expected, 0.5f)
            << "rank " << g_rank << " token " << i;
    }
    NCCL_ASSERT(ncclEpHandleDestroy(h));
}

// ── Test: combine round-trip (expert-major, no alignment) ─────────────────────
// Expert-major changes dispatch output slot order; combine uses the remapped
// S2D to route expert outputs back correctly — result identical to rank-major.

TEST_F(OutputLayoutTest, CombineExpertMajor) {
    ncclEpHandle_t h = make_handle_em(nullptr);
    ASSERT_NE(h, nullptr);

    unsigned int num_recv = 0;
    NCCL_ASSERT(ncclEpHandle_test_getNumRecvTokens(h, &num_recv));
    EXPECT_EQ(num_recv, 4u);

    auto combined = run_dispatch_combine(h, /*expert_major=*/true);

    for (int i = 0; i < kNumTokens; ++i) {
        float expected = static_cast<float>(g_rank * kNumTokens + i + 1);
        EXPECT_NEAR(combined[i], expected, 0.5f)
            << "rank " << g_rank << " token " << i;
    }
    NCCL_ASSERT(ncclEpHandleDestroy(h));
}

// ── Test: dispatch meta — offsets and padded counts ───────────────────────────

TEST_F(OutputLayoutTest, DispatchMeta) {
    constexpr size_t kAlign  = 4;
    const int        E_local = kNumExperts / g_nranks;  // local experts per rank = 2

    ncclEpHandleConfig_t cfg = NCCL_EP_HANDLE_CONFIG_INIT;
    cfg.dispatch_output_per_expert_alignment = kAlign;

    int64_t *d_off, *d_cnt;
    CUDA_ASSERT(cudaMalloc(&d_off, E_local * sizeof(int64_t)));
    CUDA_ASSERT(cudaMalloc(&d_cnt, E_local * sizeof(int64_t)));

    ncclEpTensor_t *t_off, *t_cnt;
    NCCL_ASSERT(epTensorCreate(&t_off, 1, ncclInt64, d_off, E_local));
    NCCL_ASSERT(epTensorCreate(&t_cnt, 1, ncclInt64, d_cnt, E_local));

    ncclEpLayoutInfo_t layout = NCCL_EP_LAYOUT_INFO_INIT;
    layout.expert_counters = t_cnt;
    layout.expert_offsets  = t_off;
    ncclEpHandle_t h = make_handle_em(&cfg, &layout);
    ASSERT_NE(h, nullptr);

    std::vector<int64_t> h_off(E_local), h_cnt(E_local);
    CUDA_ASSERT(cudaMemcpy(h_off.data(), d_off, E_local*sizeof(int64_t), cudaMemcpyDeviceToHost));
    CUDA_ASSERT(cudaMemcpy(h_cnt.data(), d_cnt, E_local*sizeof(int64_t), cudaMemcpyDeviceToHost));

    // Each local expert receives 2 tokens (1 per rank), padded to 4.
    // offsets[0]=0, offsets[1]=4
    EXPECT_EQ(h_cnt[0], static_cast<int64_t>(kAlign)) << "local E0: 2 tokens padded to 4";
    EXPECT_EQ(h_cnt[1], static_cast<int64_t>(kAlign)) << "local E1: 2 tokens padded to 4";
    EXPECT_EQ(h_off[0], 0LL)                          << "local E0 starts at slot 0";
    EXPECT_EQ(h_off[1], static_cast<int64_t>(kAlign)) << "local E1 starts at slot 4";

    ncclEpTensorDestroy(t_off);
    ncclEpTensorDestroy(t_cnt);
    cudaFree(d_off); cudaFree(d_cnt);
    NCCL_ASSERT(ncclEpHandleDestroy(h));
}

// ── Test: RECV_TOTAL_COUNTER_DEVICE — scalar padded total recv tokens ────────
// EM: padded total = num_local_experts * align; must equal getNumRecvTokens.

TEST_F(OutputLayoutTest, TotalCounterDeviceEM) {
    constexpr size_t kAlign  = 4;
    const int        E_local = kNumExperts / g_nranks;

    ncclEpHandleConfig_t cfg = NCCL_EP_HANDLE_CONFIG_INIT;
    cfg.dispatch_output_per_expert_alignment = kAlign;

    int64_t* d_total;
    CUDA_ASSERT(cudaMalloc(&d_total, sizeof(int64_t)));
    CUDA_ASSERT(cudaMemset(d_total, 0, sizeof(int64_t)));

    // EM requires expert_counters (per validation); pair total counter with it.
    int64_t* d_cnt;
    CUDA_ASSERT(cudaMalloc(&d_cnt, E_local * sizeof(int64_t)));

    ncclEpTensor_t *t_total, *t_cnt;
    NCCL_ASSERT(epTensorCreate(&t_total, 1, ncclInt64, d_total, 1));
    NCCL_ASSERT(epTensorCreate(&t_cnt,   1, ncclInt64, d_cnt, E_local));

    ncclEpLayoutInfo_t layout = NCCL_EP_LAYOUT_INFO_INIT;
    layout.recv_total_counter = t_total;
    layout.expert_counters    = t_cnt;
    ncclEpHandle_t h = make_handle_em(&cfg, &layout);
    ASSERT_NE(h, nullptr);

    int64_t h_total = 0;
    CUDA_ASSERT(cudaMemcpy(&h_total, d_total, sizeof(int64_t), cudaMemcpyDeviceToHost));

    unsigned int num_recv = 0;
    NCCL_ASSERT(ncclEpHandle_test_getNumRecvTokens(h, &num_recv));
    EXPECT_EQ(static_cast<unsigned int>(h_total), num_recv)
        << "RECV_TOTAL_COUNTER_DEVICE must match getNumRecvTokens (EM padded)";
    EXPECT_EQ(h_total, static_cast<int64_t>(E_local) * static_cast<int64_t>(kAlign))
        << "EM padded total = num_local_experts * align";

    ncclEpTensorDestroy(t_total);
    ncclEpTensorDestroy(t_cnt);
    cudaFree(d_total); cudaFree(d_cnt);
    NCCL_ASSERT(ncclEpHandleDestroy(h));
}

// FLAT: unpadded total = sum of per-expert recv counts (kNumTokens for our routing).
TEST_F(OutputLayoutTest, TotalCounterDeviceFLAT) {
    int32_t* d_total;
    CUDA_ASSERT(cudaMalloc(&d_total, sizeof(int32_t)));
    CUDA_ASSERT(cudaMemset(d_total, 0, sizeof(int32_t)));

    ncclEpTensor_t* t_total;
    NCCL_ASSERT(epTensorCreate(&t_total, 1, ncclInt32, d_total, 1));

    ncclEpLayoutInfo_t layout = NCCL_EP_LAYOUT_INFO_INIT;
    layout.recv_total_counter = t_total;
    ncclEpHandle_t h = make_handle(nullptr, &layout);
    ASSERT_NE(h, nullptr);

    int32_t h_total = 0;
    CUDA_ASSERT(cudaMemcpy(&h_total, d_total, sizeof(int32_t), cudaMemcpyDeviceToHost));

    unsigned int num_recv = 0;
    NCCL_ASSERT(ncclEpHandle_test_getNumRecvTokens(h, &num_recv));
    EXPECT_EQ(static_cast<unsigned int>(h_total), num_recv)
        << "FLAT RECV_TOTAL_COUNTER_DEVICE must match getNumRecvTokens (unpadded)";
    EXPECT_EQ(h_total, 4)
        << "FLAT total: 2 ranks × 2 tokens per local expert × no padding = 4";

    ncclEpTensorDestroy(t_total);
    cudaFree(d_total);
    NCCL_ASSERT(ncclEpHandleDestroy(h));
}

// ── TopK2MixedRoutingTest fixture ─────────────────────────────────────────────
// top-k=2 with a fixed routing mixing same-rank pairs (T0/T1) and cross-rank
// pairs (T2/T3):
//   T0 → E0, E1  (both rank 0 — same-rank pair)
//   T1 → E2, E3  (both rank 1 — same-rank pair)
//   T2 → E4, E6  (rank 2 and rank 3 — cross-rank pair)
//   T3 → E5, E7  (rank 2 and rank 3 — cross-rank pair)
//
// Receives (all 4 source ranks use identical routing):
//   Ranks 0,1: 4 tokens (T_{g_rank} from each source; same-rank pair → no duplication)
//   Ranks 2,3: 8 tokens (T2 and T3 from each source; one token per local expert)

static constexpr int kTopK2 = 2;

class TopK2MixedRoutingTest : public ::testing::Test {
protected:
    ncclEpTensor_t* topk_idx2_    = nullptr;
    ncclEpTensor_t* topk_idx2_em_ = nullptr;
    int64_t*       d_topk2_      = nullptr;

    void SetUp() override {
        CUDA_ASSERT(cudaMalloc(&d_topk2_, kNumTokens * kTopK2 * sizeof(int64_t)));
        const int64_t routing[kNumTokens * kTopK2] = {
            0, 1,  // T0 → E0 (rank 0 local-E0), E1 (rank 0 local-E1)  — same-rank pair
            2, 3,  // T1 → E2 (rank 1 local-E0), E3 (rank 1 local-E1)  — same-rank pair
            4, 6,  // T2 → E4 (rank 2 local-E0), E6 (rank 3 local-E0)  — cross-rank pair
            5, 7,  // T3 → E5 (rank 2 local-E1), E7 (rank 3 local-E1)  — cross-rank pair
        };
        CUDA_ASSERT(cudaMemcpy(d_topk2_, routing, sizeof(routing), cudaMemcpyHostToDevice));
        NCCL_ASSERT(epTensorCreate(&topk_idx2_, 2, ncclInt64,
                                       d_topk2_, kNumTokens, kTopK2));
        NCCL_ASSERT(epTensorCreate(&topk_idx2_em_, 2, ncclInt64,
                                       d_topk2_, kNumTokens, kTopK2));
    }

    void TearDown() override {
        if (topk_idx2_em_) ncclEpTensorDestroy(topk_idx2_em_);
        if (topk_idx2_) ncclEpTensorDestroy(topk_idx2_);
        if (d_topk2_)   cudaFree(d_topk2_);
    }

    ncclEpHandle_t make_handle2(const ncclEpHandleConfig_t* cfg,
                                 const ncclEpLayoutInfo_t* layout_info = nullptr) {
        ncclEpHandle_t h = nullptr;
        EXPECT_EQ(ncclEpCreateHandle(&h, g_ep_group, NCCL_EP_LAYOUT_FLAT,
                                     topk_idx2_, layout_info, cfg, g_stream), ncclSuccess);
        EXPECT_EQ(cudaStreamSynchronize(g_stream), cudaSuccess);
        return h;
    }

    ncclEpHandle_t make_handle2_em(const ncclEpHandleConfig_t* cfg,
                                    const ncclEpLayoutInfo_t* layout_info = nullptr) {
        ncclEpHandle_t h = nullptr;
        EXPECT_EQ(ncclEpCreateHandle(&h, g_ep_group_em, NCCL_EP_LAYOUT_EXPERT_MAJOR,
                                     topk_idx2_em_, layout_info, cfg, g_stream), ncclSuccess);
        EXPECT_EQ(cudaStreamSynchronize(g_stream), cudaSuccess);
        return h;
    }

    std::vector<float> run_dispatch2(ncclEpHandle_t handle, int num_recv, bool expert_major) {
        std::vector<nv_bfloat16> h_tok(kNumTokens * kHidden);
        for (int i = 0; i < kNumTokens; ++i) {
            float v = static_cast<float>(g_rank * kNumTokens + i + 1);
            for (int hh = 0; hh < kHidden; ++hh)
                h_tok[i * kHidden + hh] = __float2bfloat16(v);
        }

        nv_bfloat16 *d_tok, *d_recv;
        float*       d_weights;
        float*       d_recv_w;
        int64_t*     d_recv_idx;
        EXPECT_EQ(cudaMalloc(&d_tok,     kNumTokens    * kHidden * sizeof(nv_bfloat16)), cudaSuccess);
        EXPECT_EQ(cudaMalloc(&d_recv,    kMaxRecvSlots * kHidden * sizeof(nv_bfloat16)), cudaSuccess);
        EXPECT_EQ(cudaMemset(d_recv, 0,  kMaxRecvSlots * kHidden * sizeof(nv_bfloat16)), cudaSuccess);
        EXPECT_EQ(cudaMalloc(&d_weights, kNumTokens    * kTopK2  * sizeof(float)),       cudaSuccess);
        EXPECT_EQ(cudaMalloc(&d_recv_w,  kMaxRecvSlots * kTopK2  * sizeof(float)),       cudaSuccess);
        EXPECT_EQ(cudaMalloc(&d_recv_idx,kMaxRecvSlots * kTopK2  * sizeof(int64_t)),     cudaSuccess);

        EXPECT_EQ(cudaMemcpy(d_tok, h_tok.data(), kNumTokens*kHidden*sizeof(nv_bfloat16), cudaMemcpyHostToDevice), cudaSuccess);
        std::vector<float> h_w(kNumTokens * kTopK2, 1.0f);
        EXPECT_EQ(cudaMemcpy(d_weights, h_w.data(), kNumTokens*kTopK2*sizeof(float), cudaMemcpyHostToDevice), cudaSuccess);

        ncclEpTensor_t *t_tok, *t_recv, *t_w, *t_recv_w, *t_recv_idx;
        EXPECT_EQ(epTensorCreate(&t_tok,  2, ncclBfloat16, d_tok,     kNumTokens,    kHidden), ncclSuccess);
        EXPECT_EQ(epTensorCreate(&t_recv, 2, ncclBfloat16, d_recv,    kMaxRecvSlots, kHidden), ncclSuccess);
        EXPECT_EQ(epTensorCreate(&t_w,    2, ncclFloat32,  d_weights, kNumTokens,    kTopK2),  ncclSuccess);
        if (expert_major) {
            EXPECT_EQ(epTensorCreate(&t_recv_w, 1, ncclFloat32, d_recv_w, kMaxRecvSlots), ncclSuccess);
            t_recv_idx = nullptr;
        } else {
            EXPECT_EQ(epTensorCreate(&t_recv_w,   2, ncclFloat32, d_recv_w,   kMaxRecvSlots, kTopK2), ncclSuccess);
            EXPECT_EQ(epTensorCreate(&t_recv_idx, 2, ncclInt64,   d_recv_idx, kMaxRecvSlots, kTopK2), ncclSuccess);
        }

        ncclEpDispatchInputs_t  d_in_s  = NCCL_EP_DISPATCH_INPUTS_INIT;
        ncclEpDispatchOutputs_t d_out_s = NCCL_EP_DISPATCH_OUTPUTS_INIT;
        d_in_s.tokens        = t_tok;
        d_in_s.topk_weights  = t_w;
        d_out_s.tokens       = t_recv;
        d_out_s.topk_weights = t_recv_w;
        if (!expert_major) d_out_s.topk_idx = t_recv_idx;
        ncclEpDispatchConfig_t dcfg = NCCL_EP_DISPATCH_CONFIG_INIT;
        EXPECT_EQ(ncclEpDispatch(handle, &d_in_s, &d_out_s, nullptr, &dcfg, g_stream), ncclSuccess);
        EXPECT_EQ(ncclEpComplete(handle, nullptr, g_stream), ncclSuccess);
        EXPECT_EQ(cudaStreamSynchronize(g_stream), cudaSuccess);

        std::vector<nv_bfloat16> h_recv(num_recv * kHidden);
        EXPECT_EQ(cudaMemcpy(h_recv.data(), d_recv, num_recv*kHidden*sizeof(nv_bfloat16), cudaMemcpyDeviceToHost), cudaSuccess);

        ncclEpTensorDestroy(t_tok);
        ncclEpTensorDestroy(t_recv);
        ncclEpTensorDestroy(t_w);
        ncclEpTensorDestroy(t_recv_w);
        if (t_recv_idx) ncclEpTensorDestroy(t_recv_idx);
        cudaFree(d_tok); cudaFree(d_recv); cudaFree(d_weights);
        cudaFree(d_recv_w); cudaFree(d_recv_idx);

        std::vector<float> vals(num_recv);
        for (int s = 0; s < num_recv; ++s)
            vals[s] = bf16_val(h_recv[s * kHidden]);
        return vals;
    }

    // Full set of token values this rank expects to receive.
    // Ranks 0,1: T_{g_rank} from each of the 4 source ranks (same-rank pair, no duplication).
    // Ranks 2,3: T2 and T3 from each of the 4 source ranks (cross-rank pair).
    std::set<float> expected_recv_set() const {
        std::set<float> s;
        if (g_rank <= 1) {
            for (int src = 0; src < g_nranks; ++src)
                s.insert(static_cast<float>(src * kNumTokens + g_rank + 1));
        } else {
            for (int src = 0; src < g_nranks; ++src) {
                s.insert(static_cast<float>(src * kNumTokens + 3)); // T2
                s.insert(static_cast<float>(src * kNumTokens + 4)); // T3
            }
        }
        return s;
    }

    // Expected non-zero tokens in local-E0 zone (expert-major).
    // Ranks 0,1: T_{g_rank} only (break-on-first-match).
    // Ranks 2,3: T2 values (E4/E6 are both local-E0 on their respective ranks).
    std::set<float> expected_e0_zone() const {
        int tok_idx = (g_rank <= 1) ? g_rank : 2; // T_{g_rank} or T2
        std::set<float> s;
        for (int src = 0; src < g_nranks; ++src)
            s.insert(static_cast<float>(src * kNumTokens + tok_idx + 1));
        return s;
    }

    // Expected tokens in local-E1 zone (expert-major, duplication active).
    // Ranks 0,1: same as E0 zone (em_s2d carries one entry per (rank, expert) so dispatch writes both zones).
    // Ranks 2,3: T3 values (E5/E7 are both local-E1 on their respective ranks).
    std::set<float> expected_e1_zone() const {
        if (g_rank <= 1) return expected_e0_zone(); // E1 receives the same source tokens as E0 via em_s2d
        std::set<float> s;
        for (int src = 0; src < g_nranks; ++src)
            s.insert(static_cast<float>(src * kNumTokens + 4)); // T3
        return s;
    }
};

// ── Test: rank-major — correct recv counts; no duplication for same-rank pairs ─

TEST_F(TopK2MixedRoutingTest, RankMajorLayout) {
    ncclEpHandle_t h = make_handle2(nullptr);
    ASSERT_NE(h, nullptr);

    unsigned int num_recv = 0;
    NCCL_ASSERT(ncclEpHandle_test_getNumRecvTokens(h, &num_recv));
    // Same-rank pair (ranks 0,1): token sent once despite 2 local experts targeted.
    // Cross-rank pair (ranks 2,3): T2 and T3 each from 4 sources = 8 tokens.
    const unsigned int expected = (g_rank <= 1) ? 4u : 8u;
    EXPECT_EQ(num_recv, expected);

    auto slots = run_dispatch2(h, static_cast<int>(num_recv), /*expert_major=*/false);
    std::set<float> got(slots.begin(), slots.end());
    EXPECT_EQ(got, expected_recv_set());
    NCCL_ASSERT(ncclEpHandleDestroy(h));
}

// ── Test: expert-major no-align — E1 duplicated from E0 for same-rank pairs ────

TEST_F(TopK2MixedRoutingTest, ExpertMajorNoAlign) {
    ncclEpHandle_t h = make_handle2_em(nullptr);
    ASSERT_NE(h, nullptr);

    unsigned int num_recv = 0;
    NCCL_ASSERT(ncclEpHandle_test_getNumRecvTokens(h, &num_recv));
    // Same-rank pair: dispatch writes both E0 and E1 zones (em_s2d has 2 entries per token) → total 8.
    // Cross-rank pair: E0=4, E1=4 → total 8.
    EXPECT_EQ(num_recv, 8u);

    auto slots = run_dispatch2(h, static_cast<int>(num_recv), /*expert_major=*/true);
    // E0 zone [0..3] and E1 zone [4..7] both present; set union = expected_recv_set().
    std::set<float> got(slots.begin(), slots.end());
    EXPECT_EQ(got, expected_recv_set());
    NCCL_ASSERT(ncclEpHandleDestroy(h));
}

// ── Test: expert-major + align=4 — dispatch writes both zones; padding zeros trailing slots ─
// Same-rank pair (ranks 0,1): em_s2d carries one entry per (rank, expert) so dispatch writes both zones; no padding needed
//   (actual_counts[1]=4 == alignment, pad=0).
// Cross-rank pair (ranks 2,3): both zones filled with distinct tokens (no padding needed).

TEST_F(TopK2MixedRoutingTest, ExpertMajorAlignZeroPadding) {
    constexpr size_t kAlign = 4;

    ncclEpHandleConfig_t cfg = NCCL_EP_HANDLE_CONFIG_INIT;
    cfg.dispatch_output_per_expert_alignment = kAlign;

    ncclEpHandle_t h = make_handle2_em(&cfg);
    ASSERT_NE(h, nullptr);

    unsigned int num_recv = 0;
    NCCL_ASSERT(ncclEpHandle_test_getNumRecvTokens(h, &num_recv));
    EXPECT_EQ(num_recv, 2u * static_cast<unsigned>(kAlign))
        << "2 local experts × align=4 = 8 slots (all ranks)";

    auto slots = run_dispatch2(h, static_cast<int>(num_recv), /*expert_major=*/true);

    std::set<float> e0_got, e1_got;
    for (size_t s = 0;      s < kAlign;   ++s) if (slots[s] != 0.f) e0_got.insert(slots[s]);
    for (size_t s = kAlign; s < 2*kAlign; ++s) if (slots[s] != 0.f) e1_got.insert(slots[s]);

    EXPECT_EQ(e0_got, expected_e0_zone());
    // E1 zone: dispatch writes the same source token into both expert zones for same-rank pairs; cross-rank gets real E1 tokens.
    EXPECT_EQ(e1_got, expected_e1_zone());

    NCCL_ASSERT(ncclEpHandleDestroy(h));
}

// ── Test: same-rank dup — slot-by-slot E0/E1 equality for same-rank pairs ────
// Uses no alignment so zone boundaries are clean (E0=[0..3], E1=[4..7]).
// Same-rank pair (ranks 0,1): every E1 slot must equal its matching E0 slot.
// Cross-rank pair (ranks 2,3): E0 and E1 slots are distinct token values.

TEST_F(TopK2MixedRoutingTest, ExpertMajorDupTokens) {
    ncclEpHandle_t h = make_handle2_em(nullptr);
    ASSERT_NE(h, nullptr);

    unsigned int num_recv = 0;
    NCCL_ASSERT(ncclEpHandle_test_getNumRecvTokens(h, &num_recv));
    EXPECT_EQ(num_recv, 8u) << "2 experts × 4 tokens each = 8 slots";

    auto slots = run_dispatch2(h, static_cast<int>(num_recv), /*expert_major=*/true);
    // slots[0..3] = E0 zone, slots[4..7] = E1 zone.
    if (g_rank <= 1) {
        // Same-rank pair: dispatch must have written identical values into E1.
        for (int i = 0; i < 4; ++i)
            EXPECT_EQ(slots[i], slots[i + 4])
                << "E0[" << i << "]=" << slots[i] << " != E1[" << i << "]=" << slots[i+4];
    } else {
        // Cross-rank pair: E0 and E1 carry different token indices (T2 vs T3).
        std::set<float> e0_vals(slots.begin(), slots.begin() + 4);
        std::set<float> e1_vals(slots.begin() + 4, slots.end());
        // E0 ∩ E1 must be empty (T2 values ≠ T3 values).
        std::set<float> intersection;
        for (auto v : e0_vals) if (e1_vals.count(v)) intersection.insert(v);
        EXPECT_TRUE(intersection.empty())
            << "E0 and E1 zones share values for cross-rank pair (should be disjoint)";
        EXPECT_EQ(e0_vals, expected_e0_zone());
        EXPECT_EQ(e1_vals, expected_e1_zone());
    }

    NCCL_ASSERT(ncclEpHandleDestroy(h));
}

// ── Test: expert-major + align=8 — verify padding slots are zero-initialized ──
// With kAlign=8 and 4 real tokens per zone, each zone has 4 padding slots
// that must be all-zero (every hidden element = 0x0000).
// Same-rank pair (ranks 0,1): dispatch writes both zones; both have 4 real + 4 pad.
// Cross-rank pair (ranks 2,3): each zone has 4 real + 4 pad.

TEST_F(TopK2MixedRoutingTest, ExpertMajorAlignZeroPadVerified) {
    constexpr size_t kAlign = 8;

    ncclEpHandleConfig_t cfg = NCCL_EP_HANDLE_CONFIG_INIT;
    cfg.dispatch_output_per_expert_alignment = kAlign;

    ncclEpHandle_t h = make_handle2_em(&cfg);
    ASSERT_NE(h, nullptr);

    unsigned int num_recv = 0;
    NCCL_ASSERT(ncclEpHandle_test_getNumRecvTokens(h, &num_recv));
    EXPECT_EQ(num_recv, 2u * static_cast<unsigned>(kAlign))
        << "2 local experts × align=8 = 16 slots";

    // Run dispatch and read back the full recv buffer (all hidden elements).
    std::vector<nv_bfloat16> h_tok(kNumTokens * kHidden);
    for (int i = 0; i < kNumTokens; ++i) {
        float v = static_cast<float>(g_rank * kNumTokens + i + 1);
        for (int hh = 0; hh < kHidden; ++hh)
            h_tok[i * kHidden + hh] = __float2bfloat16(v);
    }

    nv_bfloat16 *d_tok, *d_recv;
    float* d_weights;
    float* d_recv_w;
    // Fill recv buffer with 0xDE pattern so we can detect un-zeroed padding.
    CUDA_ASSERT(cudaMalloc(&d_tok,     kNumTokens    * kHidden * sizeof(nv_bfloat16)));
    CUDA_ASSERT(cudaMalloc(&d_recv,    kMaxRecvSlots * kHidden * sizeof(nv_bfloat16)));
    CUDA_ASSERT(cudaMemset(d_recv, 0xDE, kMaxRecvSlots * kHidden * sizeof(nv_bfloat16)));
    CUDA_ASSERT(cudaMalloc(&d_weights, kNumTokens    * kTopK2  * sizeof(float)));
    CUDA_ASSERT(cudaMalloc(&d_recv_w,  kMaxRecvSlots * sizeof(float)));

    CUDA_ASSERT(cudaMemcpy(d_tok, h_tok.data(), kNumTokens*kHidden*sizeof(nv_bfloat16), cudaMemcpyHostToDevice));
    std::vector<float> h_w(kNumTokens * kTopK2, 1.0f);
    CUDA_ASSERT(cudaMemcpy(d_weights, h_w.data(), kNumTokens*kTopK2*sizeof(float), cudaMemcpyHostToDevice));

    ncclEpTensor_t *t_tok, *t_recv, *t_w, *t_recv_w;
    NCCL_ASSERT(epTensorCreate(&t_tok,    2, ncclBfloat16, d_tok,     kNumTokens,    kHidden));
    NCCL_ASSERT(epTensorCreate(&t_recv,   2, ncclBfloat16, d_recv,    kMaxRecvSlots, kHidden));
    NCCL_ASSERT(epTensorCreate(&t_w,      2, ncclFloat32,  d_weights, kNumTokens,    kTopK2));
    NCCL_ASSERT(epTensorCreate(&t_recv_w, 1, ncclFloat32,  d_recv_w,  kMaxRecvSlots));

    ncclEpDispatchInputs_t  d_in_s  = NCCL_EP_DISPATCH_INPUTS_INIT;
    ncclEpDispatchOutputs_t d_out_s = NCCL_EP_DISPATCH_OUTPUTS_INIT;
    d_in_s.tokens        = t_tok;
    d_in_s.topk_weights  = t_w;
    d_out_s.tokens       = t_recv;
    d_out_s.topk_weights = t_recv_w;
    ncclEpDispatchConfig_t dcfg = NCCL_EP_DISPATCH_CONFIG_INIT;
    NCCL_ASSERT(ncclEpDispatch(h, &d_in_s, &d_out_s, nullptr, &dcfg, g_stream));
    NCCL_ASSERT(ncclEpComplete(h, nullptr, g_stream));
    CUDA_ASSERT(cudaStreamSynchronize(g_stream));

    std::vector<nv_bfloat16> h_recv(num_recv * kHidden);
    CUDA_ASSERT(cudaMemcpy(h_recv.data(), d_recv, num_recv*kHidden*sizeof(nv_bfloat16), cudaMemcpyDeviceToHost));

    ncclEpTensorDestroy(t_tok);
    ncclEpTensorDestroy(t_recv);
    ncclEpTensorDestroy(t_w);
    ncclEpTensorDestroy(t_recv_w);
    cudaFree(d_tok); cudaFree(d_recv); cudaFree(d_weights); cudaFree(d_recv_w);

    // Each zone has kAlign=8 slots: 4 real tokens + 4 padding.
    // Identify real vs padding by checking first hidden element (real tokens have non-zero values).
    int pad_errors = 0;
    for (int zone = 0; zone < 2; ++zone) {
        size_t zone_start = zone * kAlign;
        int real_count = 0;
        for (size_t s = zone_start; s < zone_start + kAlign; ++s) {
            float v = __bfloat162float(h_recv[s * kHidden]);
            if (v != 0.f) { real_count++; continue; }
            // This is a padding slot — verify ALL hidden elements are zero.
            for (int hh = 0; hh < kHidden; ++hh) {
                uint16_t raw;
                memcpy(&raw, &h_recv[s * kHidden + hh], sizeof(raw));
                if (raw != 0) {
                    if (pad_errors < 5)
                        printf("[Rank %d] zone %d slot %zu h=%d: expected 0x0000, got 0x%04x\n",
                               g_rank, zone, s, hh, raw);
                    pad_errors++;
                }
            }
        }
        EXPECT_EQ(real_count, 4) << "rank " << g_rank << " zone " << zone;
    }
    EXPECT_EQ(pad_errors, 0)
        << "rank " << g_rank << ": " << pad_errors << " non-zero elements in padding slots";

    // Also verify the real tokens are correct.
    std::set<float> e0_got, e1_got;
    for (size_t s = 0;      s < kAlign;   ++s) { float v = __bfloat162float(h_recv[s*kHidden]); if (v != 0.f) e0_got.insert(v); }
    for (size_t s = kAlign; s < 2*kAlign; ++s) { float v = __bfloat162float(h_recv[s*kHidden]); if (v != 0.f) e1_got.insert(v); }
    EXPECT_EQ(e0_got, expected_e0_zone());
    EXPECT_EQ(e1_got, expected_e1_zone());

    NCCL_ASSERT(ncclEpHandleDestroy(h));
}

// ── main ──────────────────────────────────────────────────────────────────────

int main(int argc, char* argv[]) {
    if (!ep_bootstrap(argc, argv, "te_ep_layout_uid")) return 0;
    int ret = RUN_ALL_TESTS();
    ep_teardown();
    return ret;
}
