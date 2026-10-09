/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Unit tests verifying that ncclEpCreateHandle builds the correct
 * sparse-to-dense map (S2D) for rank-major vs expert-major output layouts.
 *
 * Only ncclEpCreateHandle is called — no dispatch/combine.
 *
 * Unified S2D buffer:
 *   Rank-major:  int32_t[max_tokens_per_rank][n_ranks_per_node]
 *     s2d[t][d] = recv slot at dest d, or -1.
 *   Expert-major: int32_t[max_tokens_per_rank][num_topk]
 *     s2d[t][k] = packed(rank_id<<22 | slot), or -1.
 *
 * Setup: 4 ranks, 8 experts (2 per rank), 4 tokens per rank, top-k 1.
 *   Routing: token i on rank r → expert (r * kNumTokens + i) % kNumExperts
 *
 *   Rank 0 tokens: E0→rank0, E1→rank0, E2→rank1, E3→rank1
 *   Rank 1 tokens: E4→rank2, E5→rank2, E6→rank3, E7→rank3
 *   Rank 2 tokens: same as rank 0 (same experts, later source)
 *   Rank 3 tokens: same as rank 1 (same experts, later source)
 *
 * rank-major S2D for rank D:
 *   dest_a = (D%2)*2  (first destination pair for D)
 *   dest_b = dest_a + 1
 *   base   = early ? 0 : 2  where early = (D < nranks/2)
 *   s2d[0][dest_a]=base+0, s2d[1][dest_a]=base+1,
 *   s2d[2][dest_b]=base+0, s2d[3][dest_b]=base+1
 *   (all other entries -1)
 *
 * Expert-major: unified s2d with packed (rank, slot) entries; inner_dim = num_topk.
 *
 */

#include "test_common.h"
#include "../nccl_ep_test_internal.h"

#include <algorithm>
#include <map>
#include <set>

// ── Inspector class ───────────────────────────────────────────────────────────
//
// Wraps an ncclEpHandle_t and provides test-only access to the sparse-to-dense
// map via the non-public ncclEpHandle_test_* helpers.
//
class NcclEpHandleInspector {
public:
    explicit NcclEpHandleInspector(ncclEpHandle_t h, bool expert_major = false)
        : h_(h), expert_major_(expert_major) {}

    int rows() const { return ncclEpHandle_test_getMaxTokensPerRank(h_); }
    int n_ranks() const { return ncclEpHandle_test_getNRanksPerNode(h_); }
    int epr() const { return ncclEpHandle_test_getExpertsPerRank(h_); }
    int top_k() const { return ncclEpHandle_test_getNumTopk(h_); }
    // Inner dimension of the S2D: n_ranks (rank-major) or top_k (expert-major).
    int inner_dim() const { return expert_major_ ? top_k() : n_ranks(); }

    // Copies the entire S2D from device to host. Layout: [rows][inner_dim].
    std::vector<int32_t> s2d_host() const {
        int n = rows() * inner_dim();
        std::vector<int32_t> buf(n, 0);
        EXPECT_EQ(cudaMemcpy(buf.data(),
                             ncclEpHandle_test_getSparseToDenseMap(h_),
                             n * sizeof(int32_t),
                             cudaMemcpyDeviceToHost),
                  cudaSuccess);
        return buf;
    }

    // s2d[token_row][col]: rank-major slot at dest, or packed entry in expert-major.
    int32_t at(int token_row, int col) const {
        auto v = s2d_host();
        return v[token_row * inner_dim() + col];
    }

    // Unpack em_s2d entry: rank_id from bits [31:22], slot from bits [21:0].
    // Must mirror em_s2d_pack/em_s2d_unpack_* in hybrid_ep.cuh.
    static constexpr int kEmS2dSlotBits = 22;
    static constexpr uint32_t kEmS2dSlotMask = (1u << kEmS2dSlotBits) - 1u;
    static int em_unpack_rank(int32_t v) { return static_cast<int>(static_cast<uint32_t>(v) >> kEmS2dSlotBits); }
    static int em_unpack_slot(int32_t v) { return static_cast<int>(static_cast<uint32_t>(v) & kEmS2dSlotMask); }

private:
    ncclEpHandle_t h_;
    bool expert_major_;
};

// ── Property helpers ──────────────────────────────────────────────────────────

// Local expert index at destination d for this rank's token t.
// Valid only when S2D[t][d] >= 0 (i.e. the token is actually routed to d).
static int local_expert_at_dest(int t) {
    return expert_for_token(t) % (kNumExperts / g_nranks);
}

// ── Test fixture ──────────────────────────────────────────────────────────────

class HandleMapsTest : public EpTestBase {
protected:
    // dest_a, dest_b: the two destination ranks that g_rank sends to.
    // early: true when g_rank is the first (lower-index) source for its destinations.
    // base_rank_slot: rank-major recv base slot (0 if early, 2 if late).
    void routing_params(int& dest_a, int& dest_b, bool& early, int& base_rank_slot) const {
        dest_a        = (g_rank % 2) * 2;
        dest_b        = dest_a + 1;
        early         = (g_rank < g_nranks / 2);
        base_rank_slot = early ? 0 : 2;
    }
};

// ── Test: S2D layout for rank-major ────────────────────────────────────────────

TEST_F(HandleMapsTest, S2DRankMajor) {
    ncclEpHandle_t h = make_handle(nullptr);  // default = rank-major
    ASSERT_NE(h, nullptr);

    NcclEpHandleInspector insp(h);
    ASSERT_EQ(insp.rows(), kNumTokens);
    ASSERT_EQ(insp.n_ranks(), g_nranks);

    auto s2d = insp.s2d_host();
    int dest_a, dest_b, base;
    bool early;
    routing_params(dest_a, dest_b, early, base);
    const int C = insp.inner_dim();  // n_ranks

    // rank-major: s2d[t][dest] = slot; all other entries -1.
    // Tokens 0,1 → dest_a; tokens 2,3 → dest_b.
    EXPECT_EQ(s2d[0 * C + dest_a], base + 0) << "rank " << g_rank << ": tok0 dest_a";
    EXPECT_EQ(s2d[1 * C + dest_a], base + 1) << "rank " << g_rank << ": tok1 dest_a";
    EXPECT_EQ(s2d[2 * C + dest_b], base + 0) << "rank " << g_rank << ": tok2 dest_b";
    EXPECT_EQ(s2d[3 * C + dest_b], base + 1) << "rank " << g_rank << ": tok3 dest_b";

    // All other entries must be -1.
    for (int t = 0; t < insp.rows(); ++t) {
        for (int d = 0; d < C; ++d) {
            bool is_hit = ((t <= 1 && d == dest_a) || (t >= 2 && d == dest_b));
            if (is_hit) continue;
            EXPECT_EQ(s2d[t * C + d], -1)
                << "rank " << g_rank << ": expected -1 at [" << t << "][" << d << "]";
        }
    }

    NCCL_ASSERT(ncclEpHandleDestroy(h));
}

// ── Test: S2D layout for expert-major ─────────────────────────────────────────

TEST_F(HandleMapsTest, S2DExpertMajor) {
    ncclEpHandle_t h = make_handle_em(nullptr);
    ASSERT_NE(h, nullptr);

    NcclEpHandleInspector insp(h, /*expert_major=*/true);
    ASSERT_EQ(insp.rows(), kNumTokens);
    ASSERT_EQ(insp.top_k(), kTopK);

    auto s2d = insp.s2d_host();
    int dest_a, dest_b, base;
    bool early;
    routing_params(dest_a, dest_b, early, base);
    const int C = insp.inner_dim();  // top_k

    // Expert-major slots: E_even zone [0,1], E_odd zone [2,3].
    int slot_even = early ? 0 : 1;
    int slot_odd  = early ? 2 : 3;

    // Token 0 → dest_a, expert 0, slot_even
    EXPECT_NE(s2d[0 * C + 0], -1);
    EXPECT_EQ(NcclEpHandleInspector::em_unpack_rank(s2d[0 * C + 0]), dest_a);
    EXPECT_EQ(NcclEpHandleInspector::em_unpack_slot(s2d[0 * C + 0]), slot_even);

    // Token 1 → dest_a, expert 1, slot_odd
    EXPECT_EQ(NcclEpHandleInspector::em_unpack_rank(s2d[1 * C + 0]), dest_a);
    EXPECT_EQ(NcclEpHandleInspector::em_unpack_slot(s2d[1 * C + 0]), slot_odd);

    // Token 2 → dest_b, expert 0, slot_even
    EXPECT_EQ(NcclEpHandleInspector::em_unpack_rank(s2d[2 * C + 0]), dest_b);
    EXPECT_EQ(NcclEpHandleInspector::em_unpack_slot(s2d[2 * C + 0]), slot_even);

    // Token 3 → dest_b, expert 1, slot_odd
    EXPECT_EQ(NcclEpHandleInspector::em_unpack_rank(s2d[3 * C + 0]), dest_b);
    EXPECT_EQ(NcclEpHandleInspector::em_unpack_slot(s2d[3 * C + 0]), slot_odd);

    // Remaining entries in each row should be -1 (only 1 packed entry per token with top_k=1).
    for (int t = 0; t < kNumTokens; ++t) {
        for (int k = 1; k < C; ++k) {
            EXPECT_EQ(s2d[t * C + k], -1)
                << "rank " << g_rank << ": expected -1 at s2d[" << t << "][" << k << "]";
        }
    }

    NCCL_ASSERT(ncclEpHandleDestroy(h));
}

// ── Test: No expert interleaving in expert-major S2D ─────────────────────────
//
// For each destination, tokens that route to the same local expert must occupy
// a contiguous (non-interleaved) slot range. This is the core expert-major
// invariant and is independent of the exact slot values chosen.
//
TEST_F(HandleMapsTest, S2DExpertMajorNoInterleave) {
    ncclEpHandle_t h = make_handle_em(nullptr);
    ASSERT_NE(h, nullptr);

    NcclEpHandleInspector insp(h, /*expert_major=*/true);
    auto s2d = insp.s2d_host();
    const int C = insp.inner_dim();  // top_k

    // Group (dest, slot, local_expert) from s2d packed entries.
    // For each destination, verify no interleaving of expert zones.
    std::map<int, std::vector<std::pair<int32_t, int>>> dest_entries;
    for (int t = 0; t < insp.rows(); t++) {
        int k = local_expert_at_dest(t);
        for (int e = 0; e < C; e++) {
            int32_t v = s2d[t * C + e];
            if (v == -1) break;
            int d = NcclEpHandleInspector::em_unpack_rank(v);
            int slot = NcclEpHandleInspector::em_unpack_slot(v);
            dest_entries[d].push_back({slot, k});
        }
    }

    for (auto& [d, se] : dest_entries) {
        std::sort(se.begin(), se.end());

        // No duplicate slots at this dest.
        for (int i = 1; i < (int)se.size(); i++)
            EXPECT_NE(se[i].first, se[i-1].first)
                << "rank " << g_rank << " dest " << d << ": duplicate slot " << se[i].first;

        // Once expert A ends and expert B begins, A must not reappear.
        std::set<int> closed;
        int cur = se[0].second;
        for (auto& [slot, exp] : se) {
            if (exp != cur) {
                closed.insert(cur);
                EXPECT_EQ(closed.count(exp), 0u)
                    << "rank " << g_rank << " dest " << d
                    << ": expert " << exp << " interleaves at slot " << slot;
                cur = exp;
            }
        }
    }

    NCCL_ASSERT(ncclEpHandleDestroy(h));
}

// ── Test: Expert zone boundaries with alignment ───────────────────────────────
//
// With alignment=A, each expert's slots must fall within a single zone of size A
// starting at a multiple of A, and different experts' zones must not overlap.
//
TEST_F(HandleMapsTest, S2DExpertMajorAlignmentZones) {
    constexpr size_t kAlign = 4;
    ncclEpHandleConfig_t cfg = NCCL_EP_HANDLE_CONFIG_INIT;
    cfg.dispatch_output_per_expert_alignment = kAlign;
    ncclEpHandle_t h = make_handle_em(&cfg);
    ASSERT_NE(h, nullptr);

    NcclEpHandleInspector insp(h, /*expert_major=*/true);
    auto s2d = insp.s2d_host();
    const int C = insp.inner_dim();  // top_k

    // Group (dest → expert → slots) from s2d packed entries.
    std::map<int, std::map<int, std::vector<int32_t>>> dest_expert_slots;
    for (int t = 0; t < insp.rows(); t++) {
        int k = local_expert_at_dest(t);
        for (int e = 0; e < C; e++) {
            int32_t v = s2d[t * C + e];
            if (v == -1) break;
            int d = NcclEpHandleInspector::em_unpack_rank(v);
            int slot = NcclEpHandleInspector::em_unpack_slot(v);
            dest_expert_slots[d][k].push_back(slot);
        }
    }

    for (auto& [d, by_expert] : dest_expert_slots) {
        if (by_expert.empty()) continue;

        // All slots for each expert fall within one alignment zone.
        std::vector<int32_t> zone_starts;
        for (auto& [exp, slots] : by_expert) {
            int32_t zs = (slots[0] / (int32_t)kAlign) * (int32_t)kAlign;
            zone_starts.push_back(zs);
            for (int32_t s : slots)
                EXPECT_EQ(s / (int32_t)kAlign, zs / (int32_t)kAlign)
                    << "rank " << g_rank << " dest " << d
                    << ": expert " << exp << " slot " << s
                    << " outside zone [" << zs << "," << zs + (int32_t)kAlign << ")";
        }

        // Zones for different experts do not overlap.
        std::sort(zone_starts.begin(), zone_starts.end());
        for (int i = 1; i < (int)zone_starts.size(); i++)
            EXPECT_GE(zone_starts[i], zone_starts[i-1] + (int32_t)kAlign)
                << "rank " << g_rank << " dest " << d
                << ": zone overlap at " << zone_starts[i-1] << " and " << zone_starts[i];
    }

    NCCL_ASSERT(ncclEpHandleDestroy(h));
}

// ── main ──────────────────────────────────────────────────────────────────────

int main(int argc, char* argv[]) {
    if (!ep_bootstrap(argc, argv, "te_ep_hmaps_uid")) return 0;
    int ret = RUN_ALL_TESTS();
    ep_teardown();
    return ret;
}
