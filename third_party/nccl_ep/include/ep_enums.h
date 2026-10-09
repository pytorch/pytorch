/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 * See LICENSE.txt for more license information.
 */

#pragma once

// Auto configuration constant for dynamic/automatic sizing
#define NCCL_EP_AUTO 0

// Communication algorithm (mode)
typedef enum {
    // Low-Latency (LL) mode
    NCCL_EP_ALGO_LOW_LATENCY = 0,
    // High-Throughput (HT) mode
    NCCL_EP_ALGO_HIGH_THROUGHPUT = 1
} ncclEpAlgorithm_t;

/**
 * Receive-buffer layout for the Low-Latency (LL) dispatch path.
 *
 * Controls the shape of the user-visible dispatch output tensor (recv_x) and
 * determines which side performs the expert weighted reduction before combine.
 *
 * The value is usable directly as a CUDA non-type template parameter.
 */
typedef enum {
    /**
     * Sentinel for "layout not set" (zero-init default).
     *
     * Callers must override this with one of the layout values below before
     * passing the config to ncclEpInitHandle / ncclEpHandleMemSize; the
     * library does not auto-resolve based on algorithm. Leaving the field at
     * NCCL_EP_LAYOUT_UNSET is a programmer error and trips an assertion at
     * handle-init time.
     */
    NCCL_EP_LAYOUT_UNSET = NCCL_EP_AUTO,

    /**
     * Expert-major layout.
     *
     * Dispatch output:
     *   recv_x shape:            [num_local_experts, max_dispatch_tokens_per_rank * num_ranks, hidden]
     *   recv_topk_weights shape: HT: [N] (1D, one weight per slot — each slot is per
     *                                    (source_token, local_expert), at most one match);
     *                            LL: nullptr (not populated under EM).
     *   recv_topk_idx:           not populated under EM (slot index encodes expert).
     *
     * Combine input is the post-expert activation in the same shape.
     * Each expert rank sends its post-expert activation back to the originating
     * rank as a separate message; the combine kernel there accumulates up to
     * num_topk per-expert contributions, weighted by their topk weights, as
     * they arrive (reduction on the receive side).
     */
    NCCL_EP_LAYOUT_EXPERT_MAJOR,

    /**
     * Rank-major layout.
     *
     * Dispatch output:
     *   recv_x shape:            [num_ranks, max_dispatch_tokens_per_rank, hidden]
     *   recv_topk_weights shape: [max_dispatch_tokens_per_rank * num_ranks, num_topk]
     *   recv_topk_idx shape:     [max_dispatch_tokens_per_rank * num_ranks, num_topk]
     *
     * Tokens arrive in rank-major order with no expert dimension.
     * The caller is responsible for running expert computation on each token
     * slot and pre-reducing across local experts using the per-expert weights
     * from recv_topk_weights, producing one weighted output vector per slot.
     *
     * ncclEpCombine sends these pre-reduced vectors back to each token's home
     * rank. The home rank still performs a receive-side reduction, but it
     * accumulates one contribution per source expert rank (not per expert),
     * weighted by that rank's combined weight (sum of its top-k weights for
     * the token). This is less work than expert-major, which reduces over
     * individual per-expert contributions.
     */
    NCCL_EP_LAYOUT_RANK_MAJOR,

    /**
     * Flat layout (HT mode only).
     *
     * HT dispatch output:
     *   recv_x shape:            [N(r) x hidden]
     *   recv_topk_weights shape: [N(r) x num_topk]
     *   recv_topk_idx shape:     [N(r) x num_topk]
     *
     * where N(r) is the total number of tokens targeting this rank across all
     * source ranks (num_ranks * max_dispatch_tokens_per_rank in the static case, or the
     * actual received count when max_dispatch_tokens_per_rank is NCCL_EP_AUTO).
     *
     * Tokens arrive as a single contiguous sequence with no rank-major or
     * expert-major structure.  The caller uses recv_topk_idx to route each
     * slot to the appropriate local expert(s) and recv_topk_weights to apply
     * the weighted reduction before passing pre-reduced outputs to
     * ncclEpCombine.
     *
     * This is the only layout supported by HT mode.
     */
    NCCL_EP_LAYOUT_FLAT,
} ncclEpLayout_t;

/**
 * Training pass direction for ncclEpDispatch / ncclEpCombine.
 *
 * Selects which side of the routing the call is participating in. FWD is the
 * zero-init default so callers that don't set the field get forward-pass
 * semantics.
 *
 *   FWD dispatch (HT): caller provides input topk_weights; routing is live.
 *   BWD dispatch (HT): no input topk_weights; reuses cached routing state.
 *   FWD combine  (HT): no input topk_weights; tokens only.
 *   BWD combine  (HT): caller provides input topk_weights and receives
 *                      combined topk_weights gradients alongside tokens.
 *
 * LL mode does not currently distinguish FWD/BWD and ignores this field.
 */
typedef enum {
    NCCL_EP_FWD_PASS = 0,
    NCCL_EP_BWD_PASS = 1,
} ncclEpPassDir_t;
