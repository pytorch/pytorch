/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
// Internal test-only helpers — NOT part of the installed public API.
// Include this header only from unit test sources (C++ only), never from library code.
#pragma once
#include "nccl_ep.h"
#include <stdint.h>

// Returns the device pointer to the unified sparse-to-dense map for a HT handle.
// Rank-major layout: int32_t[nNodes * max_tokens_per_rank][n_ranks_per_node]
//   s2d[token][dest] = recv slot at dest, or -1.
// Expert-major layout: int32_t[nNodes * max_tokens_per_rank][num_topk]
//   s2d[token][k] = packed(rank_id<<24 | slot), or -1.
const int32_t* ncclEpHandle_test_getSparseToDenseMap(ncclEpHandle_t handle);

// Returns num_topk stored in the handle.
int ncclEpHandle_test_getNumTopk(ncclEpHandle_t handle);

// Number of rows in the S2D: config.max_tokens_per_rank
int ncclEpHandle_test_getMaxTokensPerRank(ncclEpHandle_t handle);

// Number of ranks per NVLink node (== nRanks for single-node)
int ncclEpHandle_test_getNRanksPerNode(ncclEpHandle_t handle);

// Number of local experts per rank (num_experts / nRanks)
int ncclEpHandle_test_getExpertsPerRank(ncclEpHandle_t handle);

// HT only: total number of received tokens across all local experts.
// Reads handle->hybridep.num_tokens_for_experts via cudaMemcpy.
// Returns ncclInvalidUsage for non-HT algorithms.
ncclResult_t ncclEpHandle_test_getNumRecvTokens(ncclEpHandle_t handle, unsigned int* num_recv_tokens);

// Clear the handle's cached topk_idx pointer so the next ncclEpDispatch runs in
// scatter (backward, no-routing) mode.  HT only.
void ncclEpHandle_test_clearTopkIdx(ncclEpHandle_t handle);
