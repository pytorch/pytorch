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

#pragma once

// ============================================================================
// HT-specific configuration constants
// ============================================================================
#define HYBRIDEP_MAX_NUM_SMS_PER_RANK 16
// ============================================================================
// Dispatch configuration constants
// ============================================================================
#define HYBRIDEP_DISPATCH_NUM_OF_STAGES 12
#define HYBRIDEP_DISPATCH_NUM_OF_IN_FLIGHT_S2G 4
#define HYBRIDEP_DISPATCH_NUM_OF_BLOCKS HYBRIDEP_MAX_NUM_SMS_PER_RANK
#define HYBRIDEP_DISPATCH_NUM_OF_PIPELINES_PER_BLOCK 2
#define HYBRIDEP_DISPATCH_N2N_WARPS 2
// Maximum consecutive tokens batched into a single RDMA put in dispatch N2N.
// Larger batches reduce NIC doorbell overhead but may delay first-byte latency.
#define HYBRIDEP_DISPATCH_RDMA_BATCH_SIZE 4


// ============================================================================
// Combine configuration constants
// ============================================================================
// Single-node configuration: optimized for intra-node only (2 pipelines, deep FIFO)
#define HYBRIDEP_COMBINE_SINGLENODE_NUM_OF_STAGES_G2S 12
#define HYBRIDEP_COMBINE_SINGLENODE_NUM_OF_STAGES_S2G 2

// Multi-node configuration: optimized for inter-node RDMA (1 pipeline, shallow FIFO)
#define HYBRIDEP_COMBINE_MULTINODE_NUM_OF_STAGES_G2S 4
#define HYBRIDEP_COMBINE_MULTINODE_NUM_OF_STAGES_S2G 2

#define HYBRIDEP_COMBINE_NUM_OF_TOKENS_PER_GROUP 4
#define HYBRIDEP_COMBINE_NUM_OF_BLOCKS HYBRIDEP_MAX_NUM_SMS_PER_RANK
#define HYBRIDEP_COMBINE_NUM_OF_ADDITIONAL_IN_FLIGHT_S2G 2

// Streaming overlap: tokens between drain+signal from reduction warp to RDMA warp.
// 0 = disable streaming (fall back to chunk-level mbarrier only).
#define HYBRIDEP_COMBINE_RDMA_STREAMING_BATCH 8

// ============================================================================
// Preprocessing kernel configuration
// ============================================================================
#define HYBRIDEP_NUM_THREADS_PER_BLOCK_PREPROCESSING 512
#define HYBRIDEP_NUM_BLOCKS_PREPROCESSING HYBRIDEP_MAX_NUM_SMS_PER_RANK

// Max local experts per rank for expert-major remap kernel register arrays (runtime-asserted).
#define HYBRIDEP_MAX_LOCAL_EXPERTS_PER_RANK 64
