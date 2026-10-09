/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Shared test infrastructure for nccl_ep unit tests.
 * All declarations are static or inline — safe to include once per translation unit.
 *
 * Test parameters (4 ranks, 8 experts, 4 tokens, top-k 1):
 *   Rank 0 hosts E0 (local 0), E1 (local 1)   ← tokens from ranks 0, 2
 *   Rank 1 hosts E2 (local 0), E3 (local 1)   ← tokens from ranks 0, 2
 *   Rank 2 hosts E4 (local 0), E5 (local 1)   ← tokens from ranks 1, 3
 *   Rank 3 hosts E6 (local 0), E7 (local 1)   ← tokens from ranks 1, 3
 *   token i on rank r → expert (r * kNumTokens + i) % kNumExperts
 */
#pragma once

#include <gtest/gtest.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <nccl.h>
#include <nccl_ep.h>

#include <chrono>
#include <cstdio>
#include <string>
#include <thread>
#include <vector>

#define NCCL_ASSERT(x) ASSERT_EQ((x), ncclSuccess)
#define CUDA_ASSERT(x) ASSERT_EQ((x), cudaSuccess)

// Allocate a tensor descriptor and bind it to a caller-owned device buffer.
template<typename... Dims>
static inline ncclResult_t epTensorCreate(
    ncclEpTensor_t** t, unsigned int ndim, ncclDataType_t dtype, void* data, Dims... dims)
{
    const size_t sizes[] = {static_cast<size_t>(dims)...};
    ncclResult_t r = ncclEpTensorAlloc(t, ndim, dtype, sizes, nullptr);
    if (r != ncclSuccess) return r;
    (*t)->data = data;
    return ncclSuccess;
}

// ── Process-level state ───────────────────────────────────────────────────────

static int         g_rank    = -1;
static int         g_nranks  = -1;
static std::string g_uid_file;

static ncclComm_t    g_comm        = nullptr;
static ncclEpGroup_t g_ep_group    = nullptr;  // rank-major (default)
static ncclEpGroup_t g_ep_group_em = nullptr;  // expert-major
static cudaStream_t  g_stream      = nullptr;

// ── Test parameters ───────────────────────────────────────────────────────────

static constexpr int kNumTokens  = 4;
static constexpr int kNumExperts = 8;
static constexpr int kHidden     = 16;
static constexpr int kTopK       = 1;
// Per-rank recv-slot budget; sized for the worst-case test (EM + align=8 = 2 local experts * 8 slots).
static constexpr int kMaxRecvSlots = 16;

static int expert_for_token(int i) {
    return (g_rank * kNumTokens + i) % kNumExperts;
}

// ── UID exchange (rank-0 writes file, others poll) ────────────────────────────

static void exchange_uid(ncclUniqueId* uid) {
    const size_t sz = sizeof(ncclUniqueId);
    if (g_rank == 0) {
        ASSERT_EQ(ncclGetUniqueId(uid), ncclSuccess);
        FILE* f = fopen(g_uid_file.c_str(), "wb");
        ASSERT_NE(f, nullptr);
        fwrite(uid, 1, sz, f);
        fclose(f);
    } else {
        auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(60);
        while (true) {
            FILE* f = fopen(g_uid_file.c_str(), "rb");
            if (f) {
                fseek(f, 0, SEEK_END);
                if (static_cast<size_t>(ftell(f)) >= sz) {
                    fseek(f, 0, SEEK_SET);
                    fread(uid, 1, sz, f);
                    fclose(f);
                    break;
                }
                fclose(f);
            }
            ASSERT_LT(std::chrono::steady_clock::now(), deadline)
                << "Timed out waiting for UID file";
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
        }
    }
}

// ── Shared fixture base ───────────────────────────────────────────────────────

class EpTestBase : public ::testing::Test {
protected:
    ncclEpTensor_t* topk_idx_    = nullptr;
    ncclEpTensor_t* topk_idx_em_ = nullptr;  // for expert-major group
    int64_t*       d_topk_      = nullptr;

    void SetUp() override {
        CUDA_ASSERT(cudaMalloc(&d_topk_, kNumTokens * kTopK * sizeof(int64_t)));
        int64_t h[kNumTokens];
        for (int i = 0; i < kNumTokens; ++i) h[i] = expert_for_token(i);
        CUDA_ASSERT(cudaMemcpy(d_topk_, h, sizeof(h), cudaMemcpyHostToDevice));
        NCCL_ASSERT(epTensorCreate(&topk_idx_, 2, ncclInt64,
                                       d_topk_, kNumTokens, kTopK));
        NCCL_ASSERT(epTensorCreate(&topk_idx_em_, 2, ncclInt64,
                                       d_topk_, kNumTokens, kTopK));
    }

    void TearDown() override {
        if (topk_idx_em_) ncclEpTensorDestroy(topk_idx_em_);
        if (topk_idx_) ncclEpTensorDestroy(topk_idx_);
        if (d_topk_)   cudaFree(d_topk_);
    }

    ncclEpHandle_t make_handle(const ncclEpHandleConfig_t* cfg,
                                const ncclEpLayoutInfo_t* layout_info = nullptr) {
        ncclEpHandle_t h = nullptr;
        EXPECT_EQ(ncclEpCreateHandle(&h, g_ep_group, NCCL_EP_LAYOUT_FLAT,
                                     topk_idx_, layout_info, cfg, g_stream),
                  ncclSuccess);
        EXPECT_EQ(cudaStreamSynchronize(g_stream), cudaSuccess);
        return h;
    }

    // Create handle on the expert-major group.
    ncclEpHandle_t make_handle_em(const ncclEpHandleConfig_t* cfg,
                                   const ncclEpLayoutInfo_t* layout_info = nullptr) {
        ncclEpHandle_t h = nullptr;
        EXPECT_EQ(ncclEpCreateHandle(&h, g_ep_group_em, NCCL_EP_LAYOUT_EXPERT_MAJOR,
                                     topk_idx_em_, layout_info, cfg, g_stream),
                  ncclSuccess);
        EXPECT_EQ(cudaStreamSynchronize(g_stream), cudaSuccess);
        return h;
    }
};

// ── Bootstrap / teardown ──────────────────────────────────────────────────────

static void ep_parse_args(int argc, char* argv[], const char* uid_suffix) {
    for (int i = 1; i < argc; ++i) {
        std::string a(argv[i]);
        if      (a.rfind("--rank=",    0) == 0) g_rank   = std::stoi(a.substr(7));
        else if (a.rfind("--nranks=",  0) == 0) g_nranks = std::stoi(a.substr(9));
        else if (a.rfind("--uid-file=",0) == 0) g_uid_file = a.substr(11);
    }
    if (g_rank < 0 || g_nranks <= 0) {
        fprintf(stderr, "Usage: %s --rank=N --nranks=N [--uid-file=path] [gtest flags]\n",
                argc > 0 ? argv[0] : "test");
        exit(EXIT_FAILURE);
    }
    if (g_uid_file.empty()) {
        const char* t = getenv("TMPDIR"); if (!t) t = "/tmp";
        g_uid_file = std::string(t) + "/" + uid_suffix;
    }
}

// Returns false if the test binary should exit (wrong device / too few ranks).
static bool ep_bootstrap(int argc, char* argv[], const char* uid_suffix) {
    ep_parse_args(argc, argv, uid_suffix);
    ::testing::InitGoogleTest(&argc, argv);

    int device_count, device, major;
    cudaGetDeviceCount(&device_count);
    cudaSetDevice(g_rank % device_count);
    cudaGetDevice(&device);
    cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device);
    if (major < 9) {
        if (g_rank == 0) printf("SKIP: SM_90+ required (this device is SM_%d0)\n", major);
        return false;
    }
    if (g_nranks < 2) {
        if (g_rank == 0) printf("SKIP: at least 2 ranks required\n");
        return false;
    }

    ncclUniqueId uid{};
    exchange_uid(&uid);
    ncclResult_t comm_ret = ncclCommInitRank(&g_comm, g_nranks, uid, g_rank);
    if (comm_ret != ncclSuccess) {
        fprintf(stderr, "Rank %d: ncclCommInitRank failed (err=%d). "
                "NCCL bootstrap uses TCP between ranks on the same host; check "
                "that the loopback interface is reachable.\n", g_rank, comm_ret);
        return false;
    }
    cudaStreamCreate(&g_stream);

    ncclEpGroupConfig_t gcfg = NCCL_EP_GROUP_CONFIG_INIT;
    gcfg.algorithm                     = NCCL_EP_ALGO_HIGH_THROUGHPUT;
    gcfg.num_experts                   = kNumExperts;
    gcfg.max_dispatch_tokens_per_rank  = kNumTokens;
    gcfg.max_token_bytes               = kHidden * sizeof(nv_bfloat16);
    gcfg.rdma_buffer_size              = NCCL_EP_AUTO;
    gcfg.num_qp_per_rank               = NCCL_EP_AUTO;
    gcfg.num_channels                  = NCCL_EP_AUTO;
    gcfg.max_recv_tokens_per_rank      = static_cast<unsigned int>(kMaxRecvSlots);
    ncclResult_t grp_ret = ncclEpCreateGroup(&g_ep_group, g_comm, &gcfg);
    if (grp_ret != ncclSuccess) {
        fprintf(stderr, "Rank %d: ncclEpCreateGroup failed (err=%d).\n", g_rank, grp_ret);
        return false;
    }

    // Expert-major group (same config; layout is per-handle, not per-group)
    ncclEpGroupConfig_t gcfg_em = gcfg;
    grp_ret = ncclEpCreateGroup(&g_ep_group_em, g_comm, &gcfg_em);
    if (grp_ret != ncclSuccess) {
        fprintf(stderr, "Rank %d: ncclEpCreateGroup (expert-major) failed (err=%d).\n", g_rank, grp_ret);
        return false;
    }

    cudaStreamSynchronize(g_stream);
    return true;
}

static void ep_teardown() {
    ncclEpGroupDestroy(g_ep_group_em);
    ncclEpGroupDestroy(g_ep_group);
    cudaStreamDestroy(g_stream);
    ncclCommDestroy(g_comm);
    if (g_rank == 0) remove(g_uid_file.c_str());
}
