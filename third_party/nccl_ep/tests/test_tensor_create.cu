/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Tensor descriptor creation tests:
 *   StackAllocated     — stack-allocated ncclEpTensor_t via NCCL_EP_TENSOR_INIT
 *   HeapAllocated      — ncclEpTensorAlloc / ncclEpTensorDestroy
 *   Mixed              — both paths in a single dispatch+combine round
 *
 * All three paths must reach the same dispatch result.
 */

#include "test_common.h"

class TensorCreateTest : public EpTestBase {};

// Stack-allocated descriptor: NCCL_EP_TENSOR_INIT + caller-owned sizes[] array.
TEST_F(TensorCreateTest, StackAllocated) {
    nv_bfloat16 *d_tok, *d_recv, *d_out;
    float       *d_w, *d_recv_w;
    int64_t     *d_recv_idx;
    CUDA_ASSERT(cudaMalloc(&d_tok,      kNumTokens    * kHidden * sizeof(nv_bfloat16)));
    CUDA_ASSERT(cudaMalloc(&d_recv,     kMaxRecvSlots * kHidden * sizeof(nv_bfloat16)));
    CUDA_ASSERT(cudaMalloc(&d_out,      kNumTokens    * kHidden * sizeof(nv_bfloat16)));
    CUDA_ASSERT(cudaMalloc(&d_w,        kNumTokens    * kTopK   * sizeof(float)));
    CUDA_ASSERT(cudaMalloc(&d_recv_w,   kMaxRecvSlots * kTopK   * sizeof(float)));
    CUDA_ASSERT(cudaMalloc(&d_recv_idx, kMaxRecvSlots * kTopK   * sizeof(int64_t)));
    CUDA_ASSERT(cudaMemset(d_recv, 0,   kMaxRecvSlots * kHidden * sizeof(nv_bfloat16)));

    std::vector<nv_bfloat16> h_tok(kNumTokens * kHidden);
    for (int i = 0; i < kNumTokens; ++i) {
        float v = static_cast<float>(g_rank * kNumTokens + i + 1);
        for (int hh = 0; hh < kHidden; ++hh)
            h_tok[i * kHidden + hh] = __float2bfloat16(v);
    }
    std::vector<float> h_w(kNumTokens * kTopK, 1.0f);
    CUDA_ASSERT(cudaMemcpy(d_tok, h_tok.data(), h_tok.size() * sizeof(nv_bfloat16), cudaMemcpyHostToDevice));
    CUDA_ASSERT(cudaMemcpy(d_w,   h_w.data(),   h_w.size()   * sizeof(float),       cudaMemcpyHostToDevice));

    // Stack-allocated descriptors. Caller owns the sizes[] arrays.
    size_t tok_sz[2]     = { kNumTokens, kHidden };
    size_t recv_sz[2]    = { kMaxRecvSlots, kHidden };
    size_t out_sz[2]     = { kNumTokens, kHidden };
    size_t w_sz[2]       = { kNumTokens, kTopK };
    size_t recv_w_sz[2]  = { kMaxRecvSlots, kTopK };
    size_t recv_ix_sz[2] = { kMaxRecvSlots, kTopK };

    ncclEpTensor_t t_tok = NCCL_EP_TENSOR_INIT;
    t_tok.ndim = 2; t_tok.datatype = ncclBfloat16; t_tok.data = d_tok; t_tok.sizes = tok_sz;

    // Compound-literal form (NCCL_EP_TENSOR_INIT_INLINE).
    ncclEpTensor_t t_recv = { NCCL_EP_TENSOR_INIT_INLINE,
                              .ndim = 2, .datatype = ncclBfloat16,
                              .data = d_recv, .sizes = recv_sz };

    ncclEpTensor_t t_out = NCCL_EP_TENSOR_INIT;
    t_out.ndim = 2; t_out.datatype = ncclBfloat16; t_out.data = d_out; t_out.sizes = out_sz;

    ncclEpTensor_t t_w = NCCL_EP_TENSOR_INIT;
    t_w.ndim = 2; t_w.datatype = ncclFloat32; t_w.data = d_w; t_w.sizes = w_sz;

    ncclEpTensor_t t_recv_w = NCCL_EP_TENSOR_INIT;
    t_recv_w.ndim = 2; t_recv_w.datatype = ncclFloat32; t_recv_w.data = d_recv_w; t_recv_w.sizes = recv_w_sz;

    ncclEpTensor_t t_recv_idx = NCCL_EP_TENSOR_INIT;
    t_recv_idx.ndim = 2; t_recv_idx.datatype = ncclInt64; t_recv_idx.data = d_recv_idx; t_recv_idx.sizes = recv_ix_sz;

    EXPECT_EQ(t_tok.magic, NCCL_EP_TENSOR_MAGIC);
    EXPECT_EQ(t_tok.size, sizeof(ncclEpTensor_t));

    ncclEpHandle_t h = make_handle(nullptr);
    ASSERT_NE(h, nullptr);

    ncclEpDispatchInputs_t  d_in  = NCCL_EP_DISPATCH_INPUTS_INIT;
    ncclEpDispatchOutputs_t d_outs = NCCL_EP_DISPATCH_OUTPUTS_INIT;
    d_in.tokens         = &t_tok;
    d_in.topk_weights   = &t_w;
    d_outs.tokens       = &t_recv;
    d_outs.topk_weights = &t_recv_w;
    d_outs.topk_idx     = &t_recv_idx;
    ncclEpDispatchConfig_t dcfg = NCCL_EP_DISPATCH_CONFIG_INIT;
    NCCL_ASSERT(ncclEpDispatch(h, &d_in, &d_outs, nullptr, &dcfg, g_stream));
    NCCL_ASSERT(ncclEpComplete(h, nullptr, g_stream));

    ncclEpCombineInputs_t  c_in  = NCCL_EP_COMBINE_INPUTS_INIT;
    ncclEpCombineOutputs_t c_out = NCCL_EP_COMBINE_OUTPUTS_INIT;
    c_in.tokens  = &t_recv;
    c_out.tokens = &t_out;
    NCCL_ASSERT(ncclEpCombine(h, &c_in, &c_out, nullptr, g_stream));
    CUDA_ASSERT(cudaStreamSynchronize(g_stream));

    std::vector<nv_bfloat16> h_out(kNumTokens * kHidden);
    CUDA_ASSERT(cudaMemcpy(h_out.data(), d_out, h_out.size() * sizeof(nv_bfloat16), cudaMemcpyDeviceToHost));
    for (int i = 0; i < kNumTokens; ++i) {
        float expected = static_cast<float>(g_rank * kNumTokens + i + 1);
        EXPECT_NEAR(__bfloat162float(h_out[i * kHidden]), expected, 0.5f)
            << "rank " << g_rank << " token " << i;
    }

    NCCL_ASSERT(ncclEpHandleDestroy(h));
    cudaFree(d_tok); cudaFree(d_recv); cudaFree(d_out);
    cudaFree(d_w);   cudaFree(d_recv_w); cudaFree(d_recv_idx);
}

// Heap-allocated descriptors via ncclEpTensorAlloc / ncclEpTensorDestroy.
TEST_F(TensorCreateTest, HeapAllocated) {
    nv_bfloat16 *d_tok, *d_recv, *d_out;
    float       *d_w, *d_recv_w;
    int64_t     *d_recv_idx;
    CUDA_ASSERT(cudaMalloc(&d_tok,      kNumTokens    * kHidden * sizeof(nv_bfloat16)));
    CUDA_ASSERT(cudaMalloc(&d_recv,     kMaxRecvSlots * kHidden * sizeof(nv_bfloat16)));
    CUDA_ASSERT(cudaMalloc(&d_out,      kNumTokens    * kHidden * sizeof(nv_bfloat16)));
    CUDA_ASSERT(cudaMalloc(&d_w,        kNumTokens    * kTopK   * sizeof(float)));
    CUDA_ASSERT(cudaMalloc(&d_recv_w,   kMaxRecvSlots * kTopK   * sizeof(float)));
    CUDA_ASSERT(cudaMalloc(&d_recv_idx, kMaxRecvSlots * kTopK   * sizeof(int64_t)));
    CUDA_ASSERT(cudaMemset(d_recv, 0,   kMaxRecvSlots * kHidden * sizeof(nv_bfloat16)));

    std::vector<nv_bfloat16> h_tok(kNumTokens * kHidden);
    for (int i = 0; i < kNumTokens; ++i) {
        float v = static_cast<float>(g_rank * kNumTokens + i + 1);
        for (int hh = 0; hh < kHidden; ++hh)
            h_tok[i * kHidden + hh] = __float2bfloat16(v);
    }
    std::vector<float> h_w(kNumTokens * kTopK, 1.0f);
    CUDA_ASSERT(cudaMemcpy(d_tok, h_tok.data(), h_tok.size() * sizeof(nv_bfloat16), cudaMemcpyHostToDevice));
    CUDA_ASSERT(cudaMemcpy(d_w,   h_w.data(),   h_w.size()   * sizeof(float),       cudaMemcpyHostToDevice));

    auto alloc2d = [](ncclEpTensor_t** t, ncclDataType_t dt, void* data, size_t a, size_t b) {
        const size_t sizes[2] = { a, b };
        NCCL_ASSERT(ncclEpTensorAlloc(t, 2, dt, sizes, nullptr));
        (*t)->data = data;
    };

    ncclEpTensor_t *t_tok = nullptr, *t_recv = nullptr, *t_out = nullptr;
    ncclEpTensor_t *t_w = nullptr, *t_recv_w = nullptr, *t_recv_idx = nullptr;
    alloc2d(&t_tok,      ncclBfloat16, d_tok,      kNumTokens,    kHidden);
    alloc2d(&t_recv,     ncclBfloat16, d_recv,     kMaxRecvSlots, kHidden);
    alloc2d(&t_out,      ncclBfloat16, d_out,      kNumTokens,    kHidden);
    alloc2d(&t_w,        ncclFloat32,  d_w,        kNumTokens,    kTopK);
    alloc2d(&t_recv_w,   ncclFloat32,  d_recv_w,   kMaxRecvSlots, kTopK);
    alloc2d(&t_recv_idx, ncclInt64,    d_recv_idx, kMaxRecvSlots, kTopK);

    EXPECT_EQ(t_tok->ndim, 2u);
    EXPECT_EQ(t_tok->datatype, ncclBfloat16);
    EXPECT_EQ(t_tok->data, d_tok);
    EXPECT_EQ(t_tok->sizes[0], (size_t)kNumTokens);
    EXPECT_EQ(t_tok->sizes[1], (size_t)kHidden);

    ncclEpHandle_t h = make_handle(nullptr);
    ASSERT_NE(h, nullptr);

    ncclEpDispatchInputs_t  d_in  = NCCL_EP_DISPATCH_INPUTS_INIT;
    ncclEpDispatchOutputs_t d_outs = NCCL_EP_DISPATCH_OUTPUTS_INIT;
    d_in.tokens         = t_tok;
    d_in.topk_weights   = t_w;
    d_outs.tokens       = t_recv;
    d_outs.topk_weights = t_recv_w;
    d_outs.topk_idx     = t_recv_idx;
    ncclEpDispatchConfig_t dcfg = NCCL_EP_DISPATCH_CONFIG_INIT;
    NCCL_ASSERT(ncclEpDispatch(h, &d_in, &d_outs, nullptr, &dcfg, g_stream));
    NCCL_ASSERT(ncclEpComplete(h, nullptr, g_stream));

    ncclEpCombineInputs_t  c_in  = NCCL_EP_COMBINE_INPUTS_INIT;
    ncclEpCombineOutputs_t c_out = NCCL_EP_COMBINE_OUTPUTS_INIT;
    c_in.tokens  = t_recv;
    c_out.tokens = t_out;
    NCCL_ASSERT(ncclEpCombine(h, &c_in, &c_out, nullptr, g_stream));
    CUDA_ASSERT(cudaStreamSynchronize(g_stream));

    std::vector<nv_bfloat16> h_out(kNumTokens * kHidden);
    CUDA_ASSERT(cudaMemcpy(h_out.data(), d_out, h_out.size() * sizeof(nv_bfloat16), cudaMemcpyDeviceToHost));
    for (int i = 0; i < kNumTokens; ++i) {
        float expected = static_cast<float>(g_rank * kNumTokens + i + 1);
        EXPECT_NEAR(__bfloat162float(h_out[i * kHidden]), expected, 0.5f)
            << "rank " << g_rank << " token " << i;
    }

    NCCL_ASSERT(ncclEpHandleDestroy(h));
    ncclEpTensorDestroy(t_tok); ncclEpTensorDestroy(t_recv); ncclEpTensorDestroy(t_out);
    ncclEpTensorDestroy(t_w);   ncclEpTensorDestroy(t_recv_w); ncclEpTensorDestroy(t_recv_idx);
    cudaFree(d_tok); cudaFree(d_recv); cudaFree(d_out);
    cudaFree(d_w);   cudaFree(d_recv_w); cudaFree(d_recv_idx);
}

// Mixed: inputs use heap allocation, outputs use stack allocation.
TEST_F(TensorCreateTest, Mixed) {
    nv_bfloat16 *d_tok, *d_recv, *d_out;
    float       *d_w, *d_recv_w;
    int64_t     *d_recv_idx;
    CUDA_ASSERT(cudaMalloc(&d_tok,      kNumTokens    * kHidden * sizeof(nv_bfloat16)));
    CUDA_ASSERT(cudaMalloc(&d_recv,     kMaxRecvSlots * kHidden * sizeof(nv_bfloat16)));
    CUDA_ASSERT(cudaMalloc(&d_out,      kNumTokens    * kHidden * sizeof(nv_bfloat16)));
    CUDA_ASSERT(cudaMalloc(&d_w,        kNumTokens    * kTopK   * sizeof(float)));
    CUDA_ASSERT(cudaMalloc(&d_recv_w,   kMaxRecvSlots * kTopK   * sizeof(float)));
    CUDA_ASSERT(cudaMalloc(&d_recv_idx, kMaxRecvSlots * kTopK   * sizeof(int64_t)));
    CUDA_ASSERT(cudaMemset(d_recv, 0,   kMaxRecvSlots * kHidden * sizeof(nv_bfloat16)));

    std::vector<nv_bfloat16> h_tok(kNumTokens * kHidden);
    for (int i = 0; i < kNumTokens; ++i) {
        float v = static_cast<float>(g_rank * kNumTokens + i + 1);
        for (int hh = 0; hh < kHidden; ++hh)
            h_tok[i * kHidden + hh] = __float2bfloat16(v);
    }
    std::vector<float> h_w(kNumTokens * kTopK, 1.0f);
    CUDA_ASSERT(cudaMemcpy(d_tok, h_tok.data(), h_tok.size() * sizeof(nv_bfloat16), cudaMemcpyHostToDevice));
    CUDA_ASSERT(cudaMemcpy(d_w,   h_w.data(),   h_w.size()   * sizeof(float),       cudaMemcpyHostToDevice));

    // Heap inputs.
    auto alloc2d = [](ncclEpTensor_t** t, ncclDataType_t dt, void* data, size_t a, size_t b) {
        const size_t sizes[2] = { a, b };
        NCCL_ASSERT(ncclEpTensorAlloc(t, 2, dt, sizes, nullptr));
        (*t)->data = data;
    };
    ncclEpTensor_t *t_tok = nullptr, *t_w = nullptr;
    alloc2d(&t_tok, ncclBfloat16, d_tok, kNumTokens, kHidden);
    alloc2d(&t_w,   ncclFloat32,  d_w,   kNumTokens, kTopK);

    // Stack outputs.
    size_t recv_sz[2]    = { kMaxRecvSlots, kHidden };
    size_t out_sz[2]     = { kNumTokens, kHidden };
    size_t recv_w_sz[2]  = { kMaxRecvSlots, kTopK };
    size_t recv_ix_sz[2] = { kMaxRecvSlots, kTopK };
    ncclEpTensor_t t_recv     = NCCL_EP_TENSOR_INIT;
    ncclEpTensor_t t_out      = NCCL_EP_TENSOR_INIT;
    ncclEpTensor_t t_recv_w   = NCCL_EP_TENSOR_INIT;
    ncclEpTensor_t t_recv_idx = NCCL_EP_TENSOR_INIT;
    t_recv.ndim = 2;     t_recv.datatype = ncclBfloat16;    t_recv.data = d_recv;       t_recv.sizes = recv_sz;
    t_out.ndim = 2;      t_out.datatype = ncclBfloat16;     t_out.data = d_out;         t_out.sizes = out_sz;
    t_recv_w.ndim = 2;   t_recv_w.datatype = ncclFloat32;   t_recv_w.data = d_recv_w;   t_recv_w.sizes = recv_w_sz;
    t_recv_idx.ndim = 2; t_recv_idx.datatype = ncclInt64;   t_recv_idx.data = d_recv_idx; t_recv_idx.sizes = recv_ix_sz;

    ncclEpHandle_t h = make_handle(nullptr);
    ASSERT_NE(h, nullptr);

    ncclEpDispatchInputs_t  d_in  = NCCL_EP_DISPATCH_INPUTS_INIT;
    ncclEpDispatchOutputs_t d_outs = NCCL_EP_DISPATCH_OUTPUTS_INIT;
    d_in.tokens         = t_tok;
    d_in.topk_weights   = t_w;
    d_outs.tokens       = &t_recv;
    d_outs.topk_weights = &t_recv_w;
    d_outs.topk_idx     = &t_recv_idx;
    ncclEpDispatchConfig_t dcfg = NCCL_EP_DISPATCH_CONFIG_INIT;
    NCCL_ASSERT(ncclEpDispatch(h, &d_in, &d_outs, nullptr, &dcfg, g_stream));
    NCCL_ASSERT(ncclEpComplete(h, nullptr, g_stream));

    ncclEpCombineInputs_t  c_in  = NCCL_EP_COMBINE_INPUTS_INIT;
    ncclEpCombineOutputs_t c_out = NCCL_EP_COMBINE_OUTPUTS_INIT;
    c_in.tokens  = &t_recv;
    c_out.tokens = &t_out;
    NCCL_ASSERT(ncclEpCombine(h, &c_in, &c_out, nullptr, g_stream));
    CUDA_ASSERT(cudaStreamSynchronize(g_stream));

    std::vector<nv_bfloat16> h_out(kNumTokens * kHidden);
    CUDA_ASSERT(cudaMemcpy(h_out.data(), d_out, h_out.size() * sizeof(nv_bfloat16), cudaMemcpyDeviceToHost));
    for (int i = 0; i < kNumTokens; ++i) {
        float expected = static_cast<float>(g_rank * kNumTokens + i + 1);
        EXPECT_NEAR(__bfloat162float(h_out[i * kHidden]), expected, 0.5f)
            << "rank " << g_rank << " token " << i;
    }

    NCCL_ASSERT(ncclEpHandleDestroy(h));
    ncclEpTensorDestroy(t_tok); ncclEpTensorDestroy(t_w);
    cudaFree(d_tok); cudaFree(d_recv); cudaFree(d_out);
    cudaFree(d_w);   cudaFree(d_recv_w); cudaFree(d_recv_idx);
}

int main(int argc, char* argv[]) {
    if (!ep_bootstrap(argc, argv, "te_ep_tensor_create_uid")) return 0;
    int ret = RUN_ALL_TESTS();
    ep_teardown();
    return ret;
}
