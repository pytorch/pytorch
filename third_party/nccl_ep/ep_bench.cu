/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 * See LICENSE.txt for more license information.
 */
// Throughput and validation methodology aligned with DeepEP (https://github.com/deepseek-ai/DeepEP).

#include <getopt.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <algorithm>
#include <cmath>
#include <functional>
#include <map>
#include <numeric>
#include <random>
#include <set>
#include <string>
#include <vector>
#include <mpi.h>
#include <cuda_runtime.h>
#include <cuda_profiler_api.h>
#ifdef HAVE_CUPTI
#include <cupti.h>
#endif
#include <nvtx3/nvToolsExt.h>
#include <nccl.h>
#include <nccl_device.h>
#include "nccl_ep.h"


#define MPICHECK(cmd) do {                          \
  int e = cmd;                                      \
  if( e != MPI_SUCCESS ) {                          \
    printf("Failed: MPI error %s:%d '%d'\n",        \
        __FILE__,__LINE__, e);   \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)

#define CUDACHECK(cmd) do {                         \
  cudaError_t e = cmd;                              \
  if( e != cudaSuccess ) {                          \
    printf("Failed: Cuda error %s:%d '%s'\n",             \
        __FILE__,__LINE__,cudaGetErrorString(e));   \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)

#define NCCLCHECK(cmd) do {                         \
  ncclResult_t r = cmd;                             \
  if (r!= ncclSuccess) {                            \
    printf("Failed: NCCL error %s:%d '%s'\n",             \
        __FILE__,__LINE__,ncclGetErrorString(r));   \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)

// ============================================================================
// KernelTimer: per-kernel GPU timing (requires CUPTI)
// ============================================================================
// When HAVE_CUPTI is defined, uses the CUPTI Activity API to record per-kernel
// GPU execution times by matching kernel name substrings.  Entirely
// benchmark-side — zero impact on the production nccl_ep library.
//
// Without CUPTI, a no-op stub is provided so ep_bench still compiles and runs;
// kernel-level timing simply reports 0.

#ifdef HAVE_CUPTI

#define CUPTI_CALL(call) do {                                                  \
    CUptiResult _s = (call);                                                   \
    if (_s != CUPTI_SUCCESS) {                                                 \
        const char* _e; cuptiGetResultString(_s, &_e);                        \
        fprintf(stderr, "CUPTI error %s:%d: %s\n", __FILE__, __LINE__, _e);   \
    }                                                                          \
} while (0)

static const size_t CUPTI_BUF_SIZE = 8 * 1024 * 1024;  // 8 MB per buffer

struct KernelStat { uint64_t total_ns = 0; int count = 0; };
// Global accumulator populated by CUPTI buffer-completed callback
static std::map<std::string, KernelStat> g_kernel_stats;

static void CUPTIAPI cuptiBufferRequested(uint8_t** buf, size_t* sz, size_t* maxRecords) {
    // aligned_alloc requires size to be a multiple of alignment
    *buf = static_cast<uint8_t*>(aligned_alloc(8, CUPTI_BUF_SIZE));
    *sz = CUPTI_BUF_SIZE;
    *maxRecords = 0;
}

static void CUPTIAPI cuptiBufferCompleted(CUcontext /*ctx*/, uint32_t /*streamId*/,
                                           uint8_t* buf, size_t /*sz*/, size_t validSz) {
    CUpti_Activity* record = nullptr;
    while (cuptiActivityGetNextRecord(buf, validSz, &record) == CUPTI_SUCCESS) {
        if (record->kind == CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL) {
            auto* k = reinterpret_cast<CUpti_ActivityKernel5*>(record);
            if (k->name) {
                g_kernel_stats[k->name].total_ns += k->end - k->start;
                g_kernel_stats[k->name].count++;
            }
        }
    }
    free(buf);
}

class KernelTimer {
public:

    KernelTimer() {
        CUPTI_CALL(cuptiActivityFlushAll(0));
    }
    // Enable CUPTI kernel activity recording and clear accumulated stats.
    void start() {
        g_kernel_stats.clear();
        CUPTI_CALL(cuptiActivityRegisterCallbacks(cuptiBufferRequested, cuptiBufferCompleted));
        CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL));
    }

    // Flush all pending CUPTI buffers and disable recording.
    void stop() {
        CUPTI_CALL(cuptiActivityFlushAll(0));
        CUPTI_CALL(cuptiActivityDisable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL));
    }

    // Average GPU execution time (microseconds) across all kernels whose
    // mangled name contains substr.  Returns 0 if no matching kernel found.
    double get_avg_us(const char* substr) const {
        uint64_t total_ns = 0; int count = 0;
        for (const auto& kv : g_kernel_stats) {
            if (kv.first.find(substr) != std::string::npos) {
                total_ns += kv.second.total_ns;
                count    += kv.second.count;
            }
        }
        return count ? static_cast<double>(total_ns) / count / 1000.0 : 0.0;
    }

    // Print all captured kernel names and their stats to stdout (debug helper).
    void dump(int rank) const {
        if (rank != 0) return;
        printf("[KernelTimer] Captured %zu distinct kernel(s):\n", g_kernel_stats.size());
        for (const auto& kv : g_kernel_stats) {
            double avg_us = static_cast<double>(kv.second.total_ns) / kv.second.count / 1000.0;
            printf("  count=%3d  avg=%.2f us  %s\n", kv.second.count, avg_us, kv.first.c_str());
        }
        fflush(stdout);
    }

    inline bool is_valid() { return true;}
};

#else // !HAVE_CUPTI

class KernelTimer {
public:
    void start() {}
    void stop() {}
    double get_avg_us(const char*) const { return 0.0; }
    void dump(int) const {}
    inline bool is_valid() { return false;}
};

#endif // HAVE_CUPTI

static uint64_t getHostHash(const char* string) {
  uint64_t result = 5381;
  for (int c = 0; string[c] != '\0'; c++){
    result = ((result << 5) + result) + string[c];
  }
  return result;
}

static void getHostName(char* hostname, int maxlen) {
  gethostname(hostname, maxlen);
  for (int i=0; i< maxlen; i++) {
    if (hostname[i] == '.') {
        hostname[i] = '\0';
        return;
    }
  }
}

// CUDA allocator callbacks for ncclEpCreateGroup
static cudaError_t cudaAllocCallback(void** ptr, size_t size, void* /*context*/) {
    return cudaMalloc(ptr, size);
}

static cudaError_t cudaFreeCallback(void* ptr, void* /*context*/) {
    return cudaFree(ptr);
}

// Element size for the dtypes used in this benchmark. ncclTypeSize is internal to the EP library.
static size_t epDtypeBytes(ncclDataType_t dt) {
    switch (dt) {
        case ncclInt8:    case ncclUint8:                              return 1;
        case ncclFloat16: case ncclBfloat16:                           return 2;
        case ncclFloat32: case ncclInt32: case ncclUint32:             return 4;
        case ncclInt64:   case ncclUint64: case ncclFloat64:           return 8;
        default: return 0;
    }
}

struct RegisteredWindowEntry {
    ncclComm_t comm;
    ncclWindow_t win;
};

struct EpTensorAllocOptions {
    bool use_nccl_mem = false;
    bool use_window = false;
    ncclComm_t window_comm = nullptr;
    std::vector<RegisteredWindowEntry>* registered_windows = nullptr;
    std::vector<void*>* nccl_mem_ptrs = nullptr;
    std::map<ncclEpTensor_t*, void*>* tensor_data_ptrs = nullptr;
};

// Allocate storage and create an EP tensor via ncclEpTensorAlloc, then
// bind it to either the freshly-allocated device buffer or an NCCL window.
// Optionally uses ncclMemAlloc with a registered window for the benchmark's
// HT zero-copy path. The returned descriptor lives on the heap and must be
// released via epFreeTensor (which calls ncclEpTensorDestroy).
static ncclResult_t epMakeTensor(ncclEpTensor_t** out_tensor, unsigned int ndim,
                                 ncclDataType_t dt,
                                 unsigned int s0, unsigned int s1 = 1, unsigned int s2 = 1,
                                 unsigned int s3 = 1, unsigned int s4 = 1,
                                 const EpTensorAllocOptions* opts = nullptr) {
    if (out_tensor == nullptr) return ncclInvalidArgument;
    *out_tensor = nullptr;

    size_t dims[5] = {s0, s1, s2, s3, s4};
    size_t total = 1;
    for (unsigned int i = 0; i < ndim; i++) total *= dims[i];
    const size_t bytes = total * epDtypeBytes(dt);

    void* data = nullptr;
    const bool use_nccl_mem = opts != nullptr && opts->use_nccl_mem;
    if (use_nccl_mem) {
        ncclResult_t r = ncclMemAlloc(&data, bytes);
        if (r != ncclSuccess) return r;
        if (opts->nccl_mem_ptrs) opts->nccl_mem_ptrs->push_back(data);
    } else {
        cudaError_t e = cudaMalloc(&data, bytes);
        if (e != cudaSuccess) return ncclSystemError;
    }

    auto free_data = [&]() {
        if (data == nullptr) return;
        if (use_nccl_mem) {
            if (opts->nccl_mem_ptrs) {
                auto it = std::find(opts->nccl_mem_ptrs->begin(), opts->nccl_mem_ptrs->end(), data);
                if (it != opts->nccl_mem_ptrs->end()) opts->nccl_mem_ptrs->erase(it);
            }
            ncclMemFree(data);
        } else {
            cudaFree(data);
        }
        data = nullptr;
    };

    const bool use_win = opts != nullptr && opts->use_window;
    ncclWindow_t win{};
    if (use_win) {
        if (opts->window_comm == nullptr || opts->registered_windows == nullptr) {
            free_data();
            return ncclInvalidArgument;
        }
        ncclResult_t r = ncclCommWindowRegister(opts->window_comm, data, bytes, &win, NCCL_WIN_COLL_SYMMETRIC);
        if (r != ncclSuccess) {
            free_data();
            return r;
        }
    }

    ncclEpTensor_t* t = nullptr;
    ncclResult_t r = ncclEpTensorAlloc(&t, ndim, dt, dims, /*config=*/nullptr);
    if (r != ncclSuccess) {
        if (use_win) ncclCommWindowDeregister(opts->window_comm, win);
        free_data();
        return r;
    }

    if (use_win) {
        // Window-backed tensor: leave data unset; the EP library resolves the
        // device pointer via win_hdl/win_offset. The raw buffer is remembered
        // in tensor_data_ptrs so epGetTensorData can still hand the benchmark
        // a usable address.
        t->win_hdl = win;
        t->win_offset = 0;
        opts->registered_windows->push_back({opts->window_comm, win});
    } else {
        t->data = data;
    }
    if (opts != nullptr && opts->tensor_data_ptrs) (*opts->tensor_data_ptrs)[t] = data;

    *out_tensor = t;
    return ncclSuccess;
}

// Inverse of epMakeTensor: free the backing buffer and release the descriptor
// via ncclEpTensorDestroy. Sets *field to nullptr.
static void epFreeTensor(ncclEpTensor_t** field,
                         std::vector<void*>* nccl_mem_ptrs = nullptr,
                         std::map<ncclEpTensor_t*, void*>* tensor_data_ptrs = nullptr) {
    if (field == nullptr || *field == nullptr) return;
    ncclEpTensor_t* tensor = *field;

    void* data = tensor->data;
    if (data == nullptr && tensor_data_ptrs != nullptr) {
        auto data_it = tensor_data_ptrs->find(tensor);
        if (data_it != tensor_data_ptrs->end()) {
            data = data_it->second;
            tensor_data_ptrs->erase(data_it);
        }
    } else if (tensor_data_ptrs != nullptr) {
        tensor_data_ptrs->erase(tensor);
    }

    ncclEpTensorDestroy(tensor);
    *field = nullptr;

    if (data == nullptr) return;

    if (nccl_mem_ptrs != nullptr) {
        auto it = std::find(nccl_mem_ptrs->begin(), nccl_mem_ptrs->end(), data);
        if (it != nccl_mem_ptrs->end()) {
            NCCLCHECK(ncclMemFree(data));
            nccl_mem_ptrs->erase(it);
            return;
        }
    }
    cudaFree(data);
}

// Bookkeeping for tensors that were allocated via the zero-copy path
// (ncclMemAlloc + window registration). Lives at the top of main() and
// is threaded through setup / cleanup helpers so they can record and
// release the bound memory.
struct BenchmarkAllocState {
    std::vector<RegisteredWindowEntry> registered_windows;
    std::vector<void*> external_data_ptrs;
    std::map<ncclEpTensor_t*, void*> tensor_data_ptrs;
};

static ncclResult_t epGetTensorData(const BenchmarkAllocState& alloc, const ncclEpTensor_t* tensor, void** data) {
    if (data == nullptr || tensor == nullptr) return ncclInvalidArgument;
    if (tensor->data != nullptr) {
        *data = tensor->data;
        return ncclSuccess;
    }
    auto it = alloc.tensor_data_ptrs.find(const_cast<ncclEpTensor_t*>(tensor));
    if (it != alloc.tensor_data_ptrs.end()) {
        *data = it->second;
        return ncclSuccess;
    }
    // Empty tensor (zero-extent in some dimension) is allowed to carry a
    // null data pointer -- a zero-byte buffer has no element to address.
    // Hand back nullptr so callers can pass it to cudaMemset/cudaMemcpy
    // with count=0 (both are no-ops on a NULL pointer when count is 0).
    for (unsigned int i = 0; i < tensor->ndim; ++i) {
        if (tensor->sizes != nullptr && tensor->sizes[i] == 0) {
            *data = nullptr;
            return ncclSuccess;
        }
    }
    return ncclInvalidUsage;
}

// LL benchmark — layout-independent dispatch inputs.
//
// Initializes:
//   dispatch_inputs.tokens                          [num_tokens, hidden]
//   topk_weights                                    [num_tokens, top_k]
//                                                    (LL combine reads via outputs.topk_weights;
//                                                     rank-major also aliases to dispatch_inputs.topk_weights)
//   dispatch_layout_info.expert_counters            [num_local_experts] (LL expert-major only)
static void setupLowLatencyTensorsSharedInputs(
    ncclEpDispatchInputs_t& dispatch_inputs,
    ncclEpLayoutInfo_t&     dispatch_layout_info,
    bool&                   has_dispatch_layout_info,
    ncclEpTensor_t*&        topk_weights,
    unsigned int num_tokens,
    unsigned int hidden,
    unsigned int top_k,
    unsigned int num_local_experts
) {
    NCCLCHECK(epMakeTensor(&dispatch_inputs.tokens, 2, ncclBfloat16, num_tokens, hidden));

    NCCLCHECK(epMakeTensor(&topk_weights, 2, ncclFloat32, num_tokens, top_k));

    NCCLCHECK(epMakeTensor(&dispatch_layout_info.expert_counters, 1, ncclInt32, num_local_experts));
    has_dispatch_layout_info = true;
}

// LL benchmark — NCCL_EP_LAYOUT_EXPERT_MAJOR dispatch outputs + combine input shape.
static void setupLowLatencyTensorsExpertMajLayout(
    ncclEpDispatchOutputs_t& dispatch_outputs,
    ncclEpCombineInputs_t&   combine_inputs,
    unsigned int hidden,
    unsigned int num_local_experts,
    unsigned int max_dispatch_tokens_per_rank,
    int nRanks
) {
    // expert_counters already populated by setupLowLatencyTensorsSharedInputs.

    NCCLCHECK(epMakeTensor(&dispatch_outputs.tokens, 3, ncclBfloat16, num_local_experts,
                           (unsigned)nRanks * max_dispatch_tokens_per_rank, hidden));

    NCCLCHECK(epMakeTensor(&combine_inputs.tokens, 3, ncclBfloat16, num_local_experts,
                           (unsigned)nRanks * max_dispatch_tokens_per_rank, hidden));
}

// LL benchmark — NCCL_EP_LAYOUT_RANK_MAJOR dispatch outputs + combine input shape.
//
// Dispatch sends topk_weights so the receiving rank knows routing metadata.
// Combine receives pre-reduced expert outputs (application applies weights before combine).
static void setupLowLatencyTensorsRankMajLayout(
    ncclEpDispatchInputs_t&  dispatch_inputs,
    ncclEpDispatchOutputs_t& dispatch_outputs,
    ncclEpLayoutInfo_t&      dispatch_layout_info,
    ncclEpCombineInputs_t&   combine_inputs,
    ncclEpTensor_t*          topk_weights,
    unsigned int hidden,
    unsigned int top_k,
    unsigned int num_local_experts,
    unsigned int max_dispatch_tokens_per_rank,
    int nRanks
) {
    // Rank-major uses per-source-rank counter, not the per-expert counter that
    // setupLowLatencyTensorsSharedInputs created. Swap expert_counters out for
    // src_rank_counters on the layout_info.
    epFreeTensor(&dispatch_layout_info.expert_counters);

    dispatch_inputs.topk_weights = topk_weights;  // alias

    NCCLCHECK(epMakeTensor(&dispatch_layout_info.src_rank_counters, 1, ncclInt32, (unsigned)nRanks));

    NCCLCHECK(epMakeTensor(&dispatch_outputs.tokens, 3, ncclBfloat16,
                           (unsigned)nRanks, max_dispatch_tokens_per_rank, hidden));

    NCCLCHECK(epMakeTensor(&dispatch_outputs.topk_weights, 3, ncclFloat32,
                           (unsigned)nRanks, max_dispatch_tokens_per_rank, top_k));

    NCCLCHECK(epMakeTensor(&dispatch_outputs.topk_idx, 3, ncclInt32,
                           (unsigned)nRanks, max_dispatch_tokens_per_rank, top_k));

    NCCLCHECK(epMakeTensor(&combine_inputs.tokens, 3, ncclBfloat16,
                           (unsigned)nRanks, max_dispatch_tokens_per_rank, hidden));
}

// LL benchmark — full tensor graph for ncclEpDispatch / ncclEpCombine.
//
// topk_idx is read from handle on the LL path (signature matches setupHighThroughputTensors).
void setupLowLatencyTensors(
    ncclEpDispatchInputs_t&  dispatch_inputs,
    ncclEpDispatchOutputs_t& dispatch_outputs,
    ncclEpLayoutInfo_t&      dispatch_layout_info,
    bool&                    has_dispatch_layout_info,
    ncclEpCombineInputs_t&   combine_inputs,
    ncclEpCombineOutputs_t&  combine_outputs,
    ncclEpTensor_t*&         topk_weights,
    unsigned int num_tokens,
    unsigned int hidden,
    unsigned int top_k,
    unsigned int num_local_experts,
    unsigned int max_dispatch_tokens_per_rank,
    int nRanks,
    ncclEpLayout_t layout
) {

    setupLowLatencyTensorsSharedInputs(dispatch_inputs, dispatch_layout_info,
                                       has_dispatch_layout_info, topk_weights,
                                       num_tokens, hidden, top_k, num_local_experts);

    switch (layout) {
        case NCCL_EP_LAYOUT_EXPERT_MAJOR:
            setupLowLatencyTensorsExpertMajLayout(dispatch_outputs, combine_inputs,
                                                  hidden, num_local_experts,
                                                  max_dispatch_tokens_per_rank, nRanks);
            break;
        case NCCL_EP_LAYOUT_RANK_MAJOR:
            setupLowLatencyTensorsRankMajLayout(dispatch_inputs, dispatch_outputs,
                                                dispatch_layout_info, combine_inputs,
                                                topk_weights,
                                                hidden, top_k, num_local_experts,
                                                max_dispatch_tokens_per_rank, nRanks);
            break;
        default:
            fprintf(stderr, "setupLowLatencyTensors: unsupported layout %d\n", (int)layout);
            exit(EXIT_FAILURE);
    }

    NCCLCHECK(epMakeTensor(&combine_outputs.tokens, 2, ncclBfloat16, num_tokens, hidden));

    // LL expert-major: per-token routing weights read on receive side from
    // combine_outputs.topk_weights (see nccl_ep.h).
    if (layout == NCCL_EP_LAYOUT_EXPERT_MAJOR) {
        combine_outputs.topk_weights = topk_weights;
    }
}

// Setup tensors for HIGH_THROUGHPUT mode using epMakeTensor
void setupHighThroughputTensors(
    ncclComm_t comm,
    BenchmarkAllocState&     alloc,
    ncclEpDispatchInputs_t&  dispatch_inputs,
    ncclEpDispatchOutputs_t& dispatch_outputs,
    ncclEpLayoutInfo_t&      dispatch_layout_info,
    bool&                    has_dispatch_layout_info,
    ncclEpCombineInputs_t&   combine_inputs,
    ncclEpCombineOutputs_t&  combine_outputs,
    ncclEpTensor_t*&         topk_weights,
    unsigned int num_tokens,
    unsigned int hidden,
    unsigned int top_k,
    unsigned int num_local_experts,
    unsigned int num_recv_tokens,
    ncclEpLayout_t layout,
    bool zcopy
) {
    const bool em = (layout == NCCL_EP_LAYOUT_EXPERT_MAJOR);

    EpTensorAllocOptions zc_comm;
    zc_comm.use_nccl_mem = zcopy;
    zc_comm.use_window = zcopy;
    zc_comm.window_comm = comm;
    zc_comm.registered_windows = &alloc.registered_windows;
    zc_comm.nccl_mem_ptrs = &alloc.external_data_ptrs;
    zc_comm.tensor_data_ptrs = &alloc.tensor_data_ptrs;

    EpTensorAllocOptions zc_no_window = zc_comm;
    zc_no_window.use_window = false;
    zc_no_window.window_comm = nullptr;
    zc_no_window.registered_windows = nullptr;

    const EpTensorAllocOptions* comm_window_opts = zcopy ? &zc_comm : nullptr;
    const EpTensorAllocOptions* no_window_opts = zcopy ? &zc_no_window : nullptr;

    // Dispatch input: tokens - initialize with test pattern
    NCCLCHECK(epMakeTensor(&dispatch_inputs.tokens, 2, ncclBfloat16,
                           num_tokens, hidden, 1, 1, 1, comm_window_opts));
    {
        void* input0_data;
        NCCLCHECK(epGetTensorData(alloc, dispatch_inputs.tokens, &input0_data));
        CUDACHECK(cudaMemset(input0_data, 0, num_tokens * hidden * 2));
    }

    // Dispatch input: topk_weights - initialize with equal weights
    NCCLCHECK(epMakeTensor(&dispatch_inputs.topk_weights, 2, ncclFloat32,
                           num_tokens, top_k, 1, 1, 1, comm_window_opts));
    {
        float *topk_weights_host = new float[num_tokens * top_k];
        for (unsigned int i = 0; i < num_tokens * top_k; i++) {
            topk_weights_host[i] = 1.0f / top_k;
        }
        void* dtw_data;
        NCCLCHECK(epGetTensorData(alloc, dispatch_inputs.topk_weights, &dtw_data));
        CUDACHECK(cudaMemcpy(dtw_data, topk_weights_host,
                             num_tokens * top_k * sizeof(float), cudaMemcpyHostToDevice));
        delete[] topk_weights_host;
    }

    // Dispatch output: 2D [num_recv_tokens, hidden]
    NCCLCHECK(epMakeTensor(&dispatch_outputs.tokens, 2, ncclBfloat16,
                           num_recv_tokens, hidden, 1, 1, 1, comm_window_opts));

    // Dispatch output: recv_topk_weights — EM: 1D [N]; FLAT: 2D [N, top_k].
    if (em) {
        NCCLCHECK(epMakeTensor(&dispatch_outputs.topk_weights, 1, ncclFloat32,
                               num_recv_tokens, 1, 1, 1, 1, no_window_opts));
    } else {
        NCCLCHECK(epMakeTensor(&dispatch_outputs.topk_weights, 2, ncclFloat32,
                               num_recv_tokens, top_k, 1, 1, 1, no_window_opts));
        // Dispatch output: recv_topk_idx (FLAT only)
        NCCLCHECK(epMakeTensor(&dispatch_outputs.topk_idx, 2, ncclInt64,
                               num_recv_tokens, top_k, 1, 1, 1, no_window_opts));
    }

    // Local: expert_counters — populated by upstream dispatch metadata path.
    // (HT FLAT writes unpadded int32; HT EM writes padded.)
    NCCLCHECK(epMakeTensor(&dispatch_layout_info.expert_counters, 1, ncclInt32,
                           num_local_experts, 1, 1, 1, 1, no_window_opts));
    has_dispatch_layout_info = true;

    // Combine input: 2D expert outputs - same size as dispatch output (received token count)
    NCCLCHECK(epMakeTensor(&combine_inputs.tokens, 2, ncclBfloat16,
                           num_recv_tokens, hidden, 1, 1, 1, comm_window_opts));
    {
        void* eo_data;
        NCCLCHECK(epGetTensorData(alloc, combine_inputs.tokens, &eo_data));
        CUDACHECK(cudaMemset(eo_data, 0, num_recv_tokens * hidden * 2));
    }

    // Combine output - sized to num_tokens (original token count per rank)
    NCCLCHECK(epMakeTensor(&combine_outputs.tokens, 2, ncclBfloat16,
                           num_tokens, hidden, 1, 1, 1, comm_window_opts));

    // topk_weights kept around for HT combine validation
    NCCLCHECK(epMakeTensor(&topk_weights, 2, ncclFloat32,
                           num_tokens, top_k, 1, 1, 1, comm_window_opts));

    // HT backward combine output: per-token topk_weights aligned with combine output tokens.
    NCCLCHECK(epMakeTensor(&combine_outputs.topk_weights, 2, ncclFloat32,
                           num_tokens, top_k, 1, 1, 1, no_window_opts));
}

// Cleanup benchmark tensors created via epMakeTensor.
void cleanupBenchmarkTensors(
    BenchmarkAllocState&     alloc,
    ncclEpDispatchInputs_t&  dispatch_inputs,
    ncclEpDispatchOutputs_t& dispatch_outputs,
    ncclEpLayoutInfo_t&      dispatch_layout_info,
    ncclEpCombineInputs_t&   combine_inputs,
    ncclEpCombineOutputs_t&  combine_outputs,
    ncclEpTensor_t*&         topk_weights,
    ncclEpTensor_t*&         topk_idx,
    bool                     is_ll_mode)
{
    epFreeTensor(&topk_idx);

    for (const auto& entry : alloc.registered_windows) {
        NCCLCHECK(ncclCommWindowDeregister(entry.comm, entry.win));
    }
    alloc.registered_windows.clear();

    epFreeTensor(&dispatch_inputs.tokens, &alloc.external_data_ptrs, &alloc.tensor_data_ptrs);

    if (!is_ll_mode) {
        epFreeTensor(&dispatch_inputs.topk_weights, &alloc.external_data_ptrs, &alloc.tensor_data_ptrs);
    }

    epFreeTensor(&dispatch_outputs.tokens, &alloc.external_data_ptrs, &alloc.tensor_data_ptrs);
    epFreeTensor(&dispatch_outputs.topk_weights, &alloc.external_data_ptrs, &alloc.tensor_data_ptrs);
    epFreeTensor(&dispatch_outputs.topk_idx, &alloc.external_data_ptrs, &alloc.tensor_data_ptrs);

    epFreeTensor(&dispatch_layout_info.expert_counters, &alloc.external_data_ptrs, &alloc.tensor_data_ptrs);
    epFreeTensor(&dispatch_layout_info.src_rank_counters, &alloc.external_data_ptrs, &alloc.tensor_data_ptrs);
    epFreeTensor(&combine_inputs.tokens, &alloc.external_data_ptrs, &alloc.tensor_data_ptrs);
    epFreeTensor(&combine_outputs.tokens, &alloc.external_data_ptrs, &alloc.tensor_data_ptrs);
    epFreeTensor(&topk_weights, &alloc.external_data_ptrs, &alloc.tensor_data_ptrs);

    if (!is_ll_mode) {
        epFreeTensor(&combine_outputs.topk_weights, &alloc.external_data_ptrs, &alloc.tensor_data_ptrs);
    }

    for (auto ptr : alloc.external_data_ptrs) {
        if (ptr) NCCLCHECK(ncclMemFree(ptr));
    }
    alloc.external_data_ptrs.clear();
    alloc.tensor_data_ptrs.clear();
}

// ============================================================================
// Data Validation Support (similar to DeepEP test_internode.py / test_low_latency.py)
//
// Methodology:
//   - Input tokens are fingerprinted with (source_rank, token_id) in BF16.
//   - Dispatch validation recomputes expected routing deterministically and
//     verifies each received token's identity and integrity.
//   - Combine validation computes expected weighted sums analytically and
//     compares against actual output using a cosine-similarity metric (calc_diff).
// ============================================================================

// Rank offset for BF16 precision: integers > 256 lose precision in BF16
// Using negative values (rank - 128) allows up to 256 ranks
static const int RANK_OFFSET = 128;

// Number of columns to embed token index (for full traceability)
// Matches DeepEP's approach: last 128 columns store token index
static const int TOKEN_ID_COLS = 128;

// Helper: Convert BF16 to float (CPU-side)
static float bf16ToFloat(uint16_t bf16) {
    uint32_t bits = (static_cast<uint32_t>(bf16)) << 16;
    float f;
    memcpy(&f, &bits, sizeof(f));
    return f;
}

// Helper: Convert float to BF16 (CPU-side, truncation — used only for initialization)
static uint16_t floatToBf16(float f) {
    uint32_t bits;
    memcpy(&bits, &f, sizeof(bits));
    return static_cast<uint16_t>(bits >> 16);
}

// Cosine-similarity-based discrepancy metric in double precision
// Returns 0 for perfect match, larger values for worse match
static double calc_diff(const double* x, const double* y, size_t n) {
    double dot_xy = 0, dot_xx = 0, dot_yy = 0;
    for (size_t i = 0; i < n; i++) {
        double xi = x[i] + 1.0;
        double yi = y[i] + 1.0;
        dot_xy += xi * yi;
        dot_xx += xi * xi;
        dot_yy += yi * yi;
    }
    double denom = dot_xx + dot_yy;
    if (denom == 0) return 0;
    return 1.0 - 2.0 * dot_xy / denom;
}

// Initialize input tensors with validation-friendly patterns (DeepEP style)
// Pattern: each element = (rank - RANK_OFFSET) except last TOKEN_ID_COLS columns = token_index
void initializeValidationData(
    const BenchmarkAllocState& alloc,
    ncclEpDispatchInputs_t&    dispatch_inputs,
    ncclEpTensor_t*            topk_weights,
    unsigned int num_tokens,
    unsigned int hidden,
    unsigned int top_k,
    int myRank,
    bool is_ht_mode
) {
    // Calculate the rank value to use (handles BF16 precision limits)
    float rank_value = static_cast<float>(myRank - RANK_OFFSET);
    uint16_t rank_bf16 = floatToBf16(rank_value);

    // Allocate host buffer for token data
    size_t token_size = num_tokens * hidden;
    uint16_t* token_data_host = new uint16_t[token_size];

    // Fill token data with rank value, embed token index in last TOKEN_ID_COLS columns.
    // Token ID is split into high (t/256) and low (t%256) bytes to stay within BF16's
    // exact integer range (0-255). First TOKEN_ID column = high byte, rest = low byte.
    for (unsigned int t = 0; t < num_tokens; t++) {
        uint16_t token_hi = floatToBf16(static_cast<float>(t / 256));
        uint16_t token_lo = floatToBf16(static_cast<float>(t % 256));
        for (unsigned int h = 0; h < hidden; h++) {
            if (h == hidden - TOKEN_ID_COLS) {
                token_data_host[t * hidden + h] = token_hi;
            } else if (h > hidden - TOKEN_ID_COLS) {
                token_data_host[t * hidden + h] = token_lo;
            } else {
                token_data_host[t * hidden + h] = rank_bf16;
            }
        }
    }

    // Copy to GPU
    {
        void* input0_data;
        NCCLCHECK(epGetTensorData(alloc, dispatch_inputs.tokens, &input0_data));
        CUDACHECK(cudaMemcpy(input0_data, token_data_host,
                             token_size * sizeof(uint16_t), cudaMemcpyHostToDevice));
    }

    // Generate random positive topk_weights: abs(randn)
    // LL: weights applied during combine → affects combined output
    // HT: weights forwarded during dispatch → does NOT affect combined output
    float* topk_weights_host = new float[num_tokens * top_k];
    std::mt19937 rng(42 + myRank);
    std::normal_distribution<float> normal(0.0f, 1.0f);
    for (unsigned int i = 0; i < num_tokens * top_k; i++) {
        topk_weights_host[i] = std::abs(normal(rng));
        if (topk_weights_host[i] < 1e-6f) topk_weights_host[i] = 1e-6f;
    }

    if (is_ht_mode) {
        void* dtw_data;
        NCCLCHECK(epGetTensorData(alloc, dispatch_inputs.topk_weights, &dtw_data));
        CUDACHECK(cudaMemcpy(dtw_data, topk_weights_host,
                             num_tokens * top_k * sizeof(float), cudaMemcpyHostToDevice));
    }
    // Also initialize the combine topk_weights (used by both modes)
    {
        void* tw_data;
        NCCLCHECK(epGetTensorData(alloc, topk_weights, &tw_data));
        CUDACHECK(cudaMemcpy(tw_data, topk_weights_host,
                             num_tokens * top_k * sizeof(float), cudaMemcpyHostToDevice));
    }

    delete[] topk_weights_host;
    delete[] token_data_host;
}

// Validation result structure
struct ValidationResult {
    bool passed;
    int errors;
    double max_diff;
    std::string message;
};

// Forward declaration (defined later in the file)
void generateRandomTopkIndicesLL(
    int64_t* topk_idx_host, unsigned int num_tokens, unsigned int num_experts,
    unsigned int top_k, int rank, int seed = 1);

// Generate HT topk_idx for a given rank (deterministic)
// Randperm routing (uniform), consistent with Hybrid-EP (test_hybrid_ep.py)
static void generateTopkIndicesHT(
    int64_t* topk_idx_host,
    unsigned int num_tokens,
    unsigned int num_experts,
    unsigned int top_k,
    int rank
) {
    std::mt19937 gen(rank + 42);
    std::vector<int64_t> expert_perm(num_experts);
    std::iota(expert_perm.begin(), expert_perm.end(), 0);
    for (unsigned int i = 0; i < num_tokens; i++) {
        std::shuffle(expert_perm.begin(), expert_perm.end(), gen);
        for (unsigned int j = 0; j < top_k; j++) {
            topk_idx_host[i * top_k + j] = expert_perm[j];
        }
    }
}

// Extract (source_rank, token_id) from a received token row using first and last columns.
// `max_token_id` is the upper bound on encoded token id used as a sanity check; pass the
// group-wide max_tokens_per_rank (== num_tokens under uniform; >= every rank's count under
// non-uniform).
static bool extractTokenIdentity(
    const uint16_t* row,
    unsigned int hidden,
    int nRanks,
    unsigned int max_token_id,
    int* out_source_rank,
    int* out_token_id
) {
    float rank_val = bf16ToFloat(row[0]);
    *out_source_rank = static_cast<int>(rank_val + RANK_OFFSET + 0.5f);

    float token_hi = bf16ToFloat(row[hidden - TOKEN_ID_COLS]);
    float token_lo = bf16ToFloat(row[hidden - 1]);
    *out_token_id = static_cast<int>(token_hi + 0.5f) * 256 + static_cast<int>(token_lo + 0.5f);

    return (*out_source_rank >= 0 && *out_source_rank < nRanks &&
            *out_token_id >= 0 && *out_token_id < static_cast<int>(max_token_id));
}

// Verify a received token row has consistent data (all rank cols match, all token_id cols match)
static bool verifyTokenIntegrity(
    const uint16_t* row,
    unsigned int hidden
) {
    uint16_t expected_rank_bf16 = row[0];
    for (unsigned int h = 1; h < hidden - TOKEN_ID_COLS; h++) {
        if (row[h] != expected_rank_bf16) return false;
    }
    // First TOKEN_ID column is the high byte (standalone), rest are low byte
    uint16_t expected_token_lo_bf16 = row[hidden - 1];
    for (unsigned int h = hidden - TOKEN_ID_COLS + 1; h < hidden - 1; h++) {
        if (row[h] != expected_token_lo_bf16) return false;
    }
    return true;
}

// Caps error-print volume while still counting every error.
struct ErrorReporter {
    int errors = 0;
    int printed = 0;
    int max_print;
    explicit ErrorReporter(int cap = 10) : max_print(cap) {}
    __attribute__((format(printf, 2, 3)))
    void error(const char* fmt, ...) {
        if (printed < max_print) {
            va_list ap; va_start(ap, fmt); vprintf(fmt, ap); va_end(ap);
            printed++;
        }
        errors++;
    }
};

// Decode+integrity+expected-match over [zone_offset, zone_offset+zone_count); returns decoded keys.
//   skip_invalid_identity=true  → invalid rows silently skipped (HT-EM padding).
//   skip_invalid_identity=false → invalid identity reported as error (LL-EM).
static std::set<std::pair<int,int>> scanExpertZone(
    const uint16_t* recv_data,
    int64_t zone_offset,
    int64_t zone_count,
    unsigned int hidden,
    int nRanks,
    unsigned int max_token_id,
    const std::set<std::pair<int,int>>& expected,
    bool skip_invalid_identity,
    const char* tag,
    int myRank,
    unsigned int expert_idx,
    ErrorReporter& rep)
{
    std::set<std::pair<int,int>> found;
    for (int64_t s = 0; s < zone_count; s++) {
        const uint16_t* row = recv_data + (zone_offset + s) * hidden;
        int source_rank = -1, token_id = -1;
        if (!extractTokenIdentity(row, hidden, nRanks, max_token_id, &source_rank, &token_id)) {
            if (!skip_invalid_identity) {
                rep.error("[Rank %d] %s: expert %u slot %ld: invalid identity (rank=%d, token=%d)\n",
                          myRank, tag, expert_idx, (long)s, source_rank, token_id);
            }
            continue;
        }
        if (!verifyTokenIntegrity(row, hidden)) {
            rep.error("[Rank %d] %s: expert %u slot %ld: data corruption (rank=%d, token=%d)\n",
                      myRank, tag, expert_idx, (long)s, source_rank, token_id);
        }
        auto key = std::make_pair(source_rank, token_id);
        if (expected.find(key) == expected.end()) {
            rep.error("[Rank %d] %s: expert %u slot %ld: unexpected token (rank=%d, token=%d)\n",
                      myRank, tag, expert_idx, (long)s, source_rank, token_id);
        }
        found.insert(key);
    }
    return found;
}

// ==================== LL expert-major dispatch validation ====================
// Output: 3D [num_local_experts, max_tokens_per_expert, hidden].
// Token counts per expert come from local_tensors[0] (RECV_EXPERT_COUNTER_DEVICE).
static ValidationResult validateDispatchOutputLLExpertMaj(
    const BenchmarkAllocState&     alloc,
    const ncclEpDispatchOutputs_t& dispatch_outputs,
    const ncclEpLayoutInfo_t&      dispatch_layout_info,
    unsigned int max_tokens_per_rank,
    const unsigned int*            num_tokens_per_rank,
    unsigned int hidden,
    unsigned int top_k,
    unsigned int num_experts,
    unsigned int num_local_experts,
    int myRank,
    int nRanks
) {
    ValidationResult result = {true, 0, 0.0, ""};
    ErrorReporter rep;
    int64_t* src_topk = new int64_t[max_tokens_per_rank * top_k];

    const size_t* out0_sizes = dispatch_outputs.tokens->sizes;
    unsigned int max_tpe = out0_sizes[1];
    size_t total_size = static_cast<size_t>(num_local_experts) * max_tpe * hidden;
    uint16_t* recv_data = new uint16_t[total_size];
    void* output0_data;
    NCCLCHECK(epGetTensorData(alloc, dispatch_outputs.tokens, &output0_data));
    CUDACHECK(cudaMemcpy(recv_data, output0_data,
                         total_size * sizeof(uint16_t), cudaMemcpyDeviceToHost));

    int* tokens_per_expert = new int[num_local_experts];
    void* local0_data;
    NCCLCHECK(epGetTensorData(alloc, dispatch_layout_info.expert_counters, &local0_data));
    CUDACHECK(cudaMemcpy(tokens_per_expert, local0_data,
                         num_local_experts * sizeof(int), cudaMemcpyDeviceToHost));

    // Build expected set: expected[local_expert] = set of (source_rank, token_id)
    std::vector<std::set<std::pair<int,int>>> expected(num_local_experts);
    for (int r = 0; r < nRanks; r++) {
        unsigned int r_tokens = num_tokens_per_rank[r];
        generateRandomTopkIndicesLL(src_topk, r_tokens, num_experts, top_k, r);
        for (unsigned int t = 0; t < r_tokens; t++) {
            for (unsigned int k = 0; k < top_k; k++) {
                int64_t expert_id = src_topk[t * top_k + k];
                if (expert_id < 0) continue;
                int expert_rank = static_cast<int>(expert_id) / static_cast<int>(num_local_experts);
                int local_expert = static_cast<int>(expert_id) % static_cast<int>(num_local_experts);
                if (expert_rank == myRank)
                    expected[local_expert].insert({r, static_cast<int>(t)});
            }
        }
    }

    // Scan output and match against expected
    for (unsigned int e = 0; e < num_local_experts; e++) {
        int count = tokens_per_expert[e];
        if (count < 0 || count > static_cast<int>(max_tpe)) {
            rep.error("[Rank %d] LL-EM dispatch: expert %u has invalid count %d (max %u)\n",
                      myRank, e, count, max_tpe);
            continue;
        }
        auto found = scanExpertZone(recv_data, static_cast<int64_t>(e) * max_tpe, count,
                                    hidden, nRanks, max_tokens_per_rank, expected[e],
                                    /*skip_invalid_identity=*/false, "LL-EM dispatch",
                                    myRank, e, rep);
        for (const auto& key : expected[e]) {
            if (found.find(key) == found.end()) {
                rep.error("[Rank %d] LL-EM dispatch: expert %u: missing token (rank=%d, token=%d)\n",
                          myRank, e, key.first, key.second);
            }
        }
    }

    delete[] tokens_per_expert;
    delete[] recv_data;
    delete[] src_topk;

    result.errors = rep.errors;
    result.passed = (rep.errors == 0);
    if (!result.passed) {
        char buf[256];
        snprintf(buf, sizeof(buf), "LL-EM dispatch validation: %d errors", rep.errors);
        result.message = buf;
    }
    return result;
}

// ==================== LL rank-major dispatch validation ====================
// Output: 3D [nRanks, max_dispatch_tokens_per_rank, hidden], one slot per received token packed by source rank.
// outputs[1] = recv_topk_weights [nRanks, max_tpr, top_k]: all top-k weights from the source token.
// outputs[2] = recv_topk_idx     [nRanks, max_tpr, top_k]: local expert index on myRank, or -1.
// Slots within each rank's block are contiguous from index 0; first invalid slot ends the block.
static ValidationResult validateDispatchOutputLLRankMaj(
    const BenchmarkAllocState&     alloc,
    const ncclEpDispatchOutputs_t& dispatch_outputs,
    const ncclEpLayoutInfo_t&      dispatch_layout_info,
    unsigned int max_tokens_per_rank,
    const unsigned int*            num_tokens_per_rank,
    unsigned int hidden,
    unsigned int top_k,
    unsigned int num_experts,
    unsigned int num_local_experts,
    int myRank,
    int nRanks
) {
    ValidationResult result = {true, 0, 0.0, ""};
    int errors = 0;
    const int max_errors_to_print = 10;
    int errors_printed = 0;

    // Output tensor sizes: tokens is 3D [nRanks, max_tpr, hidden] in LL rank-major.
    const size_t* out0_sizes = dispatch_outputs.tokens->sizes;
    const size_t max_tpr     = out0_sizes[1];
    const size_t total_slots = out0_sizes[0] * max_tpr;

    // Copy recv_x, recv_topk_weights, recv_topk_idx, recv_rank_counter to host
    uint16_t* recv_data = new uint16_t[total_slots * hidden];
    float*    recv_wgt  = new float   [total_slots * top_k];
    int32_t*  recv_idx  = new int32_t [total_slots * top_k];
    int32_t*  recv_cnt  = new int32_t [(size_t)nRanks];

    void *out0_data, *out1_data, *out2_data, *local0_data;
    NCCLCHECK(epGetTensorData(alloc, dispatch_outputs.tokens, &out0_data));
    NCCLCHECK(epGetTensorData(alloc, dispatch_outputs.topk_weights, &out1_data));
    NCCLCHECK(epGetTensorData(alloc, dispatch_outputs.topk_idx, &out2_data));
    NCCLCHECK(epGetTensorData(alloc, dispatch_layout_info.src_rank_counters, &local0_data));
    CUDACHECK(cudaMemcpy(recv_data, out0_data,    total_slots * hidden * sizeof(uint16_t), cudaMemcpyDeviceToHost));
    CUDACHECK(cudaMemcpy(recv_wgt,  out1_data,    total_slots * top_k  * sizeof(float),   cudaMemcpyDeviceToHost));
    CUDACHECK(cudaMemcpy(recv_idx,  out2_data,    total_slots * top_k  * sizeof(int32_t), cudaMemcpyDeviceToHost));
    CUDACHECK(cudaMemcpy(recv_cnt,  local0_data,  (size_t)nRanks       * sizeof(int32_t), cudaMemcpyDeviceToHost));

    int64_t* src_topk = new int64_t[max_tokens_per_rank * top_k];
    float*   src_wgt  = new float  [max_tokens_per_rank * top_k];

    for (int r = 0; r < nRanks; r++) {
        unsigned int r_tokens = num_tokens_per_rank[r];
        // Regenerate expected topk indices and weights for rank r (must match initializeValidationData)
        generateRandomTopkIndicesLL(src_topk, r_tokens, num_experts, top_k, r);
        {
            std::mt19937 rng(42 + r);
            std::normal_distribution<float> normal_dist(0.0f, 1.0f);
            for (unsigned int i = 0; i < r_tokens * top_k; i++) {
                src_wgt[i] = std::abs(normal_dist(rng));
                if (src_wgt[i] < 1e-6f) src_wgt[i] = 1e-6f;
            }
        }

        // Build ordered list of tokens from rank r that map at least one expert to myRank
        std::vector<int> expected_tokens;
        for (unsigned int t = 0; t < r_tokens; t++) {
            for (unsigned int k = 0; k < top_k; k++) {
                int64_t eid = src_topk[t * top_k + k];
                if (eid >= 0 && (int)(eid / num_local_experts) == myRank) {
                    expected_tokens.push_back((int)t);
                    break;
                }
            }
        }

        // Scan exactly recv_cnt[r] slots — the authoritative count written by the dispatch kernel.
        const int expected_slot_count = recv_cnt[r];
        int slot_count = 0;
        std::set<int> found_tokens;

        for (int s = 0; s < expected_slot_count; s++) {
            unsigned int slot = (unsigned)r * max_tpr + (unsigned)s;
            const uint16_t* row = recv_data + slot * hidden;
            const int32_t*  idx = recv_idx  + slot * top_k;
            const float*    wgt = recv_wgt  + slot * top_k;

            int source_rank = -1, token_id = -1;
            if (!extractTokenIdentity(row, hidden, nRanks, max_tokens_per_rank, &source_rank, &token_id)) {
                if (errors_printed++ < max_errors_to_print)
                    printf("[Rank %d] LL-RM dispatch: rank %d slot %d: invalid token identity\n",
                           myRank, r, s);
                errors++; continue;
            }
            slot_count++;

            if (source_rank != r) {
                if (errors_printed++ < max_errors_to_print)
                    printf("[Rank %d] LL-RM dispatch: rank %d slot %u: wrong source rank %d\n",
                           myRank, r, s, source_rank);
                errors++; continue;
            }
            if (!verifyTokenIntegrity(row, hidden)) {
                if (errors_printed++ < max_errors_to_print)
                    printf("[Rank %d] LL-RM dispatch: rank %d slot %u (token %d): data corruption\n",
                           myRank, r, s, token_id);
                errors++;
            }

            // Verify recv_topk_idx: local expert index for experts on myRank, -1 otherwise
            for (unsigned int k = 0; k < top_k; k++) {
                int64_t eid = src_topk[token_id * top_k + k];
                int32_t expected_idx;
                if (eid < 0) {
                    expected_idx = -1;
                } else {
                    int expert_rank = (int)(eid / num_local_experts);
                    expected_idx = (expert_rank == myRank) ? (int32_t)(eid % num_local_experts) : (int32_t)-1;
                }
                if (idx[k] != expected_idx) {
                    if (errors_printed++ < max_errors_to_print)
                        printf("[Rank %d] LL-RM dispatch: rank %d slot %u token %d: topk[%u] idx=%d expected=%d\n",
                               myRank, r, s, token_id, k, idx[k], expected_idx);
                    errors++;
                }
            }

            // Verify recv_topk_weights: should exactly match source weights (float, no precision loss)
            for (unsigned int k = 0; k < top_k; k++) {
                float expected_w = src_wgt[token_id * top_k + k];
                if (std::abs(wgt[k] - expected_w) > 1e-5f * expected_w) {
                    if (errors_printed++ < max_errors_to_print)
                        printf("[Rank %d] LL-RM dispatch: rank %d slot %u token %d: weight[%u]=%.6f expected=%.6f\n",
                               myRank, r, s, token_id, k, wgt[k], expected_w);
                    errors++;
                }
            }

            found_tokens.insert(token_id);
        }

        // Verify token count: recv_cnt[r] must match expected
        if (expected_slot_count != (int)expected_tokens.size()) {
            if (errors_printed++ < max_errors_to_print)
                printf("[Rank %d] LL-RM dispatch: rank %d: recv_cnt=%d, expected %d\n",
                       myRank, r, expected_slot_count, (int)expected_tokens.size());
            errors++;
        } else if (slot_count != expected_slot_count) {
            if (errors_printed++ < max_errors_to_print)
                printf("[Rank %d] LL-RM dispatch: rank %d: decoded %d valid slots of %d\n",
                       myRank, r, slot_count, expected_slot_count);
            errors++;
        }

        // Verify coverage: all expected tokens were received
        for (int t : expected_tokens) {
            if (found_tokens.find(t) == found_tokens.end()) {
                if (errors_printed++ < max_errors_to_print)
                    printf("[Rank %d] LL-RM dispatch: rank %d: missing token %d\n",
                           myRank, r, t);
                errors++;
            }
        }
    }

    delete[] src_wgt;
    delete[] src_topk;
    delete[] recv_cnt;
    delete[] recv_idx;
    delete[] recv_wgt;
    delete[] recv_data;

    result.errors = errors;
    result.passed = (errors == 0);
    if (!result.passed) {
        char buf[256];
        snprintf(buf, sizeof(buf), "LL-RM dispatch: %d errors", errors);
        result.message = buf;
    }
    return result;
}

// ==================== LL rank-major pre-reduction ====================
// Multiplies each received expert output slot by the per-rank weight sum before combine.
// The rank-major combine kernel uses weight=1; the caller is responsible for applying
// the weights so that combined[t] = sum_R(weight_sum_R * expert_output_R).
//
// For each valid slot s from source rank r:
//   weight_sum = sum of recv_topk_weights[slot, k] for all k where recv_topk_idx[slot, k] >= 0
//   expert_outputs[slot] *= weight_sum
//
// Uses RECV_RANK_COUNTER_DEVICE (local_tensors[0]) as the authoritative per-rank slot count.
static void preReduceRankMajor(
    const BenchmarkAllocState&     alloc,
    const ncclEpDispatchOutputs_t& dispatch_outputs,
    const ncclEpLayoutInfo_t&      dispatch_layout_info,
    const ncclEpCombineInputs_t&   combine_inputs,
    unsigned int top_k,
    int nRanks
) {
    // tokens is 3D [nRanks, max_tpr, hidden] in LL rank-major.
    const size_t* out0_sizes = dispatch_outputs.tokens->sizes;
    const unsigned int max_tpr     = out0_sizes[1];
    const unsigned int hidden      = out0_sizes[2];
    const unsigned int total_slots = out0_sizes[0] * max_tpr;

    void *out1_data, *out2_data, *local0_data;
    NCCLCHECK(epGetTensorData(alloc, dispatch_outputs.topk_weights, &out1_data));
    NCCLCHECK(epGetTensorData(alloc, dispatch_outputs.topk_idx, &out2_data));
    NCCLCHECK(epGetTensorData(alloc, dispatch_layout_info.src_rank_counters, &local0_data));

    float*   recv_wgt = new float  [total_slots * top_k];
    int32_t* recv_idx = new int32_t[total_slots * top_k];
    int32_t* recv_cnt = new int32_t[nRanks];
    CUDACHECK(cudaMemcpy(recv_wgt, out1_data, total_slots * top_k * sizeof(float),   cudaMemcpyDeviceToHost));
    CUDACHECK(cudaMemcpy(recv_idx, out2_data, total_slots * top_k * sizeof(int32_t), cudaMemcpyDeviceToHost));
    CUDACHECK(cudaMemcpy(recv_cnt, local0_data, (size_t)nRanks      * sizeof(int32_t), cudaMemcpyDeviceToHost));

    void* eo_data;
    NCCLCHECK(epGetTensorData(alloc, combine_inputs.tokens, &eo_data));
    uint16_t* eo_host = new uint16_t[total_slots * hidden];
    CUDACHECK(cudaMemcpy(eo_host, eo_data, total_slots * hidden * sizeof(uint16_t), cudaMemcpyDeviceToHost));

    for (int r = 0; r < nRanks; r++) {
        for (int s = 0; s < recv_cnt[r]; s++) {
            unsigned int slot = (unsigned)r * max_tpr + (unsigned)s;
            // Sum weights for all top-k entries that routed to this rank (non-negative local idx)
            float weight_sum = 0.0f;
            for (unsigned int k = 0; k < top_k; k++) {
                if (recv_idx[slot * top_k + k] >= 0)
                    weight_sum += recv_wgt[slot * top_k + k];
            }
            for (unsigned int h = 0; h < hidden; h++) {
                float val = bf16ToFloat(eo_host[slot * hidden + h]);
                eo_host[slot * hidden + h] = floatToBf16(val * weight_sum);
            }
        }
    }

    CUDACHECK(cudaMemcpy(eo_data, eo_host, total_slots * hidden * sizeof(uint16_t), cudaMemcpyHostToDevice));

    delete[] eo_host;
    delete[] recv_cnt;
    delete[] recv_idx;
    delete[] recv_wgt;
}

// ==================== HT rank-major dispatch validation ====================
// Output: 2D [nRanks*max_tokens_per_rank, hidden]; row valid iff any recv_topk_idx[k] >= 0.
// FIXME: ncclEpHandleGetNumRecvTokens returns buffer max, not actual count — scan recv_topk_idx as workaround.
static ValidationResult validateDispatchOutputHTRankMaj(
    const BenchmarkAllocState&     alloc,
    const ncclEpDispatchOutputs_t& dispatch_outputs,
    unsigned int max_tokens_per_rank,
    const unsigned int*            num_tokens_per_rank,
    unsigned int hidden,
    unsigned int top_k,
    unsigned int num_experts,
    unsigned int num_local_experts,
    int myRank,
    int nRanks
) {
    ValidationResult result = {true, 0, 0.0, ""};
    int errors = 0;
    const int max_errors_to_print = 10;
    int errors_printed = 0;
    int64_t* src_topk = new int64_t[max_tokens_per_rank * top_k];

    const size_t* out0_sizes = dispatch_outputs.tokens->sizes;
    unsigned int buf_rows = out0_sizes[0];
    size_t recv_size = static_cast<size_t>(buf_rows) * hidden;
    uint16_t* recv_data = new uint16_t[recv_size];
    void* output0_data;
    NCCLCHECK(epGetTensorData(alloc, dispatch_outputs.tokens, &output0_data));
    CUDACHECK(cudaMemcpy(recv_data, output0_data,
                         recv_size * sizeof(uint16_t), cudaMemcpyDeviceToHost));

    bool* valid_slot = new bool[buf_rows]();
    int64_t* recv_topk_idx = new int64_t[static_cast<size_t>(buf_rows) * top_k];
    void* output2_data;
    NCCLCHECK(epGetTensorData(alloc, dispatch_outputs.topk_idx, &output2_data));
    CUDACHECK(cudaMemcpy(recv_topk_idx, output2_data,
                         static_cast<size_t>(buf_rows) * top_k * sizeof(int64_t),
                         cudaMemcpyDeviceToHost));
    for (unsigned int j = 0; j < buf_rows; j++) {
        for (unsigned int k = 0; k < top_k; k++) {
            if (recv_topk_idx[j * top_k + k] >= 0) { valid_slot[j] = true; break; }
        }
    }
    delete[] recv_topk_idx;

    std::set<std::pair<int,int>> expected;
    for (int r = 0; r < nRanks; r++) {
        unsigned int r_tokens = num_tokens_per_rank[r];
        generateTopkIndicesHT(src_topk, r_tokens, num_experts, top_k, r);
        for (unsigned int t = 0; t < r_tokens; t++) {
            for (unsigned int k = 0; k < top_k; k++) {
                int64_t expert_id = src_topk[t * top_k + k];
                int expert_rank = static_cast<int>(expert_id) / static_cast<int>(num_local_experts);
                if (expert_rank == myRank) {
                    expected.insert({r, static_cast<int>(t)});
                    break;
                }
            }
        }
    }

    std::set<std::pair<int,int>> found;
    for (unsigned int j = 0; j < buf_rows; j++) {
        if (!valid_slot[j]) continue;

        const uint16_t* row = recv_data + j * hidden;
        int source_rank = -1, token_id = -1;

        // RM: every marked slot must decode.
        if (!extractTokenIdentity(row, hidden, nRanks, max_tokens_per_rank, &source_rank, &token_id)) {
            if (errors_printed < max_errors_to_print) {
                printf("[Rank %d] HT dispatch: slot %u: invalid identity (rank=%d, token=%d)\n",
                       myRank, j, source_rank, token_id);
                errors_printed++;
            }
            errors++;
            continue;
        }

        if (!verifyTokenIntegrity(row, hidden)) {
            if (errors_printed < max_errors_to_print) {
                printf("[Rank %d] HT dispatch: slot %u: data corruption (rank=%d, token=%d)\n",
                       myRank, j, source_rank, token_id);
                errors_printed++;
            }
            errors++;
        }

        auto key = std::make_pair(source_rank, token_id);
        if (expected.find(key) == expected.end()) {
            if (errors_printed < max_errors_to_print) {
                printf("[Rank %d] HT dispatch: slot %u: unexpected token (rank=%d, token=%d)\n",
                       myRank, j, source_rank, token_id);
                errors_printed++;
            }
            errors++;
        }
        found.insert(key);
    }

    for (const auto& key : expected) {
        if (found.find(key) == found.end()) {
            if (errors_printed < max_errors_to_print) {
                printf("[Rank %d] HT dispatch: missing token (rank=%d, token=%d)\n",
                       myRank, key.first, key.second);
                errors_printed++;
            }
            errors++;
        }
    }

    delete[] valid_slot;
    delete[] recv_data;
    delete[] src_topk;

    result.errors = errors;
    result.passed = (errors == 0);
    if (!result.passed) {
        char buf[256];
        snprintf(buf, sizeof(buf), "HT dispatch validation: %d errors", errors);
        result.message = buf;
    }
    return result;
}

// ==================== HT expert-major dispatch validation ====================
// Output: 2D [budget, hidden] split into per-expert zones (meta_expert_offsets[e], counts_padded[e]).
// Phase A: pad slots (decode rank=128) must be all-zero. Phase B: dup tokens across zones byte-identical.
static ValidationResult validateDispatchOutputHTExpertMaj(
    const BenchmarkAllocState&     alloc,
    const ncclEpDispatchOutputs_t& dispatch_outputs,
    unsigned int max_tokens_per_rank,
    const unsigned int*            num_tokens_per_rank,
    unsigned int hidden,
    unsigned int top_k,
    unsigned int num_experts,
    unsigned int num_local_experts,
    int myRank,
    int nRanks,
    const int64_t* meta_expert_counts_padded,
    const int64_t* meta_expert_offsets
) {
    ValidationResult result = {true, 0, 0.0, ""};
    ErrorReporter rep;
    int64_t* src_topk = new int64_t[max_tokens_per_rank * top_k];

    const size_t* out0_sizes = dispatch_outputs.tokens->sizes;
    unsigned int buf_rows = out0_sizes[0];
    size_t recv_size = static_cast<size_t>(buf_rows) * hidden;
    uint16_t* recv_data = new uint16_t[recv_size];
    void* output0_data;
    NCCLCHECK(epGetTensorData(alloc, dispatch_outputs.tokens, &output0_data));
    CUDACHECK(cudaMemcpy(recv_data, output0_data,
                         recv_size * sizeof(uint16_t), cudaMemcpyDeviceToHost));

    // HT-EM expected: flat (src_rank, token_id) — token reaches at least one local expert.
    std::set<std::pair<int,int>> expected;
    for (int r = 0; r < nRanks; r++) {
        unsigned int r_tokens = num_tokens_per_rank[r];
        generateTopkIndicesHT(src_topk, r_tokens, num_experts, top_k, r);
        for (unsigned int t = 0; t < r_tokens; t++) {
            for (unsigned int k = 0; k < top_k; k++) {
                int64_t expert_id = src_topk[t * top_k + k];
                int expert_rank = static_cast<int>(expert_id) / static_cast<int>(num_local_experts);
                if (expert_rank == myRank) {
                    expected.insert({r, static_cast<int>(t)});
                    break;
                }
            }
        }
    }

    // Decode + match per expert zone (padding rows silently skipped).
    std::set<std::pair<int,int>> found;
    for (unsigned int e = 0; e < num_local_experts; e++) {
        auto z = scanExpertZone(recv_data, meta_expert_offsets[e], meta_expert_counts_padded[e],
                                hidden, nRanks, max_tokens_per_rank, expected,
                                /*skip_invalid_identity=*/true, "HT dispatch",
                                myRank, e, rep);
        found.insert(z.begin(), z.end());
    }

    for (const auto& key : expected) {
        if (found.find(key) == found.end()) {
            rep.error("[Rank %d] HT dispatch: missing token (rank=%d, token=%d)\n",
                      myRank, key.first, key.second);
        }
    }

    // Phase A/B: per-expert-zone padding zero-check and dup-token cross-zone consistency.
    using TokenKey = std::pair<int, int>;
    std::map<TokenKey, std::vector<std::pair<unsigned int, int64_t>>> locs;
    for (unsigned int e = 0; e < num_local_experts; e++) {
        int64_t off = meta_expert_offsets[e];
        int64_t cnt = meta_expert_counts_padded[e];
        for (int64_t s = 0; s < cnt; s++) {
            const uint16_t* row = recv_data + (off + s) * hidden;
            int src_rank = -1, tok_id = -1;
            if (!extractTokenIdentity(row, hidden, nRanks, max_tokens_per_rank, &src_rank, &tok_id)) {
                // Phase A: padding slot must be all-zero.
                for (unsigned int h = 0; h < hidden; h++) {
                    if (row[h] != 0) {
                        rep.error("[Rank %d] HT dispatch: expert %u pad slot %ld: non-zero at h=%u (0x%04x)\n",
                                  myRank, e, (long)s, h, row[h]);
                        break;
                    }
                }
                continue;
            }
            locs[{src_rank, tok_id}].push_back({e, s});
        }
    }

    // Phase B: duplicated tokens must be data-identical across all expert zones.
    for (const auto& kv : locs) {
        if (kv.second.size() < 2) continue;
        const auto& ref = kv.second[0];
        const uint16_t* base = recv_data + (meta_expert_offsets[ref.first] + ref.second) * hidden;
        for (size_t i = 1; i < kv.second.size(); i++) {
            const auto& loc = kv.second[i];
            const uint16_t* cmp = recv_data + (meta_expert_offsets[loc.first] + loc.second) * hidden;
            if (memcmp(base, cmp, hidden * sizeof(uint16_t)) != 0) {
                rep.error("[Rank %d] HT dispatch: dup-zone mismatch token (src=%d tok=%d) E%u[%ld]!=E%u[%ld]\n",
                          myRank, kv.first.first, kv.first.second,
                          ref.first, (long)ref.second, loc.first, (long)loc.second);
            }
        }
    }

    delete[] recv_data;
    delete[] src_topk;

    result.errors = rep.errors;
    result.passed = (rep.errors == 0);
    if (!result.passed) {
        char buf[256];
        snprintf(buf, sizeof(buf), "HT dispatch validation: %d errors", rep.errors);
        result.message = buf;
    }
    return result;
}

// Dispatcher: routes to the appropriate per-mode validation function.
ValidationResult validateDispatchOutput(
    const BenchmarkAllocState&     alloc,
    const ncclEpDispatchOutputs_t& dispatch_outputs,
    const ncclEpLayoutInfo_t&      dispatch_layout_info,
    unsigned int max_tokens_per_rank,
    const unsigned int*            num_tokens_per_rank,
    unsigned int hidden,
    unsigned int top_k,
    unsigned int num_experts,
    unsigned int num_local_experts,
    int myRank,
    int nRanks,
    bool is_ht_mode,
    bool is_expert_major,
    size_t expert_major_alignment,
    const int64_t* meta_expert_counts_padded = nullptr,
    const int64_t* meta_expert_offsets       = nullptr
) {
    (void)expert_major_alignment;
    if (is_ht_mode) {
        // EM requires both meta arrays; fall back to RM scan if missing.
        if (is_expert_major && meta_expert_offsets != nullptr && meta_expert_counts_padded != nullptr) {
            return validateDispatchOutputHTExpertMaj(alloc, dispatch_outputs,
                                                     max_tokens_per_rank, num_tokens_per_rank,
                                                     hidden, top_k,
                                                     num_experts, num_local_experts, myRank, nRanks,
                                                     meta_expert_counts_padded, meta_expert_offsets);
        }
        return validateDispatchOutputHTRankMaj(alloc, dispatch_outputs,
                                               max_tokens_per_rank, num_tokens_per_rank,
                                               hidden, top_k,
                                               num_experts, num_local_experts, myRank, nRanks);
    } else {
        if (is_expert_major) {
            return validateDispatchOutputLLExpertMaj(alloc, dispatch_outputs, dispatch_layout_info,
                                                     max_tokens_per_rank, num_tokens_per_rank,
                                                     hidden, top_k,
                                                     num_experts, num_local_experts, myRank, nRanks);
        } else {
            return validateDispatchOutputLLRankMaj(alloc, dispatch_outputs, dispatch_layout_info,
                                                   max_tokens_per_rank, num_tokens_per_rank,
                                                   hidden, top_k,
                                                   num_experts, num_local_experts, myRank, nRanks);
        }
    }
}

// Compute is_token_in_rank.sum() - count of unique ranks each token is sent to
// This matches DeepEP's validation approach
// Returns array of size num_tokens with unique rank count per token
int* countUniqueRanksPerToken(
    const int64_t* topk_idx_host,
    unsigned int num_tokens,
    unsigned int num_experts,
    unsigned int top_k,
    int nRanks
) {
    int* unique_ranks = new int[num_tokens]();  // Zero-initialized
    unsigned int num_local_experts = num_experts / nRanks;

    for (unsigned int t = 0; t < num_tokens; t++) {
        std::set<int> ranks_set;
        for (unsigned int k = 0; k < top_k; k++) {
            int64_t expert_id = topk_idx_host[t * top_k + k];
            if (expert_id >= 0) {
                int target_rank = expert_id / num_local_experts;
                ranks_set.insert(target_rank);
            }
        }
        unique_ranks[t] = ranks_set.size();
    }
    return unique_ranks;
}

// Count valid experts for each token (experts with topk_idx >= 0)
int countValidExperts(const int64_t* topk_idx_host, unsigned int token_idx, unsigned int top_k) {
    int count = 0;
    for (unsigned int k = 0; k < top_k; k++) {
        if (topk_idx_host[token_idx * top_k + k] >= 0) {
            count++;
        }
    }
    return count;
}

// Validate combine output for Low Latency mode
// DeepEP formula: check = combined / is_token_in_rank.sum()
// LL combine applies weighted sum: combined[t] = x[t] * sum(valid weights)
// Compared using calc_diff in double precision with threshold 1e-5
ValidationResult validateCombineOutputLL(
    const BenchmarkAllocState&    alloc,
    const ncclEpCombineOutputs_t& combine_outputs,
    ncclEpTensor_t*               topk_weights,
    unsigned int num_tokens,
    unsigned int hidden,
    unsigned int num_experts,
    unsigned int top_k,
    int myRank,
    int nRanks,
    int64_t* topk_idx_host
) {
    (void)num_experts;
    (void)nRanks;

    ValidationResult result = {true, 0, 0.0, ""};

    size_t output_size = num_tokens * hidden;
    uint16_t* combined_data = new uint16_t[output_size];
    {
        void* co_data;
        NCCLCHECK(epGetTensorData(alloc, combine_outputs.tokens, &co_data));
        CUDACHECK(cudaMemcpy(combined_data, co_data,
                             output_size * sizeof(uint16_t), cudaMemcpyDeviceToHost));
    }

    float* topk_weights_host = new float[num_tokens * top_k];
    void* tw_data_ll;
    NCCLCHECK(epGetTensorData(alloc, topk_weights, &tw_data_ll));
    CUDACHECK(cudaMemcpy(topk_weights_host, tw_data_ll,
                         num_tokens * top_k * sizeof(float), cudaMemcpyDeviceToHost));

    float original_rank_val = static_cast<float>(myRank - RANK_OFFSET);

    size_t num_elements = 0;
    for (unsigned int t = 0; t < num_tokens; t++) {
        if (countValidExperts(topk_idx_host, t, top_k) > 0)
            num_elements += hidden;
    }

    double* ref = new double[num_elements];
    double* actual = new double[num_elements];
    size_t idx = 0;

    bool has_nan = false;
    for (unsigned int t = 0; t < num_tokens; t++) {
        int nv = countValidExperts(topk_idx_host, t, top_k);
        if (nv == 0) continue;

        double weight_sum = 0;
        for (unsigned int k = 0; k < top_k; k++) {
            if (topk_idx_host[t * top_k + k] >= 0)
                weight_sum += static_cast<double>(topk_weights_host[t * top_k + k]);
        }

        double rank_val = static_cast<double>(original_rank_val);
        double token_hi_val = static_cast<double>(bf16ToFloat(floatToBf16(static_cast<float>(t / 256))));
        double token_lo_val = static_cast<double>(bf16ToFloat(floatToBf16(static_cast<float>(t % 256))));

        for (unsigned int h = 0; h < hidden; h++) {
            double orig;
            if (h == hidden - TOKEN_ID_COLS)
                orig = token_hi_val;
            else if (h > hidden - TOKEN_ID_COLS)
                orig = token_lo_val;
            else
                orig = rank_val;
            ref[idx] = orig * weight_sum;
            float actual_f = bf16ToFloat(combined_data[t * hidden + h]);
            actual[idx] = static_cast<double>(actual_f);
            if (std::isnan(actual_f)) has_nan = true;
            idx++;
        }
    }

    double diff = calc_diff(ref, actual, num_elements);
    result.max_diff = diff;
    result.passed = (diff < 1e-5) && !has_nan;

    if (!result.passed) {
        char buf[256];
        snprintf(buf, sizeof(buf), "LL combine: calc_diff=%.6e (threshold=1e-5)%s",
                 diff, has_nan ? ", NaN detected" : "");
        result.message = buf;
    }

    delete[] ref;
    delete[] actual;
    delete[] topk_weights_host;
    delete[] combined_data;
    return result;
}

// Validate combine output for High Throughput mode
// rank-major:    combined[t] = x[t] * num_unique_ranks  (one slot per dest rank)
// Expert-major: combined[t] = x[t] * num_valid_experts (one slot per expert, S2G-driven dup)
// Compared using calc_diff in double precision with threshold 5e-6
ValidationResult validateCombineOutputHT(
    const BenchmarkAllocState&    alloc,
    const ncclEpCombineOutputs_t& combine_outputs,
    unsigned int num_tokens,
    unsigned int hidden,
    unsigned int num_experts,
    unsigned int top_k,
    int myRank,
    int nRanks,
    int64_t* topk_idx_host,
    bool expert_major
) {
    ValidationResult result = {true, 0, 0.0, ""};

    size_t output_size = num_tokens * hidden;
    uint16_t* combined_data = new uint16_t[output_size];
    {
        void* co_data;
        NCCLCHECK(epGetTensorData(alloc, combine_outputs.tokens, &co_data));
        CUDACHECK(cudaMemcpy(combined_data, co_data,
                             output_size * sizeof(uint16_t), cudaMemcpyDeviceToHost));
    }

    // rank-major: one dispatch slot per destination rank → scale by unique ranks.
    // Expert-major: one dispatch slot per expert (S2G-driven dup) → scale by valid experts.
    int* unique_ranks = countUniqueRanksPerToken(topk_idx_host, num_tokens,
                                                  num_experts, top_k, nRanks);

    float original_rank_val = static_cast<float>(myRank - RANK_OFFSET);

    size_t num_elements = 0;
    for (unsigned int t = 0; t < num_tokens; t++) {
        int nr = expert_major ? countValidExperts(topk_idx_host, t, top_k)
                              : unique_ranks[t];
        if (nr > 0) num_elements += hidden;
    }

    double* ref = new double[num_elements];
    double* actual = new double[num_elements];
    size_t idx = 0;

    bool has_nan = false;
    for (unsigned int t = 0; t < num_tokens; t++) {
        int nr = expert_major ? countValidExperts(topk_idx_host, t, top_k)
                              : unique_ranks[t];
        if (nr == 0) continue;

        double rank_val = static_cast<double>(original_rank_val);
        double token_hi_val = static_cast<double>(bf16ToFloat(floatToBf16(static_cast<float>(t / 256))));
        double token_lo_val = static_cast<double>(bf16ToFloat(floatToBf16(static_cast<float>(t % 256))));
        double scale = static_cast<double>(nr);

        for (unsigned int h = 0; h < hidden; h++) {
            double orig;
            if (h == hidden - TOKEN_ID_COLS)
                orig = token_hi_val;
            else if (h > hidden - TOKEN_ID_COLS)
                orig = token_lo_val;
            else
                orig = rank_val;
            ref[idx] = orig * scale;
            float actual_f = bf16ToFloat(combined_data[t * hidden + h]);
            actual[idx] = static_cast<double>(actual_f);
            if (std::isnan(actual_f)) has_nan = true;
            idx++;
        }
    }

    double diff = calc_diff(ref, actual, num_elements);
    result.max_diff = diff;
    result.passed = (diff < 5e-6) && !has_nan;

    if (!result.passed) {
        char buf[256];
        snprintf(buf, sizeof(buf), "HT combine: calc_diff=%.6e (threshold=5e-6)%s",
                 diff, has_nan ? ", NaN detected" : "");
        result.message = buf;
    }

    delete[] ref;
    delete[] actual;
    delete[] unique_ranks;
    delete[] combined_data;
    return result;
}

// Wrapper that calls appropriate validation based on mode
ValidationResult validateCombineOutput(
    const BenchmarkAllocState&    alloc,
    const ncclEpCombineOutputs_t& combine_outputs,
    ncclEpTensor_t*               topk_weights,
    unsigned int num_tokens,
    unsigned int hidden,
    unsigned int top_k,
    unsigned int num_experts,
    int myRank,
    int nRanks,
    bool is_ht_mode,
    int64_t* topk_idx_host,
    bool expert_major = false
) {
    if (is_ht_mode) {
        return validateCombineOutputHT(alloc, combine_outputs,
                                       num_tokens, hidden, num_experts,
                                       top_k, myRank, nRanks, topk_idx_host,
                                       expert_major);
    } else {
        return validateCombineOutputLL(alloc, combine_outputs, topk_weights,
                                       num_tokens, hidden, num_experts,
                                       top_k, myRank, nRanks, topk_idx_host);
    }
}

// Benchmark result structure
struct BenchResult {
    double avg_ms;
    double min_ms;
    double max_ms;
    double throughput_gbps;
};

// Structure to hold paired dispatch+combine benchmark results
struct PairedBenchResult {
    BenchResult dispatch;
    BenchResult combine;
    BenchResult total;
};

// Run paired dispatch+combine benchmark with separate timing for each phase
// This ensures dispatch and combine are always paired (required for correctness)
// while still measuring individual performance
PairedBenchResult runPairedBenchmark(
    std::function<void()> dispatch_fn,
    std::function<void()> combine_fn,
    int num_warmup,
    int num_iters,
    size_t dispatch_bytes,
    size_t combine_bytes,
    KernelTimer& ktimer,
    cudaStream_t stream
) {
    // Warmup with paired dispatch+combine
    // Note: cudaStreamSynchronize between dispatch and combine is required for HT mode
    // MPI_Barrier at end of each iteration ensures all ranks stay in sync (critical for HT mode)
    for (int i = 0; i < num_warmup; i++) {
        dispatch_fn();
        CUDACHECK(cudaStreamSynchronize(stream));
        combine_fn();
        CUDACHECK(cudaStreamSynchronize(stream));
        MPICHECK(MPI_Barrier(MPI_COMM_WORLD));
    }

    // Create events for dispatch, combine, and total timing
    std::vector<cudaEvent_t> dispatch_start(num_iters);
    std::vector<cudaEvent_t> dispatch_end(num_iters);
    std::vector<cudaEvent_t> combine_start(num_iters);
    std::vector<cudaEvent_t> combine_end(num_iters);

    for (int i = 0; i < num_iters; i++) {
        CUDACHECK(cudaEventCreate(&dispatch_start[i]));
        CUDACHECK(cudaEventCreate(&dispatch_end[i]));
        CUDACHECK(cudaEventCreate(&combine_start[i]));
        CUDACHECK(cudaEventCreate(&combine_end[i]));
    }

    // Start CUPTI kernel timer
    ktimer.start();

    // Run paired benchmark with individual timing
    // Events are recorded immediately after kernel launch (before sync) to measure GPU time only
    // Sync happens after event recording to not affect timing
    // MPI_Barrier at end of each iteration ensures all ranks stay in sync (critical for HT mode)
    for (int i = 0; i < num_iters; i++) {
        CUDACHECK(cudaEventRecord(dispatch_start[i], stream));
        dispatch_fn();
        CUDACHECK(cudaEventRecord(dispatch_end[i], stream));    // Record before sync
        CUDACHECK(cudaStreamSynchronize(stream));              // Sync outside timing
        CUDACHECK(cudaEventRecord(combine_start[i], stream));  // Record after sync, before combine
        combine_fn();
        CUDACHECK(cudaEventRecord(combine_end[i], stream));    // Record before sync
        CUDACHECK(cudaStreamSynchronize(stream));             // Sync outside timing
        MPICHECK(MPI_Barrier(MPI_COMM_WORLD));
    }

    // Stop CUPTI kernel timer
    ktimer.stop();

    // Collect times
    std::vector<float> dispatch_times(num_iters);
    std::vector<float> combine_times(num_iters);
    std::vector<float> total_times(num_iters);

    for (int i = 0; i < num_iters; i++) {
        CUDACHECK(cudaEventElapsedTime(&dispatch_times[i], dispatch_start[i], dispatch_end[i]));
        CUDACHECK(cudaEventElapsedTime(&combine_times[i], combine_start[i], combine_end[i]));
        CUDACHECK(cudaEventElapsedTime(&total_times[i], dispatch_start[i], combine_end[i]));
    }

    // Cleanup events
    for (int i = 0; i < num_iters; i++) {
        CUDACHECK(cudaEventDestroy(dispatch_start[i]));
        CUDACHECK(cudaEventDestroy(dispatch_end[i]));
        CUDACHECK(cudaEventDestroy(combine_start[i]));
        CUDACHECK(cudaEventDestroy(combine_end[i]));
    }

    // Helper to calculate stats from times vector (skip first iteration if we have more than 1)
    auto calc_stats = [](const std::vector<float>& times, size_t data_bytes) -> BenchResult {
        // For HT mode with only 1 iteration, don't skip any - use all data
        // For LL mode with multiple iterations, skip the first (warmup outlier)
        std::vector<float> times_trimmed;
        if (times.size() > 1) {
            times_trimmed.assign(times.begin() + 1, times.end());
        } else {
            times_trimmed = times;  // Use all data when we only have 1 iteration
        }

        BenchResult result;
        if (times_trimmed.empty()) {
            result.avg_ms = 0;
            result.min_ms = 0;
            result.max_ms = 0;
            result.throughput_gbps = 0;
        } else {
            result.avg_ms = std::accumulate(times_trimmed.begin(), times_trimmed.end(), 0.0) / times_trimmed.size();
            result.min_ms = *std::min_element(times_trimmed.begin(), times_trimmed.end());
            result.max_ms = *std::max_element(times_trimmed.begin(), times_trimmed.end());
            result.throughput_gbps = (data_bytes / 1e9) / (result.avg_ms / 1000.0);
        }
        return result;
    };

    PairedBenchResult result;
    result.dispatch = calc_stats(dispatch_times, dispatch_bytes);
    result.combine = calc_stats(combine_times, combine_bytes);
    result.total = calc_stats(total_times, dispatch_bytes + combine_bytes);

    return result;
}

// Structure to hold Low Latency byte calculation
// Matches DeepEP test_low_latency.py methodology
struct LowLatencyBytes {
    size_t dispatch_bytes;  // BF16 format per selection
    size_t combine_bytes;   // BF16 format: hidden * 2 per selection
    unsigned int num_valid_selections;
};

// Calculate bytes for Low Latency mode (dispatch and combine are both BF16)
LowLatencyBytes calculateLowLatencyBytes(
    const int64_t* topk_idx_host,
    unsigned int num_tokens,
    unsigned int top_k,
    unsigned int hidden
) {
    LowLatencyBytes bytes = {0, 0, 0};

    // Count valid selections (non-masked entries)
    for (unsigned int i = 0; i < num_tokens * top_k; i++) {
        if (topk_idx_host[i] >= 0) {
            bytes.num_valid_selections++;
        }
    }

    // BF16 bytes per selection: hidden * 2
    const size_t bf16_bytes_per_selection = hidden * 2;

    // Dispatch and combine are both BF16
    bytes.dispatch_bytes = static_cast<size_t>(bytes.num_valid_selections) * bf16_bytes_per_selection;
    bytes.combine_bytes = static_cast<size_t>(bytes.num_valid_selections) * bf16_bytes_per_selection;

    return bytes;
}

// Six bandwidth metrics for High Throughput mode, all dividing by measured time t:
//
//  Send-side (this rank dispatching tokens to experts):
//   total_send  = total_send_bytes / t   — all destinations (NVL+RDMA)
//   nvl_send    = nvl_send_bytes / t     — local node only (NVLink)
//   rdma_send   = rdma_send_bytes / t    — remote nodes only (RDMA outbound)
//
//  Recv-side (this rank's experts receiving tokens):
//   total_recv  = total_recv_bytes / t   — all sources (NVL+RDMA)
//   nvl_recv    = nvl_recv_bytes / t     — from local ranks (NVLink)
//   rdma_recv   = rdma_recv_bytes / t    — from remote ranks (RDMA inbound)
//
//  Derived: nvl_send = total_send - rdma_send
//           nvl_recv = total_recv - rdma_recv
struct HighThroughputBytes {
    size_t total_send_bytes;     // NVL + RDMA outbound
    size_t rdma_send_bytes;      // RDMA outbound only
    size_t total_recv_bytes;     // NVL + RDMA inbound
    size_t rdma_recv_bytes;      // RDMA inbound only (from remote ranks)
    unsigned int total_send_tokens;
    unsigned int rdma_send_tokens;
    unsigned int rdma_recv_tokens;
    unsigned int total_recv_tokens;
};

// Calculate all six byte metrics from topk_idx for High Throughput mode.
//
// Send side: count unique (token, node) pairs this rank sends to.
//   total_send_tokens = all nodes (local + remote)
//   rdma_send_tokens  = remote nodes only
//
// Recv side: simulate all source ranks' randperm routing (deterministic from
// seed = src_rank + 42) to count unique (src_rank, token) pairs where at least
// one selected expert belongs to myRank.
//   total_recv_tokens = all source ranks (NVL + RDMA)
//   rdma_recv_tokens = remote source ranks only
HighThroughputBytes calculateHighThroughputBytes(
    const int64_t* topk_idx_host,
    unsigned int num_tokens,
    const unsigned int* num_tokens_per_rank,
    unsigned int top_k,
    unsigned int num_experts,
    unsigned int hidden,
    int myRank,
    int nRanks,
    int num_ranks_per_node
) {
    HighThroughputBytes bytes = {0, 0, 0, 0, 0, 0, 0, 0};

    int num_nodes = (nRanks + num_ranks_per_node - 1) / num_ranks_per_node;
    unsigned int num_experts_per_node = static_cast<unsigned int>(num_experts / num_nodes);
    int local_node = myRank / num_ranks_per_node;
    unsigned int num_experts_per_rank = num_experts / static_cast<unsigned int>(nRanks);

    // Send side: count unique (token, node) pairs from this rank's topk_idx
    // A token routed to multiple experts on the same node is counted only once, even though
    // NCCL EP sends it to each target rank individually via NVLink P2P (not once per node).
    // TODO: switch to per-rank counting for both nvl_send and nvl_recv.
    for (unsigned int t = 0; t < num_tokens; t++) {
        std::set<int> nodes_for_token;
        for (unsigned int k = 0; k < top_k; k++) {
            int64_t expert_id = topk_idx_host[t * top_k + k];
            if (expert_id < 0) continue;
            int target_node = static_cast<int>(expert_id / num_experts_per_node);
            if (nodes_for_token.insert(target_node).second) {
                bytes.total_send_tokens++;
                if (target_node != local_node)
                    bytes.rdma_send_tokens++;
            }
        }
    }

    // Recv side: replay every source rank's randperm routing to count tokens
    // received by myRank. This is deterministic because each rank uses the
    // same seed (src_rank + 42) and same shuffle algorithm.
    // Each (src_rank, token) pair is counted once regardless of how many experts on myRank it targets.
    std::vector<int64_t> src_perm(num_experts);
    for (int src_rank = 0; src_rank < nRanks; src_rank++) {
        int src_node = src_rank / num_ranks_per_node;
        bool is_rdma = (src_node != local_node);
        unsigned int src_tokens = num_tokens_per_rank[src_rank];

        std::mt19937 src_gen(src_rank + 42);
        std::iota(src_perm.begin(), src_perm.end(), 0);
        for (unsigned int t = 0; t < src_tokens; t++) {
            std::shuffle(src_perm.begin(), src_perm.end(), src_gen);
            for (unsigned int k = 0; k < top_k; k++) {
                int target_rank = static_cast<int>(src_perm[k] / num_experts_per_rank);
                if (target_rank == myRank) {
                    bytes.total_recv_tokens++;
                    if (is_rdma) bytes.rdma_recv_tokens++;
                    break;
                }
            }
        }
    }

    const size_t bytes_per_token = hidden * 2;  // BF16

    bytes.total_send_bytes = bytes.total_send_tokens * bytes_per_token;
    bytes.rdma_send_bytes  = bytes.rdma_send_tokens  * bytes_per_token;
    bytes.total_recv_bytes = bytes.total_recv_tokens   * bytes_per_token;
    bytes.rdma_recv_bytes  = bytes.rdma_recv_tokens   * bytes_per_token;

    return bytes;
}

// Print benchmark results with MPI aggregation across ranks
// Print benchmark results for Low Latency mode
// Uses BF16 bytes for both dispatch and combine
void printLowLatencyResults(
    int myRank,
    int nRanks,
    const BenchResult& dispatch_result,
    const BenchResult& combine_result,
    const BenchResult& combined_result,
    KernelTimer& ktimer,
    const LowLatencyBytes& ll_bytes
) {

    // Uncomment for detailed per-rank results
    // // Print per-rank results
    // printf("[Rank %d] Dispatch:         avg=%.2f us, min=%.2f us, max=%.2f us, throughput=%.2f GB/s\n",
    //        myRank,
    //        dispatch_result.avg_ms * 1000, dispatch_result.min_ms * 1000, dispatch_result.max_ms * 1000,
    //        dispatch_result.throughput_gbps);

    // printf("[Rank %d] Combine:          avg=%.2f us, min=%.2f us, max=%.2f us, throughput=%.2f GB/s\n",
    //        myRank,
    //        combine_result.avg_ms * 1000, combine_result.min_ms * 1000, combine_result.max_ms * 1000,
    //        combine_result.throughput_gbps);

    // printf("[Rank %d] Dispatch+Combine: avg=%.2f us, min=%.2f us, max=%.2f us, throughput=%.2f GB/s\n",
    //        myRank,
    //        combined_result.avg_ms * 1000, combined_result.min_ms * 1000, combined_result.max_ms * 1000,
    //        combined_result.throughput_gbps);

    // Aggregate latency results across ranks
    double local_dispatch_avg = dispatch_result.avg_ms;
    double local_dispatch_min = dispatch_result.min_ms;
    double local_dispatch_max = dispatch_result.max_ms;
    double local_combine_avg = combine_result.avg_ms;
    double local_combine_min = combine_result.min_ms;
    double local_combine_max = combine_result.max_ms;
    double local_total_avg = combined_result.avg_ms;
    double local_total_min = combined_result.min_ms;
    double local_total_max = combined_result.max_ms;

    double global_dispatch_avg, global_dispatch_min, global_dispatch_max;
    double global_combine_avg, global_combine_min, global_combine_max;
    double global_total_avg, global_total_min, global_total_max;

    MPI_Reduce(&local_dispatch_avg, &global_dispatch_avg, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_dispatch_min, &global_dispatch_min, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_dispatch_max, &global_dispatch_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_combine_avg, &global_combine_avg, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_combine_min, &global_combine_min, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_combine_max, &global_combine_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_total_avg, &global_total_avg, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_total_min, &global_total_min, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_total_max, &global_total_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    // Gather throughput min/max with rank info using MPI_MINLOC/MPI_MAXLOC
    struct { double value; int rank; } local_dispatch_tp, local_combine_tp, local_total_tp;
    struct { double value; int rank; } global_dispatch_tp_min, global_dispatch_tp_max;
    struct { double value; int rank; } global_combine_tp_min, global_combine_tp_max;
    struct { double value; int rank; } global_total_tp_min, global_total_tp_max;

    local_dispatch_tp.value = dispatch_result.throughput_gbps;
    local_dispatch_tp.rank = myRank;
    local_combine_tp.value = combine_result.throughput_gbps;
    local_combine_tp.rank = myRank;
    local_total_tp.value = combined_result.throughput_gbps;
    local_total_tp.rank = myRank;

    MPI_Reduce(&local_dispatch_tp, &global_dispatch_tp_min, 1, MPI_DOUBLE_INT, MPI_MINLOC, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_dispatch_tp, &global_dispatch_tp_max, 1, MPI_DOUBLE_INT, MPI_MAXLOC, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_combine_tp, &global_combine_tp_min, 1, MPI_DOUBLE_INT, MPI_MINLOC, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_combine_tp, &global_combine_tp_max, 1, MPI_DOUBLE_INT, MPI_MAXLOC, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_total_tp, &global_total_tp_min, 1, MPI_DOUBLE_INT, MPI_MINLOC, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_total_tp, &global_total_tp_max, 1, MPI_DOUBLE_INT, MPI_MAXLOC, 0, MPI_COMM_WORLD);



    double dispatch_kernel_avg = 0.0, combine_kernel_avg = 0.0;
    double dispatch_kernel_min = 0.0, combine_kernel_min = 0.0;
    double dispatch_kernel_max = 0.0, combine_kernel_max = 0.0;
    if (ktimer.is_valid()) {
        double local_disp_kern = ktimer.get_avg_us("dispatch");
        double local_comb_kern = ktimer.get_avg_us("combine");
        double global_disp_kern = 0.0, global_comb_kern = 0.0;
        MPI_Reduce(&local_disp_kern, &global_disp_kern, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
        MPI_Reduce(&local_comb_kern, &global_comb_kern, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
        dispatch_kernel_avg = global_disp_kern / nRanks;
        combine_kernel_avg = global_comb_kern / nRanks;
        MPI_Reduce(&local_disp_kern, &dispatch_kernel_min, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
        MPI_Reduce(&local_comb_kern, &combine_kernel_min, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
        MPI_Reduce(&local_disp_kern, &dispatch_kernel_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        MPI_Reduce(&local_comb_kern, &combine_kernel_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    }


    // Print summary on rank 0
    if (myRank == 0) {
        global_dispatch_avg /= nRanks;
        global_combine_avg /= nRanks;
        global_total_avg /= nRanks;

        double total_data_bytes = ll_bytes.dispatch_bytes + ll_bytes.combine_bytes;
        double avg_dispatch_tp = (ll_bytes.dispatch_bytes / 1e9) / (global_dispatch_avg / 1000.0);
        double avg_combine_tp = (ll_bytes.combine_bytes / 1e9) / (global_combine_avg / 1000.0);
        double avg_total_tp = (total_data_bytes / 1e9) / (global_total_avg / 1000.0);

        printf("\n=== Summary (Low Latency, across %d ranks) ===\n", nRanks);

        printf("\n--- Host-observed performance ---\n");

        printf("Dispatch (BF16):  avg=%.2f us, min=%.2f us, max=%.2f us\n",
               global_dispatch_avg * 1000,
               global_dispatch_min * 1000,
               global_dispatch_max * 1000);
        printf("                  throughput: avg=%.2f GB/s, min=%.2f GB/s (rank %d), max=%.2f GB/s (rank %d)\n",
               avg_dispatch_tp,
               global_dispatch_tp_min.value, global_dispatch_tp_min.rank,
               global_dispatch_tp_max.value, global_dispatch_tp_max.rank);
        printf("Combine (BF16):   avg=%.2f us, min=%.2f us, max=%.2f us\n",
               global_combine_avg * 1000,
               global_combine_min * 1000,
               global_combine_max * 1000);
        printf("                  throughput: avg=%.2f GB/s, min=%.2f GB/s (rank %d), max=%.2f GB/s (rank %d)\n",
               avg_combine_tp,
               global_combine_tp_min.value, global_combine_tp_min.rank,
               global_combine_tp_max.value, global_combine_tp_max.rank);
        printf("Total (D+C):      avg=%.2f us, min=%.2f us, max=%.2f us\n",
               global_total_avg * 1000,
               global_total_min * 1000,
               global_total_max * 1000);
        printf("                  throughput: avg=%.2f GB/s, min=%.2f GB/s (rank %d), max=%.2f GB/s (rank %d)\n",
               avg_total_tp,
               global_total_tp_min.value, global_total_tp_min.rank,
               global_total_tp_max.value, global_total_tp_max.rank);

        printf("\n--- Kernel-only performance ---\n");
        if (ktimer.is_valid()) {
            printf("Dispatch:    avg=%.2f us, min=%.2f us, max=%.2f us\n",
                dispatch_kernel_avg,
                dispatch_kernel_min,
                dispatch_kernel_max);
            printf("                  throughput: avg=%.2f GB/s, min=%.2f GB/s, max=%.2f GB/s\n",
                    (ll_bytes.dispatch_bytes / 1e9) / (dispatch_kernel_avg / 1e6),
                    (ll_bytes.dispatch_bytes / 1e9) / (dispatch_kernel_min / 1e6),
                    (ll_bytes.dispatch_bytes / 1e9) / (dispatch_kernel_max / 1e6));
            printf("Combine:     avg=%.2f us, min=%.2f us, max=%.2f us\n",
                combine_kernel_avg,
                combine_kernel_min,
                combine_kernel_max);
            printf("                  throughput: avg=%.2f GB/s, min=%.2f GB/s, max=%.2f GB/s\n",
                    (ll_bytes.combine_bytes / 1e9) / (combine_kernel_avg / 1e6),
                    (ll_bytes.combine_bytes / 1e9) / (combine_kernel_min / 1e6),
                    (ll_bytes.combine_bytes / 1e9) / (combine_kernel_max / 1e6));
        } else {
            printf("  NOTE: CUPTI support was not compiled.\n");
        }

        printf("\nByte counts: dispatch=%.2f MB (BF16), combine=%.2f MB (BF16), selections=%u\n",
               ll_bytes.dispatch_bytes / 1e6,
               ll_bytes.combine_bytes / 1e6,
               ll_bytes.num_valid_selections);
        fflush(stdout);
    }
}

// Print results for High Throughput mode.
// local_kernel_dk_us / local_kernel_ck_us: per-rank CUPTI kernel times for the per-rank lines.
// All "global_*" parameters are raw MPI_SUM across all ranks (rank 0 only; 0 on other ranks).
// The function averages them by nRanks before use.
void printHighThroughputResults(
    int myRank,
    int nRanks,
    const BenchResult& dispatch_result,
    const BenchResult& combine_result,
    const BenchResult& combined_result,
    KernelTimer& ktimer,
    const HighThroughputBytes& ht_bytes,
    size_t global_total_send, size_t global_rdma_send,
    size_t global_total_recv, size_t global_rdma_recv
) {
    double local_dispatch_avg = dispatch_result.avg_ms;
    double local_dispatch_min = dispatch_result.min_ms;
    double local_dispatch_max = dispatch_result.max_ms;
    double local_combine_avg = combine_result.avg_ms;
    double local_combine_min = combine_result.min_ms;
    double local_combine_max = combine_result.max_ms;
    double local_total_avg = combined_result.avg_ms;
    double local_total_min = combined_result.min_ms;
    double local_total_max = combined_result.max_ms;

    double global_dispatch_avg, global_dispatch_min, global_dispatch_max;
    double global_combine_avg, global_combine_min, global_combine_max;
    double global_total_avg, global_total_min, global_total_max;

    MPI_Reduce(&local_dispatch_avg, &global_dispatch_avg, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_dispatch_min, &global_dispatch_min, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_dispatch_max, &global_dispatch_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_combine_avg, &global_combine_avg, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_combine_min, &global_combine_min, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_combine_max, &global_combine_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_total_avg, &global_total_avg, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_total_min, &global_total_min, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_total_max, &global_total_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);


    // Obtain kernel times from CUPTI if available
    double global_kernel_dk_us = 0.0, global_kernel_ck_us = 0.0;
    double local_dispatch_kernel_us = 0.0;
    double local_combine_kernel_us  = 0.0;
    if (ktimer.is_valid()) {
        local_dispatch_kernel_us = ktimer.get_avg_us("dispatch_kernel");
        local_combine_kernel_us  = ktimer.get_avg_us("combine_kernel");
        MPI_Reduce(&local_dispatch_kernel_us, &global_kernel_dk_us, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
        MPI_Reduce(&local_combine_kernel_us,  &global_kernel_ck_us, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    }

    // Uncomment for debugging
    // printf("[Rank %d] Dispatch:         total=%.2f us  kernel=%.2f us\n",
    //     myRank, dispatch_result.avg_ms * 1000, local_dispatch_kernel_us);
    // printf("[Rank %d] Combine:          total=%.2f us  kernel=%.2f us\n",
    //     myRank, combine_result.avg_ms * 1000, local_combine_kernel_us);
    // printf("[Rank %d] Dispatch+Combine: total=%.2f us\n", myRank, combined_result.avg_ms * 1000);


    if (myRank == 0) {
        global_dispatch_avg /= nRanks;
        global_combine_avg  /= nRanks;
        global_total_avg    /= nRanks;

        double avg_total_send = static_cast<double>(global_total_send) / nRanks;
        double avg_rdma_send  = static_cast<double>(global_rdma_send)  / nRanks;
        double avg_total_recv = static_cast<double>(global_total_recv) / nRanks;
        double avg_rdma_recv  = static_cast<double>(global_rdma_recv)  / nRanks;
        double avg_nvl_send   = avg_total_send - avg_rdma_send;
        double avg_nvl_recv   = avg_total_recv - avg_rdma_recv;

        double dk_total_s = global_dispatch_avg * 1e-3;  // avg total dispatch time in seconds
        double ck_total_s = global_combine_avg  * 1e-3;

        printf("\n=== Summary (High Throughput BF16, across %d ranks) ===\n", nRanks);
        printf("NOTE: total time = kernel time + memcpyD2D + misc\n");

        // --- BW based on total time ---
        printf("--- BW based on total time ---\n");
        printf("Dispatch:    total=%.2f us (min=%.2f, max=%.2f)\n",
               global_dispatch_avg * 1000, global_dispatch_min * 1000, global_dispatch_max * 1000);
        if (dk_total_s > 0) {
            printf("             recv: total_bw=%.2f  nvl_bw=%.2f  rdma_bw=%.2f GB/s\n",
                   (avg_total_recv / 1e9) / dk_total_s, (avg_nvl_recv / 1e9) / dk_total_s, (avg_rdma_recv / 1e9) / dk_total_s);
            printf("             send: total_bw=%.2f  nvl_bw=%.2f  rdma_bw=%.2f GB/s\n",
                   (avg_total_send / 1e9) / dk_total_s, (avg_nvl_send / 1e9) / dk_total_s, (avg_rdma_send / 1e9) / dk_total_s);
        }
        printf("Combine:     total=%.2f us (min=%.2f, max=%.2f)\n",
               global_combine_avg * 1000, global_combine_min * 1000, global_combine_max * 1000);
        if (ck_total_s > 0) {
            printf("             send: total_bw=%.2f  nvl_bw=%.2f  rdma_bw=%.2f GB/s\n",
                   (avg_total_recv / 1e9) / ck_total_s, (avg_nvl_recv / 1e9) / ck_total_s, (avg_rdma_recv / 1e9) / ck_total_s);
            printf("             recv: total_bw=%.2f  nvl_bw=%.2f  rdma_bw=%.2f GB/s\n",
                   (avg_total_send / 1e9) / ck_total_s, (avg_nvl_send / 1e9) / ck_total_s, (avg_rdma_send / 1e9) / ck_total_s);
        }
        printf("Total (D+C): avg=%.2f us, min=%.2f us, max=%.2f us\n",
               global_total_avg * 1000, global_total_min * 1000, global_total_max * 1000);

        // --- BW based on kernel time ---
        printf("\n--- BW based on kernel time ---\n");
        if (ktimer.is_valid()) {
            double avg_kernel_dk_us = global_kernel_dk_us / nRanks;
            double avg_kernel_ck_us = global_kernel_ck_us / nRanks;
            double dk_s       = avg_kernel_dk_us / 1e6;
            double ck_s       = avg_kernel_ck_us / 1e6;
            printf("Dispatch:    kernel=%.2f us\n", avg_kernel_dk_us);
            printf("             recv: total_bw=%.2f  nvl_bw=%.2f  rdma_bw=%.2f GB/s\n",
                (avg_total_recv / 1e9) / dk_s, (avg_nvl_recv / 1e9) / dk_s, (avg_rdma_recv / 1e9) / dk_s);
            printf("             send: total_bw=%.2f  nvl_bw=%.2f  rdma_bw=%.2f GB/s\n",
                (avg_total_send / 1e9) / dk_s, (avg_nvl_send / 1e9) / dk_s, (avg_rdma_send / 1e9) / dk_s);

            printf("Combine:     kernel=%.2f us\n", avg_kernel_ck_us);
            printf("             send: total_bw=%.2f  nvl_bw=%.2f  rdma_bw=%.2f GB/s\n",
                (avg_total_recv / 1e9) / ck_s, (avg_nvl_recv / 1e9) / ck_s, (avg_rdma_recv / 1e9) / ck_s);
            printf("             recv: total_bw=%.2f  nvl_bw=%.2f  rdma_bw=%.2f GB/s\n",
                (avg_total_send / 1e9) / ck_s, (avg_nvl_send / 1e9) / ck_s, (avg_rdma_send / 1e9) / ck_s);
            printf("Total (D+C): kernel=%.2f us\n", avg_kernel_dk_us + avg_kernel_ck_us);
        } else {
            printf("  NOTE: CUPTI support was not compiled.\n");
        }

        printf("\nByte counts (per rank avg): total_send=%.2f MB (%u tokens), rdma_send=%.2f MB (%u tokens), "
                "rdma_recv=%.2f MB (%u tokens), total_recv=%.2f MB (%u tokens)\n",
                avg_total_send / 1e6, ht_bytes.total_send_tokens,
                avg_rdma_send  / 1e6, ht_bytes.rdma_send_tokens,
                avg_rdma_recv  / 1e6, ht_bytes.rdma_recv_tokens,
                avg_total_recv / 1e6, ht_bytes.total_recv_tokens);

    }
}

// Run NVTX profiling with labeled ranges for nsys analysis.
// Profiles one HandleCreate (to see AG + metadata processing) followed by
// num_iters paired Dispatch+Combine iterations.
void runNvtxProfiling(
    int myRank,
    int num_iters,
    std::function<void()> dispatch_fn,
    std::function<void()> combine_fn,
    std::function<void()> handle_create_fn,
    cudaStream_t stream
) {
    if (myRank == 0) {
        printf("\n=== NVTX Profiling Mode ===\n");
        printf("Run with: nsys profile --stats=true mpirun ...\n\n");
    }

    MPICHECK(MPI_Barrier(MPI_COMM_WORLD));
    CUDACHECK(cudaStreamSynchronize(stream));

    // Start CUDA profiler (for nsys --capture-range=cudaProfilerApi)
    cudaProfilerStart();

    // Profile HandleCreate to expose AG and metadata processing phases
    nvtxRangePush("HandleCreate");
    handle_create_fn();
    CUDACHECK(cudaStreamSynchronize(stream));
    nvtxRangePop();

    MPICHECK(MPI_Barrier(MPI_COMM_WORLD));

    // Profile paired dispatch+combine iterations with individual labels
    // Note: cudaStreamSynchronize between dispatch and combine is required for HT mode
    // MPI_Barrier at end of each iteration ensures all ranks stay in sync (critical for HT mode)
    nvtxRangePush("Paired Dispatch+Combine Benchmark");
    for (int i = 0; i < num_iters; i++) {
        nvtxRangePush("Dispatch");
        dispatch_fn();
        CUDACHECK(cudaStreamSynchronize(stream));
        nvtxRangePop();
        nvtxRangePush("Combine");
        combine_fn();
        CUDACHECK(cudaStreamSynchronize(stream));
        nvtxRangePop();
        MPICHECK(MPI_Barrier(MPI_COMM_WORLD));
    }
    nvtxRangePop();  // Paired Dispatch+Combine Benchmark

    cudaProfilerStop();

    if (myRank == 0) {
        printf("Profiling complete. Analyze with nsys-ui or nsys stats.\n");
    }
}

// Generate topk indices for LL mode (consistent with DeepEP test_low_latency.py)
// abs(randn)+1 scores → topk selection → random -1 masking (simulates dropped tokens)
void generateRandomTopkIndicesLL(
    int64_t* topk_idx_host,
    unsigned int num_tokens,
    unsigned int num_experts,
    unsigned int top_k,
    int rank,
    int seed
) {
    // Seed with (seed + rank) for reproducibility across ranks
    std::mt19937 gen(seed + rank);
    std::normal_distribution<float> dist(0.0f, 1.0f);

    std::vector<std::pair<float, int>> score_idx(num_experts);

    for (unsigned int i = 0; i < num_tokens; i++) {
        // Generate random scores: abs(randn) + 1
        for (unsigned int e = 0; e < num_experts; e++) {
            float score = std::abs(dist(gen)) + 1.0f;
            score_idx[e] = {score, static_cast<int>(e)};
        }

        // Partial sort to get top-k (largest scores first)
        std::partial_sort(score_idx.begin(), score_idx.begin() + top_k, score_idx.end(),
                          [](const auto& a, const auto& b) { return a.first > b.first; });

        // Extract top-k expert indices (sorted by score, descending)
        for (unsigned int j = 0; j < top_k; j++) {
            topk_idx_host[i * top_k + j] = score_idx[j].second;
        }
    }

    // Randomly mask 10 positions with -1 (simulates dropped tokens).
    // Guarded on num_tokens > 0: zero-token ranks have no slots to mask, and
    // the distribution upper bound `num_tokens - 1` would underflow to
    // UINT_MAX on an unsigned 0, then index out-of-bounds into the empty
    // topk_idx_host buffer.
    if (num_tokens > 0) {
        std::uniform_int_distribution<unsigned int> token_dist(0, num_tokens - 1);
        std::uniform_int_distribution<unsigned int> topk_dist(0, top_k - 1);
        for (int i = 0; i < 10; i++) {
            unsigned int token_idx = token_dist(gen);
            unsigned int k_idx = topk_dist(gen);
            topk_idx_host[token_idx * top_k + k_idx] = -1;
        }
    }
}

// Per-rank token counts for the non-uniform-tokens sub-test.
// Same seed on every rank so all ranks agree without MPI exchange.
// Rank 0 pinned to max_tokens; last rank pinned to 1 (epMakeTensor rejects 0-dim).
static std::vector<unsigned int> computeNonUniformTokensPerRank(
    unsigned int max_tokens,
    int nRanks,
    unsigned int seed = 0xEB12345u
) {
    std::vector<unsigned int> out(nRanks);
    if (max_tokens == 0) {
        std::fill(out.begin(), out.end(), 0u);
        return out;
    }
    std::mt19937 rng(seed);
    // Sample inclusive of 0 so the asymmetric zero-tokens path is exercised.
    // The kernel's per-token loops degrade to no-ops at num_tokens=0 and
    // tensorHasBinding now accepts empty tensors, so a zero-tokens rank is
    // a valid configuration that must not regress.
    std::uniform_int_distribution<unsigned int> dist(0u, max_tokens);
    for (int r = 0; r < nRanks; r++) {
        out[r] = dist(rng);
    }
    out[0] = max_tokens;
    // Force at least one rank to 0 so the regression case is always exercised
    // (the random sample alone may not hit 0 for small nRanks).
    if (nRanks > 1) out[nRanks - 1] = 0u;
    return out;
}

void printUsage(const char* programName, int myRank) {
    if (myRank == 0) {
        printf("Usage: %s [OPTIONS]\n", programName);
        printf("Performance benchmark for NCCL EP operations\n\n");
        printf("Options:\n");
        printf("  --algorithm <mode>      Algorithm mode (default: ll)\n");
        printf("                          ll or low-latency:  Low latency mode\n");
        printf("                          ht or high-throughput:  High throughput mode\n");
        printf("  --layout <layout>       Buffer layout\n");
        printf("                          em or expert-major:  Expert-major layout (LL only, default for LL)\n");
        printf("                          rm or rank-major:    Rank-major layout (LL only)\n");
        printf("                          fl or flat:          Flat layout (HT only, default for HT)\n");
        printf("  --tokens <num>          Number of tokens (default: LL=128, HT=4096)\n");
        printf("  --dispatch-less-than-max-tokens <M>  Per-rank dispatch count M (M in [1, --tokens]; default = --tokens)\n");
        printf("  --non-uniform-tokens    Per-rank dispatch count random in [1, --tokens]; mutually exclusive with --dispatch-less-than-max-tokens\n");
        printf("  --hidden <num>          Hidden dimension (default: 7168)\n");
        printf("  --top-k <num>           Top-k experts per token (default: 8)\n");
        printf("  --experts <num>         Total number of experts (default: 256)\n");
        printf("  --warmup <num>          Warmup iterations (default: 10)\n");
        printf("  --iters <num>           Benchmark iterations (default: 50)\n");
        printf("  --user-handle-mem       Use caller-owned buffer via ncclEpInitHandle+ncclEpUpdateHandle\n");
        printf("  --profile               Enable NVTX profiling mode (use with nsys)\n");
        printf("  --disable-nvlink        Disable NVLink, force RDMA for intranode communication (LL only)\n");
        printf("  --validate              Validate dispatch/combine data correctness\n");
        printf("  --dispatch-only         With --validate, run and validate dispatch only (skip combine)\n");
        printf("  --dynamic-tokens        Enable dynamic token allocation (HT only, required for random topk)\n");
        printf("  --expert-major-alignment <N>      Per-expert zone alignment in tokens (expert-major only, power of 2)\n");
        printf("  --max-recv-token-slots-per-rank <N>  Per-rank recv-slot budget (0 = auto; bench default for EM = nRanks*tokens*top_k/2)\n");
        printf("  --zcopy                 Use ncclMemAlloc buffers + windows for HT tensors that need peer access\n");
        printf("  --max-num-sms <N>       Maximum SMs for EP kernels (0 = auto, default: 0)\n");
        printf("  --mask-test             Simulate rank failures and test active-mask (LL only, implies --validate)\n");
        printf("  --help                  Show this help message\n");
    }
}

int main(int argc, char* argv[]) {
    int myRank, nRanks, localRank = 0;

    // Default parameters
    ncclEpAlgorithm_t algorithm = NCCL_EP_ALGO_LOW_LATENCY;
    ncclEpLayout_t layout = NCCL_EP_LAYOUT_EXPERT_MAJOR;
    bool layout_set = false;
    unsigned int max_tokens_per_rank = 0;  // 0 means use algorithm-specific default
    unsigned int num_dispatch_tokens = UINT_MAX;  // UINT_MAX = unset
    unsigned int hidden = 7168;
    unsigned int top_k = 8;
    unsigned int num_experts = 256;
    int num_warmup = 10;
    int num_iters = 50;
    bool profile_mode = false;  // Enable NVTX profiling with nsys
    bool disable_nvlink = false;  // Force RDMA instead of NVLink
    bool user_handle_mem = false;  // Use caller-owned buffer via ncclEpInitHandle+ncclEpUpdateHandle
    bool validate_data = false;  // Validate dispatch/combine correctness
    bool dispatch_only = false;  // Skip combine run and validation (use with --validate)
    bool dynamic_tokens = false;  // Enable dynamic token allocation (HT only, for random topk)
    size_t expert_major_alignment = 0;  // 0 = no padding; >1 aligns each expert zone
    unsigned int max_recv_tokens_per_rank = UINT_MAX;  // UINT_MAX = unset -> bench auto; 0 = lib auto (worst case)
    bool zcopy = false;  // Use ncclMemAlloc + windows for HT tensors that need peer access
    unsigned int max_num_sms = NCCL_EP_AUTO;  // 0 = auto (resolved to HYBRIDEP_MAX_NUM_SMS_PER_RANK)
    bool mask_test = false;       // Simulate rank failures and test active-mask (LL only)
    bool include_uniform_less_than_max = false;
    bool include_non_uniform_tokens    = false;
    // Initialize MPI
    MPICHECK(MPI_Init(&argc, &argv));
    MPICHECK(MPI_Comm_rank(MPI_COMM_WORLD, &myRank));
    MPICHECK(MPI_Comm_size(MPI_COMM_WORLD, &nRanks));

    // Parse command line arguments
    static struct option long_options[] = {
        {"algorithm",      required_argument, 0, 'a'},
        {"layout",         required_argument, 0, 'L'},
        {"tokens",         required_argument, 0, 't'},
        {"hidden",         required_argument, 0, 'd'},
        {"top-k",          required_argument, 0, 'k'},
        {"experts",        required_argument, 0, 'e'},
        {"warmup",         required_argument, 0, 'w'},
        {"iters",          required_argument, 0, 'i'},
        {"profile",        no_argument,       0, 'p'},
        {"disable-nvlink", no_argument,       0, 'n'},
        {"user-handle-mem",no_argument,       0, 'U'},
        {"validate",       no_argument,       0, 'V'},
        {"dispatch-only",  no_argument,       0, 'D'},
        {"dynamic-tokens", no_argument,       0, 'M'},
        {"expert-major-alignment",   required_argument, 0, 'A'},
        {"max-recv-token-slots-per-rank", required_argument, 0, 'R'},
        {"zcopy",          no_argument,       0, 'z'},
        {"max-num-sms",    required_argument, 0, 'S'},
        {"mask-test",      no_argument,       0, 'T'},
        {"dispatch-less-than-max-tokens", required_argument, 0, 'l'},
        {"non-uniform-tokens", no_argument, 0, 'N'},
        {"help",           no_argument,       0, 'h'},
        {0, 0, 0, 0}
    };

    int opt;
    int option_index = 0;
    while ((opt = getopt_long(argc, argv, "a:L:t:d:k:e:w:i:pnUVDMA:R:zS:Tl:Nh", long_options, &option_index)) != -1) {
        switch (opt) {
            case 'a':
                if (strcmp(optarg, "ll") == 0 || strcmp(optarg, "low-latency") == 0) {
                    algorithm = NCCL_EP_ALGO_LOW_LATENCY;
                } else if (strcmp(optarg, "ht") == 0 || strcmp(optarg, "high-throughput") == 0) {
                    algorithm = NCCL_EP_ALGO_HIGH_THROUGHPUT;
                } else {
                    if (myRank == 0) {
                        printf("Error: Invalid algorithm '%s'. Use 'll', 'low-latency', 'ht', or 'high-throughput'\n", optarg);
                    }
                    MPI_Finalize();
                    return 1;
                }
                break;
            case 'L':
                layout_set = true;
                if (strcmp(optarg, "em") == 0 || strcmp(optarg, "expert-major") == 0) {
                    layout = NCCL_EP_LAYOUT_EXPERT_MAJOR;
                } else if (strcmp(optarg, "rm") == 0 || strcmp(optarg, "rank-major") == 0) {
                    layout = NCCL_EP_LAYOUT_RANK_MAJOR;
                } else if (strcmp(optarg, "fl") == 0 || strcmp(optarg, "flat") == 0) {
                    layout = NCCL_EP_LAYOUT_FLAT;
                } else {
                    if (myRank == 0) {
                        printf("Error: Invalid layout '%s'. Use 'em'/'expert-major', 'rm'/'rank-major', or 'fl'/'flat'\n", optarg);
                    }
                    MPI_Finalize();
                    return 1;
                }
                layout_set = true;
                break;
            case 't':
                max_tokens_per_rank = static_cast<unsigned int>(atoi(optarg));
                break;
            case 'd':
                hidden = static_cast<unsigned int>(atoi(optarg));
                break;
            case 'k':
                top_k = static_cast<unsigned int>(atoi(optarg));
                break;
            case 'e':
                num_experts = static_cast<unsigned int>(atoi(optarg));
                break;
            case 'w':
                num_warmup = atoi(optarg);
                break;
            case 'n':
                disable_nvlink = true;
                break;
            case 'i':
                num_iters = atoi(optarg);
                break;
            case 'p':
                profile_mode = true;
                break;
            case 'U':
                user_handle_mem = true;
                break;
            case 'V':
                validate_data = true;
                break;
            case 'D':
                dispatch_only = true;
                break;
            case 'M':
                dynamic_tokens = true;
                break;
            case 'A':
                expert_major_alignment = static_cast<size_t>(atoi(optarg));
                break;
            case 'R':
                max_recv_tokens_per_rank = static_cast<unsigned int>(atoi(optarg));
                break;
            case 'z':
                zcopy = true;
                break;
            case 'S':
                max_num_sms = static_cast<unsigned int>(atoi(optarg));
                break;
            case 'T':
                mask_test = true;
                validate_data = true;
                break;
            case 'l':
                num_dispatch_tokens = static_cast<unsigned int>(atoi(optarg));
                include_uniform_less_than_max = true;
                break;
            case 'N':
                include_non_uniform_tokens = true;
                break;
            case 'h':
                printUsage(argv[0], myRank);
                MPI_Finalize();
                return 0;
            default:
                printUsage(argv[0], myRank);
                MPI_Finalize();
                return 1;
        }
    }

    // Set algorithm-specific default for max_tokens_per_rank if not explicitly provided
    if (max_tokens_per_rank == 0) {
        max_tokens_per_rank = (algorithm == NCCL_EP_ALGO_HIGH_THROUGHPUT) ? 4096 : 128;
    }

    if (include_uniform_less_than_max && include_non_uniform_tokens) {
        if (myRank == 0) {
            printf("Error: --dispatch-less-than-max-tokens and --non-uniform-tokens are mutually exclusive\n");
        }
        MPI_Finalize();
        return 1;
    }
    if (include_uniform_less_than_max &&
        (num_dispatch_tokens == 0 || num_dispatch_tokens > max_tokens_per_rank)) {
        if (myRank == 0) {
            printf("Error: --dispatch-less-than-max-tokens (%u) must be > 0 and <= --tokens (%u)\n",
                   num_dispatch_tokens, max_tokens_per_rank);
        }
        MPI_Finalize();
        return 1;
    }

    // Set algorithm-specific default layout if user didn't specify
    if (!layout_set) {
        layout = (algorithm == NCCL_EP_ALGO_HIGH_THROUGHPUT)
                 ? NCCL_EP_LAYOUT_FLAT
                 : NCCL_EP_LAYOUT_EXPERT_MAJOR;
    }

    // Validate parameters
    if (num_experts % nRanks != 0) {
        if (myRank == 0) {
            printf("Error: num_experts (%u) must be divisible by nRanks (%d)\n", num_experts, nRanks);
        }
        MPI_Finalize();
        return 1;
    }

    // --mask-test is only supported for LL mode and requires at least 4 ranks
    if (mask_test) {
        if (algorithm != NCCL_EP_ALGO_LOW_LATENCY) {
            if (myRank == 0) printf("Error: --mask-test is only supported for LL mode\n");
            MPI_Finalize();
            return 1;
        }
        if (nRanks < 4) {
            if (myRank == 0) printf("Error: --mask-test requires at least 4 ranks (simulates failures on ranks 1 and 3)\n");
            MPI_Finalize();
            return 1;
        }
    }

    // --dynamic-tokens (NCCL_EP_AUTO for max_dispatch_tokens_per_rank) is intended for HT mode only.
    // Not yet supported in the current release; code paths are kept for future use.
    if (dynamic_tokens) {
        if (myRank == 0) {
            if (algorithm != NCCL_EP_ALGO_HIGH_THROUGHPUT)
                printf("Error: --dynamic-tokens is only applicable to HT mode (--algorithm ht)\n");
            else
                printf("Error: --dynamic-tokens (NCCL_EP_AUTO for max_dispatch_tokens_per_rank) is not yet supported.\n"
                       "       This feature will be available in a future release for HT mode.\n");
        }
        MPI_Finalize();
        return 1;
    }

    // Validate user-specified layout against algorithm.
    // HT supports flat and expert-major; LL supports expert-major and rank-major.
    if (layout_set) {
        if (algorithm == NCCL_EP_ALGO_HIGH_THROUGHPUT &&
            layout != NCCL_EP_LAYOUT_FLAT && layout != NCCL_EP_LAYOUT_EXPERT_MAJOR) {
            if (myRank == 0)
                printf("Error: HT mode supports flat or expert-major layout.\n");
            MPI_Finalize();
            return 1;
        }
        if (algorithm == NCCL_EP_ALGO_LOW_LATENCY &&
            (layout != NCCL_EP_LAYOUT_EXPERT_MAJOR && layout != NCCL_EP_LAYOUT_RANK_MAJOR)) {
            if (myRank == 0)
                printf("Error: LL mode only supports expert-major layout.\n");
            MPI_Finalize();
            return 1;
        }
    }
    if (zcopy &&
        algorithm != NCCL_EP_ALGO_HIGH_THROUGHPUT) {
        if (myRank == 0)
            printf("Error: Zero-copy is only applicable to HT mode (--algorithm ht)\n");
        MPI_Finalize();
        return 1;
    }

    unsigned int num_local_experts = num_experts / nRanks;

    // Calculate local rank based on hostname
    uint64_t hostHashs[nRanks];
    char hostname[1024];
    getHostName(hostname, 1024);
    hostHashs[myRank] = getHostHash(hostname);
    MPICHECK(MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, hostHashs, sizeof(uint64_t), MPI_BYTE, MPI_COMM_WORLD));
    for (int p = 0; p < nRanks; p++) {
        if (p == myRank) break;
        if (hostHashs[p] == hostHashs[myRank]) localRank++;
    }

    // Print configuration
    if (myRank == 0) {
        const char* algo_name = (algorithm == NCCL_EP_ALGO_LOW_LATENCY) ? "LOW_LATENCY" : "HIGH_THROUGHPUT";
        printf("=== NCCL EP Performance Benchmark ===\n");
        printf("Configuration:\n");
        printf("  Algorithm:       %s\n", algo_name);
        printf("  Layout:          %s\n",
               layout == NCCL_EP_LAYOUT_FLAT ? "flat" :
               layout == NCCL_EP_LAYOUT_RANK_MAJOR ? "rank-major" : "expert-major");
        printf("  Ranks:           %d\n", nRanks);
        if (max_num_sms != NCCL_EP_AUTO) {
            printf("  Max num SMs:     %u\n", max_num_sms);
        } else {
            printf("  Max num SMs:     auto\n");
        }
        printf("  Tokens:          %u\n", max_tokens_per_rank);
        if (include_uniform_less_than_max) {
            printf("  Sub-test:        Uniform tokens (num<max=%u)\n", num_dispatch_tokens);
        } else if (include_non_uniform_tokens) {
            printf("  Sub-test:        Non-uniform tokens in [0, %u] (last rank forced to 0)\n", max_tokens_per_rank);
        }
        printf("  Hidden:          %u\n", hidden);
        printf("  Top-k:           %u\n", top_k);
        printf("  Experts:         %u (local: %u)\n", num_experts, num_local_experts);
        printf("  Warmup iters:    %d\n", num_warmup);
        printf("  Benchmark iters: %d\n", num_iters);
        printf("  Dispatch dtype:  BF16\n");
        printf("  Profile mode:    %s\n", profile_mode ? "enabled" : "disabled");
        printf("  NVLink:          %s\n", disable_nvlink ? "disabled (force RDMA intranode, LL only)" : "enabled");
        printf("  Validate mode:   %s\n", validate_data ? "enabled" : "disabled");
        printf("  Dynamic tokens:  %s\n", dynamic_tokens ? "enabled (NCCL_EP_AUTO)" : "disabled");
#ifdef HAVE_CUPTI
        printf("  CUPTI:           enabled (kernel-level GPU timing available)\n");
#else
        printf("  CUPTI:           not available (kernel timing will report 0; ensure CUDA Toolkit with CUPTI headers is installed)\n");
#endif
        if (algorithm == NCCL_EP_ALGO_HIGH_THROUGHPUT) {
            const char* layout_str =
                (layout == NCCL_EP_LAYOUT_EXPERT_MAJOR)
                    ? (expert_major_alignment > 0 ? "expert-major (with alignment)" : "expert-major")
                    : "flat";
            printf("  Output layout:   %s\n", layout_str);
            if (expert_major_alignment > 0)
                printf("  Align (tokens):  %zu\n", expert_major_alignment);
        }
        printf("  Use zero-copy: %s\n",
               (zcopy && algorithm == NCCL_EP_ALGO_HIGH_THROUGHPUT)
                   ? "enabled (ncclMemAlloc + TensorCreateFromWindow)"
                   : "disabled");
        printf("\n");
    }

    // Disable NVLink/P2P if requested (LL mode only)
    // This forces RDMA communication even for intra-node communication
    // LL kernels use NCCL GIN, so NCCL_P2P_DISABLE is the relevant flag
    // NCCL_SHM_DISABLE is also set to avoid shared memory issues at scale
    if (disable_nvlink && algorithm == NCCL_EP_ALGO_LOW_LATENCY) {
        setenv("NCCL_P2P_DISABLE", "1", 1);
        setenv("NCCL_SHM_DISABLE", "1", 1);
    }

    // Setup CUDA
    CUDACHECK(cudaSetDevice(localRank));
    cudaStream_t stream;
    CUDACHECK(cudaStreamCreate(&stream));

    // Initialize NCCL
    ncclUniqueId id;
    ncclComm_t comm;
    if (myRank == 0) ncclGetUniqueId(&id);
    MPICHECK(MPI_Bcast(static_cast<void*>(&id), sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD));
    NCCLCHECK(ncclCommInitRank(&comm, nRanks, id, myRank));

    // Create EP group
    if (myRank == 0) { printf("[DEBUG] Creating EP group...\n"); fflush(stdout); }
    ncclEpGroup_t ep_group;
    ncclEpGroupConfig_t config = NCCL_EP_GROUP_CONFIG_INIT;
    config.algorithm = algorithm;
    config.num_experts = num_experts;
    // max_dispatch_tokens_per_rank is the per-rank batch size (max tokens any single rank will send).
    config.max_dispatch_tokens_per_rank = dynamic_tokens ? NCCL_EP_AUTO : max_tokens_per_rank;

    config.max_token_bytes = hidden * 2;  // bfloat16 worst case
    // Use NCCL_EP_AUTO for buffer sizes (required for dynamic tokens with larger batches)
    // For LL mode with disable_nvlink: NCCL_P2P_DISABLE env var handles NCCL GIN P2P
    config.rdma_buffer_size = NCCL_EP_AUTO;
    // num_qp_per_rank: LL mode requires >= num_local_experts, HT mode uses auto
    config.num_qp_per_rank = (algorithm == NCCL_EP_ALGO_LOW_LATENCY) ? num_local_experts : NCCL_EP_AUTO;
    config.num_channels = NCCL_EP_AUTO;
    // Uniform-routing estimate: per-rank recv ≈ max_tokens_per_rank*top_k (2x safety for EM dup variance).
    if (max_recv_tokens_per_rank == UINT_MAX) {
        const unsigned int est = std::max(1u, max_tokens_per_rank * top_k *
                                              std::max(1u, num_local_experts) /
                                              std::max(1u, static_cast<unsigned int>(num_experts)) *
                                              static_cast<unsigned int>(nRanks));
        const unsigned int safety = (layout == NCCL_EP_LAYOUT_EXPERT_MAJOR) ? 2u : 1u;
        max_recv_tokens_per_rank = std::max(1u, est * safety);
    }
    config.max_recv_tokens_per_rank = max_recv_tokens_per_rank;
    config.max_num_sms = max_num_sms;
    config.alloc.alloc_fn = cudaAllocCallback;
    config.alloc.free_fn  = cudaFreeCallback;
    config.alloc.context  = nullptr;
    config.enable_mask = mask_test;

    printf("Rank %d: Testing ncclEpCreateGroup with algorithm: %s%s\n", myRank,
           (algorithm == NCCL_EP_ALGO_LOW_LATENCY) ? "LOW_LATENCY" : "HIGH_THROUGHPUT",
           mask_test ? " (mask-test mode)" : "");
    MPICHECK(MPI_Barrier(MPI_COMM_WORLD));
    // Baseline GPU memory before any EP allocations (group buffer, handle mem,
    // staging tensors). Compared against a post-combine snapshot below.
    size_t gpu_mem_free_pre = 0, gpu_mem_total = 0;
    CUDACHECK(cudaMemGetInfo(&gpu_mem_free_pre, &gpu_mem_total));
    double group_create_start = MPI_Wtime();
    NCCLCHECK(ncclEpCreateGroup(&ep_group, comm, &config));
    double group_create_end = MPI_Wtime();
    double group_create_ms = (group_create_end - group_create_start) * 1000.0;
    printf("Rank %d: ncclEpCreateGroup took %.2f ms\n", myRank, group_create_ms);

    std::vector<unsigned int> num_tokens_per_rank = include_non_uniform_tokens
        ? computeNonUniformTokensPerRank(max_tokens_per_rank, nRanks)
        : std::vector<unsigned int>(nRanks, include_uniform_less_than_max ? num_dispatch_tokens : max_tokens_per_rank);
    unsigned int num_tokens = num_tokens_per_rank[myRank];

    if (myRank == 0 && include_non_uniform_tokens) {
        printf("Per-rank token counts:");
        for (int r = 0; r < nRanks; r++) printf(" r%d=%u", r, num_tokens_per_rank[r]);
        printf("\n");
        fflush(stdout);
    }

    // Initialize topk_idx tensor
    ncclEpTensor_t* topk_idx = nullptr;
    NCCLCHECK(epMakeTensor(&topk_idx, 2, ncclInt64, num_tokens, top_k));

    // Generate topk indices
    // HT: randperm (uniform), consistent with Hybrid-EP (test_hybrid_ep.py)
    // LL: abs(randn)+1 scores + topk + -1 masking, consistent with DeepEP (test_low_latency.py)
    int64_t *topk_idx_host = new int64_t[num_tokens * top_k];

    if (algorithm == NCCL_EP_ALGO_HIGH_THROUGHPUT) {
        generateTopkIndicesHT(topk_idx_host, num_tokens, num_experts, top_k, myRank);
        if (myRank == 0) {
            printf("Using randperm topk_idx for HT mode (uniform distribution)\n\n");
        }
    } else {
        generateRandomTopkIndicesLL(topk_idx_host, num_tokens, num_experts, top_k, myRank);
        if (myRank == 0) {
            printf("Using random topk_idx for LL mode (with -1 masking)\n\n");
        }
    }

    // Count valid token-expert pairs (excluding -1 masked entries)
    unsigned int num_valid_selections = 0;
    for (unsigned int i = 0; i < num_tokens * top_k; i++) {
        if (topk_idx_host[i] != -1) {
            num_valid_selections++;
        }
    }

    // Calculate byte metrics based on algorithm mode (BF16)
    LowLatencyBytes ll_bytes = {};
    HighThroughputBytes ht_bytes = {};
    if (algorithm == NCCL_EP_ALGO_LOW_LATENCY) {
        ll_bytes = calculateLowLatencyBytes(topk_idx_host, num_tokens, top_k, hidden);
    } else {
        ht_bytes = calculateHighThroughputBytes(
            topk_idx_host, num_tokens, num_tokens_per_rank.data(),
            top_k, num_experts, hidden, myRank, nRanks,
            ncclTeamLsa(comm).nRanks);
    }

    {
        void* topk_idx_data;
        topk_idx_data = topk_idx->data;
        CUDACHECK(cudaMemcpy(topk_idx_data, topk_idx_host, num_tokens * top_k * sizeof(int64_t), cudaMemcpyHostToDevice));
    }
    // Note: topk_idx_host is kept for validation, deleted at end

    // RECV_EXPERT_COUNTER_DEVICE: per-expert counts.
    //   HT flat: int32 unpadded counts (only needed when dynamic_tokens).
    //   HT expert-major: int64 padded counts (needed for dynamic_tokens AND validation).
    // RECV_EXPERT_OFFSETS_DEVICE: int64 padded offsets (HT expert-major only, for validation).
    int64_t* dispatch_meta_counts_host  = nullptr;
    int64_t* dispatch_meta_offsets_host = nullptr;
    const bool ht_em = (algorithm == NCCL_EP_ALGO_HIGH_THROUGHPUT &&
                        layout == NCCL_EP_LAYOUT_EXPERT_MAJOR);
    const bool need_dispatch_meta = ht_em && validate_data;

    ncclEpTensor_t* recv_expert_counter_tensor = nullptr;
    ncclEpTensor_t* recv_total_counter_tensor = nullptr;
    if (ht_em && (dynamic_tokens || need_dispatch_meta)) {
        NCCLCHECK(epMakeTensor(&recv_expert_counter_tensor, 1, ncclInt64, num_local_experts));
    } else if (dynamic_tokens) {
        NCCLCHECK(epMakeTensor(&recv_expert_counter_tensor, 1, ncclInt32, num_local_experts));
        NCCLCHECK(epMakeTensor(&recv_total_counter_tensor, 1, ncclInt32, 1));
    }
    ncclEpTensor_t* meta_offsets_tensor = nullptr;
    if (need_dispatch_meta) {
        NCCLCHECK(epMakeTensor(&meta_offsets_tensor, 1, ncclInt64, num_local_experts));
    }

    // Create handle — populate the layout_info struct with the optional counter / offset tensors.
    ncclEpLayoutInfo_t handle_layout_info = NCCL_EP_LAYOUT_INFO_INIT;
    if (recv_expert_counter_tensor != nullptr) handle_layout_info.expert_counters    = recv_expert_counter_tensor;
    if (recv_total_counter_tensor != nullptr)  handle_layout_info.recv_total_counter = recv_total_counter_tensor;
    if (meta_offsets_tensor != nullptr)        handle_layout_info.expert_offsets     = meta_offsets_tensor;
    const bool has_handle_layout_info =
        handle_layout_info.expert_counters != nullptr ||
        handle_layout_info.recv_total_counter != nullptr ||
        handle_layout_info.expert_offsets != nullptr;

    const bool ht_expert_major = (algorithm == NCCL_EP_ALGO_HIGH_THROUGHPUT &&
                                  layout == NCCL_EP_LAYOUT_EXPERT_MAJOR);
    ncclEpHandleConfig_t handle_cfg = NCCL_EP_HANDLE_CONFIG_INIT;
    handle_cfg.dispatch_output_per_expert_alignment = expert_major_alignment;
    // Pass config only when a non-default field is set.
    const bool need_cfg = (ht_expert_major && expert_major_alignment > 0);
    const ncclEpHandleConfig_t* cfg_ptr = need_cfg ? &handle_cfg : nullptr;

    // Optional caller-owned buffer (--user-handle-mem)
    ncclEpTensor_t* handle_mem_tensor = nullptr;
    if (user_handle_mem) {
        size_t handle_mem_size;
        NCCLCHECK(ncclEpHandleMemSize(ep_group, layout, cfg_ptr, &handle_mem_size, static_cast<int>(top_k)));
        NCCLCHECK(epMakeTensor(&handle_mem_tensor, 1, ncclUint8, static_cast<unsigned int>(handle_mem_size)));
        if (myRank == 0)
            printf("Rank 0: ncclEpHandleMemSize = %zu bytes\n", handle_mem_size);
    }

    ncclEpHandle_t ep_handle;
    MPICHECK(MPI_Barrier(MPI_COMM_WORLD));
    double handle_create_start = MPI_Wtime();
    if (user_handle_mem) {
        NCCLCHECK(ncclEpInitHandle(&ep_handle, ep_group, layout, cfg_ptr, static_cast<int>(top_k), handle_mem_tensor));
        NCCLCHECK(ncclEpUpdateHandle(ep_handle, topk_idx,
                                     has_handle_layout_info ? &handle_layout_info : nullptr, stream));
    } else {
        NCCLCHECK(ncclEpCreateHandle(&ep_handle, ep_group, layout, topk_idx,
                                     has_handle_layout_info ? &handle_layout_info : nullptr, cfg_ptr, stream));
    }
    CUDACHECK(cudaStreamSynchronize(stream));
    double handle_create_end = MPI_Wtime();
    double handle_create_ms = (handle_create_end - handle_create_start) * 1000.0;
    printf("Rank %d: handle creation took %.2f ms\n", myRank, handle_create_ms);

    // max_dispatch_tokens_per_rank is the per-rank dispatch count.
    // num_recv_tokens is the max tokens this rank can receive (nRanks * max_dispatch_tokens_per_rank).
    unsigned int num_recv_tokens = 0;
    if (dynamic_tokens) {
        void* total_data = nullptr;
        total_data = recv_total_counter_tensor->data;
        int32_t total_host = 0;
        CUDACHECK(cudaMemcpy(&total_host, total_data, sizeof(int32_t), cudaMemcpyDeviceToHost));
        assert(total_host >= 0);
        num_recv_tokens = static_cast<unsigned int>(total_host);
        if (myRank == 0) {
            printf("[DEBUG] Dynamic tokens: num_recv_tokens=%u\n", num_recv_tokens);
            fflush(stdout);
        }
    } else {
        // num_recv_tokens = total per-rank slot budget = config.max_recv_tokens_per_rank (resolved by lib).
        num_recv_tokens = config.max_recv_tokens_per_rank;
    }
    assert(num_recv_tokens);

    // HT recv bytes are pre-computed in calculateHighThroughputBytes via routing simulation
    if (algorithm == NCCL_EP_ALGO_HIGH_THROUGHPUT && myRank == 0) {
        printf("[DEBUG] HT bytes: send=%u tokens, rdma_send=%u, total_recv=%u tokens, rdma_recv=%u (buffer=%u)\n",
               ht_bytes.total_send_tokens, ht_bytes.rdma_send_tokens,
               ht_bytes.total_recv_tokens, ht_bytes.rdma_recv_tokens, num_recv_tokens);
        fflush(stdout);
    }

    // Setup benchmark tensors based on algorithm mode. Tensor handles live
    // inside the named-struct fields (dispatch_inputs/outputs/layout_info /
    // combine_inputs/outputs); setup writes directly there and validation /
    // cleanup reads from there. `topk_weights` is a side handle aliased into
    // different struct fields per layout. The `alloc` state tracks zero-copy bookkeeping
    // (NCCL window registrations, ncclMemAlloc'd pointers).
    BenchmarkAllocState     alloc;
    ncclEpDispatchInputs_t  dispatch_inputs          = NCCL_EP_DISPATCH_INPUTS_INIT;
    ncclEpDispatchOutputs_t dispatch_outputs         = NCCL_EP_DISPATCH_OUTPUTS_INIT;
    ncclEpLayoutInfo_t      dispatch_layout_info     = NCCL_EP_LAYOUT_INFO_INIT;
    bool                    has_dispatch_layout_info = false;
    ncclEpCombineInputs_t   combine_inputs           = NCCL_EP_COMBINE_INPUTS_INIT;
    ncclEpCombineOutputs_t  combine_outputs          = NCCL_EP_COMBINE_OUTPUTS_INIT;
    ncclEpTensor_t*         topk_weights             = nullptr;
    const bool              is_ll_mode               = (algorithm == NCCL_EP_ALGO_LOW_LATENCY);

    if (myRank == 0) { printf("[DEBUG] Setting up tensors...\n"); fflush(stdout); }
    if (is_ll_mode) {
        setupLowLatencyTensors(dispatch_inputs, dispatch_outputs, dispatch_layout_info,
                               has_dispatch_layout_info, combine_inputs, combine_outputs,
                               topk_weights,
                               num_tokens, hidden, top_k,
                               num_local_experts, config.max_dispatch_tokens_per_rank,
                               nRanks, layout);
    } else {
        setupHighThroughputTensors(comm, alloc,
                                   dispatch_inputs, dispatch_outputs, dispatch_layout_info,
                                   has_dispatch_layout_info,
                                   combine_inputs, combine_outputs,
                                   topk_weights,
                                   num_tokens, hidden, top_k,
                                   num_local_experts, num_recv_tokens, layout, zcopy);
    }
    if (myRank == 0) { printf("[DEBUG] Tensors set up\n"); fflush(stdout); }

    // Initialize validation data if enabled (fills tensors with rank-based patterns)
    if (validate_data) {
        if (myRank == 0) { printf("[DEBUG] Initializing validation data...\n"); fflush(stdout); }
        initializeValidationData(alloc, dispatch_inputs, topk_weights,
                                 num_tokens, hidden, top_k, myRank, !is_ll_mode);
        if (myRank == 0) { printf("[DEBUG] Validation data initialized\n"); fflush(stdout); }
    }

    ncclEpDispatchConfig_t dispatch_config = NCCL_EP_DISPATCH_CONFIG_INIT;
    dispatch_config.round_scales = 0;

    // Synchronize before benchmarking
    MPICHECK(MPI_Barrier(MPI_COMM_WORLD));
    CUDACHECK(cudaStreamSynchronize(stream));

    // Calculate data sizes for bandwidth calculation based on algorithm mode
    size_t dispatch_data_bytes, combine_data_bytes;
    if (algorithm == NCCL_EP_ALGO_LOW_LATENCY) {
        // LL mode: BF16 for both dispatch and combine
        dispatch_data_bytes = ll_bytes.dispatch_bytes;
        combine_data_bytes = ll_bytes.combine_bytes;
    } else {
        // HT mode: RDMA_send + total_recv (matches DeepEP methodology)
        dispatch_data_bytes = ht_bytes.rdma_send_bytes + ht_bytes.total_recv_bytes;
        combine_data_bytes = dispatch_data_bytes;  // Symmetric
    }

    // ==================== Paired Dispatch + Combine Benchmark ====================
    // Always run dispatch and combine paired to ensure correct internal state
    // (matching DeepEP's benchmarking approach)

    if (myRank == 0) { printf("[DEBUG] Starting benchmark...\n"); fflush(stdout); }

    ncclEpCombineConfig_t combine_config = NCCL_EP_COMBINE_CONFIG_INIT;
    auto dispatch_fn = [&]() {
        NCCLCHECK(ncclEpDispatch(ep_handle,
                                  &dispatch_inputs, &dispatch_outputs,
                                  has_dispatch_layout_info ? &dispatch_layout_info : nullptr,
                                  &dispatch_config, stream));
        NCCLCHECK(ncclEpComplete(ep_handle, nullptr, stream));
    };

    auto combine_fn = [&]() {
        NCCLCHECK(ncclEpCombine(ep_handle, &combine_inputs, &combine_outputs,
                                 &combine_config, stream));
        NCCLCHECK(ncclEpComplete(ep_handle, nullptr, stream));
    };

    // Use the requested number of iterations for both modes
    // HT mode uses "cached" mode for iterations after the first (handle state is reused)
    int actual_warmup = num_warmup;
    int actual_iters = num_iters;

    // CUPTI wraps the benchmark loop — records kernel GPU timestamps in hardware
    // alongside the cudaEvent timing, with zero interference.
    KernelTimer ktimer;

    MPICHECK(MPI_Barrier(MPI_COMM_WORLD));
    PairedBenchResult paired_result = runPairedBenchmark(
        dispatch_fn, combine_fn, actual_warmup, actual_iters,
        dispatch_data_bytes, combine_data_bytes,
        ktimer,
        stream);

    // Post-combine GPU memory snapshot. By this point the EP group has been
    // created, the handle has been initialized (and the rdma_buffer grown to
    // fit the active layout, if needed), and dispatch+combine have run. The
    // delta vs gpu_mem_free_pre reflects total EP-induced device allocations
    // (mostly rdma_buffer + handle mem + bench-time tensors).
    size_t gpu_mem_free_post = 0;
    {
        size_t total_ignored = 0;
        CUDACHECK(cudaMemGetInfo(&gpu_mem_free_post, &total_ignored));
    }

    // Extract individual results for printing
    BenchResult dispatch_result = paired_result.dispatch;
    BenchResult combine_result = paired_result.combine;
    BenchResult combined_result = paired_result.total;

    // ==================== NVTX Profiling Mode ====================
    if (profile_mode) {
        auto handle_create_fn = [&]() {
            NCCLCHECK(ncclEpHandleDestroy(ep_handle));
            const ncclEpLayoutInfo_t* layout_info_ptr = has_handle_layout_info ? &handle_layout_info : nullptr;
            if (user_handle_mem) {
                NCCLCHECK(ncclEpInitHandle(&ep_handle, ep_group, layout, cfg_ptr, static_cast<int>(top_k), handle_mem_tensor));
                NCCLCHECK(ncclEpUpdateHandle(ep_handle, topk_idx, layout_info_ptr, stream));
            } else {
                NCCLCHECK(ncclEpCreateHandle(&ep_handle, ep_group, layout, topk_idx,
                                              layout_info_ptr, cfg_ptr, stream));
            }
        };
        runNvtxProfiling(myRank, actual_iters, dispatch_fn, combine_fn, handle_create_fn, stream);
    }

    // ==================== CUPTI Kernel-Only Timing (reduce before print) ====================
    // Debug: show all captured kernels on rank 0 (uncomment to inspect names)
    // ktimer.dump(myRank);

    // Aggregate byte counts across ranks (HT only)
    size_t global_total_send = 0, global_rdma_send = 0;
    size_t global_total_recv = 0, global_rdma_recv = 0;
    if (algorithm == NCCL_EP_ALGO_HIGH_THROUGHPUT) {
        MPI_Reduce(&ht_bytes.total_send_bytes, &global_total_send, 1, MPI_UNSIGNED_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
        MPI_Reduce(&ht_bytes.rdma_send_bytes,  &global_rdma_send,  1, MPI_UNSIGNED_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
        MPI_Reduce(&ht_bytes.total_recv_bytes, &global_total_recv, 1, MPI_UNSIGNED_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
        MPI_Reduce(&ht_bytes.rdma_recv_bytes,  &global_rdma_recv,  1, MPI_UNSIGNED_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
    }

    // Print results and summary based on algorithm mode
    if (algorithm == NCCL_EP_ALGO_LOW_LATENCY) {
        printLowLatencyResults(myRank, nRanks,
                              dispatch_result, combine_result, combined_result,
                              ktimer,
                              ll_bytes);
    } else {
        printHighThroughputResults(myRank, nRanks, dispatch_result, combine_result, combined_result,
                                   ktimer,
                                   ht_bytes,
                                   global_total_send, global_rdma_send,
                                   global_total_recv, global_rdma_recv);
    }

    // Aggregate group/handle creation times across ranks
    {
        double local_group_ms = group_create_ms;
        double local_handle_ms = handle_create_ms;
        double global_group_avg, global_group_min, global_group_max;
        double global_handle_avg, global_handle_min, global_handle_max;

        MPI_Reduce(&local_group_ms, &global_group_avg, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
        MPI_Reduce(&local_group_ms, &global_group_min, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
        MPI_Reduce(&local_group_ms, &global_group_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        MPI_Reduce(&local_handle_ms, &global_handle_avg, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
        MPI_Reduce(&local_handle_ms, &global_handle_min, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
        MPI_Reduce(&local_handle_ms, &global_handle_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

        if (myRank == 0) {
            global_group_avg /= nRanks;
            global_handle_avg /= nRanks;

            printf("\n=== Setup Timing (across %d ranks) ===\n", nRanks);
            printf("ncclEpCreateGroup:   avg=%.2f ms, min=%.2f ms, max=%.2f ms\n",
                   global_group_avg, global_group_min, global_group_max);
            printf("Handle creation:     avg=%.2f ms, min=%.2f ms, max=%.2f ms\n",
                   global_handle_avg, global_handle_min, global_handle_max);
        }
    }

    // GPU memory usage (rank 0 snapshot; same device, identical layout across ranks).
    // Captured before group creation and right after the paired dispatch+combine.
    if (myRank == 0) {
        const double MB = 1024.0 * 1024.0;
        const size_t used_pre  = gpu_mem_total  - gpu_mem_free_pre;
        const size_t used_post = gpu_mem_total  - gpu_mem_free_post;
        const long long delta  = static_cast<long long>(used_post) -
                                 static_cast<long long>(used_pre);
        printf("\n=== GPU Memory (rank 0) ===\n");
        printf("Total device memory: %.2f MB\n", gpu_mem_total / MB);
        printf("Used pre-create:     %.2f MB\n", used_pre  / MB);
        printf("Used post-combine:   %.2f MB\n", used_post / MB);
        printf("EP-induced delta:    %+.2f MB\n", delta / MB);
    }

    // ==================== Data Validation ====================
    if (validate_data) {
        MPICHECK(MPI_Barrier(MPI_COMM_WORLD));

        // Run one more dispatch+combine with validation
        if (myRank == 0) { printf("\n=== Data Validation ===\n"); fflush(stdout); }

        // Re-initialize validation data (benchmark may have modified it)
        initializeValidationData(alloc, dispatch_inputs, topk_weights,
                                 num_tokens, hidden, top_k, myRank, !is_ll_mode);

        // Run dispatch
        dispatch_fn();
        CUDACHECK(cudaStreamSynchronize(stream));
        MPICHECK(MPI_Barrier(MPI_COMM_WORLD));

        // Copy per-expert metadata from device to host for validation
        if (need_dispatch_meta) {
            dispatch_meta_counts_host  = new int64_t[num_local_experts];
            dispatch_meta_offsets_host = new int64_t[num_local_experts];
            void* counts_ptr;
            void* offsets_ptr;
            counts_ptr = recv_expert_counter_tensor->data;
            offsets_ptr = meta_offsets_tensor->data;
            CUDACHECK(cudaMemcpy(dispatch_meta_counts_host,  counts_ptr,
                                 num_local_experts * sizeof(int64_t), cudaMemcpyDeviceToHost));
            CUDACHECK(cudaMemcpy(dispatch_meta_offsets_host, offsets_ptr,
                                 num_local_experts * sizeof(int64_t), cudaMemcpyDeviceToHost));
        }

        ValidationResult dispatch_valid = validateDispatchOutput(
            alloc, dispatch_outputs, dispatch_layout_info,
            max_tokens_per_rank, num_tokens_per_rank.data(),
            hidden, top_k, num_experts, num_local_experts, myRank, nRanks,
            !is_ll_mode,
            layout == NCCL_EP_LAYOUT_EXPERT_MAJOR,
            expert_major_alignment,
            dispatch_meta_counts_host,
            dispatch_meta_offsets_host);

        if (!dispatch_only) {
            // Simulate expert FFN processing: copy dispatch output into expert_outputs,
            // then apply per-rank weight sums for rank-major (kernel uses weight=1).
            {
                void* eo_data;
                void* output0_data;
                NCCLCHECK(epGetTensorData(alloc, combine_inputs.tokens, &eo_data));
                NCCLCHECK(epGetTensorData(alloc, dispatch_outputs.tokens, &output0_data));

                if (!is_ll_mode) {
                    // HT: 2D [num_recv_tokens, hidden]
                    const size_t* eo_sizes = combine_inputs.tokens->sizes;
                    size_t data_size = eo_sizes[0] * eo_sizes[1] * sizeof(uint16_t);
                    CUDACHECK(cudaMemcpy(eo_data, output0_data, data_size, cudaMemcpyDeviceToDevice));
                } else if (layout == NCCL_EP_LAYOUT_EXPERT_MAJOR) {
                    // LL expert-major: 3D [num_local_experts, max_tokens_per_expert, hidden]
                    const size_t* out0_sizes = dispatch_outputs.tokens->sizes;
                    size_t data_size = out0_sizes[0] * out0_sizes[1] * out0_sizes[2] * sizeof(uint16_t);
                    CUDACHECK(cudaMemcpy(eo_data, output0_data, data_size, cudaMemcpyDeviceToDevice));
                } else {
                    // LL rank-major: 3D [nRanks, max_tpr, hidden] — copy then apply
                    // per-rank weight sums before combine (kernel uses weight=1).
                    const size_t* out0_sizes = dispatch_outputs.tokens->sizes;
                    size_t data_size = out0_sizes[0] * out0_sizes[1] * out0_sizes[2] * sizeof(uint16_t);
                    CUDACHECK(cudaMemcpy(eo_data, output0_data, data_size, cudaMemcpyDeviceToDevice));
                    preReduceRankMajor(alloc, dispatch_outputs, dispatch_layout_info,
                                       combine_inputs, top_k, nRanks);
                }
            }

            // Run combine
            combine_fn();
            CUDACHECK(cudaStreamSynchronize(stream));
            MPICHECK(MPI_Barrier(MPI_COMM_WORLD));
        }  // if (!dispatch_only) — copy + combine

        // Validate combine output (skipped in dispatch-only mode)
        ValidationResult combine_valid = {true, 0, 0.0, "skipped (dispatch-only)"};
        if (!dispatch_only) {
            combine_valid = validateCombineOutput(
                alloc, combine_outputs, topk_weights,
                num_tokens, hidden, top_k, num_experts, myRank, nRanks,
                !is_ll_mode, topk_idx_host,
                layout == NCCL_EP_LAYOUT_EXPERT_MAJOR);
        }

        // Print validation results (rank 0 only to avoid clutter)
        if (myRank == 0) {
            printf("Dispatch validation: %s", dispatch_valid.passed ? "PASSED" : "FAILED");
            if (!dispatch_valid.passed) {
                printf(" (%s)", dispatch_valid.message.c_str());
            }
            printf("\n");

            if (!dispatch_only) {
                printf("Combine validation:  %s", combine_valid.passed ? "PASSED" : "FAILED");
                if (!combine_valid.passed) {
                    printf(" (%s)", combine_valid.message.c_str());
                }
                printf(" (calc_diff=%.6e)\n", combine_valid.max_diff);
            }
            fflush(stdout);
        }

        // Collect validation results across all ranks
        int local_dispatch_pass = dispatch_valid.passed ? 1 : 0;
        int local_combine_pass = combine_valid.passed ? 1 : 0;
        int global_dispatch_pass, global_combine_pass;

        MPICHECK(MPI_Allreduce(&local_dispatch_pass, &global_dispatch_pass, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD));
        MPICHECK(MPI_Allreduce(&local_combine_pass, &global_combine_pass, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD));

        if (myRank == 0) {
            if (dispatch_only) {
                printf("\nGlobal validation: Dispatch=%s\n",
                       global_dispatch_pass ? "PASSED" : "FAILED");
            } else {
                printf("\nGlobal validation: Dispatch=%s, Combine=%s\n",
                       global_dispatch_pass ? "PASSED" : "FAILED",
                       global_combine_pass ? "PASSED" : "FAILED");
            }
            fflush(stdout);
        }
    }

    // Destroy HT per-expert metadata local tensors and free host copies
    if (need_dispatch_meta) {
        epFreeTensor(&meta_offsets_tensor);
    }
    delete[] dispatch_meta_counts_host;
    delete[] dispatch_meta_offsets_host;

    // ==================== Active-Mask Test ====================
    // Simulates rank failures during dispatch/combine and verifies that the
    // kernel's timeout mechanism correctly masks failed ranks.
    if (mask_test) {
        MPICHECK(MPI_Barrier(MPI_COMM_WORLD));
        if (myRank == 0) { printf("\n=== Active-Mask Test ===\n"); fflush(stdout); }

        // Ranks designated to fail at each phase
        const int dispatch_fail_rank = 1;
        const int combine_fail_rank = 3;

        // Re-initialize validation data
        initializeValidationData(alloc, dispatch_inputs, topk_weights,
                                 num_tokens, hidden, top_k, myRank, !is_ll_mode);

        // --- Phase 1: Dispatch with rank 1 failing ---
        if (myRank == dispatch_fail_rank) {
            printf("Rank %d: simulating failure (skipping dispatch)\n", myRank);
            fflush(stdout);
        } else {
            dispatch_fn();
            CUDACHECK(cudaStreamSynchronize(stream));
        }

        // Surviving ranks wait; failed rank also reaches barrier via MPI
        MPICHECK(MPI_Barrier(MPI_COMM_WORLD));

        // Poll async error first (lightweight, no GPU sync), then query mask for details
        if (myRank != dispatch_fail_rank) {
            int async_err = 0;
            NCCLCHECK(ncclEpGetAsyncError(ep_group, &async_err));
            printf("Rank %d: async error after dispatch: %d (%s)\n",
                   myRank, async_err, async_err ? "PASSED" : "FAILED");
            fflush(stdout);

            // Error detected -- query mask to find which ranks failed
            int* mask_status_d;
            CUDACHECK(cudaMalloc(reinterpret_cast<void**>(&mask_status_d), nRanks * sizeof(int)));
            NCCLCHECK(ncclEpMaskQuery(ep_group, mask_status_d, stream));
            CUDACHECK(cudaStreamSynchronize(stream));

            int* mask_status_h = new int[nRanks];
            CUDACHECK(cudaMemcpy(mask_status_h, mask_status_d, nRanks * sizeof(int), cudaMemcpyDeviceToHost));

            printf("Rank %d: mask after dispatch = [", myRank);
            for (int r = 0; r < nRanks; r++) printf("%d%s", mask_status_h[r], r < nRanks-1 ? "," : "");
            printf("]\n");

            // 0 = masked/failed, 1 = active
            bool dispatch_mask_ok = (mask_status_h[dispatch_fail_rank] == 0);
            printf("Rank %d: dispatch mask check: %s (rank %d mask=%d)\n",
                   myRank, dispatch_mask_ok ? "PASSED" : "FAILED",
                   dispatch_fail_rank, mask_status_h[dispatch_fail_rank]);
            fflush(stdout);

            delete[] mask_status_h;
            CUDACHECK(cudaFree(mask_status_d));
        }

        // --- Phase 2: Combine with rank 3 failing ---
        // Copy dispatch output to combine input (passthrough)
        if (myRank != dispatch_fail_rank && myRank != combine_fail_rank) {
            void* eo_data;
            void* output0_data;
            NCCLCHECK(epGetTensorData(alloc, combine_inputs.tokens, &eo_data));
            NCCLCHECK(epGetTensorData(alloc, dispatch_outputs.tokens, &output0_data));
            const size_t* out0_sizes = dispatch_outputs.tokens->sizes;
            size_t data_size = out0_sizes[0] * out0_sizes[1] * out0_sizes[2] * sizeof(uint16_t);
            CUDACHECK(cudaMemcpy(eo_data, output0_data, data_size, cudaMemcpyDeviceToDevice));
        }

        if (myRank == dispatch_fail_rank || myRank == combine_fail_rank) {
            printf("Rank %d: simulating failure (skipping combine)\n", myRank);
            fflush(stdout);
        } else {
            combine_fn();
            CUDACHECK(cudaStreamSynchronize(stream));
        }

        MPICHECK(MPI_Barrier(MPI_COMM_WORLD));

        // Poll async error first, then query mask for details
        if (myRank != dispatch_fail_rank && myRank != combine_fail_rank) {
            int async_err = 0;
            NCCLCHECK(ncclEpGetAsyncError(ep_group, &async_err));
            printf("Rank %d: async error after combine: %d (%s)\n",
                   myRank, async_err, async_err ? "PASSED" : "FAILED");
            fflush(stdout);

            // Error detected -- query mask to find which ranks failed
            int* mask_status_d;
            CUDACHECK(cudaMalloc(reinterpret_cast<void**>(&mask_status_d), nRanks * sizeof(int)));
            NCCLCHECK(ncclEpMaskQuery(ep_group, mask_status_d, stream));
            CUDACHECK(cudaStreamSynchronize(stream));

            int* mask_status_h = new int[nRanks];
            CUDACHECK(cudaMemcpy(mask_status_h, mask_status_d, nRanks * sizeof(int), cudaMemcpyDeviceToHost));

            printf("Rank %d: mask after combine = [", myRank);
            for (int r = 0; r < nRanks; r++) printf("%d%s", mask_status_h[r], r < nRanks-1 ? "," : "");
            printf("]\n");

            // 0 = masked/failed, 1 = active
            bool combine_mask_ok = (mask_status_h[dispatch_fail_rank] == 0) &&
                                   (mask_status_h[combine_fail_rank] == 0);
            printf("Rank %d: combine mask check: %s (rank %d=%d, rank %d=%d)\n",
                   myRank, combine_mask_ok ? "PASSED" : "FAILED",
                   dispatch_fail_rank, mask_status_h[dispatch_fail_rank],
                   combine_fail_rank, mask_status_h[combine_fail_rank]);
            fflush(stdout);

            delete[] mask_status_h;
            CUDACHECK(cudaFree(mask_status_d));
        }

        // --- Phase 3: Clean mask buffer and verify ---
        if (myRank != dispatch_fail_rank && myRank != combine_fail_rank) {
            NCCLCHECK(ncclEpMaskClean(ep_group, stream));
            NCCLCHECK(ncclEpErrorClear(ep_group));
            CUDACHECK(cudaStreamSynchronize(stream));

            int* mask_status_d;
            CUDACHECK(cudaMalloc(reinterpret_cast<void**>(&mask_status_d), nRanks * sizeof(int)));
            NCCLCHECK(ncclEpMaskQuery(ep_group, mask_status_d, stream));
            CUDACHECK(cudaStreamSynchronize(stream));

            int* mask_status_h = new int[nRanks];
            CUDACHECK(cudaMemcpy(mask_status_h, mask_status_d, nRanks * sizeof(int), cudaMemcpyDeviceToHost));

            // After clean, all ranks should be active (1)
            bool clean_ok = true;
            for (int r = 0; r < nRanks; r++) {
                if (mask_status_h[r] != 1) { clean_ok = false; break; }
            }
            printf("Rank %d: mask clean check: %s\n", myRank, clean_ok ? "PASSED" : "FAILED");

            // Verify async error flag is cleared after ncclEpErrorClear
            int async_err = 0;
            NCCLCHECK(ncclEpGetAsyncError(ep_group, &async_err));
            printf("Rank %d: async error after clean: %d (%s)\n",
                   myRank, async_err, async_err == 0 ? "PASSED" : "FAILED");
            fflush(stdout);

            delete[] mask_status_h;
            CUDACHECK(cudaFree(mask_status_d));
        }

        MPICHECK(MPI_Barrier(MPI_COMM_WORLD));
        if (myRank == 0) { printf("=== Active-Mask Test Complete ===\n"); fflush(stdout); }
    }

    // Cleanup (order matters: tensors -> handle -> group -> comm)
    cleanupBenchmarkTensors(alloc, dispatch_inputs, dispatch_outputs, dispatch_layout_info,
                            combine_inputs, combine_outputs,
                            topk_weights, topk_idx, is_ll_mode);
    delete[] topk_idx_host;  // Now safe to delete after validation

    NCCLCHECK(ncclEpHandleDestroy(ep_handle));

    if (handle_mem_tensor != nullptr)
        epFreeTensor(&handle_mem_tensor);

    // Cleanup recv_expert_counter if allocated (must be before group destroy)
    if (dynamic_tokens && recv_expert_counter_tensor != nullptr) {
        epFreeTensor(&recv_expert_counter_tensor);
    }
    if (dynamic_tokens && recv_total_counter_tensor != nullptr) {
        epFreeTensor(&recv_total_counter_tensor);
    }

    NCCLCHECK(ncclEpGroupDestroy(ep_group));
    ncclCommDestroy(comm);

    CUDACHECK(cudaStreamDestroy(stream));

    MPICHECK(MPI_Finalize());
    cudaDeviceReset();

    return 0;
}
