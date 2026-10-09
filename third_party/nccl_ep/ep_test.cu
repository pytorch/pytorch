/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 * See LICENSE.txt for more license information.
 */

#include <stdio.h>
#include <stdlib.h>
#include "cuda_runtime.h"
#include "nccl.h"
#include "nccl_ep.h"
#include "mpi.h"
#include <unistd.h>
#include <stdint.h>
#include <string.h>
#include <getopt.h>

// Custom allocator wrappers (can be replaced with custom memory pool)
static cudaError_t torchMalloc(void** ptr, size_t size, void* /*context*/) {
    return cudaMalloc(ptr, size);
}

static cudaError_t torchFree(void* ptr, void* /*context*/) {
    return cudaFree(ptr);
}

// Element size for the dtypes used in this test. ncclTypeSize is internal to the EP library.
static size_t epDtypeBytes(ncclDataType_t dt) {
    switch (dt) {
        case ncclInt8:    case ncclUint8:                              return 1;
        case ncclFloat16: case ncclBfloat16:                           return 2;
        case ncclFloat32: case ncclInt32: case ncclUint32:             return 4;
        case ncclInt64:   case ncclUint64: case ncclFloat64:           return 8;
        default: return 0;
    }
}

// cudaMalloc the backing buffer and wrap it in a heap-allocated descriptor via
// ncclEpTensorAlloc. epFreeTensor releases both the descriptor and the buffer.
static ncclResult_t epMakeTensor(ncclEpTensor_t** out_tensor, unsigned int ndim,
                                 ncclDataType_t dt,
                                 unsigned int s0, unsigned int s1 = 1, unsigned int s2 = 1,
                                 unsigned int s3 = 1, unsigned int s4 = 1) {
    if (out_tensor == nullptr) return ncclInvalidArgument;
    *out_tensor = nullptr;
    size_t dims[5] = {s0, s1, s2, s3, s4};
    size_t total = 1;
    for (unsigned int i = 0; i < ndim; i++) total *= dims[i];
    void* data = nullptr;
    cudaError_t e = cudaMalloc(&data, total * epDtypeBytes(dt));
    if (e != cudaSuccess) return ncclSystemError;
    ncclResult_t r = ncclEpTensorAlloc(out_tensor, ndim, dt, dims, /*config=*/nullptr);
    if (r != ncclSuccess) {
        cudaFree(data);
        return r;
    }
    (*out_tensor)->data = data;
    return ncclSuccess;
}

// Inverse of epMakeTensor: free backing buffer and release the descriptor.
static void epFreeTensor(ncclEpTensor_t** field) {
    if (field == nullptr || *field == nullptr) return;
    void* data = (*field)->data;
    ncclEpTensorDestroy(*field);
    *field = nullptr;
    if (data) cudaFree(data);
}

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

static uint64_t getHostHash(const char* string) {
    // Based on DJB2, result = result * 33 + char
    uint64_t result = 5381;
    for (int c = 0; string[c] != '\0'; c++) {
        result = ((result << 5) + result) + string[c];
    }
    return result;
}

// Convert float to bfloat16 (as uint16_t)
static inline uint16_t float_to_bf16(float f) {
    uint32_t x = *reinterpret_cast<uint32_t*>(&f);
    // Round to nearest even
    uint32_t rounding_bias = 0x00007FFF + ((x >> 16) & 1);
    return static_cast<uint16_t>((x + rounding_bias) >> 16);
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


// Allocate and fill the combine-side tensors. expert_outputs is the per-rank
// MoE expert output (combine input); combined_output is the per-token reduction
// (combine output). Shapes diverge by algorithm: LL is 3D [num_local_experts,
// max_dispatch_tokens_per_rank * nRanks, hidden]; HT is 2D [num_recv_tokens, hidden].
// The first elements_tested_per_token slots of each token are seeded with
// float_to_bf16((j+1)*2) so combine validation can recompute the weighted sum.
static void epSetupCombineTensors(
    ncclEpAlgorithm_t algorithm,
    unsigned int num_local_experts,
    unsigned int num_recv_tokens,
    unsigned int max_dispatch_tokens_per_rank,
    int nRanks,
    unsigned int hidden,
    unsigned int num_tokens,
    int elements_tested_per_token,
    ncclEpTensor_t** expert_outputs,
    ncclEpTensor_t** combined_output) {
  if (algorithm == NCCL_EP_ALGO_LOW_LATENCY) {
    NCCLCHECK(epMakeTensor(expert_outputs, 3, ncclBfloat16,
                           num_local_experts,
                           static_cast<unsigned int>(max_dispatch_tokens_per_rank * nRanks),
                           hidden));
    uint16_t *host = new uint16_t[max_dispatch_tokens_per_rank * hidden]();
    for (unsigned int t = 0; t < max_dispatch_tokens_per_rank; ++t) {
      for (int j = 0; j < elements_tested_per_token; ++j) {
        host[t * hidden + j] = float_to_bf16(static_cast<float>((j + 1) * 2));
      }
    }
    void* data;
    data = (*expert_outputs)->data;
    for (unsigned int e = 0; e < num_local_experts; ++e) {
      size_t offset_bytes = static_cast<size_t>(e) * max_dispatch_tokens_per_rank * hidden * nRanks * 2;
      CUDACHECK(cudaMemcpy(static_cast<uint8_t*>(data) + offset_bytes,
                           host, max_dispatch_tokens_per_rank * hidden * 2,
                           cudaMemcpyHostToDevice));
    }
    delete[] host;
  } else {
    NCCLCHECK(epMakeTensor(expert_outputs, 2, ncclBfloat16, num_recv_tokens, hidden));
    uint16_t *host = new uint16_t[num_recv_tokens * hidden]();
    for (unsigned int t = 0; t < num_recv_tokens; ++t) {
      for (int j = 0; j < elements_tested_per_token; ++j) {
        host[t * hidden + j] = float_to_bf16(static_cast<float>((j + 1) * 2));
      }
    }
    void* data;
    data = (*expert_outputs)->data;
    CUDACHECK(cudaMemcpy(data, host, num_recv_tokens * hidden * 2, cudaMemcpyHostToDevice));
    delete[] host;
  }

  NCCLCHECK(epMakeTensor(combined_output, 2, ncclBfloat16, num_tokens, hidden));
}

void printUsage(const char* programName, int myRank) {
    if (myRank == 0) {
        printf("Usage: %s [OPTIONS]\n", programName);
        printf("Options:\n");
        printf("  -a <ll|ht>                   Set algorithm mode (default: ll)\n");
        printf("                               ll:  Low latency mode (NCCL_EP_ALGO_LOW_LATENCY)\n");
        printf("                               ht:  High throughput mode (NCCL_EP_ALGO_HIGH_THROUGHPUT)\n");
        printf("  -L <fl|em>                   Layout (HT only; default: fl)\n");
        printf("                               fl:  Flat layout (default)\n");
        printf("                               em:  Expert-major layout (recv_topk_weights is 1D)\n");
        printf("  -m                           Disable max_dispatch_tokens_per_rank (only supported with HT mode)\n");
        printf("  -s <none|dispatch|combine|both>  Set send_only mode (default: none)\n");
        printf("                               none:     send_only=false for both dispatch and combine\n");
        printf("                               dispatch: send_only=true for dispatch only\n");
        printf("                               combine:  send_only=true for combine only\n");
        printf("                               both:     send_only=true for both dispatch and combine\n");
        printf("  -c                           Enable cached mode (default: disabled, only supported for HT)\n");
        printf("  -r                           Enable random mode (random topk_idx, skip correctness checks)\n");
        printf("  -g                           Capture dispatch+combine into a single CUDA Graph (CreateHandle excluded)\n");
        printf("  -t <num>                     Set number of tokens (default: 50)\n");
        printf("  -d <num>                     Set hidden dimension size (default: 7168)\n");
        printf("  -e <num>                     Set total number of experts (default: top_k * nRanks)\n");
        printf("  -h                           Show this help message\n");
    }
}

int main(int argc, char* argv[])
{
  int myRank, nRanks, localRank = 0;
  ncclEpAlgorithm_t algorithm = NCCL_EP_ALGO_LOW_LATENCY; // Default to 'll' (low latency)
  ncclEpLayout_t ht_layout = NCCL_EP_LAYOUT_FLAT; // HT layout (overridable via -L)
  bool disable_max_tokens = false; // Flag to disable max_dispatch_tokens_per_rank
  bool dispatch_send_only = false; // send_only flag for dispatch
  bool combine_send_only = false;  // send_only flag for combine
  bool cached_mode = false;        // cached mode flag
  bool random_mode = false;        // random mode flag (random topk_idx, skip checks)
  bool use_cuda_graph = false;     // capture dispatch and combine into a single CUDA Graph
  unsigned int num_tokens = 50; // number of tokens (default)
  unsigned int hidden = 7168;      // hidden dimension size (default)
  unsigned int num_experts = 0;    // 0 = auto (top_k * nRanks)

  // initializing MPI
  MPICHECK(MPI_Init(&argc, &argv));
  MPICHECK(MPI_Comm_rank(MPI_COMM_WORLD, &myRank));
  MPICHECK(MPI_Comm_size(MPI_COMM_WORLD, &nRanks));

  if (nRanks != 2 && nRanks % 4 != 0) {
    printf("Error: nRanks must be 2 or a multiple of 4 for this test\n");
    MPI_Finalize();
    exit(EXIT_FAILURE);
  }

  // Parse command line arguments using getopt
  int opt;
  while ((opt = getopt(argc, argv, "a:L:ms:crgt:d:e:h")) != -1) {
    switch (opt) {
      case 'a':
        if (strcmp(optarg, "ll") == 0) {
          algorithm = NCCL_EP_ALGO_LOW_LATENCY;
        } else if (strcmp(optarg, "ht") == 0) {
          algorithm = NCCL_EP_ALGO_HIGH_THROUGHPUT;
        } else {
          if (myRank == 0) {
            printf("Error: Invalid algorithm '%s'. Use 'll' or 'ht'\n", optarg);
            printUsage(argv[0], myRank);
          }
          MPI_Finalize();
          exit(EXIT_FAILURE);
        }
        break;
      case 'L':
        if (strcmp(optarg, "fl") == 0 || strcmp(optarg, "flat") == 0) {
          ht_layout = NCCL_EP_LAYOUT_FLAT;
        } else if (strcmp(optarg, "em") == 0 || strcmp(optarg, "expert-major") == 0) {
          ht_layout = NCCL_EP_LAYOUT_EXPERT_MAJOR;
        } else {
          if (myRank == 0) {
            printf("Error: Invalid layout '%s'. Use 'fl' or 'em'\n", optarg);
            printUsage(argv[0], myRank);
          }
          MPI_Finalize();
          exit(EXIT_FAILURE);
        }
        break;
      case 'm':
        disable_max_tokens = true;
        break;
      case 's':
        if (strcmp(optarg, "none") == 0) {
          dispatch_send_only = false;
          combine_send_only = false;
        } else if (strcmp(optarg, "dispatch") == 0) {
          dispatch_send_only = true;
          combine_send_only = false;
        } else if (strcmp(optarg, "combine") == 0) {
          dispatch_send_only = false;
          combine_send_only = true;
        } else if (strcmp(optarg, "both") == 0) {
          dispatch_send_only = true;
          combine_send_only = true;
        } else {
          if (myRank == 0) {
            printf("Error: Invalid send_only mode '%s'. Use 'none', 'dispatch', 'combine', or 'both'\n", optarg);
            printUsage(argv[0], myRank);
          }
          MPI_Finalize();
          exit(EXIT_FAILURE);
        }
        break;
      case 'c':
        cached_mode = true;
        break;
      case 'r':
        random_mode = true;
        break;
      case 'g':
        use_cuda_graph = true;
        break;
      case 't':
        num_tokens = static_cast<unsigned int>(atoi(optarg));
        if (num_tokens == 0) {
          if (myRank == 0) {
            printf("Error: Invalid num_tokens '%s'. Must be a positive integer.\n", optarg);
            printUsage(argv[0], myRank);
          }
          MPI_Finalize();
          exit(EXIT_FAILURE);
        }
        break;
      case 'd':
        hidden = static_cast<unsigned int>(atoi(optarg));
        if (hidden == 0) {
          if (myRank == 0) {
            printf("Error: Invalid hidden size '%s'. Must be a positive integer.\n", optarg);
            printUsage(argv[0], myRank);
          }
          MPI_Finalize();
          exit(EXIT_FAILURE);
        }
        break;
      case 'e':
        num_experts = static_cast<unsigned int>(atoi(optarg));
        break;
      case 'h':
        printUsage(argv[0], myRank);
        MPI_Finalize();
        exit(EXIT_SUCCESS);
      case '?':
        if (myRank == 0) {
          printUsage(argv[0], myRank);
        }
        MPI_Finalize();
        exit(EXIT_FAILURE);
      default:
        if (myRank == 0) {
          printUsage(argv[0], myRank);
        }
        MPI_Finalize();
        exit(EXIT_FAILURE);
    }
  }

  // -m (NCCL_EP_AUTO for max_dispatch_tokens_per_rank) is intended for HT mode only.
  // Not yet supported in the current release; code paths are kept for future use.
  if (disable_max_tokens) {
    if (myRank == 0) {
      if (algorithm != NCCL_EP_ALGO_HIGH_THROUGHPUT)
        printf("Error: -m is only applicable to HT mode (-a ht)\n");
      else
        printf("Error: -m (NCCL_EP_AUTO for max_dispatch_tokens_per_rank) is not yet supported.\n"
               "       This feature will be available in a future release for HT mode.\n");
    }
    MPI_Finalize();
    exit(EXIT_FAILURE);
  }

  // calculating localRank based on hostname which is used in selecting a GPU
  uint64_t hostHashs[nRanks];

  const unsigned int ELEMENTS_TESTED_PER_TOKEN = 10;
  unsigned int top_k = std::min(8, nRanks);
  if (num_experts == 0)
    num_experts = top_k * static_cast<unsigned int>(nRanks);
  unsigned int num_local_experts = num_experts / nRanks;
  unsigned int local_experts_start = num_local_experts * myRank;
  unsigned int local_experts_end = local_experts_start + num_local_experts; // exclusive

  if (num_experts == 0 || (num_experts & (num_experts - 1)) != 0) {
    if (myRank == 0) printf("Error: num_experts (%u) must be a power of 2\n", num_experts);
    MPI_Finalize();
    exit(EXIT_FAILURE);
  }
  if (num_experts % nRanks != 0) {
    if (myRank == 0) printf("Error: num_experts (%u) must be divisible by nRanks (%d)\n", num_experts, nRanks);
    MPI_Finalize();
    exit(EXIT_FAILURE);
  }
  if (num_experts < top_k * static_cast<unsigned int>(nRanks)) {
    if (myRank == 0) printf("Error: num_experts (%u) must be >= top_k * nRanks (%u)\n", num_experts, top_k * static_cast<unsigned int>(nRanks));
    MPI_Finalize();
    exit(EXIT_FAILURE);
  }

  char hostname[1024];
  getHostName(hostname, 1024);
  hostHashs[myRank] = getHostHash(hostname);
  MPICHECK(MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, hostHashs, sizeof(uint64_t), MPI_BYTE, MPI_COMM_WORLD));
  for (int p = 0; p < nRanks; p++) {
     if (p == myRank) break;
     if (hostHashs[p] == hostHashs[myRank]) localRank++;
  }

  ncclUniqueId id;
  ncclComm_t comm;
  cudaStream_t s;

  // get NCCL unique ID at rank 0 and broadcast it to all others
  if (myRank == 0) ncclGetUniqueId(&id);
  MPICHECK(MPI_Bcast(static_cast<void*>(&id), sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD));

  // picking a GPU based on localRank, allocate device buffers
  CUDACHECK(cudaSetDevice(localRank));
  CUDACHECK(cudaStreamCreate(&s));

  // initializing NCCL
  NCCLCHECK(ncclCommInitRank(&comm, nRanks, id, myRank));

  ncclEpGroup_t ep_group;
  ncclEpGroupConfig_t config = NCCL_EP_GROUP_CONFIG_INIT;
  config.algorithm = algorithm;                          // Algorithm type (set by command line)
  config.num_experts = num_experts;
  // max_dispatch_tokens_per_rank is the per-rank batch size (max tokens any single rank will send).
  config.max_dispatch_tokens_per_rank = disable_max_tokens ? NCCL_EP_AUTO : num_tokens;
  config.max_token_bytes = hidden * 2;                   // bfloat16
  config.rdma_buffer_size = NCCL_EP_AUTO;               // NCCL_EP_AUTO for auto configuration, internally uses the hint
  config.num_qp_per_rank = NCCL_EP_AUTO;                // Default is 24, see internode_ll.cu:181 for the minimum
  config.num_channels = NCCL_EP_AUTO;                   // Number of communication channels
  if (algorithm == NCCL_EP_ALGO_HIGH_THROUGHPUT) {
    // HT requires max_recv_tokens_per_rank > 0. Worst-case = nRanks * num_tokens (for top_k=1)
    // or nRanks * num_tokens * top_k for EM under heavy fan-out; use the latter for safety.
    config.max_recv_tokens_per_rank = static_cast<unsigned int>(nRanks) * num_tokens * top_k;
  }
  config.alloc.alloc_fn = torchMalloc;
  config.alloc.free_fn  = torchFree;
  config.alloc.context  = nullptr;

  const char* algorithm_name = (algorithm == NCCL_EP_ALGO_LOW_LATENCY) ? "LOW_LATENCY" : "HIGH_THROUGHPUT";
  printf("Rank %d: Testing ncclEpCreateGroup with algorithm: %s%s\n", myRank, algorithm_name,
         disable_max_tokens ? " (no max_dispatch_tokens_per_rank)" : "");
  NCCLCHECK(ncclEpCreateGroup(&ep_group, comm, &config));

  ncclEpTensor_t* topk_idx = nullptr;
  NCCLCHECK(epMakeTensor(&topk_idx, 2, ncclInt64, static_cast<unsigned int>(num_tokens), static_cast<unsigned int>(top_k)));
  int64_t *topk_idx_host = new int64_t[num_tokens * top_k];

  if (random_mode) {
    // Random mode: first expert is random, rest are deterministic to avoid repetitions
    srand(myRank + 42);  // Seed with rank for reproducibility
    for (int i = 0; i < num_tokens; i++) {
      // First expert is random
      int64_t first_expert = rand() % num_experts;
      topk_idx_host[i * top_k + 0] = first_expert;

      // Remaining experts are deterministic based on first choice (no repetitions)
      for (int j = 1; j < top_k; j++) {
        topk_idx_host[i * top_k + j] = (first_expert + j) % num_experts;
      }
    }
    if (myRank == 0) {
      printf("Random mode enabled: first expert random, rest deterministic (no repetitions)\n");
    }
  } else {
    // Deterministic mode: send each token to top_k experts, cycling through all
    // num_local_experts on the target rank for equal distribution
    for (int i = 0; i < num_tokens; i++) {
      for (int j = 0; j < top_k; j++) {
        topk_idx_host[i * top_k + j] = (local_experts_end + (i * top_k + j) % num_local_experts) % num_experts;
      }
    }
  }
  void* topk_idx_data;
  topk_idx_data = topk_idx->data;
  CUDACHECK(cudaMemcpy(topk_idx_data, topk_idx_host, num_tokens * top_k * sizeof(int64_t), cudaMemcpyHostToDevice));

  // Create recv-counter tensors for ncclEpCreateHandle (only when disable_max_tokens is true)
  ncclEpLayoutInfo_t handle_layout_info = NCCL_EP_LAYOUT_INFO_INIT;
  ncclEpTensor_t* handle_recv_expert_counter = nullptr;
  ncclEpTensor_t* handle_recv_total_counter = nullptr;
  if (disable_max_tokens) {
    NCCLCHECK(epMakeTensor(&handle_recv_expert_counter, 1, ncclInt32, num_local_experts));
    handle_layout_info.expert_counters = handle_recv_expert_counter;
    NCCLCHECK(epMakeTensor(&handle_recv_total_counter, 1, ncclInt32, 1));
    handle_layout_info.recv_total_counter = handle_recv_total_counter;
  }

  // Non-graph mode tests CreateHandle (combined Init+Update). Graph mode tests
  // the Init+Update split, where InitHandle stays outside the capture (host-side
  // allocation only) and UpdateHandle is recorded inside the captured region.
  ncclEpHandle_t ep_handle;
  const ncclEpLayout_t handle_layout = (algorithm == NCCL_EP_ALGO_HIGH_THROUGHPUT)
                                       ? ht_layout
                                       : NCCL_EP_LAYOUT_EXPERT_MAJOR;
  if (use_cuda_graph) {
    printf("Rank %d: Testing ncclEpInitHandle\n", myRank);
    NCCLCHECK(ncclEpInitHandle(&ep_handle, ep_group, handle_layout, /*config=*/nullptr, static_cast<int>(top_k), /*handle_mem=*/nullptr));
  } else {
    printf("Rank %d: Testing ncclEpCreateHandle\n", myRank);
    NCCLCHECK(ncclEpCreateHandle(&ep_handle, ep_group, handle_layout, topk_idx,
                                 disable_max_tokens ? &handle_layout_info : nullptr, /*config=*/nullptr, s));
    CUDACHECK(cudaStreamSynchronize(s));
  }

  unsigned int num_recv_tokens = 0;
  if (disable_max_tokens) {
    void* total_data = nullptr;
    total_data = handle_recv_total_counter->data;
    int32_t total_host = 0;
    CUDACHECK(cudaMemcpy(&total_host, total_data, sizeof(int32_t), cudaMemcpyDeviceToHost));
    assert(total_host >= 0);
    num_recv_tokens = static_cast<unsigned int>(total_host);
  }
  else if (algorithm == NCCL_EP_ALGO_HIGH_THROUGHPUT) {
    // HT recv buffer is sized by max_recv_tokens_per_rank.
    num_recv_tokens = config.max_recv_tokens_per_rank;
  }
  else {
    num_recv_tokens = config.max_dispatch_tokens_per_rank * num_local_experts;
  }
  assert(num_recv_tokens);

  ncclEpDispatchConfig_t dispatch_config = NCCL_EP_DISPATCH_CONFIG_INIT;
  dispatch_config.round_scales = 0; // Not testing this parameter atm

  // Build named-struct dispatch arguments. HT FLAT also populates topk_weights/topk_idx
  // outputs; HT EM populates only topk_weights (1D). LL allocates only tokens.
  const bool ht_em = (algorithm == NCCL_EP_ALGO_HIGH_THROUGHPUT &&
                       handle_layout == NCCL_EP_LAYOUT_EXPERT_MAJOR);

  ncclEpDispatchInputs_t  dispatch_inputs  = NCCL_EP_DISPATCH_INPUTS_INIT;
  ncclEpDispatchOutputs_t dispatch_outputs = NCCL_EP_DISPATCH_OUTPUTS_INIT;
  ncclEpLayoutInfo_t     dispatch_layout_info   = NCCL_EP_LAYOUT_INFO_INIT;

  ncclEpTensor_t* input_tokens = nullptr;
  ncclEpTensor_t* recv_x = nullptr;
  ncclEpTensor_t* topk_weights = nullptr;
  ncclEpTensor_t* recv_topk_weights = nullptr;
  ncclEpTensor_t* recv_topk_idx = nullptr;
  ncclEpTensor_t* ll_recv_expert_counter = nullptr;

  NCCLCHECK(epMakeTensor(&input_tokens, 2, ncclBfloat16,
                         static_cast<unsigned int>(num_tokens), static_cast<unsigned int>(hidden)));
  dispatch_inputs.tokens = input_tokens;

  NCCLCHECK(epMakeTensor(&topk_weights, 2, ncclFloat32,
                         static_cast<unsigned int>(num_tokens), static_cast<unsigned int>(top_k)));
  if (algorithm != NCCL_EP_ALGO_LOW_LATENCY) {
    // HT: topk_weights is a dispatch input; topk_idx is taken from the handle.
    dispatch_inputs.topk_weights = topk_weights;
  }

  if (algorithm == NCCL_EP_ALGO_LOW_LATENCY) {
    // LL mode: recv_x is [num_local_experts, max_recv_tokens * nRanks, hidden]
    NCCLCHECK(epMakeTensor(&recv_x, 3, ncclBfloat16,
                           static_cast<unsigned int>(num_local_experts),
                           config.max_dispatch_tokens_per_rank * nRanks,
                           static_cast<unsigned int>(hidden)));
    NCCLCHECK(epMakeTensor(&ll_recv_expert_counter, 1, ncclInt32,
                           static_cast<unsigned int>(num_local_experts)));
    dispatch_layout_info.expert_counters = ll_recv_expert_counter;
  } else {
    // HT mode: recv_x is [num_recv_tokens, hidden]
    NCCLCHECK(epMakeTensor(&recv_x, 2, ncclBfloat16,
                           static_cast<unsigned int>(num_recv_tokens),
                           static_cast<unsigned int>(hidden)));
  }
  dispatch_outputs.tokens = recv_x;

  // HT recv_topk_weights: EM is 1D [num_recv_tokens], FLAT is 2D [num_recv_tokens, top_k].
  // HT recv_topk_idx: only allocated under FLAT.
  if (algorithm != NCCL_EP_ALGO_LOW_LATENCY) {
    if (ht_em) {
      NCCLCHECK(epMakeTensor(&recv_topk_weights, 1, ncclFloat32,
                             static_cast<unsigned int>(num_recv_tokens)));
    } else {
      NCCLCHECK(epMakeTensor(&recv_topk_weights, 2, ncclFloat32,
                             static_cast<unsigned int>(num_recv_tokens),
                             static_cast<unsigned int>(top_k)));
      NCCLCHECK(epMakeTensor(&recv_topk_idx, 2, ncclInt64,
                             static_cast<unsigned int>(num_recv_tokens),
                             static_cast<unsigned int>(top_k)));
      dispatch_outputs.topk_idx = recv_topk_idx;
    }
    dispatch_outputs.topk_weights = recv_topk_weights;
  }

  // Fill the first ELEMENTS_TESTED_PER_TOKEN elements of each token with a special value based on the current rank
  uint16_t *input_host = new uint16_t[num_tokens * hidden]();
  for (int i = 0; i < num_tokens; ++i) {
    for (int j = 0; j < ELEMENTS_TESTED_PER_TOKEN; ++j) {
      input_host[i * hidden + j] = static_cast<uint16_t>(0x1000 + myRank);
    }
  }
  void* input0_data;
  input0_data = input_tokens->data;
  CUDACHECK(cudaMemcpy(input0_data, input_host, num_tokens * hidden * 2, cudaMemcpyHostToDevice));
  delete[] input_host;

  // Fill topk_weights (used as input for HT mode, kept around for LL mode validation)
  float *topk_weights_host = new float[num_tokens * top_k];
  for (int i = 0; i < num_tokens; ++i) {
    for (int j = 0; j < top_k; ++j) {
      topk_weights_host[i * top_k + j] = 1.0f / top_k; // Equal weights
    }
  }
  void* topk_weights_data;
  topk_weights_data = topk_weights->data;
  CUDACHECK(cudaMemcpy(topk_weights_data, topk_weights_host, num_tokens * top_k * sizeof(float), cudaMemcpyHostToDevice));
  delete[] topk_weights_host;

  // Host buffer for first phase dispatch output (kept for cached mode comparison in HT mode)
  uint16_t *first_dispatch_output0_host = nullptr;

  // Combine-side tensors are allocated above the dispatch call so the entire
  // dispatch+combine sequence can optionally be recorded into a single CUDA Graph.
  // Combine inputs do not depend on dispatch outputs in either LL or HT mode.
  ncclEpTensor_t* expert_outputs = nullptr;
  ncclEpTensor_t* combined_output = nullptr;
  epSetupCombineTensors(algorithm, num_local_experts, num_recv_tokens,
                        config.max_dispatch_tokens_per_rank, nRanks, hidden, num_tokens,
                        ELEMENTS_TESTED_PER_TOKEN, &expert_outputs, &combined_output);

  // Setup combine inputs and outputs (named-struct API)
  ncclEpCombineInputs_t  combine_inputs  = NCCL_EP_COMBINE_INPUTS_INIT;
  ncclEpCombineOutputs_t combine_outputs = NCCL_EP_COMBINE_OUTPUTS_INIT;
  combine_inputs.tokens  = expert_outputs;
  combine_outputs.tokens = combined_output;
  // LL expert-major reads per-token routing weights from outputs.topk_weights
  // (used as receive-side scaling, not a write target). See nccl_ep.h.
  if (algorithm == NCCL_EP_ALGO_LOW_LATENCY) {
    combine_outputs.topk_weights = topk_weights;
  }

  ncclEpCombineConfig_t combine_config = NCCL_EP_COMBINE_CONFIG_INIT;
  combine_config.send_only = combine_send_only ? 1 : 0;

  dispatch_config.send_only = dispatch_send_only ? 1 : 0;

  // Single graph spans dispatch + combine when -g is set.
  cudaGraph_t graph = nullptr;
  cudaGraphExec_t graph_exec = nullptr;

  if (use_cuda_graph) {
    printf("Rank %d: [CUDA Graph: begin capture]\n", myRank);
    CUDACHECK(cudaStreamBeginCapture(s, cudaStreamCaptureModeRelaxed));
    printf("Rank %d: Testing ncclEpUpdateHandle\n", myRank);
    NCCLCHECK(ncclEpUpdateHandle(ep_handle, topk_idx,
                                 disable_max_tokens ? &handle_layout_info : nullptr, s));
  }

  printf("Rank %d: Testing ncclEpDispatch (send_only=%s)\n", myRank, dispatch_send_only ? "true" : "false");
  NCCLCHECK(ncclEpDispatch(ep_handle,
                           &dispatch_inputs, &dispatch_outputs,
                           (algorithm == NCCL_EP_ALGO_LOW_LATENCY) ? &dispatch_layout_info : nullptr,
                           &dispatch_config, s));

  printf("Rank %d: Testing ncclEpComplete\n", myRank);
  NCCLCHECK(ncclEpComplete(ep_handle, nullptr, s));

  printf("Rank %d: Testing %s Combine flow\n", myRank, algorithm_name);

  printf("Rank %d: Testing ncclEpCombine (send_only=%s)\n", myRank, combine_send_only ? "true" : "false");
  NCCLCHECK(ncclEpCombine(ep_handle, &combine_inputs, &combine_outputs, &combine_config, s));

  NCCLCHECK(ncclEpComplete(ep_handle, nullptr, s));
  if (use_cuda_graph) {
    printf("Rank %d: [CUDA Graph: end capture, instantiate, launch]\n", myRank);
    CUDACHECK(cudaStreamEndCapture(s, &graph));
    CUDACHECK(cudaGraphInstantiate(&graph_exec, graph, NULL, NULL, 0));
    CUDACHECK(cudaGraphLaunch(graph_exec, s));
  }
  CUDACHECK(cudaStreamSynchronize(s));
  // Read recv_count tensor to use for validation
  // LL mode: allocated and copied from device ll_recv_expert_counter
  // HT mode with disable_max_tokens: points to handle_recv_expert_counter (already host memory)
  // HT mode without disable_max_tokens: nullptr (no validation available)
  int *recv_count_host = nullptr;
  bool should_free_recv_count = false;
  if (algorithm == NCCL_EP_ALGO_LOW_LATENCY) {
    recv_count_host = new int[num_local_experts];
    void* local_tensor0_data;
    local_tensor0_data = ll_recv_expert_counter->data;
    CUDACHECK(cudaMemcpy(recv_count_host, local_tensor0_data, num_local_experts * sizeof(int), cudaMemcpyDeviceToHost));
    should_free_recv_count = true;
  } else if (disable_max_tokens && handle_recv_expert_counter != nullptr) {
    void* handle_local_tensor0_data;
    handle_local_tensor0_data = handle_recv_expert_counter->data;
    recv_count_host = static_cast<int*>(handle_local_tensor0_data);
  }

  unsigned int recv_from_expert_start = (local_experts_start + num_experts - num_local_experts) % num_experts;
  unsigned int recv_rank = recv_from_expert_start / num_local_experts;

  // Check the first ELEMENTS_TESTED_PER_TOKEN elements of each token
  // For LL mode: use recv_count to validate per-expert token counts
  // For HT mode: use the simple approach
  bool dispatch_check_passed = true;

  if (!random_mode && algorithm == NCCL_EP_ALGO_LOW_LATENCY && recv_count_host != nullptr) {
    // LL mode: recv_x is [num_local_experts, config.max_dispatch_tokens_per_rank * nRanks, hidden]
    uint16_t *output_host = new uint16_t[num_local_experts * config.max_dispatch_tokens_per_rank * nRanks * hidden]();
    void* output0_data;
    output0_data = recv_x->data;
    CUDACHECK(cudaMemcpy(output_host, output0_data,
                         num_local_experts * config.max_dispatch_tokens_per_rank * nRanks * hidden * 2,
                         cudaMemcpyDeviceToHost));

    for (unsigned int e = 0; e < num_local_experts; e++) {
      int expected_count = static_cast<int>(num_tokens) * top_k / num_local_experts;
      if (recv_count_host[e] != expected_count) {
        printf("Recv_count check failed! Rank %d, expert %d: expected %d, got %d\n",
               myRank, e, expected_count, recv_count_host[e]);
        dispatch_check_passed = false;
        break;
      }

      // Verify the first recv_count_host[e] tokens for this expert
      for (int t = 0; t < recv_count_host[e] && t < static_cast<int>(config.max_dispatch_tokens_per_rank * nRanks); t++) {
        size_t token_offset = (e * config.max_dispatch_tokens_per_rank * nRanks + t) * hidden;
        for (int j = 0; j < ELEMENTS_TESTED_PER_TOKEN; ++j) {
          uint16_t expected = static_cast<uint16_t>(0x1000 + recv_rank);
          uint16_t actual = output_host[token_offset + j];
          if (actual != expected) {
            printf("Dispatch data check failed! Rank %d, expert %d, token %d, element %d: expected %d, got %d\n",
                   myRank, e, t, j, expected, actual);
            dispatch_check_passed = false;
            break;
          }
        }
        if (!dispatch_check_passed) break;
      }
      if (!dispatch_check_passed) break;
    }
    delete[] output_host;
  } else if (!random_mode) {
    // HT mode or fallback (only check if not in random mode)
    // Check recv_count (only if available)
    if (recv_count_host != nullptr) {
      for (unsigned int e = 0; e < num_local_experts; e++) {
        int expected_count = static_cast<int>(num_tokens) * top_k / num_local_experts;
        if (recv_count_host[e] != expected_count) {
          printf("Recv_count check failed! Rank %d, expert %d: expected %d, got %d\n",
                  myRank, e, expected_count, recv_count_host[e]);
          dispatch_check_passed = false;
          break;
        }
      }
    }

    uint16_t *output_host = new uint16_t[num_recv_tokens * hidden]();
    void* output0_data;
    output0_data = recv_x->data;
    CUDACHECK(cudaMemcpy(output_host, output0_data, num_recv_tokens * hidden * 2, cudaMemcpyDeviceToHost));
    int expected_count = disable_max_tokens ? num_recv_tokens : num_tokens;
    for (int i = 0; i < expected_count; ++i) {
      for (int j = 0; j < ELEMENTS_TESTED_PER_TOKEN; ++j) {
        uint16_t expected = static_cast<uint16_t>(0x1000 + recv_rank);
        uint16_t actual = output_host[i * hidden + j];
        if (actual != expected) {
          printf("Dispatch check failed! Rank %d, token %d, element %d: expected %d, got %d\n",
                 myRank, i, j, expected, actual);
          dispatch_check_passed = false;
          break;
        }
      }
      if (!dispatch_check_passed) break;
    }
    // Keep output_host for cached mode comparison, delete later
    first_dispatch_output0_host = output_host;
  } else {
    // Random mode: skip checks, just clean up
  }

  // Consolidated cleanup for recv_count_host
  if (should_free_recv_count && recv_count_host != nullptr) {
    delete[] recv_count_host;
  }

  // In HT mode, copy and verify recv_topk_weights (recv_topk_weights) and (FLAT only) recv_topk_idx (recv_topk_idx)
  if (!random_mode && algorithm != NCCL_EP_ALGO_LOW_LATENCY) {
    // EM: 1D [num_recv_tokens]; FLAT: 2D [num_recv_tokens, top_k].
    const size_t weights_count = ht_em ? static_cast<size_t>(num_recv_tokens)
                                        : static_cast<size_t>(num_recv_tokens) * top_k;
    float *recv_topk_weights_host = new float[weights_count]();
    void* output1_data;
    output1_data = recv_topk_weights->data;
    CUDACHECK(cudaMemcpy(recv_topk_weights_host, output1_data,
                         weights_count * sizeof(float), cudaMemcpyDeviceToHost));

    int64_t *recv_topk_idx_host = nullptr;
    if (!ht_em) {
      recv_topk_idx_host = new int64_t[static_cast<size_t>(num_recv_tokens) * top_k]();
      void* output2_data;
      output2_data = recv_topk_idx->data;
      CUDACHECK(cudaMemcpy(recv_topk_idx_host, output2_data,
                           static_cast<size_t>(num_recv_tokens) * top_k * sizeof(int64_t),
                           cudaMemcpyDeviceToHost));
    }

    bool ht_outputs_valid = true;
    printf("Rank %d: Verifying recv_topk_weights%s\n", myRank,
           ht_em ? " (HT+EM, 1D)" : " and recv_topk_idx (HT+FLAT, 2D)");

    float expected_weight = 1.0f / top_k;
    int weight_errors = 0;
    int idx_errors = 0;

    if (ht_em) {
      // EM: each filled slot has a single weight == 1/top_k. Padded slots default to 0.
      for (int i = 0; i < num_recv_tokens; i++) {
        float w = recv_topk_weights_host[i];
        if (w != 0.0f && w != expected_weight) {
          if (weight_errors < 5)
            printf("Rank %d: recv_topk_weights[%d] = %f, expected 0 or %f\n",
                   myRank, i, w, expected_weight);
          weight_errors++;
          ht_outputs_valid = false;
        }
      }
    } else {
      for (int i = 0; i < num_tokens; i++) {
        for (int j = 0; j < top_k; j++) {
          int offset = i * top_k + j;
          if (recv_topk_weights_host[offset] != expected_weight) {
            if (weight_errors < 5)
              printf("Rank %d: recv_topk_weights[%d][%d] = %f, expected %f\n",
                     myRank, i, j, recv_topk_weights_host[offset], expected_weight);
            weight_errors++;
            ht_outputs_valid = false;
          }
          int64_t idx_val = recv_topk_idx_host[offset];
          if (idx_val < 0 || idx_val >= static_cast<int64_t>(num_experts)) {
            if (idx_errors < 5)
              printf("Rank %d: recv_topk_idx[%d][%d] = %ld, expected range [0, %u)\n",
                     myRank, i, j, static_cast<long>(idx_val), num_experts);
            idx_errors++;
            ht_outputs_valid = false;
          }
        }
      }
    }

    if (weight_errors > 0) {
      printf("Rank %d: recv_topk_weights verification failed with %d errors\n", myRank, weight_errors);
    }
    if (idx_errors > 0) {
      printf("Rank %d: recv_topk_idx verification failed with %d errors\n", myRank, idx_errors);
    }

    if (ht_outputs_valid) {
      printf("Rank %d: %s mode recv_topk_weights%s verification passed\n",
             myRank, algorithm_name, ht_em ? "" : " and recv_topk_idx");
    } else {
      dispatch_check_passed = false;
    }

    delete[] recv_topk_weights_host;
    delete[] recv_topk_idx_host;
  }

  if (random_mode) {
    printf("Rank %d: %s Dispatch flow completed (random mode, checks skipped)\n", myRank, algorithm_name);
  } else if (dispatch_check_passed) {
    printf("Rank %d: %s Dispatch flow passed successfully\n", myRank, algorithm_name);
  } else {
    printf("Rank %d: Exiting test due to dispatch failure\n", myRank);
    exit(EXIT_FAILURE);
  }

  // Verify combine output - check ELEMENTS_TESTED_PER_TOKEN elements per token.
  // ncclEpCombine sums slot contributions unweighted; caller is expected to pre-weight
  // expert_outputs. ep_test passes a CONSTANT expert_outputs, so result = N * (j+1)*2
  // where N = slot count per home token, which varies by HT layout:
  //   FLAT: 1 slot per token (all top_k matches collapse into one source-rank slot
  //         in this deterministic routing).
  //   EM:   top_k slots per token (one per (token, local_expert) pair).
  int combine_errors = 0;
  if (!random_mode) {
    uint16_t *combined_output_host = new uint16_t[num_tokens * hidden]();
    void* combined_output_data;
    combined_output_data = combined_output->data;
    CUDACHECK(cudaMemcpy(combined_output_host, combined_output_data, num_tokens * hidden * 2, cudaMemcpyDeviceToHost));

    const int slot_count = ht_em ? top_k : 1;
    for (int i = 0; i < num_tokens; ++i) {
      for (int j = 0; j < ELEMENTS_TESTED_PER_TOKEN; ++j) {
        uint16_t expected = float_to_bf16(static_cast<float>(slot_count * (j + 1) * 2));
        uint16_t actual = (combined_output_host[i * hidden + j]);
        if (actual != expected) {
          printf("Combine check failed! Rank %d, token %d, element %d: expected %d, got %d\n",
                 myRank, i, j, expected, actual);
          combine_errors++;
          if (combine_errors >= 5) break; // Limit error output
        }
      }
      if (combine_errors >= 5) break;
    }
    delete[] combined_output_host;
  }

  if (random_mode) {
    printf("Rank %d: Combine flow completed (random mode, checks skipped)\n", myRank);
  } else if (combine_errors == 0) {
    printf("Rank %d: Combine verification PASSED! All %d tokens with %d elements each correctly combined\n",
           myRank, num_tokens, hidden);
  } else {
    printf("Rank %d: Combine verification FAILED with %d errors\n", myRank, combine_errors);
    printf("Rank %d: Exiting test due to combine failure\n", myRank);
    exit(EXIT_FAILURE);
  }

  if (use_cuda_graph) {
    CUDACHECK(cudaGraphExecDestroy(graph_exec));
    CUDACHECK(cudaGraphDestroy(graph));
  }

  // Cached mode test: repeat dispatch and combine calls with the same inputs but new outputs
  // Compare results between first and second phase to verify cached mode correctness
  if (cached_mode) {
    // Cached mode is only supported in HT modes (not LL)
    if (algorithm == NCCL_EP_ALGO_LOW_LATENCY) {
      printf("Rank %d: Error - cached mode is only supported in HT modes (not LL)\n", myRank);
      exit(EXIT_FAILURE);
    }
    // HT+EM cached path uses backward combine which expects 2D combine_input topk_weights;
    // dispatch recv_topk_weights under EM is 1D. Skip until EM combine bwd path is wired separately.
    if (ht_em) {
      printf("Rank %d: cached mode skipped under HT+EM (1D recv_topk_weights not yet "
             "wired into combine bwd input)\n", myRank);
    } else {

    printf("Rank %d: Testing cached mode (%s)\n", myRank, algorithm_name);

    // Save first phase dispatch outputs to host for comparison (only if not in random mode)
    uint16_t *first_dispatch_output0 = nullptr;
    float *first_dispatch_output1 = nullptr;
    int64_t *first_dispatch_output2 = nullptr;
    uint16_t *first_combine_output = nullptr;

    if (!random_mode) {
      // Per-layout sizes: HT+EM weights are 1D [N], no recv_topk_idx.
      const size_t weights_count = ht_em ? static_cast<size_t>(num_recv_tokens)
                                          : static_cast<size_t>(num_recv_tokens) * top_k;
      first_dispatch_output0 = new uint16_t[num_recv_tokens * hidden];
      first_dispatch_output1 = new float[weights_count];
      if (!ht_em) first_dispatch_output2 = new int64_t[num_recv_tokens * top_k];

      void* out0_data;
      out0_data = recv_x->data;
      CUDACHECK(cudaMemcpy(first_dispatch_output0, out0_data,
                           num_recv_tokens * hidden * sizeof(uint16_t), cudaMemcpyDeviceToHost));
      void* out1_data;
      out1_data = recv_topk_weights->data;
      CUDACHECK(cudaMemcpy(first_dispatch_output1, out1_data,
                           weights_count * sizeof(float), cudaMemcpyDeviceToHost));
      if (!ht_em) {
        void* out2_data;
        out2_data = recv_topk_idx->data;
        CUDACHECK(cudaMemcpy(first_dispatch_output2, out2_data,
                             num_recv_tokens * top_k * sizeof(int64_t), cudaMemcpyDeviceToHost));
      }

      // Save first phase combine output to host for comparison
      first_combine_output = new uint16_t[num_tokens * hidden];
      void* comb_out_data;
      comb_out_data = combined_output->data;
      CUDACHECK(cudaMemcpy(first_combine_output, comb_out_data,
                           num_tokens * hidden * sizeof(uint16_t), cudaMemcpyDeviceToHost));
    }

    // Allocate new output tensors for second phase dispatch.
    // HT+FL forward dispatch requires recv_topk_weights [N, top_k] and recv_topk_idx [N, top_k].
    ncclEpTensor_t* cached_recv_x = nullptr;
    NCCLCHECK(epMakeTensor(&cached_recv_x, 2, ncclBfloat16,
                           static_cast<unsigned int>(num_recv_tokens), static_cast<unsigned int>(hidden)));
    ncclEpTensor_t* cached_recv_topk_weights = nullptr;
    NCCLCHECK(epMakeTensor(&cached_recv_topk_weights, 2, ncclFloat32,
                           static_cast<unsigned int>(num_recv_tokens), static_cast<unsigned int>(top_k)));
    ncclEpTensor_t* cached_recv_topk_idx = nullptr;
    NCCLCHECK(epMakeTensor(&cached_recv_topk_idx, 2, ncclInt64,
                           static_cast<unsigned int>(num_recv_tokens), static_cast<unsigned int>(top_k)));
    ncclEpDispatchOutputs_t cached_dispatch_outputs = NCCL_EP_DISPATCH_OUTPUTS_INIT;
    cached_dispatch_outputs.tokens         = cached_recv_x;
    cached_dispatch_outputs.topk_weights   = cached_recv_topk_weights;
    cached_dispatch_outputs.topk_idx       = cached_recv_topk_idx;

    // Allocate new output tensors for second phase combine
    ncclEpTensor_t* cached_combined_output = nullptr;
    NCCLCHECK(epMakeTensor(&cached_combined_output, 2, ncclBfloat16,
                           static_cast<unsigned int>(num_tokens), static_cast<unsigned int>(hidden)));
    ncclEpTensor_t* cached_combined_topk_weights = nullptr;
    NCCLCHECK(epMakeTensor(&cached_combined_topk_weights, 2, ncclFloat32,
                           static_cast<unsigned int>(num_tokens), static_cast<unsigned int>(top_k)));
    ncclEpCombineOutputs_t cached_combine_outputs = NCCL_EP_COMBINE_OUTPUTS_INIT;
    cached_combine_outputs.tokens       = cached_combined_output;
    cached_combine_outputs.topk_weights = cached_combined_topk_weights;

    // Setup combine inputs with per-received-token topk_weights from dispatch output.
    // HT backward combine expects combine_inputs.topk_weights to align with combine_inputs.tokens.
    ncclEpTensor_t* cached_combine_topk_weights_input = nullptr;
    NCCLCHECK(epMakeTensor(&cached_combine_topk_weights_input, 2, ncclFloat32,
                           static_cast<unsigned int>(num_recv_tokens), static_cast<unsigned int>(top_k)));
    void* cached_ctwi_data;
    cached_ctwi_data = cached_combine_topk_weights_input->data;
    void* out1_data_for_copy;
    out1_data_for_copy = recv_topk_weights->data;
    const size_t cached_w_count = ht_em ? static_cast<size_t>(num_recv_tokens)
                                          : static_cast<size_t>(num_recv_tokens) * top_k;
    CUDACHECK(cudaMemcpy(cached_ctwi_data, out1_data_for_copy,
                         cached_w_count * sizeof(float), cudaMemcpyDeviceToDevice));

    ncclEpCombineInputs_t cached_combine_inputs = NCCL_EP_COMBINE_INPUTS_INIT;
    cached_combine_inputs.tokens       = expert_outputs;
    cached_combine_inputs.topk_weights = cached_combine_topk_weights_input;

    printf("Rank %d: Testing cached mode - second ncclEpDispatch call (send_only=%s)\n",
           myRank, dispatch_send_only ? "true" : "false");
    NCCLCHECK(ncclEpDispatch(ep_handle, &dispatch_inputs, &cached_dispatch_outputs,
                             nullptr, &dispatch_config, s));

    printf("Rank %d: Testing cached mode - second ncclEpComplete (dispatch)\n", myRank);
    NCCLCHECK(ncclEpComplete(ep_handle, nullptr, s));
    CUDACHECK(cudaStreamSynchronize(s));

    printf("Rank %d: Testing cached mode - second ncclEpCombine call (send_only=%s)\n",
           myRank, combine_send_only ? "true" : "false");
    // Cached combine exercises the BWD path: topk_weights provided on both sides.
    ncclEpCombineConfig_t cached_combine_config = combine_config;
    cached_combine_config.pass_direction = NCCL_EP_BWD_PASS;
    NCCLCHECK(ncclEpCombine(ep_handle, &cached_combine_inputs, &cached_combine_outputs,
                            &cached_combine_config, s));

    printf("Rank %d: Testing cached mode - second ncclEpComplete (combine)\n", myRank);
    NCCLCHECK(ncclEpComplete(ep_handle, nullptr, s));
    CUDACHECK(cudaStreamSynchronize(s));

    // Copy second phase outputs to host for comparison (only if not in random mode)
    int cached_dispatch_errors = 0;
    int cached_combine_errors = 0;

    if (!random_mode) {
      uint16_t *second_dispatch_output0 = new uint16_t[num_recv_tokens * hidden];
      void* cached_out0_data;
      cached_out0_data = cached_recv_x->data;
      CUDACHECK(cudaMemcpy(second_dispatch_output0, cached_out0_data,
                           num_recv_tokens * hidden * sizeof(uint16_t), cudaMemcpyDeviceToHost));

      uint16_t *second_combine_output = new uint16_t[num_tokens * hidden];
      void* cached_comb_out_data;
      cached_comb_out_data = cached_combined_output->data;
      CUDACHECK(cudaMemcpy(second_combine_output, cached_comb_out_data,
                           num_tokens * hidden * sizeof(uint16_t), cudaMemcpyDeviceToHost));
      float *second_combine_topk_weights = new float[num_tokens * top_k];
      void* cached_comb_tw_data;
      cached_comb_tw_data = cached_combined_topk_weights->data;
      CUDACHECK(cudaMemcpy(second_combine_topk_weights, cached_comb_tw_data,
                           num_tokens * top_k * sizeof(float), cudaMemcpyDeviceToHost));

      // Compare dispatch outputs between first and second phase
      for (unsigned int i = 0; i < num_recv_tokens * hidden; ++i) {
        if (first_dispatch_output0[i] != second_dispatch_output0[i]) {
          if (cached_dispatch_errors < 5) {
            printf("Rank %d: Cached dispatch output0 mismatch at %u: first=%u, second=%u\n",
                   myRank, i, first_dispatch_output0[i], second_dispatch_output0[i]);
          }
          cached_dispatch_errors++;
        }
      }

      // Compare combine outputs between first and second phase
      for (unsigned int i = 0; i < num_tokens * hidden; ++i) {
        if (first_combine_output[i] != second_combine_output[i]) {
          if (cached_combine_errors < 5) {
            printf("Rank %d: Cached combine output mismatch at %u: first=%u, second=%u\n",
                   myRank, i, first_combine_output[i], second_combine_output[i]);
          }
          cached_combine_errors++;
        }
      }

      // Verify combined topk_weights output (should match input topk_weights: 1.0f / top_k)
      float expected_weight = 1.0f / top_k;
      for (int i = 0; i < num_tokens * top_k; ++i) {
        if (second_combine_topk_weights[i] != expected_weight) {
          if (cached_combine_errors < 5) {
            printf("Rank %d: Cached combine topk_weights mismatch at %d: expected=%f, got=%f\n",
                   myRank, i, expected_weight, second_combine_topk_weights[i]);
          }
          cached_combine_errors++;
        }
      }

      delete[] second_dispatch_output0;
      delete[] second_combine_output;
      delete[] second_combine_topk_weights;
    }

    if (random_mode) {
      printf("Rank %d: Cached mode completed (random mode, checks skipped)\n", myRank);
    } else if (cached_dispatch_errors == 0 && cached_combine_errors == 0) {
      printf("Rank %d: Cached mode verification PASSED - dispatch and combine outputs match\n", myRank);
    } else {
      printf("Rank %d: Cached mode verification FAILED - dispatch errors: %d, combine errors: %d\n",
             myRank, cached_dispatch_errors, cached_combine_errors);
      exit(EXIT_FAILURE);
    }

    // Clean up cached mode tensors
    if (!random_mode) {
      delete[] first_dispatch_output0;
      delete[] first_dispatch_output1;
      delete[] first_dispatch_output2;
      delete[] first_combine_output;
    }
    epFreeTensor(&cached_recv_x);
    epFreeTensor(&cached_recv_topk_weights);
    epFreeTensor(&cached_recv_topk_idx);
    epFreeTensor(&cached_combined_output);
    epFreeTensor(&cached_combined_topk_weights);
    epFreeTensor(&cached_combine_topk_weights_input);

    printf("Rank %d: Cached mode - second dispatch and combine calls completed successfully\n", myRank);
    }  // end !ht_em else branch
  }

  // Clean up first phase host buffer (kept for cached mode comparison in HT mode)
  if (first_dispatch_output0_host) delete[] first_dispatch_output0_host;

  // Clean up combine tensors
  epFreeTensor(&expert_outputs);
  epFreeTensor(&topk_weights);
  epFreeTensor(&combined_output);

  NCCLCHECK(ncclEpHandleDestroy(ep_handle));

  NCCLCHECK(ncclEpGroupDestroy(ep_group));

  // finalizing NCCL
  ncclCommDestroy(comm);

  delete[] topk_idx_host;
  epFreeTensor(&topk_idx);
  if (disable_max_tokens && handle_recv_expert_counter != nullptr) {
    epFreeTensor(&handle_recv_expert_counter);
  }
  epFreeTensor(&input_tokens);
  epFreeTensor(&recv_x);
  if (algorithm != NCCL_EP_ALGO_LOW_LATENCY) {
    epFreeTensor(&recv_topk_weights);
    if (!ht_em) epFreeTensor(&recv_topk_idx);
  }
  if (algorithm == NCCL_EP_ALGO_LOW_LATENCY) {
    epFreeTensor(&ll_recv_expert_counter);
  }

  // finalizing MPI
  MPICHECK(MPI_Finalize());

  // Needed for cuda-memcheck --leak-check full
  cudaDeviceReset();

  printf("[MPI Rank %d] Success \n", myRank);
  return 0;
}
