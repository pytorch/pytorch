/*************************************************************************
 * Copyright (c) 2016, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 ************************************************************************/

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <float.h>

#include "nccl.h"
#include "test_utilities.h"
#include <nvToolsExt.h>

void showUsage(const char* bin) {
  printf("\n"
         "Usage: %s <type> <n_min> <n_max> [delta] [gpus] [gpu0 [gpu1 [...]]]\n"
         "Where:\n"
#ifdef CUDA_HAS_HALF
         "    type   =   [char|int|half|float|double|int64|uint64]\n"
#else
         "    type   =   [char|int|float|double|int64|uint64]\n"
#endif
         "    n_min  >   0\n"
         "    n_max  >=  n_min\n"
         "    delta  >   0\n\n", bin);
  return;
}

int main(int argc, char* argv[]) {
  int nvis = 0;
  CUDACHECK(cudaGetDeviceCount(&nvis));
  if (nvis == 0) {
    printf("No GPUs found\n");
    showUsage(argv[0]);
    exit(EXIT_FAILURE);
  }

  ncclDataType_t type;
  int n_min;
  int n_max;
  int delta;
  int gpus;
  int* list = NULL;

  if (argc < 4) {
    showUsage(argv[0]);
    exit(EXIT_FAILURE);
  }

  type = strToType(argv[1]);
  if (type == nccl_NUM_TYPES) {
    printf("Invalid <type> '%s'\n", argv[1]);
    showUsage(argv[0]);
    exit(EXIT_FAILURE);
  }

  n_min = strToPosInt(argv[2]);
  if (n_min < 1) {
    printf("Invalid <n_min> '%s'\n", argv[2]);
    showUsage(argv[0]);
    exit(EXIT_FAILURE);
  }

  n_max = strToPosInt(argv[3]);
  if (n_max < n_min) {
    printf("Invalid <n_max> '%s'\n", argv[3]);
    showUsage(argv[0]);
    exit(EXIT_FAILURE);
  }

  if (argc > 4) {
    delta = strToPosInt(argv[4]);
    if (delta < 1) {
      printf("Invalid <delta> '%s'\n", argv[4]);
      showUsage(argv[0]);
      exit(EXIT_FAILURE);
    }
  } else {
    delta = (n_max == n_min) ? 1 : (n_max - n_min+9) / 10;
  }

  if (argc > 5) {
    gpus = strToPosInt(argv[5]);
    if (gpus < 1) {
      printf("Invalid <gpus> '%s'\n", argv[5]);
      showUsage(argv[0]);
      exit(EXIT_FAILURE);
    }
  } else {
    gpus = nvis;
  }

  list = (int*)malloc(gpus*sizeof(int));

  if (argc > 6 && argc != 6+gpus) {
    printf("If given, GPU list must be fully specified.\n");
    showUsage(argv[0]);
    exit(EXIT_FAILURE);
  }

  for(int g=0; g<gpus; ++g) {
    if(argc > 6) {
      list[g] = strToNonNeg(argv[6+g]);
      if (list[g] < 0) {
        printf("Invalid GPU%d '%s'\n", g, argv[6+g]);
        showUsage(argv[0]);
        exit(EXIT_FAILURE);
      } else if (list[g] >= nvis) {
        printf("GPU%d (%d) exceeds visible devices (%d)\n", g, list[g], nvis);
        showUsage(argv[0]);
        exit(EXIT_FAILURE);
      }
    } else {
      list[g] = g % nvis;
    }
  }

  size_t word = wordSize(type);
  size_t max_input = n_max * word;
  size_t max_output = max_input * gpus;
  void* refout;
  CUDACHECK(cudaMallocHost(&refout, max_output));

  void **input, **output;
  double** localError;
  ncclComm_t* comm;
  cudaStream_t* stream;

  input = (void**)malloc(gpus*sizeof(void*));
  output = (void**)malloc(gpus*sizeof(void*));
  localError = (double**)malloc(gpus*sizeof(double*));
  comm = (ncclComm_t*)malloc(gpus*sizeof(ncclComm_t));
  stream = (cudaStream_t*)malloc(gpus*sizeof(cudaStream_t));

  for(int g=0; g<gpus; ++g) {
    char busid[32] = {0};
    CUDACHECK(cudaDeviceGetPCIBusId(busid, 32, list[g]));
    printf("# Rank %d using device %d [%s]\n", g, list[g], busid);

    CUDACHECK(cudaSetDevice(list[g]));
    CUDACHECK(cudaMalloc(&input[g],  max_input));
    CUDACHECK(cudaMalloc(&output[g], max_output));
    CUDACHECK(cudaMallocHost(&localError[g], sizeof(double)));
    CUDACHECK(cudaStreamCreate(&stream[g]));
    makeRandom(input[g], n_max, type, 42+g);

    CUDACHECK(cudaMemcpy((char*)refout+max_input*g, input[g], max_input, cudaMemcpyDeviceToHost));
  }

  NCCLCHECK(ncclCommInitAll(comm, gpus, list));

  printf("       BYTES ERROR       MSEC     BW\n");

  for(int n=n_min; n<=n_max; n+=delta) {
    size_t out_bytes = word * n * gpus;

    for(int g=0; g<gpus; ++g) {
      CUDACHECK(cudaSetDevice(list[g]));
      CUDACHECK(cudaMemsetAsync(output[g], 0, out_bytes, stream[g]));
      CUDACHECK(cudaStreamSynchronize(stream[g]));
    }

    auto start = std::chrono::high_resolution_clock::now();
    for(int g=0; g<gpus; ++g) {
      CUDACHECK(cudaSetDevice(list[g]));
      NCCLCHECK(ncclAllGather(input[g], n, type, output[g], comm[g], stream[g]));
    }
    for(int g=0; g<gpus; ++g) {
      CUDACHECK(cudaSetDevice(list[g]));
      CUDACHECK(cudaStreamSynchronize(stream[g]));
    }
    auto stop = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration_cast<std::chrono::duration<double>>
        (stop - start).count() * 1000.0;

    double max_error = 0.0;
    for(int slice=0; slice<gpus; ++slice) {
      void* refSlice = (void*)((char*)refout + slice*max_input);
      for(int g=0; g<gpus; ++g) {
        CUDACHECK(cudaSetDevice(list[g]));
        void* mySlice = (void*)((char*)output[g] + slice*n*word);
        maxDiff(localError[g], mySlice, refSlice, n, type, stream[g]);
      }
      for(int g=0; g<gpus; ++g) {
        CUDACHECK(cudaSetDevice(list[g]));
        CUDACHECK(cudaStreamSynchronize(stream[g]));
        max_error = max(max_error, *localError[g]);
      }
    }

    double mb = (double)(n*word * (gpus-1)) * 1.e-6;
    double algbw = mb / ms;
    printf("%12lu %5.0le %10.3lf %6.2lf\n",
        n*word, max_error, ms, algbw);
  }

  for(int g=0; g<gpus; ++g) {
    CUDACHECK(cudaSetDevice(list[g]));
    CUDACHECK(cudaStreamDestroy(stream[g]));
    ncclCommDestroy(comm[g]);
    CUDACHECK(cudaFree(input[g]));
    CUDACHECK(cudaFree(output[g]));
    CUDACHECK(cudaFreeHost(localError[g]));
  }

  free(localError);
  free(output);
  free(input);
  free(comm);
  free(stream);
  CUDACHECK(cudaFreeHost(refout));
  exit(EXIT_SUCCESS);
}

