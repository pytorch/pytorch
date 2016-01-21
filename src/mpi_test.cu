/*************************************************************************
 * Copyright (c) 2015-2016, NVIDIA CORPORATION. All rights reserved.
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

#include <sys/types.h>
#include <unistd.h>

#include "nccl.h"
#include "mpi.h"

#define CUDACHECK(cmd) do {                              \
    cudaError_t e = cmd;                                 \
    if( e != cudaSuccess ) {                             \
        printf("Cuda failure %s:%d '%s'\n",              \
               __FILE__,__LINE__,cudaGetErrorString(e)); \
        exit(EXIT_FAILURE);                              \
    }                                                    \
} while(false)

int main(int argc, char *argv[]) {
  ncclUniqueId commId;
  int size, rank;
  int ret;

  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  int gpu = atoi(argv[rank+1]);
  printf("MPI Rank %d running on GPU %d\n", rank, gpu);
  // We have to set our device before NCCL init
  CUDACHECK(cudaSetDevice(gpu));
  MPI_Barrier(MPI_COMM_WORLD);

  ncclComm_t comm;
  // Let's use rank 0 PID as job ID
  ncclGetUniqueId(&commId);
  MPI_Bcast(&commId, NCCL_UNIQUE_ID_BYTES, MPI_CHAR, 0, MPI_COMM_WORLD);
  ret = ncclCommInitRank(&comm, size, commId, rank);
  if (ret != ncclSuccess) {
    printf("NCCL Init failed : %d\n", ret);
    exit(1);
  }

  int *dptr;
  CUDACHECK(cudaMalloc(&dptr, 1024*2*sizeof(int)));
  int val = rank;
  CUDACHECK(cudaMemcpy(dptr, &val, sizeof(int), cudaMemcpyHostToDevice));

  ncclAllReduce((const void*)dptr, (void*)(dptr+1024), 1024, ncclInt, ncclSum, comm, cudaStreamDefault);

  CUDACHECK(cudaMemcpy(&val, (dptr+1024), sizeof(int), cudaMemcpyDeviceToHost));
  printf("Sum is %d\n", val);
  CUDACHECK(cudaFree(dptr));

  MPI_Finalize();
  ncclCommDestroy(comm);
  return 0;
}
