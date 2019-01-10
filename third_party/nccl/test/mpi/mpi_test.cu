/*************************************************************************
 * Copyright (c) 2015-2016, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENCE.txt for license information
 ************************************************************************/

#include <sys/types.h>
#include <unistd.h>
#include <stdio.h>

#include "nccl.h"
#include "mpi.h"
#include "test_utilities.h"

#define SIZE 128
#define NITERS 1

int main(int argc, char *argv[]) {
  ncclUniqueId commId;
  int size, rank;
  ncclResult_t ret;

  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  if (argc < size) {
    if (rank == 0)
      printf("Usage : %s <GPU list per rank>\n", argv[0]);
    exit(1);
  }

  int gpu = atoi(argv[rank+1]);

  // We have to set our device before NCCL init
  CUDACHECK(cudaSetDevice(gpu));
  MPI_Barrier(MPI_COMM_WORLD);

  // NCCL Communicator creation
  ncclComm_t comm;
  NCCLCHECK(ncclGetUniqueId(&commId));
  MPI_Bcast(&commId, NCCL_UNIQUE_ID_BYTES, MPI_CHAR, 0, MPI_COMM_WORLD);
  ret = ncclCommInitRank(&comm, size, commId, rank);
  if (ret != ncclSuccess) {
    printf("NCCL Init failed (%d) '%s'\n", ret, ncclGetErrorString(ret));
    exit(1);
  }

  // CUDA stream creation
  cudaStream_t stream;
  CUDACHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

  // Initialize input values
  int *dptr;
  CUDACHECK(cudaMalloc(&dptr, SIZE*2*sizeof(int)));
  int *val = (int*) malloc(SIZE*sizeof(int));
  for (int v=0; v<SIZE; v++) {
    val[v] = rank + 1;
  }
  CUDACHECK(cudaMemcpy(dptr, val, SIZE*sizeof(int), cudaMemcpyHostToDevice));

  // Compute final value
  int ref = size*(size+1)/2;

  // Run allreduce
  int errors = 0;
  for (int i=0; i<NITERS; i++) {
    NCCLCHECK(ncclAllReduce((const void*)dptr, (void*)(dptr+SIZE), SIZE, ncclInt, ncclSum, comm, stream));
  }

  // Check results
  CUDACHECK(cudaStreamSynchronize(stream));
  CUDACHECK(cudaMemcpy(val, (dptr+SIZE), SIZE*sizeof(int), cudaMemcpyDeviceToHost));
  for (int v=0; v<SIZE; v++) {
    if (val[v] != ref) {
      errors++;
      printf("[%d] Error at %d : got %d instead of %d\n", rank, v, val[v], ref);
    }
  }
  CUDACHECK(cudaFree(dptr));

  MPI_Allreduce(MPI_IN_PLACE, &errors, 1, MPI_INTEGER, MPI_SUM, MPI_COMM_WORLD);
  if (rank == 0) {
    if (errors)
      printf("%d errors. Test FAILED.\n", errors);
    else
      printf("Test PASSED.\n");
  }

  MPI_Finalize();
  ncclCommDestroy(comm);
  return errors ? 1 : 0;
}
