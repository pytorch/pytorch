/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <cuda.h>
#include <cuda_runtime_api.h>

#include <stdexcept>
#include <string>

#include "nccl.h"

namespace kineto_stress_test {

inline void checkCudaStatus(cudaError_t status, int lineNumber = -1) {
  if (status != cudaSuccess) {
    printf(
        "PID %d --> CUDA API failed with status %d: %s at line %d\n",
        getpid(),
        status,
        cudaGetErrorString(status),
        lineNumber);
    exit(EXIT_FAILURE);
  }
}

#define CUDA_CHECK(EXPR)                            \
  do {                                              \
    const cudaError_t err = EXPR;                   \
    if (err == cudaSuccess) {                       \
      break;                                        \
    }                                               \
    std::string error_message;                      \
    error_message.append(__FILE__);                 \
    error_message.append(":");                      \
    error_message.append(std::to_string(__LINE__)); \
    error_message.append(" CUDA error: ");          \
    error_message.append(cudaGetErrorString(err));  \
    throw std::runtime_error(error_message);        \
  } while (0)

#define CUDA_KERNEL_LAUNCH_CHECK() CUDA_CHECK(cudaGetLastError())

#define MPICHECK(cmd)                                  \
  do {                                                 \
    int e = cmd;                                       \
    if (e != MPI_SUCCESS) {                            \
      printf(                                          \
          "PID %d --> Failed: MPI error %s:%d '%d'\n", \
          getpid(),                                    \
          __FILE__,                                    \
          __LINE__,                                    \
          e);                                          \
      exit(EXIT_FAILURE);                              \
    }                                                  \
  } while (0)

#define NCCLCHECK(cmd)                                  \
  do {                                                  \
    ncclResult_t r = cmd;                               \
    if (r != ncclSuccess) {                             \
      printf(                                           \
          "PID %d --> Failed, NCCL error %s:%d '%s'\n", \
          getpid(),                                     \
          __FILE__,                                     \
          __LINE__,                                     \
          ncclGetErrorString(r));                       \
      exit(EXIT_FAILURE);                               \
    }                                                   \
  } while (0)

} // namespace kineto_stress_test
