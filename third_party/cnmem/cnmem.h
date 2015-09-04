/* ********************************************************************** 
 * Copyright (c) 2015, NVIDIA CORPORATION. All rights reserved.
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
 * ********************************************************************** */
#pragma once

#ifdef __cplusplus
#include "cstdio"
#else
#include "stdio.h"
#endif
#include "cuda_runtime_api.h"

#if defined(_MSC_VER) || defined(WIN32)
#ifdef CNMEM_DLLEXPORT
#define CNMEM_API __declspec(dllexport)
#else
#define CNMEM_API __declspec(dllimport)
#endif
#else
#ifdef CNMEM_DLLEXPORT
#define CNMEM_API __attribute__((visibility ("default")))
#else
#define CNMEM_API
#endif
#endif

#define CNMEM_VERSION 100 // It corresponds to 1.0.0

#ifdef __cplusplus
extern "C" {
#endif

/* ********************************************************************************************* */

typedef enum
{
  CNMEM_STATUS_SUCCESS = 0,
  CNMEM_STATUS_CUDA_ERROR,
  CNMEM_STATUS_INVALID_ARGUMENT,
  CNMEM_STATUS_NOT_INITIALIZED,
  CNMEM_STATUS_OUT_OF_MEMORY,
  CNMEM_STATUS_UNKNOWN_ERROR
} cnmemStatus_t;

/* ********************************************************************************************* */

typedef enum
{
  CNMEM_FLAGS_DEFAULT = 0,       /// Default flags.
  CNMEM_FLAGS_CANNOT_GROW = 1,   /// Prevent the manager from growing its memory consumption.
  CNMEM_FLAGS_CANNOT_STEAL = 2,  /// Prevent the manager from stealing memory.
} cnmemManagerFlags_t;

/* ********************************************************************************************* */

typedef struct cnmemDevice_t_
{
  /** The device number. */
  int device;
  /** The size to allocate for that device. If 0, the implementation chooses the size. */
  size_t size;
  /** The number of named streams associated with the device. The NULL stream is not counted. */
  int numStreams;
  /** The streams associated with the device. It can be NULL. The NULL stream is managed. */
  cudaStream_t *streams;
  /** The size reserved for each streams. It can be 0. */
  size_t *streamSizes;

} cnmemDevice_t;

/**
 * \brief Initialize the library and allocate memory on the listed devices.
 *
 * For each device, an internal memory manager is created and the specified amount of memory is 
 * allocated (it is the size defined in device[i].size). For each, named stream an additional 
 * memory manager is created. Currently, it is implemented as a tree of memory managers: A root 
 * manager for the device and a list of children, one for each named stream.
 * 
 * This function must be called before any other function in the library. It has to be called 
 * by a single thread since it is not thread-safe.
 *
 * \return 
 * CNMEM_STATUS_SUCCESS,          if everything goes fine,
 * CNMEM_STATUS_INVALID_ARGUMENT, if one of the argument is invalid,
 * CNMEM_STATUS_OUT_OF_MEMORY,    if the requested size exceeds the available memory,
 * CNMEM_STATUS_CUDA_ERROR,       if an error happens in a CUDA function.
 */
cnmemStatus_t CNMEM_API cnmemInit(int numDevices, const cnmemDevice_t *devices, unsigned flags);

/**
 * \brief Release all the allocated memory. 
 * 
 * This function must be called by a single thread and after all threads that called 
 * cnmemMalloc/cnmemFree have joined. This function is not thread-safe.
 *
 * \return 
 * CNMEM_STATUS_SUCCESS,          if everything goes fine,
 * CNMEM_STATUS_NOT_INITIALIZED,  if the ::cnmemInit function has not been called,
 * CNMEM_STATUS_CUDA_ERROR,       if an error happens in one of the CUDA functions.
 */
cnmemStatus_t CNMEM_API cnmemFinalize();

/**
 * \brief Increase the internal reference counter of the context object.
 * 
 * This function increases the internal reference counter of the library. The purpose of that
 * reference counting mechanism is to give more control to the user over the lifetime of the 
 * library. It is useful with scoped memory allocation which may be destroyed in a final 
 * memory collection after the end of main(). That function is thread-safe.
 *
 * \return 
 * CNMEM_STATUS_SUCCESS,          if everything goes fine,
 * CNMEM_STATUS_NOT_INITIALIZED,  if the ::cnmemInit function has not been called,
 */
cnmemStatus_t CNMEM_API cnmemRetain();

/**
 * \brief Decrease the internal reference counter of the context object.
 * 
 * This function decreases the internal reference counter of the library. The purpose of that
 * reference counting mechanism is to give more control to the user over the lifetime of the 
 * library. It is useful with scoped memory allocation which may be destroyed in a final 
 * memory collection after the end of main(). That function is thread-safe.
 *
 * You can use \c cnmemRelease to explicitly finalize the library.
 *
 * \return 
 * CNMEM_STATUS_SUCCESS,          if everything goes fine,
 * CNMEM_STATUS_NOT_INITIALIZED,  if the ::cnmemInit function has not been called,
 */
cnmemStatus_t CNMEM_API cnmemRelease();

/**
 * \brief Add a new stream to the pool of managed streams on a device.
 *
 * This function registers a new stream into a device memory manager. It is thread-safe.
 *
 * \return 
 * CNMEM_STATUS_SUCCESS,          if everything goes fine,
 * CNMEM_STATUS_INVALID_ARGUMENT, if one of the argument is invalid,
 */
cnmemStatus_t CNMEM_API cnmemRegisterStream(cudaStream_t stream);

/**
 * \brief Allocate memory. 
 * 
 * This function allocates memory and initializes a pointer to device memory. If no memory 
 * is available, it returns a CNMEM_STATUS_OUT_OF_MEMORY error. This function is thread safe.
 *
 * The behavior of that function is the following: 
 *
 * - If the stream is NULL, the root memory manager is asked to allocate a buffer of device 
 *   memory. If there's a buffer of size larger or equal to the requested size in the list of 
 *   free blocks, it is returned. If there's no such buffer but the manager is allowed to grow 
 *   its memory usage (the CNMEM_FLAGS_CANNOT_GROW flag is not set), the memory manager calls 
 *   cudaMalloc. If cudaMalloc fails due to no more available memory or the manager is not 
 *   allowed to grow, the manager attempts to steal memory from one of its children (unless 
 *   CNMEM_FLAGS_CANNOT_STEAL is set). If that attempt also fails, the manager returns 
 *   CNMEM_STATUS_OUT_OF_MEMORY.
 * 
 * - If the stream is a named stream, the initial request goes to the memory manager associated 
 *   with that stream. If a free node is available in the lists of that manager, it is returned. 
 *   Otherwise, the request is passed to the root node and works as if the request were made on 
 *   the NULL stream.
 *
 * The calls to cudaMalloc are potentially costly and may induce GPU synchronizations. Also the 
 * mechanism to steal memory from the children induces GPU synchronizations (the manager has to 
 * make sure no kernel uses a given buffer before stealing it) and it the execution is 
 * sequential (in a multi-threaded context, the code is executed in a critical section inside
 * the cnmem library - no need for the user to wrap cnmemMalloc with locks).
 *
 * \return 
 * CNMEM_STATUS_SUCCESS,          if everything goes fine,
 * CNMEM_STATUS_NOT_INITIALIZED,  if the ::cnmemInit function has not been called,
 * CNMEM_STATUS_INVALID_ARGUMENT, if one of the argument is invalid. For example, ptr == 0,
 * CNMEM_STATUS_OUT_OF_MEMORY,    if there is not enough memory available,
 * CNMEM_STATUS_CUDA_ERROR,       if an error happens in one of the CUDA functions.
 */
cnmemStatus_t CNMEM_API cnmemMalloc(void **ptr, size_t size, cudaStream_t stream);

/**
 * \brief Release memory. 
 * 
 * This function releases memory and recycles a memory block in the manager. This function is 
 * thread safe. 
 *
 * \return 
 * CNMEM_STATUS_SUCCESS,          if everything goes fine,
 * CNMEM_STATUS_NOT_INITIALIZED,  if the ::cnmemInit function has not been called,
 * CNMEM_STATUS_INVALID_ARGUMENT, if one of the argument is invalid. For example, ptr == 0,
 * CNMEM_STATUS_CUDA_ERROR,       if an error happens in one of the CUDA functions.
 */
cnmemStatus_t CNMEM_API cnmemFree(void *ptr, cudaStream_t stream);

/* ********************************************************************************************* */
/* Utility functions.                                                                            */
/* ********************************************************************************************* */

/**
 * \brief Returns the amount of memory managed by the memory manager associated with a stream.
 * 
 * The pointers totalMem and freeMem must be valid. At the moment, this function has a comple-
 * xity linear in the number of allocated blocks so do not call it in performance critical 
 * sections. 
 *
 * \return 
 * CNMEM_STATUS_SUCCESS,          if everything goes fine,
 * CNMEM_STATUS_NOT_INITIALIZED,  if the ::cnmemInit function has not been called,
 * CNMEM_STATUS_INVALID_ARGUMENT, if one of the argument is invalid,
 * CNMEM_STATUS_CUDA_ERROR,       if an error happens in one of the CUDA functions.
 */
cnmemStatus_t CNMEM_API cnmemMemGetInfo(size_t *freeMem, size_t *totalMem, cudaStream_t stream);

/**
 * \brief Print a list of nodes to a file. 
 * 
 * This function is intended to be used in case of complex scenarios to help understand the 
 * behaviour of the memory managers/application. It is thread safe.
 *
 * \return 
 * CNMEM_STATUS_SUCCESS,          if everything goes fine,
 * CNMEM_STATUS_NOT_INITIALIZED,  if the ::cnmemInit function has not been called,
 * CNMEM_STATUS_INVALID_ARGUMENT, if one of the argument is invalid. For example, used_mem == 0 
 *                                or free_mem == 0,
 * CNMEM_STATUS_CUDA_ERROR,       if an error happens in one of the CUDA functions.
 */
cnmemStatus_t CNMEM_API cnmemPrintMemoryState(FILE *file, cudaStream_t stream);

/**
 * \brief Converts a cnmemStatus_t value to a string.
 */
const char CNMEM_API * cnmemGetErrorString(cnmemStatus_t status);

/* ********************************************************************************************* */

#ifdef __cplusplus
} // extern "C"
#endif

