// pthreadpool header from https://github.com/Maratyszcza/pthreadpool
// for NNPACK
#ifndef CAFFE2_UTILS_PTHREADPOOL_H_
#define CAFFE2_UTILS_PTHREADPOOL_H_

#include "ThreadPoolCommon.h"


#include <stddef.h> // for size_t

typedef struct pthreadpool* pthreadpool_t;

typedef void (*pthreadpool_function_1d_t)(void*, size_t);
typedef void (*pthreadpool_function_1d_tiled_t)(void*, size_t, size_t);
typedef void (*pthreadpool_function_2d_t)(void*, size_t, size_t);
typedef void (*pthreadpool_function_2d_tiled_t)(void*, size_t, size_t, size_t, size_t);
typedef void (*pthreadpool_function_3d_tiled_t)(
    void*,
    size_t,
    size_t,
    size_t,
    size_t,
    size_t,
    size_t);
typedef void (*pthreadpool_function_4d_tiled_t)(
    void*,
    size_t,
    size_t,
    size_t,
    size_t,
    size_t,
    size_t,
    size_t,
    size_t);

#include <stdint.h> // for uint32_t

typedef void (*pthreadpool_task_1d_t)(void*, size_t);
typedef void (*pthreadpool_task_1d_tile_1d_t)(void*, size_t, size_t);
typedef void (*pthreadpool_task_2d_t)(void*, size_t, size_t);
typedef void (*pthreadpool_task_2d_tile_1d_t)(void*, size_t, size_t, size_t);
typedef void (*pthreadpool_task_2d_tile_2d_t)(void*, size_t, size_t, size_t, size_t);
typedef void (*pthreadpool_task_3d_tile_2d_t)(
    void*,
    size_t,
    size_t,
    size_t,
    size_t,
    size_t);
typedef void (*pthreadpool_task_4d_tile_2d_t)(
    void*,
    size_t,
    size_t,
    size_t,
    size_t,
    size_t,
    size_t);
typedef void (*pthreadpool_task_5d_tile_2d_t)(
    void*,
    size_t,
    size_t,
    size_t,
    size_t,
    size_t,
    size_t,
    size_t);
typedef void (*pthreadpool_task_6d_tile_2d_t)(
    void*,
    size_t,
    size_t,
    size_t,
    size_t,
    size_t,
    size_t,
    size_t,
    size_t);

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Creates a thread pool with the specified number of threads.
 *
 * @param[in]  threads_count  The number of threads in the thread pool.
 *    A value of 0 has special interpretation: it creates a thread for each
 *    processor core available in the system.
 *
 * @returns  A pointer to an opaque thread pool object.
 *    On error the function returns NULL and sets errno accordingly.
 */

//Returns internal threadpool impl.
pthreadpool_t pthreadpool_create(size_t threads_count);

/**
 * Queries the number of threads in a thread pool.
 *
 * @param[in]  threadpool  The thread pool to query.
 *
 * @returns  The number of threads in the thread pool.
 */
size_t pthreadpool_get_threads_count(pthreadpool_t threadpool);

/**
 * Processes items in parallel using threads from a thread pool.
 *
 * When the call returns, all items have been processed and the thread pool is
 * ready for a new task.
 *
 * @note If multiple threads call this function with the same thread pool, the
 *    calls are serialized.
 *
 * @param[in]  threadpool  The thread pool to use for parallelisation.
 * @param[in]  function    The function to call for each item.
 * @param[in]  argument    The first argument passed to the @a function.
 * @param[in]  items       The number of items to process. The @a function
 *    will be called once for each item.
 */
void pthreadpool_compute_1d(
    pthreadpool_t threadpool,
    pthreadpool_function_1d_t function,
    void* argument,
    size_t range);

void pthreadpool_compute_1d_tiled(
    pthreadpool_t threadpool,
    pthreadpool_function_1d_tiled_t function,
    void* argument,
    size_t range,
    size_t tile);

void pthreadpool_compute_2d(
    pthreadpool_t threadpool,
    pthreadpool_function_2d_t function,
    void* argument,
    size_t range_i,
    size_t range_j);

void pthreadpool_compute_2d_tiled(
    pthreadpool_t threadpool,
    pthreadpool_function_2d_tiled_t function,
    void* argument,
    size_t range_i,
    size_t range_j,
    size_t tile_i,
    size_t tile_j);

void pthreadpool_compute_3d_tiled(
    pthreadpool_t threadpool,
    pthreadpool_function_3d_tiled_t function,
    void* argument,
    size_t range_i,
    size_t range_j,
    size_t range_k,
    size_t tile_i,
    size_t tile_j,
    size_t tile_k);

void pthreadpool_compute_4d_tiled(
    pthreadpool_t threadpool,
    pthreadpool_function_4d_tiled_t function,
    void* argument,
    size_t range_i,
    size_t range_j,
    size_t range_k,
    size_t range_l,
    size_t tile_i,
    size_t tile_j,
    size_t tile_k,
    size_t tile_l);

/**
 * Terminates threads in the thread pool and releases associated resources.
 *
 * @warning  Accessing the thread pool after a call to this function constitutes
 *    undefined behaviour and may cause data corruption.
 *
 * @param[in,out]  threadpool  The thread pool to destroy.
 */
void pthreadpool_destroy(pthreadpool_t threadpool);

// New interface copy/pasted from pthreadpool.
// We will merge the internal and third-party/pthreadpool eventually.
// For now copy-paste to get past build issues.

#define PTHREADPOOL_FLAG_DISABLE_DENORMALS 0x00000001

// Returns the copied threadpool impl of third-party/pthreadpool
pthreadpool_t pthreadpool_create_xnnpack(size_t threads_count);

// Copied third-party impl.
size_t pthreadpool_get_threads_count_xnnpack(pthreadpool_t threadpool);

// Copied third-party impl.
void pthreadpool_destroy_xnnpack(pthreadpool_t threadpool);

/**
 * Processes items in parallel using threads from a thread pool.
 *
 * When the call returns, all items have been processed and the thread pool is
 * ready for a new task.
 *
 * @note If multiple threads call this function with the same thread pool, the
 *    calls are serialized.
 *
 * @param[in]  threadpool  The thread pool to use for parallelisation.
 * @param[in]  function    The function to call for each item.
 * @param[in]  argument    The first argument passed to the @a function.
 * @param[in]  items       The number of items to process. The @a function
 *    will be called once for each item.
 */
void pthreadpool_parallelize_1d(
    pthreadpool_t threadpool,
    pthreadpool_task_1d_t function,
    void* argument,
    size_t range,
    uint32_t flags);

void pthreadpool_parallelize_1d_tile_1d(
    pthreadpool_t threadpool,
    pthreadpool_task_1d_tile_1d_t function,
    void* argument,
    size_t range,
    size_t tile,
    uint32_t flags);

void pthreadpool_parallelize_2d(
    pthreadpool_t threadpool,
    pthreadpool_task_2d_t function,
    void* argument,
    size_t range_i,
    size_t range_j,
    uint32_t flags);

void pthreadpool_parallelize_2d_tile_1d(
    pthreadpool_t threadpool,
    pthreadpool_task_2d_tile_1d_t function,
    void* argument,
    size_t range_i,
    size_t range_j,
    size_t tile_j,
    uint32_t flags);

void pthreadpool_parallelize_2d_tile_2d(
    pthreadpool_t threadpool,
    pthreadpool_task_2d_tile_2d_t function,
    void* argument,
    size_t range_i,
    size_t range_j,
    size_t tile_i,
    size_t tile_j,
    uint32_t flags);

void pthreadpool_parallelize_3d_tile_2d(
    pthreadpool_t threadpool,
    pthreadpool_task_3d_tile_2d_t function,
    void* argument,
    size_t range_i,
    size_t range_j,
    size_t range_k,
    size_t tile_j,
    size_t tile_k,
    uint32_t flags);

void pthreadpool_parallelize_4d_tile_2d(
    pthreadpool_t threadpool,
    pthreadpool_task_4d_tile_2d_t function,
    void* argument,
    size_t range_i,
    size_t range_j,
    size_t range_k,
    size_t range_l,
    size_t tile_k,
    size_t tile_l,
    uint32_t flags);

void pthreadpool_parallelize_5d_tile_2d(
    pthreadpool_t threadpool,
    pthreadpool_task_5d_tile_2d_t function,
    void* argument,
    size_t range_i,
    size_t range_j,
    size_t range_k,
    size_t range_l,
    size_t range_m,
    size_t tile_l,
    size_t tile_m,
    uint32_t flags);

void pthreadpool_parallelize_6d_tile_2d(
    pthreadpool_t threadpool,
    pthreadpool_task_6d_tile_2d_t function,
    void* argument,
    size_t range_i,
    size_t range_j,
    size_t range_k,
    size_t range_l,
    size_t range_m,
    size_t range_n,
    size_t tile_m,
    size_t tile_n,
    uint32_t flags);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif // CAFFE2_UTILS_PTHREADPOOL_H_
