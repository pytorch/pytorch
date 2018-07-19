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
typedef void (*pthreadpool_function_3d_t)(void*, size_t, size_t, size_t);

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

/**
 * Terminates threads in the thread pool and releases associated resources.
 *
 * @warning  Accessing the thread pool after a call to this function constitutes
 *    undefined behaviour and may cause data corruption.
 *
 * @param[in,out]  threadpool  The thread pool to destroy.
 */
void pthreadpool_destroy(pthreadpool_t threadpool);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif // CAFFE2_UTILS_PTHREADPOOL_H_
