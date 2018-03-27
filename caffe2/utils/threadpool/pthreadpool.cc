/* Standard C headers */
#include <stdint.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

/* POSIX headers */
#include <pthread.h>
#include <unistd.h>

/* Library header */
#include "caffe2/core/logging.h"
#include "caffe2/utils/fixed_divisor.h"
#include "caffe2/utils/threadpool/pthreadpool.h"


static inline size_t divide_round_up(size_t dividend, size_t divisor) {
  if (dividend % divisor == 0) {
    return dividend / divisor;
  } else {
    return dividend / divisor + 1;
  }
}

static inline size_t min(size_t a, size_t b) {
  return a < b ? a : b;
}

struct compute_1d_tiled_context {
  pthreadpool_function_1d_tiled_t function;
  void* argument;
  size_t range;
  size_t tile;
};

static void compute_1d_tiled(const struct compute_1d_tiled_context* context, size_t linear_index) {
  const size_t tile_index = linear_index;
  const size_t index = tile_index * context->tile;
  const size_t tile = min(context->tile, context->range - index);
  context->function(context->argument, index, tile);
}

void pthreadpool_compute_1d_tiled(
  pthreadpool_t threadpool,
  pthreadpool_function_1d_tiled_t function,
  void* argument,
  size_t range,
  size_t tile)
{
  if (threadpool == NULL) {
    /* No thread pool provided: execute function sequentially on the calling thread */
    for (size_t i = 0; i < range; i += tile) {
      function(argument, i, min(range - i, tile));
    }
  } else {
    /* Execute in parallel on the thread pool using linearized index */
    const size_t tile_range = divide_round_up(range, tile);
    struct compute_1d_tiled_context context = {
      .function = function,
      .argument = argument,
      .range = range,
      .tile = tile
    };
    pthreadpool_compute_1d(threadpool, (pthreadpool_function_1d_t) compute_1d_tiled, &context, tile_range);
  }
}

struct compute_2d_context {
  pthreadpool_function_2d_t function;
  void* argument;
  caffe2::FixedDivisor<int> range_j;
};

static void compute_2d(const struct compute_2d_context* context, size_t linear_index) {
  DCHECK_LE(linear_index, std::numeric_limits<int>::max());

  int q;
  int r;
  context->range_j.divMod((int) linear_index, q, r);
  context->function(context->argument, q, r);
}

void pthreadpool_compute_2d(
  struct pthreadpool* threadpool,
  pthreadpool_function_2d_t function,
  void* argument,
  size_t range_i,
  size_t range_j)
{
  if (threadpool == NULL) {
    /* No thread pool provided: execute function sequentially on the calling thread */
    for (size_t i = 0; i < range_i; i++) {
      for (size_t j = 0; j < range_j; j++) {
        function(argument, i, j);
      }
    }
  } else {
    DCHECK_LE(range_i * range_j, (size_t) std::numeric_limits<int>::max());
    /* Execute in parallel on the thread pool using linearized index */
    struct compute_2d_context context = {
      .function = function,
      .argument = argument,
      .range_j = caffe2::FixedDivisor<int>(range_j)
    };
    pthreadpool_compute_1d(threadpool, (pthreadpool_function_1d_t) compute_2d, &context, range_i * range_j);
  }
}

struct compute_2d_tiled_context {
  pthreadpool_function_2d_tiled_t function;
  void* argument;
  caffe2::FixedDivisor<int> tile_range_j;
  size_t range_i;
  size_t range_j;
  size_t tile_i;
  size_t tile_j;
};

static void compute_2d_tiled(const struct compute_2d_tiled_context* context, size_t linear_index) {
  int q;
  int r;

  context->tile_range_j.divMod(linear_index, q, r);
  const size_t max_tile_i = context->tile_i;
  const size_t max_tile_j = context->tile_j;
  const size_t index_i = q * max_tile_i;
  const size_t index_j = r * max_tile_j;
  const size_t tile_i = min(max_tile_i, context->range_i - index_i);
  const size_t tile_j = min(max_tile_j, context->range_j - index_j);
  context->function(context->argument, index_i, index_j, tile_i, tile_j);
}

void pthreadpool_compute_2d_tiled(
  pthreadpool_t threadpool,
  pthreadpool_function_2d_tiled_t function,
  void* argument,
  size_t range_i,
  size_t range_j,
  size_t tile_i,
  size_t tile_j)
{
  if (threadpool == NULL) {
    /* No thread pool provided: execute function sequentially on the calling thread */
    for (size_t i = 0; i < range_i; i += tile_i) {
      for (size_t j = 0; j < range_j; j += tile_j) {
        function(argument, i, j, min(range_i - i, tile_i), min(range_j - j, tile_j));
      }
    }
  } else {
    /* Execute in parallel on the thread pool using linearized index */
    const size_t tile_range_i = divide_round_up(range_i, tile_i);
    const size_t tile_range_j = divide_round_up(range_j, tile_j);
    DCHECK_LE(tile_range_i * tile_range_j, (size_t) std::numeric_limits<int>::max());
    struct compute_2d_tiled_context context = {
      .function = function,
      .argument = argument,
      .tile_range_j = caffe2::FixedDivisor<int>(tile_range_j),
      .range_i = range_i,
      .range_j = range_j,
      .tile_i = tile_i,
      .tile_j = tile_j
    };
    pthreadpool_compute_1d(threadpool, (pthreadpool_function_1d_t) compute_2d_tiled, &context, tile_range_i * tile_range_j);
  }
}
