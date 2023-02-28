/* Standard C headers */
#include <stdint.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

#ifdef _MSC_VER
#include <cstdio>
#undef min
#else
/* POSIX headers */
#include <unistd.h>
#endif

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
  legacy_pthreadpool_function_1d_tiled_t function;
  void* argument;
  size_t range;
  size_t tile;
};

static void compute_1d_tiled(void* context_, size_t linear_index) {
  const struct compute_1d_tiled_context* context = (compute_1d_tiled_context*) context_;
  const size_t tile_index = linear_index;
  const size_t index = tile_index * context->tile;
  const size_t tile = min(context->tile, context->range - index);
  context->function(context->argument, index, tile);
}

void legacy_pthreadpool_compute_1d_tiled(
  legacy_pthreadpool_t threadpool,
  legacy_pthreadpool_function_1d_tiled_t function,
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
    struct compute_1d_tiled_context context = {/*.function = */ function,
                                               /*.argument = */ argument,
                                               /*.range = */ range,
                                               /*.tile = */ tile};
    legacy_pthreadpool_compute_1d(threadpool, (legacy_pthreadpool_function_1d_t) compute_1d_tiled, &context, tile_range);
  }
}

struct compute_2d_context {
  legacy_pthreadpool_function_2d_t function;
  void* argument;
  caffe2::FixedDivisor<int32_t> range_j;
};

static void compute_2d(void* context_, size_t linear_index) {
  TORCH_DCHECK_LE(linear_index, std::numeric_limits<int32_t>::max());

  const struct compute_2d_context* context = static_cast<compute_2d_context*>(context_);
  int32_t q;
  int32_t r;
  context->range_j.DivMod(static_cast<int32_t>(linear_index), &q, &r);
  context->function(context->argument, q, r);
}

void legacy_pthreadpool_compute_2d(
  legacy_pthreadpool_t threadpool,
  legacy_pthreadpool_function_2d_t function,
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
    TORCH_DCHECK_LE(range_i * range_j, (size_t)std::numeric_limits<int32_t>::max());
    /* Execute in parallel on the thread pool using linearized index */
    struct compute_2d_context context = {
        /*.function = */ function,
        /*.argument = */ argument,
        /*.range_j = */ caffe2::FixedDivisor<int32_t>(range_j)};
    legacy_pthreadpool_compute_1d(threadpool, (legacy_pthreadpool_function_1d_t) compute_2d, &context, range_i * range_j);
  }
}

struct compute_2d_tiled_context {
  legacy_pthreadpool_function_2d_tiled_t function;
  void* argument;
  caffe2::FixedDivisor<int32_t> tile_range_j;
  size_t range_i;
  size_t range_j;
  size_t tile_i;
  size_t tile_j;
};

static void compute_2d_tiled(void* context_, size_t linear_index) {
  int32_t q;
  int32_t r;

  const struct compute_2d_tiled_context* context = static_cast<compute_2d_tiled_context*>(context_);
  context->tile_range_j.DivMod(linear_index, &q, &r);
  const size_t max_tile_i = context->tile_i;
  const size_t max_tile_j = context->tile_j;
  const size_t index_i = q * max_tile_i;
  const size_t index_j = r * max_tile_j;
  const size_t tile_i = min(max_tile_i, context->range_i - index_i);
  const size_t tile_j = min(max_tile_j, context->range_j - index_j);
  context->function(context->argument, index_i, index_j, tile_i, tile_j);
}

void legacy_pthreadpool_compute_2d_tiled(
  legacy_pthreadpool_t threadpool,
  legacy_pthreadpool_function_2d_tiled_t function,
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
    TORCH_DCHECK_LE(
        tile_range_i * tile_range_j,
        (size_t)std::numeric_limits<int32_t>::max());
    struct compute_2d_tiled_context context = {
        /*.function = */ function,
        /*.argument = */ argument,
        /*.tile_range_j = */ caffe2::FixedDivisor<int32_t>(tile_range_j),
        /*.range_i = */ range_i,
        /*.range_j = */ range_j,
        /*.tile_i = */ tile_i,
        /*.tile_j = */ tile_j};
    legacy_pthreadpool_compute_1d(threadpool, (legacy_pthreadpool_function_1d_t) compute_2d_tiled, &context, tile_range_i * tile_range_j);
  }
}

struct compute_3d_tiled_context {
  legacy_pthreadpool_function_3d_tiled_t function;
  void* argument;
  caffe2::FixedDivisor<int32_t> tile_range_j;
  caffe2::FixedDivisor<int32_t> tile_range_k;
  size_t range_i;
  size_t range_j;
  size_t range_k;
  size_t tile_i;
  size_t tile_j;
  size_t tile_k;
};

static void compute_3d_tiled(
    void* context_,
    size_t linear_index) {
  int32_t tile_index_ij, tile_index_k;
  const struct compute_3d_tiled_context* context = static_cast<compute_3d_tiled_context*>(context_);
  context->tile_range_k.DivMod(
      static_cast<int32_t>(linear_index), &tile_index_ij, &tile_index_k);
  int32_t tile_index_i, tile_index_j;
  context->tile_range_j.DivMod(tile_index_ij, &tile_index_i, &tile_index_j);
  const size_t max_tile_i = context->tile_i;
  const size_t max_tile_j = context->tile_j;
  const size_t max_tile_k = context->tile_k;
  const size_t index_i = static_cast<uint32_t>(tile_index_i) * max_tile_i;
  const size_t index_j = static_cast<uint32_t>(tile_index_j) * max_tile_j;
  const size_t index_k = static_cast<uint32_t>(tile_index_k) * max_tile_k;
  const size_t tile_i = min(max_tile_i, context->range_i - index_i);
  const size_t tile_j = min(max_tile_j, context->range_j - index_j);
  const size_t tile_k = min(max_tile_k, context->range_k - index_k);
  context->function(
      context->argument, index_i, index_j, index_k, tile_i, tile_j, tile_k);
}

void legacy_pthreadpool_compute_3d_tiled(
    legacy_pthreadpool_t threadpool,
    legacy_pthreadpool_function_3d_tiled_t function,
    void* argument,
    size_t range_i,
    size_t range_j,
    size_t range_k,
    size_t tile_i,
    size_t tile_j,
    size_t tile_k) {
  if (threadpool == NULL) {
    /* No thread pool provided: execute function sequentially on the calling
     * thread */
    for (size_t i = 0; i < range_i; i += tile_i) {
      for (size_t j = 0; j < range_j; j += tile_j) {
        for (size_t k = 0; k < range_k; k += tile_k) {
          function(
              argument,
              i,
              j,
              k,
              min(range_i - i, tile_i),
              min(range_j - j, tile_j),
              min(range_k - k, tile_k));
        }
      }
    }
  } else {
    /* Execute in parallel on the thread pool using linearized index */
    const size_t tile_range_i = divide_round_up(range_i, tile_i);
    const size_t tile_range_j = divide_round_up(range_j, tile_j);
    const size_t tile_range_k = divide_round_up(range_k, tile_k);
    TORCH_DCHECK_LE(
        tile_range_i * tile_range_j * tile_range_k,
        (size_t)std::numeric_limits<int>::max());
    struct compute_3d_tiled_context context = {
        /*.function = */ function,
        /*.argument = */ argument,
        /*.tile_range_j = */ caffe2::FixedDivisor<int>(tile_range_j),
        /*.tile_range_k = */ caffe2::FixedDivisor<int>(tile_range_k),
        /*.range_i = */ range_i,
        /*.range_j = */ range_j,
        /*.range_k = */ range_k,
        /*.tile_i = */ tile_i,
        /*.tile_j = */ tile_j,
        /*.tile_k = */ tile_k};
    legacy_pthreadpool_compute_1d(
        threadpool,
        (legacy_pthreadpool_function_1d_t)compute_3d_tiled,
        &context,
        tile_range_i * tile_range_j * tile_range_k);
  }
}

struct compute_4d_tiled_context {
  legacy_pthreadpool_function_4d_tiled_t function;
  void* argument;
  caffe2::FixedDivisor<int32_t> tile_range_kl;
  caffe2::FixedDivisor<int32_t> tile_range_j;
  caffe2::FixedDivisor<int32_t> tile_range_l;
  size_t range_i;
  size_t range_j;
  size_t range_k;
  size_t range_l;
  size_t tile_i;
  size_t tile_j;
  size_t tile_k;
  size_t tile_l;
};

static void compute_4d_tiled(
    void* context_,
    size_t linear_index) {
  int32_t tile_index_ij, tile_index_kl;
  const struct compute_4d_tiled_context* context = static_cast<compute_4d_tiled_context*>(context_);
  context->tile_range_kl.DivMod(
      static_cast<int32_t>(linear_index), &tile_index_ij, &tile_index_kl);
  int32_t tile_index_i, tile_index_j;
  context->tile_range_j.DivMod(tile_index_ij, &tile_index_i, &tile_index_j);
  int32_t tile_index_k, tile_index_l;
  context->tile_range_l.DivMod(tile_index_kl, &tile_index_k, &tile_index_l);
  const size_t max_tile_i = context->tile_i;
  const size_t max_tile_j = context->tile_j;
  const size_t max_tile_k = context->tile_k;
  const size_t max_tile_l = context->tile_l;
  const size_t index_i = static_cast<uint32_t>(tile_index_i) * max_tile_i;
  const size_t index_j = static_cast<uint32_t>(tile_index_j) * max_tile_j;
  const size_t index_k = static_cast<uint32_t>(tile_index_k) * max_tile_k;
  const size_t index_l = static_cast<uint32_t>(tile_index_l) * max_tile_l;
  const size_t tile_i = min(max_tile_i, context->range_i - index_i);
  const size_t tile_j = min(max_tile_j, context->range_j - index_j);
  const size_t tile_k = min(max_tile_k, context->range_k - index_k);
  const size_t tile_l = min(max_tile_l, context->range_l - index_l);
  context->function(
      context->argument,
      index_i,
      index_j,
      index_k,
      index_l,
      tile_i,
      tile_j,
      tile_k,
      tile_l);
}

void legacy_pthreadpool_compute_4d_tiled(
    legacy_pthreadpool_t threadpool,
    legacy_pthreadpool_function_4d_tiled_t function,
    void* argument,
    size_t range_i,
    size_t range_j,
    size_t range_k,
    size_t range_l,
    size_t tile_i,
    size_t tile_j,
    size_t tile_k,
    size_t tile_l) {
  if (threadpool == NULL) {
    /* No thread pool provided: execute function sequentially on the calling
     * thread */
    for (size_t i = 0; i < range_i; i += tile_i) {
      for (size_t j = 0; j < range_j; j += tile_j) {
        for (size_t k = 0; k < range_k; k += tile_k) {
          for (size_t l = 0; l < range_l; l += tile_l) {
            function(
                argument,
                i,
                j,
                k,
                l,
                min(range_i - i, tile_i),
                min(range_j - j, tile_j),
                min(range_k - k, tile_k),
                min(range_l - l, tile_l));
          }
        }
      }
    }
  } else {
    /* Execute in parallel on the thread pool using linearized index */
    const size_t tile_range_i = divide_round_up(range_i, tile_i);
    const size_t tile_range_j = divide_round_up(range_j, tile_j);
    const size_t tile_range_k = divide_round_up(range_k, tile_k);
    const size_t tile_range_l = divide_round_up(range_l, tile_l);
    TORCH_DCHECK_LE(
        tile_range_i * tile_range_j * tile_range_k * tile_range_l,
        (size_t)std::numeric_limits<int>::max());
    struct compute_4d_tiled_context context = {
        /*.function = */ function,
        /*.argument = */ argument,
        /*.tile_range_kl = */
        caffe2::FixedDivisor<int>(tile_range_k * tile_range_l),
        /*.tile_range_j = */ caffe2::FixedDivisor<int>(tile_range_j),
        /*.tile_range_l = */ caffe2::FixedDivisor<int>(tile_range_l),
        /*.range_i = */ range_i,
        /*.range_j = */ range_j,
        /*.range_k = */ range_k,
        /*.range_l = */ range_l,
        /*.tile_i = */ tile_i,
        /*.tile_j = */ tile_j,
        /*.tile_k = */ tile_k,
        /*.tile_l = */ tile_l};
    legacy_pthreadpool_compute_1d(
        threadpool,
        (legacy_pthreadpool_function_1d_t)compute_4d_tiled,
        &context,
        tile_range_i * tile_range_j * tile_range_k * tile_range_l);
  }
}
