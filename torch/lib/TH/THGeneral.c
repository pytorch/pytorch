#include "THGeneral.h"
#include "THAtomic.h"

#ifdef _OPENMP
#include <omp.h>
#endif

#ifndef TH_HAVE_THREAD
#define __thread
#elif _MSC_VER
#define __thread __declspec( thread )
#endif

#if (defined(__unix) || defined(_WIN32))
  #if defined(__FreeBSD__)
    #include <malloc_np.h>
  #else
    #include <malloc.h>
  #endif
#elif defined(__APPLE__)
#include <malloc/malloc.h>
#endif

/* Torch Error Handling */
static void defaultErrorHandlerFunction(const char *msg, void *data)
{
  printf("$ Error: %s\n", msg);
  exit(-1);
}

static THErrorHandlerFunction defaultErrorHandler = defaultErrorHandlerFunction;
static void *defaultErrorHandlerData;
static __thread THErrorHandlerFunction threadErrorHandler = NULL;
static __thread void *threadErrorHandlerData;

void _THError(const char *file, const int line, const char *fmt, ...)
{
  char msg[2048];
  va_list args;

  /* vasprintf not standard */
  /* vsnprintf: how to handle if does not exists? */
  va_start(args, fmt);
  int n = vsnprintf(msg, 2048, fmt, args);
  va_end(args);

  if(n < 2048) {
    snprintf(msg + n, 2048 - n, " at %s:%d", file, line);
  }

  if (threadErrorHandler)
    (*threadErrorHandler)(msg, threadErrorHandlerData);
  else
    (*defaultErrorHandler)(msg, defaultErrorHandlerData);
}

void _THAssertionFailed(const char *file, const int line, const char *exp, const char *fmt, ...) {
  char msg[1024];
  va_list args;
  va_start(args, fmt);
  vsnprintf(msg, 1024, fmt, args);
  va_end(args);
  _THError(file, line, "Assertion `%s' failed. %s", exp, msg);
}

void THSetErrorHandler(THErrorHandlerFunction new_handler, void *data)
{
  threadErrorHandler = new_handler;
  threadErrorHandlerData = data;
}

void THSetDefaultErrorHandler(THErrorHandlerFunction new_handler, void *data)
{
  if (new_handler)
    defaultErrorHandler = new_handler;
  else
    defaultErrorHandler = defaultErrorHandlerFunction;
  defaultErrorHandlerData = data;
}

/* Torch Arg Checking Handling */
static void defaultArgErrorHandlerFunction(int argNumber, const char *msg, void *data)
{
  if(msg)
    printf("$ Invalid argument %d: %s\n", argNumber, msg);
  else
    printf("$ Invalid argument %d\n", argNumber);
  exit(-1);
}

static THArgErrorHandlerFunction defaultArgErrorHandler = defaultArgErrorHandlerFunction;
static void *defaultArgErrorHandlerData;
static __thread THArgErrorHandlerFunction threadArgErrorHandler = NULL;
static __thread void *threadArgErrorHandlerData;

void _THArgCheck(const char *file, int line, int condition, int argNumber, const char *fmt, ...)
{
  if(!condition) {
    char msg[2048];
    va_list args;

    /* vasprintf not standard */
    /* vsnprintf: how to handle if does not exists? */
    va_start(args, fmt);
    int n = vsnprintf(msg, 2048, fmt, args);
    va_end(args);

    if(n < 2048) {
      snprintf(msg + n, 2048 - n, " at %s:%d", file, line);
    }

    if (threadArgErrorHandler)
      (*threadArgErrorHandler)(argNumber, msg, threadArgErrorHandlerData);
    else
      (*defaultArgErrorHandler)(argNumber, msg, defaultArgErrorHandlerData);
  }
}

void THSetArgErrorHandler(THArgErrorHandlerFunction new_handler, void *data)
{
  threadArgErrorHandler = new_handler;
  threadArgErrorHandlerData = data;
}

void THSetDefaultArgErrorHandler(THArgErrorHandlerFunction new_handler, void *data)
{
  if (new_handler)
    defaultArgErrorHandler = new_handler;
  else
    defaultArgErrorHandler = defaultArgErrorHandlerFunction;
  defaultArgErrorHandlerData = data;
}

static __thread void (*torchGCFunction)(void *data) = NULL;
static __thread void *torchGCData;
static ptrdiff_t heapSize = 0;
static __thread ptrdiff_t heapDelta = 0;
static const ptrdiff_t heapMaxDelta = (ptrdiff_t)1e6; // limit to +/- 1MB before updating heapSize
static const ptrdiff_t heapMinDelta = (ptrdiff_t)-1e6;
static __thread ptrdiff_t heapSoftmax = (ptrdiff_t)3e8; // 300MB, adjusted upward dynamically
static const double heapSoftmaxGrowthThresh = 0.8; // grow softmax if >80% max after GC
static const double heapSoftmaxGrowthFactor = 1.4; // grow softmax by 40%

/* Optional hook for integrating with a garbage-collected frontend.
 *
 * If torch is running with a garbage-collected frontend (e.g. Lua),
 * the GC isn't aware of TH-allocated memory so may not know when it
 * needs to run. These hooks trigger the GC to run in two cases:
 *
 * (1) When a memory allocation (malloc, realloc, ...) fails
 * (2) When the total TH-allocated memory hits a dynamically-adjusted
 *     soft maximum.
 */
void THSetGCHandler( void (*torchGCFunction_)(void *data), void *data )
{
  torchGCFunction = torchGCFunction_;
  torchGCData = data;
}

/* it is guaranteed the allocated size is not bigger than PTRDIFF_MAX */
static ptrdiff_t getAllocSize(void *ptr) {
#if defined(__unix) && defined(HAVE_MALLOC_USABLE_SIZE)
  return malloc_usable_size(ptr);
#elif defined(__APPLE__)
  return malloc_size(ptr);
#elif defined(_WIN32)
  if(ptr) { return _msize(ptr); } else { return 0; }
#else
  return 0;
#endif
}

static ptrdiff_t applyHeapDelta() {
  ptrdiff_t oldHeapSize = THAtomicAddPtrdiff(&heapSize, heapDelta);
#ifdef DEBUG
  if (heapDelta > 0 && oldHeapSize > PTRDIFF_MAX - heapDelta)
    THError("applyHeapDelta: heapSize(%td) + increased(%td) > PTRDIFF_MAX, heapSize overflow!", oldHeapSize, heapDelta);
  if (heapDelta < 0 && oldHeapSize < PTRDIFF_MIN - heapDelta)
    THError("applyHeapDelta: heapSize(%td) + decreased(%td) < PTRDIFF_MIN, heapSize underflow!", oldHeapSize, heapDelta);
#endif
  ptrdiff_t newHeapSize = oldHeapSize + heapDelta;
  heapDelta = 0;
  return newHeapSize;
}

/* (1) if the torch-allocated heap size exceeds the soft max, run GC
 * (2) if post-GC heap size exceeds 80% of the soft max, increase the
 *     soft max by 40%
 */
static void maybeTriggerGC(ptrdiff_t curHeapSize) {
  if (torchGCFunction && curHeapSize > heapSoftmax) {
    torchGCFunction(torchGCData);

    // ensure heapSize is accurate before updating heapSoftmax
    ptrdiff_t newHeapSize = applyHeapDelta();

    if (newHeapSize > heapSoftmax * heapSoftmaxGrowthThresh) {
      heapSoftmax = (ptrdiff_t)(heapSoftmax * heapSoftmaxGrowthFactor);
    }
  }
}

// hooks into the TH heap tracking
void THHeapUpdate(ptrdiff_t size) {
#ifdef DEBUG
  if (size > 0 && heapDelta > PTRDIFF_MAX - size)
    THError("THHeapUpdate: heapDelta(%td) + increased(%td) > PTRDIFF_MAX, heapDelta overflow!", heapDelta, size);
  if (size < 0 && heapDelta < PTRDIFF_MIN - size)
    THError("THHeapUpdate: heapDelta(%td) + decreased(%td) < PTRDIFF_MIN, heapDelta underflow!", heapDelta, size);
#endif

  heapDelta += size;

  // batch updates to global heapSize to minimize thread contention
  if (heapDelta < heapMaxDelta && heapDelta > heapMinDelta) {
    return;
  }

  ptrdiff_t newHeapSize = applyHeapDelta();

  if (size > 0) {
    maybeTriggerGC(newHeapSize);
  }
}

static void* THAllocInternal(ptrdiff_t size)
{
  void *ptr;

  if (size > 5120)
  {
#if (defined(__unix) || defined(__APPLE__)) && (!defined(DISABLE_POSIX_MEMALIGN))
    if (posix_memalign(&ptr, 64, size) != 0)
      ptr = NULL;
/*
#elif defined(_WIN32)
    ptr = _aligned_malloc(size, 64);
*/
#else
    ptr = malloc(size);
#endif
  }
  else
  {
    ptr = malloc(size);
  }

  THHeapUpdate(getAllocSize(ptr));
  return ptr;
}

void* THAlloc(ptrdiff_t size)
{
  void *ptr;

  if(size < 0)
    THError("$ Torch: invalid memory size -- maybe an overflow?");

  if(size == 0)
    return NULL;

  ptr = THAllocInternal(size);

  if(!ptr && torchGCFunction) {
    torchGCFunction(torchGCData);
    ptr = THAllocInternal(size);
  }

  if(!ptr)
    THError("$ Torch: not enough memory: you tried to allocate %dGB. Buy new RAM!", size/1073741824);

  return ptr;
}

void* THRealloc(void *ptr, ptrdiff_t size)
{
  if(!ptr)
    return(THAlloc(size));

  if(size == 0)
  {
    THFree(ptr);
    return NULL;
  }

  if(size < 0)
    THError("$ Torch: invalid memory size -- maybe an overflow?");

  ptrdiff_t oldSize = -getAllocSize(ptr);
  void *newptr = realloc(ptr, size);

  if(!newptr && torchGCFunction) {
    torchGCFunction(torchGCData);
    newptr = realloc(ptr, size);
  }

  if(!newptr)
    THError("$ Torch: not enough memory: you tried to reallocate %dGB. Buy new RAM!", size/1073741824);

  // update heapSize only after successfully reallocated
  THHeapUpdate(oldSize + getAllocSize(newptr));

  return newptr;
}

void THFree(void *ptr)
{
  THHeapUpdate(-getAllocSize(ptr));
  free(ptr);
}

double THLog1p(const double x)
{
#if (defined(_MSC_VER) || defined(__MINGW32__))
  volatile double y = 1 + x;
  return log(y) - ((y-1)-x)/y ;  /* cancels errors with IEEE arithmetic */
#else
  return log1p(x);
#endif
}

void THSetNumThreads(int num_threads)
{
#ifdef _OPENMP
  omp_set_num_threads(num_threads);
#endif
}

int THGetNumThreads(void)
{
#ifdef _OPENMP
  return omp_get_max_threads();
#else
  return 1;
#endif
}

int THGetNumCores(void)
{
#ifdef _OPENMP
  return omp_get_num_procs();
#else
  return 1;
#endif
}

#ifdef TH_BLAS_MKL
extern int mkl_get_max_threads(void);
#endif

TH_API void THInferNumThreads(void)
{
#if defined(_OPENMP) && defined(TH_BLAS_MKL)
  // If we are using MKL an OpenMP make sure the number of threads match.
  // Otherwise, MKL and our OpenMP-enabled functions will keep changing the
  // size of the OpenMP thread pool, resulting in worse performance (and memory
  // leaks in GCC 5.4)
  omp_set_num_threads(mkl_get_max_threads());
#endif
}

TH_API THDescBuff _THSizeDesc(const long *size, const long ndim) {
  const int L = TH_DESC_BUFF_LEN;
  THDescBuff buf;
  char *str = buf.str;
  int n = 0;
  n += snprintf(str, L-n, "[");
  int i;
  for(i = 0; i < ndim; i++) {
    if(n >= L) break;
    n += snprintf(str+n, L-n, "%ld", size[i]);
    if(i < ndim-1) {
      n += snprintf(str+n, L-n, " x ");
    }
  }
  if(n < L - 2) {
    snprintf(str+n, L-n, "]");
  } else {
    snprintf(str+L-5, 5, "...]");
  }
  return buf;
}

