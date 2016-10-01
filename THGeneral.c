#include "THGeneral.h"
#include "THAtomic.h"

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

    if (threadArgErrorHandlerData)
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
static long heapSize = 0;
static __thread long heapDelta = 0;
static const long heapMaxDelta = 1e6; // limit to +/- 1MB before updating heapSize
static __thread long heapSoftmax = 3e8; // 300MB, adjusted upward dynamically
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

static long getAllocSize(void *ptr) {
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

static long applyHeapDelta() {
  long newHeapSize = THAtomicAddLong(&heapSize, heapDelta) + heapDelta;
  heapDelta = 0;
  return newHeapSize;
}

/* (1) if the torch-allocated heap size exceeds the soft max, run GC
 * (2) if post-GC heap size exceeds 80% of the soft max, increase the
 *     soft max by 40%
 */
static void maybeTriggerGC(long curHeapSize) {
  if (torchGCFunction && curHeapSize > heapSoftmax) {
    torchGCFunction(torchGCData);

    // ensure heapSize is accurate before updating heapSoftmax
    long newHeapSize = applyHeapDelta();

    if (newHeapSize > heapSoftmax * heapSoftmaxGrowthThresh) {
      heapSoftmax = heapSoftmax * heapSoftmaxGrowthFactor;
    }
  }
}

// hooks into the TH heap tracking
void THHeapUpdate(long size) {
  heapDelta += size;

  // batch updates to global heapSize to minimize thread contention
  if (labs(heapDelta) < heapMaxDelta) {
    return;
  }

  long newHeapSize = applyHeapDelta();

  if (size > 0) {
    maybeTriggerGC(newHeapSize);
  }
}

static void* THAllocInternal(long size)
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

void* THAlloc(long size)
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

void* THRealloc(void *ptr, long size)
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

  THHeapUpdate(-getAllocSize(ptr));
  void *newptr = realloc(ptr, size);

  if(!newptr && torchGCFunction) {
    torchGCFunction(torchGCData);
    newptr = realloc(ptr, size);
  }
  THHeapUpdate(getAllocSize(newptr ? newptr : ptr));

  if(!newptr)
    THError("$ Torch: not enough memory: you tried to reallocate %dGB. Buy new RAM!", size/1073741824);

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
