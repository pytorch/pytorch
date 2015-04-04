#include "THGeneral.h"

#ifndef TH_HAVE_THREAD
#define __thread
#endif
/* Torch Error Handling */
static void defaultTorchErrorHandlerFunction(const char *msg, void *data)
{
  printf("$ Error: %s\n", msg);
  exit(-1);
}

static __thread void (*torchErrorHandlerFunction)(const char *msg, void *data) = defaultTorchErrorHandlerFunction;
static __thread void *torchErrorHandlerData;

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

  (*torchErrorHandlerFunction)(msg, torchErrorHandlerData);
}

void _THAssertionFailed(const char *file, const int line, const char *exp, const char *fmt, ...) {
  char msg[1024];
  va_list args;
  va_start(args, fmt);
  vsnprintf(msg, 1024, fmt, args);
  va_end(args);
  _THError(file, line, "Assertion `%s' failed. %s", exp, msg);
}

void THSetErrorHandler( void (*torchErrorHandlerFunction_)(const char *msg, void *data), void *data )
{
  if(torchErrorHandlerFunction_)
    torchErrorHandlerFunction = torchErrorHandlerFunction_;
  else
    torchErrorHandlerFunction = defaultTorchErrorHandlerFunction;
  torchErrorHandlerData = data;
}

/* Torch Arg Checking Handling */
static void defaultTorchArgErrorHandlerFunction(int argNumber, const char *msg, void *data)
{
  if(msg)
    printf("$ Invalid argument %d: %s\n", argNumber, msg);
  else
    printf("$ Invalid argument %d\n", argNumber);
  exit(-1);
}

static __thread void (*torchArgErrorHandlerFunction)(int argNumber, const char *msg, void *data) = defaultTorchArgErrorHandlerFunction;
static __thread void *torchArgErrorHandlerData;

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

    (*torchArgErrorHandlerFunction)(argNumber, msg, torchArgErrorHandlerData);
  }
}

void THSetArgErrorHandler( void (*torchArgErrorHandlerFunction_)(int argNumber, const char *msg, void *data), void *data )
{
  if(torchArgErrorHandlerFunction_)
    torchArgErrorHandlerFunction = torchArgErrorHandlerFunction_;
  else
    torchArgErrorHandlerFunction = defaultTorchArgErrorHandlerFunction;
  torchArgErrorHandlerData = data;
}

void* THAlloc(long size)
{
  void *ptr;

  if(size < 0)
    THError("$ Torch: invalid memory size -- maybe an overflow?");

  if(size == 0)
    return NULL;

  ptr = malloc(size);
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

  ptr = realloc(ptr, size);
  if(!ptr)
    THError("$ Torch: not enough memory: you tried to reallocate %dGB. Buy new RAM!", size/1073741824);
  return ptr;
}

void THFree(void *ptr)
{
  free(ptr);
}

double THLog1p(const double x)
{
#ifdef _MSC_VER
  volatile double y = 1 + x;
  return log(y) - ((y-1)-x)/y ;  /* cancels errors with IEEE arithmetic */
#else
  return log1p(x);
#endif
}
