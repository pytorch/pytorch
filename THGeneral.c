#include "THGeneral.h"

/* Torch Error Handling */
static void defaultTorchErrorHandlerFunction(const char *msg)
{
  printf("$ Error: %s\n", msg);
  exit(-1);
}

static void (*torchErrorHandlerFunction)(const char *msg) = defaultTorchErrorHandlerFunction;

void THError(const char *fmt, ...)
{
  static char msg[1024];
  va_list args;

  /* vasprintf not standard */
  /* vsnprintf: how to handle if does not exists? */
  va_start(args, fmt);
  vsnprintf(msg, 1024, fmt, args);
  va_end(args);

  (*torchErrorHandlerFunction)(msg);
}

void THSetErrorHandler( void (*torchErrorHandlerFunction_)(const char *msg) )
{
  if(torchErrorHandlerFunction_)
    torchErrorHandlerFunction = torchErrorHandlerFunction_;
  else
    torchErrorHandlerFunction = defaultTorchErrorHandlerFunction;
}

/* Torch Arg Checking Handling */
static void defaultTorchArgErrorHandlerFunction(int argNumber, const char *msg)
{
  if(msg)
    printf("$ Invalid argument %d: %s\n", argNumber, msg);
  else
    printf("$ Invalid argument %d\n", argNumber);
  exit(-1);
}

static void (*torchArgErrorHandlerFunction)(int argNumber, const char *msg) = defaultTorchArgErrorHandlerFunction;

void THArgCheck(int condition, int argNumber, const char *msg)
{
  if(!condition)
    (*torchArgErrorHandlerFunction)(argNumber, msg);
}

void THSetArgErrorHandler( void (*torchArgErrorHandlerFunction_)(int argNumber, const char *msg) )
{
  if(torchArgErrorHandlerFunction_)
    torchArgErrorHandlerFunction = torchArgErrorHandlerFunction_;
  else
    torchArgErrorHandlerFunction = defaultTorchArgErrorHandlerFunction;
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

#ifdef _MSC_VER
double log1p(const double x)
{
  volatile double y;
  y = 1 + x;
  return log(y) - ((y-1)-x)/y ;  /* cancels errors with IEEE arithmetic */
}
#endif
