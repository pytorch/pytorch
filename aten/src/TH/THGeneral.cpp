#include <TH/THGeneral.h>

#ifdef __cplusplus
#include <c10/core/CPUAllocator.h>
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
  throw std::runtime_error(msg);
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
static THErrorHandlerFunction defaultErrorHandler = defaultErrorHandlerFunction;
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
static void *defaultErrorHandlerData;
// NOLINTNEXTLINE(modernize-use-nullptr,cppcoreguidelines-avoid-non-const-global-variables)
static __thread THErrorHandlerFunction threadErrorHandler = NULL;
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
static __thread void *threadErrorHandlerData;

void _THError(const char *file, const int line, const char *fmt, ...)
{
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays,cppcoreguidelines-avoid-magic-numbers)
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
  TH_UNREACHABLE;
}

void _THAssertionFailed(const char *file, const int line, const char *exp, const char *fmt, ...) {
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays,cppcoreguidelines-avoid-magic-numbers)
  char msg[1024];
  va_list args;
  va_start(args, fmt);
  vsnprintf(msg, 1024, fmt, args);
  va_end(args);
  _THError(file, line, "Assertion `%s' failed. %s", exp, msg);
}

/* Torch Arg Checking Handling */
static void defaultArgErrorHandlerFunction(int argNumber, const char *msg, void *data)
{
  std::stringstream new_error;
  new_error << "invalid argument " << argNumber << ": " << msg;
  throw std::runtime_error(new_error.str());
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
static THArgErrorHandlerFunction defaultArgErrorHandler = defaultArgErrorHandlerFunction;
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
static void *defaultArgErrorHandlerData;
// NOLINTNEXTLINE(modernize-use-nullptr,cppcoreguidelines-avoid-non-const-global-variables)
static __thread THArgErrorHandlerFunction threadArgErrorHandler = NULL;
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
static __thread void *threadArgErrorHandlerData;

void _THArgCheck(const char *file, int line, int condition, int argNumber, const char *fmt, ...)
{
  if(!condition) {
    // NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays,cppcoreguidelines-avoid-magic-numbers)
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
    TH_UNREACHABLE;
  }
}

void* THAlloc(ptrdiff_t size)
{
  if(size < 0)
    THError("$ Torch: invalid memory size -- maybe an overflow?");

  return c10::alloc_cpu(size);
}

void THFree(void *ptr)
{
  c10::free_cpu(ptr);
}
