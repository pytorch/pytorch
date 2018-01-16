#include "THAtomic.h"

/*
  Note: I thank Leon Bottou for his useful comments.
  Ronan.
*/

#if defined(USE_C11_ATOMICS)
#include <stdatomic.h>
#endif

#if defined(USE_MSC_ATOMICS)
#include <intrin.h>
#include <assert.h>
#endif

#if !defined(USE_MSC_ATOMICS) && !defined(USE_GCC_ATOMICS) && defined(USE_PTHREAD_ATOMICS)
#include <pthread.h>
static pthread_mutex_t ptm = PTHREAD_MUTEX_INITIALIZER;
#endif

void THAtomicSet(int32_t volatile *a, int32_t newvalue)
{
#if defined(USE_C11_ATOMICS)
  atomic_store(a, newvalue);
#elif defined(USE_MSC_ATOMICS)
  assert(sizeof(int) == sizeof(int32_t));
  _InterlockedExchange((int32_t*)a, newvalue);
#elif defined(USE_GCC_ATOMICS)
  __sync_lock_test_and_set(a, newvalue);
#else
  int32_t oldvalue;
  do {
    oldvalue = *a;
  } while (!THAtomicCompareAndSwap(a, oldvalue, newvalue));
#endif
}

int THAtomicGet(int32_t volatile *a)
{
#if defined(USE_C11_ATOMICS)
  return atomic_load(a);
#else
  int32_t value;
  do {
    value = *a;
  } while (!THAtomicCompareAndSwap(a, value, value));
  return value;
#endif
}

int THAtomicAdd(int32_t volatile *a, int32_t value)
{
#if defined(USE_C11_ATOMICS)
  return atomic_fetch_add(a, value);
#elif defined(USE_MSC_ATOMICS)
  return _InterlockedExchangeAdd((int32_t*)a, value);
#elif defined(USE_GCC_ATOMICS)
  return __sync_fetch_and_add(a, value);
#else
  int32_t oldvalue;
  do {
    oldvalue = *a;
  } while (!THAtomicCompareAndSwap(a, oldvalue, (oldvalue + value)));
  return oldvalue;
#endif
}

void THAtomicIncrementRef(int32_t volatile *a)
{
  THAtomicAdd(a, 1);
}

int THAtomicDecrementRef(int32_t volatile *a)
{
  return (THAtomicAdd(a, -1) == 1);
}

int THAtomicCompareAndSwap(int32_t volatile *a, int32_t oldvalue, int32_t newvalue)
{
#if defined(USE_C11_ATOMICS)
  return atomic_compare_exchange_strong(a, &oldvalue, newvalue);
#elif defined(USE_MSC_ATOMICS)
  return (_InterlockedCompareExchange((int32_t*)a, (int32_t)newvalue, (int32_t)oldvalue) == (int32_t)oldvalue);
#elif defined(USE_GCC_ATOMICS)
  return __sync_bool_compare_and_swap(a, oldvalue, newvalue);
#elif defined(USE_PTHREAD_ATOMICS)
  int32_t ret = 0;
  pthread_mutex_lock(&ptm);
  if(*a == oldvalue) {
    *a = newvalue;
    ret = 1;
  }
  pthread_mutex_unlock(&ptm);
  return ret;
#else
#warning THAtomic is not thread safe
  if(*a == oldvalue) {
    *a = newvalue;
    return 1;
  }
  else
    return 0;
#endif
}

void THAtomicSetLong(int64_t volatile *a, int64_t newvalue)
{
#if defined(USE_C11_ATOMICS)
  atomic_store(a, newvalue);
#elif defined(USE_MSC_ATOMICS)
  _InterlockedExchange64(a, newvalue);
#elif defined(USE_GCC_ATOMICS)
  __sync_lock_test_and_set(a, newvalue);
#else
  int64_t oldvalue;
  do {
    oldvalue = *a;
  } while (!THAtomicCompareAndSwapLong(a, oldvalue, newvalue));
#endif
}

int64_t THAtomicGetLong(int64_t volatile *a)
{
#if defined(USE_C11_ATOMICS)
  return atomic_load(a);
#else
  int64_t value;
  do {
    value = *a;
  } while (!THAtomicCompareAndSwapLong(a, value, value));
  return value;
#endif
}

int64_t THAtomicAddLong(int64_t volatile *a, int64_t value)
{
#if defined(USE_C11_ATOMICS)
  return atomic_fetch_add(a, value);
#elif defined(USE_MSC_ATOMICS)
  return _InterlockedExchangeAdd64(a, value);
#elif defined(USE_GCC_ATOMICS)
  return __sync_fetch_and_add(a, value);
#else
  int64_t oldvalue;
  do {
    oldvalue = *a;
  } while (!THAtomicCompareAndSwapLong(a, oldvalue, (oldvalue + value)));
  return oldvalue;
#endif
}

int64_t THAtomicCompareAndSwapLong(int64_t volatile *a, int64_t oldvalue, int64_t newvalue)
{
#if defined(USE_C11_ATOMICS)
  return atomic_compare_exchange_strong(a, &oldvalue, newvalue);
#elif defined(USE_MSC_ATOMICS)
  return (_InterlockedCompareExchange64(a, newvalue, oldvalue) == oldvalue);
#elif defined(USE_GCC_ATOMICS)
  return __sync_bool_compare_and_swap(a, oldvalue, newvalue);
#elif defined(USE_PTHREAD_ATOMICS)
  int64_t ret = 0;
  pthread_mutex_lock(&ptm);
  if(*a == oldvalue) {
    *a = newvalue;
    ret = 1;
  }
  pthread_mutex_unlock(&ptm);
  return ret;
#else
#warning THAtomic is not thread safe
  if(*a == oldvalue) {
    *a = newvalue;
    return 1;
  }
  else
    return 0;
#endif
}

void THAtomicSetPtrdiff(ptrdiff_t volatile *a, ptrdiff_t newvalue)
{
#if defined(USE_C11_ATOMICS)
  atomic_store(a, newvalue);
#elif defined(USE_MSC_ATOMICS)
#ifdef _WIN64
  _InterlockedExchange64(a, newvalue);
#else
  _InterlockedExchange(a, newvalue);
#endif
#elif defined(USE_GCC_ATOMICS)
  __sync_lock_test_and_set(a, newvalue);
#else
  ptrdiff_t oldvalue;
  do {
    oldvalue = *a;
  } while (!THAtomicCompareAndSwapPtrdiff(a, oldvalue, newvalue));
#endif
}

ptrdiff_t THAtomicGetPtrdiff(ptrdiff_t volatile *a)
{
#if defined(USE_C11_ATOMICS)
  return atomic_load(a);
#else
  ptrdiff_t value;
  do {
    value = *a;
  } while (!THAtomicCompareAndSwapPtrdiff(a, value, value));
  return value;
#endif
}

ptrdiff_t THAtomicAddPtrdiff(ptrdiff_t volatile *a, ptrdiff_t value)
{
#if defined(USE_C11_ATOMICS)
  return atomic_fetch_add(a, value);
#elif defined(USE_MSC_ATOMICS)
#ifdef _WIN64
  return _InterlockedExchangeAdd64(a, value);
#else
  return _InterlockedExchangeAdd(a, value);
#endif
#elif defined(USE_GCC_ATOMICS)
  return __sync_fetch_and_add(a, value);
#else
  ptrdiff_t oldvalue;
  do {
    oldvalue = *a;
  } while (!THAtomicCompareAndSwapPtrdiff(a, oldvalue, (oldvalue + value)));
  return oldvalue;
#endif
}

ptrdiff_t THAtomicCompareAndSwapPtrdiff(ptrdiff_t volatile *a, ptrdiff_t oldvalue, ptrdiff_t newvalue)
{
#if defined(USE_C11_ATOMICS)
  return atomic_compare_exchange_strong(a, &oldvalue, newvalue);
#elif defined(USE_MSC_ATOMICS)
#ifdef _WIN64
  return (_InterlockedCompareExchange64(a, newvalue, oldvalue) == oldvalue);
#else
  return (_InterlockedCompareExchange(a, newvalue, oldvalue) == oldvalue);
#endif
#elif defined(USE_GCC_ATOMICS)
  return __sync_bool_compare_and_swap(a, oldvalue, newvalue);
#elif defined(USE_PTHREAD_ATOMICS)
  ptrdiff_t ret = 0;
  pthread_mutex_lock(&ptm);
  if(*a == oldvalue) {
    *a = newvalue;
    ret = 1;
  }
  pthread_mutex_unlock(&ptm);
  return ret;
#else
#warning THAtomic is not thread safe
  if(*a == oldvalue) {
    *a = newvalue;
    return 1;
  }
  else
    return 0;
#endif
}
