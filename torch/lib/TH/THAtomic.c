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
#endif

#if !defined(USE_MSC_ATOMICS) && !defined(USE_GCC_ATOMICS) && defined(USE_PTHREAD_ATOMICS)
#include <pthread.h>
static pthread_mutex_t ptm = PTHREAD_MUTEX_INITIALIZER;
#endif

void THAtomicSet(int volatile *a, int newvalue)
{
#if defined(USE_C11_ATOMICS)
  atomic_store(a, newvalue);
#elif defined(USE_MSC_ATOMICS)
  _InterlockedExchange((long*)a, newvalue);
#elif defined(USE_GCC_ATOMICS)
  __sync_lock_test_and_set(a, newvalue);
#else
  int oldvalue;
  do {
    oldvalue = *a;
  } while (!THAtomicCompareAndSwap(a, oldvalue, newvalue));
#endif
}

int THAtomicGet(int volatile *a)
{
#if defined(USE_C11_ATOMICS)
  return atomic_load(a);
#else
  int value;
  do {
    value = *a;
  } while (!THAtomicCompareAndSwap(a, value, value));
  return value;
#endif
}

int THAtomicAdd(int volatile *a, int value)
{
#if defined(USE_C11_ATOMICS)
  return atomic_fetch_add(a, value);
#elif defined(USE_MSC_ATOMICS)
  return _InterlockedExchangeAdd((long*)a, value);
#elif defined(USE_GCC_ATOMICS)
  return __sync_fetch_and_add(a, value);
#else
  int oldvalue;
  do {
    oldvalue = *a;
  } while (!THAtomicCompareAndSwap(a, oldvalue, (oldvalue + value)));
  return oldvalue;
#endif
}

void THAtomicIncrementRef(int volatile *a)
{
  THAtomicAdd(a, 1);
}

int THAtomicDecrementRef(int volatile *a)
{
  return (THAtomicAdd(a, -1) == 1);
}

int THAtomicCompareAndSwap(int volatile *a, int oldvalue, int newvalue)
{
#if defined(USE_C11_ATOMICS)
  return atomic_compare_exchange_strong(a, &oldvalue, newvalue);
#elif defined(USE_MSC_ATOMICS)
  return (_InterlockedCompareExchange((long*)a, (long)newvalue, (long)oldvalue) == (long)oldvalue);
#elif defined(USE_GCC_ATOMICS)
  return __sync_bool_compare_and_swap(a, oldvalue, newvalue);
#elif defined(USE_PTHREAD_ATOMICS)
  int ret = 0;
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

void THAtomicSetLong(long volatile *a, long newvalue)
{
#if defined(USE_C11_ATOMICS)
  atomic_store(a, newvalue);
#elif defined(USE_MSC_ATOMICS)
  _InterlockedExchange(a, newvalue);
#elif defined(USE_GCC_ATOMICS)
  __sync_lock_test_and_set(a, newvalue);
#else
  long oldvalue;
  do {
    oldvalue = *a;
  } while (!THAtomicCompareAndSwapLong(a, oldvalue, newvalue));
#endif
}

long THAtomicGetLong(long volatile *a)
{
#if defined(USE_C11_ATOMICS)
  return atomic_load(a);
#else
  long value;
  do {
    value = *a;
  } while (!THAtomicCompareAndSwapLong(a, value, value));
  return value;
#endif
}

long THAtomicAddLong(long volatile *a, long value)
{
#if defined(USE_C11_ATOMICS)
  return atomic_fetch_add(a, value);
#elif defined(USE_MSC_ATOMICS)
  return _InterlockedExchangeAdd(a, value);
#elif defined(USE_GCC_ATOMICS)
  return __sync_fetch_and_add(a, value);
#else
  long oldvalue;
  do {
    oldvalue = *a;
  } while (!THAtomicCompareAndSwapLong(a, oldvalue, (oldvalue + value)));
  return oldvalue;
#endif
}

long THAtomicCompareAndSwapLong(long volatile *a, long oldvalue, long newvalue)
{
#if defined(USE_C11_ATOMICS)
  return atomic_compare_exchange_strong(a, &oldvalue, newvalue);
#elif defined(USE_MSC_ATOMICS)
  return (_InterlockedCompareExchange(a, newvalue, oldvalue) == oldvalue);
#elif defined(USE_GCC_ATOMICS)
  return __sync_bool_compare_and_swap(a, oldvalue, newvalue);
#elif defined(USE_PTHREAD_ATOMICS)
  long ret = 0;
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
