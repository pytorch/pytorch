check_c_source_runs("
#include <stdatomic.h>
// ATOMIC_INT_LOCK_FREE is flaky on some older gcc versions
// so if this define is not usable a preprocessor definition
// we fail this check and fall back to GCC atomics
#if ATOMIC_INT_LOCK_FREE == 2
#define TH_ATOMIC_IPC_REFCOUNT 1
#endif
int main()
{
  int a;
  int oa;
  atomic_store(&a, 1);
  atomic_fetch_add(&a, 1);
  oa = atomic_load(&a);
  if(!atomic_compare_exchange_strong(&a, &oa, 3))
    return -1;
  return 0;
}
" HAS_C11_ATOMICS)

if(NOT HAS_C11_ATOMICS)
  check_c_source_runs("
#include <intrin.h>
int main()
{
  long a;
  _InterlockedExchange(&a, 1);
  _InterlockedExchangeAdd(&a, 1);
  if(_InterlockedCompareExchange(&a, 3, 2) != 2)
    return -1;
  return 0;
}
" HAS_MSC_ATOMICS)

  check_c_source_runs("
int main()
{
  int a;
  __sync_lock_test_and_set(&a, 1);
  __sync_fetch_and_add(&a, 1);
  if(!__sync_bool_compare_and_swap(&a, 2, 3))
    return -1;
  return 0;
}
" HAS_GCC_ATOMICS)
endif()

if(HAS_C11_ATOMICS)
  add_definitions(-DUSE_C11_ATOMICS=1)
  message(STATUS "Atomics: using C11 intrinsics")
elseif(HAS_MSC_ATOMICS)
  add_definitions(-DUSE_MSC_ATOMICS=1)
  message(STATUS "Atomics: using MSVC intrinsics")
elseif(HAS_GCC_ATOMICS)
  add_definitions(-DUSE_GCC_ATOMICS=1)
    message(STATUS "Atomics: using GCC intrinsics")
else()
  # TODO: I'm pretty sure this no longer works
  set(CMAKE_THREAD_PREFER_PTHREAD TRUE)
  find_package(Threads)
  if(THREADS_FOUND)
    add_definitions(-DUSE_PTHREAD_ATOMICS=1)
    target_link_libraries(TH ${CMAKE_THREAD_LIBS_INIT})
    message(STATUS "Atomics: using pthread")
  endif()
endif()
