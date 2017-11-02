#ifndef TH_ATOMIC_INC
#define TH_ATOMIC_INC

#include "THGeneral.h"

/******************************************************************************
 * Atomic operations for TH
 *  Five backends are integrated:
 *  - C11 atomic operations
 *  - MSVC intrinsics
 *  - GCC intrinsics
 *  - Pthread if none of the above is available
 *  - Unsafe mode in none of the above is available
 ******************************************************************************/


/******************************************************************************
 * all-purpose functions
 ******************************************************************************/

/*
 * *a = newvalue
*/
TH_API void THAtomicSet(int32_t volatile *a, int32_t newvalue);

/*
 * return *a
*/
TH_API int32_t THAtomicGet(int32_t volatile *a);

/*
 * *a += value,
 * return previous *a
*/
TH_API int32_t THAtomicAdd(int32_t volatile *a, int32_t value);

/*
 * check if (*a == oldvalue)
 * if true: set *a to newvalue, return 1
 * if false: return 0
*/
TH_API int32_t THAtomicCompareAndSwap(int32_t volatile *a, int32_t oldvalue, int32_t newvalue);


/******************************************************************************
 * refcounting functions
 ******************************************************************************/

/*
 * *a++
*/
TH_API void THAtomicIncrementRef(int32_t volatile *a);

/*
 * *a--,
 * return 1 if *a == 0 after the operation, 0 otherwise
*/
TH_API int32_t THAtomicDecrementRef(int32_t volatile *a);



/******************************************************************************
 * functions for long type
 ******************************************************************************/

/*
 * *a = newvalue
*/
TH_API void THAtomicSetLong(int64_t volatile *a, int64_t newvalue);

/*
 * return *a
*/
TH_API int64_t THAtomicGetLong(int64_t volatile *a);

/*
 * *a += value,
 * return previous *a
*/
TH_API int64_t THAtomicAddLong(int64_t volatile *a, int64_t value);

/*
 * check if (*a == oldvalue)
 * if true: set *a to newvalue, return 1
 * if false: return 0
*/
TH_API int64_t THAtomicCompareAndSwapLong(int64_t volatile *a, int64_t oldvalue, int64_t newvalue);



/******************************************************************************
 * functions for ptrdiff_t type
 ******************************************************************************/

/*
 * *a = newvalue
*/
TH_API void THAtomicSetPtrdiff(ptrdiff_t volatile *a, ptrdiff_t newvalue);

/*
 * return *a
*/
TH_API ptrdiff_t THAtomicGetPtrdiff(ptrdiff_t volatile *a);

/*
 * *a += value,
 * return previous *a
*/
TH_API ptrdiff_t THAtomicAddPtrdiff(ptrdiff_t volatile *a, ptrdiff_t value);

/*
 * check if (*a == oldvalue)
 * if true: set *a to newvalue, return 1
 * if false: return 0
*/
TH_API ptrdiff_t THAtomicCompareAndSwapPtrdiff(ptrdiff_t volatile *a, ptrdiff_t oldvalue, ptrdiff_t newvalue);

#endif
