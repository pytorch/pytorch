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
TH_API void THAtomicSet(int volatile *a, int newvalue);

/*
 * return *a
*/
TH_API int THAtomicGet(int volatile *a);

/*
 * *a += value,
 * return previous *a
*/
TH_API int THAtomicAdd(int volatile *a, int value);

/*
 * check if (*a == oldvalue)
 * if true: set *a to newvalue, return 1
 * if false: return 0
*/
TH_API int THAtomicCompareAndSwap(int volatile *a, int oldvalue, int newvalue);


/******************************************************************************
 * refcounting functions
 ******************************************************************************/

/*
 * *a++
*/
TH_API void THAtomicIncrementRef(int volatile *a);

/*
 * *a--,
 * return 1 if *a == 0 after the operation, 0 otherwise
*/
TH_API int THAtomicDecrementRef(int volatile *a);



/******************************************************************************
 * functions for long type
 ******************************************************************************/

/*
 * return *a
*/
TH_API long THAtomicGetLong(long volatile *a);

/*
 * *a += value,
 * return previous *a
*/
TH_API long THAtomicAddLong(long volatile *a, long value);

/*
 * check if (*a == oldvalue)
 * if true: set *a to newvalue, return 1
 * if false: return 0
*/
TH_API long THAtomicCompareAndSwapLong(long volatile *a, long oldvalue, long newvalue);

#endif
