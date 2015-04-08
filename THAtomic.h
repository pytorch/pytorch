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
void THAtomicSet(int volatile *a, int newvalue);

/*
 * return *a
*/
int THAtomicGet(int volatile *a);

/*
 * *a += value,
 * return previous *a
*/
int THAtomicAdd(int volatile *a, int value);

/*
 * check if (*a == oldvalue)
 * if true: set *a to newvalue, return 1
 * if false: return 0
*/
int THAtomicCompareAndSwap(int volatile *a, int oldvalue, int newvalue);


/******************************************************************************
 * refcounting functions
 ******************************************************************************/

/*
 * *a++
*/
void THAtomicIncrementRef(int volatile *a);

/*
 * *a--,
 * return 1 if *a == 0 after the operation, 0 otherwise
*/
int THAtomicDecrementRef(int volatile *a);

#endif
