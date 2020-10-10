#ifndef _NPY_NUMPYCONFIG_H_
#define _NPY_NUMPYCONFIG_H_

#include "_numpyconfig.h"

/*
 * On Mac OS X, because there is only one configuration stage for all the archs
 * in universal builds, any macro which depends on the arch needs to be
 * hardcoded
 */
#ifdef __APPLE__
    #undef NPY_SIZEOF_LONG
    #undef NPY_SIZEOF_PY_INTPTR_T

    #ifdef __LP64__
        #define NPY_SIZEOF_LONG         8
        #define NPY_SIZEOF_PY_INTPTR_T  8
    #else
        #define NPY_SIZEOF_LONG         4
        #define NPY_SIZEOF_PY_INTPTR_T  4
    #endif
#endif

/**
 * To help with the NPY_NO_DEPRECATED_API macro, we include API version
 * numbers for specific versions of NumPy. To exclude all API that was
 * deprecated as of 1.7, add the following before #including any NumPy
 * headers:
 *   #define NPY_NO_DEPRECATED_API  NPY_1_7_API_VERSION
 */
#define NPY_1_7_API_VERSION 0x00000007
#define NPY_1_8_API_VERSION 0x00000008
#define NPY_1_9_API_VERSION 0x00000008
#define NPY_1_10_API_VERSION 0x00000008
#define NPY_1_11_API_VERSION 0x00000008
#define NPY_1_12_API_VERSION 0x00000008
#define NPY_1_13_API_VERSION 0x00000008
#define NPY_1_14_API_VERSION 0x00000008
#define NPY_1_15_API_VERSION 0x00000008
#define NPY_1_16_API_VERSION 0x00000008
#define NPY_1_17_API_VERSION 0x00000008
#define NPY_1_18_API_VERSION 0x00000008
#define NPY_1_19_API_VERSION 0x00000008

#endif
