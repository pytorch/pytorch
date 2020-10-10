#ifndef __NUMPY_UTILS_HEADER__
#define __NUMPY_UTILS_HEADER__

#ifndef __COMP_NPY_UNUSED
        #if defined(__GNUC__)
                #define __COMP_NPY_UNUSED __attribute__ ((__unused__))
        # elif defined(__ICC)
                #define __COMP_NPY_UNUSED __attribute__ ((__unused__))
        # elif defined(__clang__)
                #define __COMP_NPY_UNUSED __attribute__ ((unused))
        #else
                #define __COMP_NPY_UNUSED
        #endif
#endif

/* Use this to tag a variable as not used. It will remove unused variable
 * warning on support platforms (see __COM_NPY_UNUSED) and mangle the variable
 * to avoid accidental use */
#define NPY_UNUSED(x) (__NPY_UNUSED_TAGGED ## x) __COMP_NPY_UNUSED

#endif
