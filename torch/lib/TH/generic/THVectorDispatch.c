#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/THVectorDispatch.c"
#else

/* For now there are only SIMD implementations for FLOAT and DOUBLE.
 * Hopefully in the future this can be made totally generic (e.g, there are SIMD implementations
 * for a lot of functions */
/* Each function with multiple implementations has:
 * 1. A DISPATCHPTR which will be initialized to point to the best available implementation for the host
 * 2. A DISPATCHTABLE which holds pointers to each implementation of a function, and a value indicating
 *    which SIMD extension a given implementation uses
 * 3. A dispatch stub, which is what is actually called by clients, that simply wraps the dispatch pointer.
 */

static void (*THVector_(fill_DISPATCHPTR))(real *, const real, const ptrdiff_t) = &THVector_(fill_DEFAULT);
static FunctionDescription THVector_(fill_DISPATCHTABLE)[] = {
  #if defined(__NEON__)
    #if defined(TH_REAL_IS_FLOAT)
      FUNCTION_IMPL(THVector_(fill_NEON), SIMDExtension_NEON),
    #endif
  #endif

  #if defined(__PPC64__)
    #if defined(TH_REAL_IS_DOUBLE) || defined(TH_REAL_IS_FLOAT)
      FUNCTION_IMPL(THVector_(fill_VSX), SIMDExtension_VSX),
    #endif
  #endif

  #if defined(USE_AVX)
    #if defined(TH_REAL_IS_DOUBLE) || defined(TH_REAL_IS_FLOAT)
      FUNCTION_IMPL(THVector_(fill_AVX), SIMDExtension_AVX),
    #endif
  #endif

  #if defined(USE_SSE2) || defined(USE_SSE3) || defined(USE_SSSE3) \
          || defined(USE_SSE4_1) || defined(USE_SSE4_2)
    #if defined(TH_REAL_IS_DOUBLE) || defined(TH_REAL_IS_FLOAT)
      FUNCTION_IMPL(THVector_(fill_SSE), SIMDExtension_SSE),
    #endif
  #endif
  FUNCTION_IMPL(THVector_(fill_DEFAULT), SIMDExtension_DEFAULT)
};
void THVector_(fill)(real *x, const real c, const ptrdiff_t n) {
  THVector_(fill_DISPATCHPTR)(x, c, n);
}

static void (*THVector_(cadd_DISPATCHPTR))(real *, const real *, const real *, const real, const ptrdiff_t) = &THVector_(cadd_DEFAULT);
static FunctionDescription THVector_(cadd_DISPATCHTABLE)[] = {
  #if defined(__NEON__)
    #if defined(TH_REAL_IS_FLOAT)
      FUNCTION_IMPL(THVector_(cadd_NEON), SIMDExtension_NEON),
    #endif
  #endif

  #if defined(USE_AVX2)
    #if defined(TH_REAL_IS_DOUBLE) || defined(TH_REAL_IS_FLOAT)
      FUNCTION_IMPL(THVector_(cadd_AVX2), SIMDExtension_AVX2),
    #endif
  #endif

  #if defined(USE_AVX)
    #if defined(TH_REAL_IS_DOUBLE) || defined(TH_REAL_IS_FLOAT)
      FUNCTION_IMPL(THVector_(cadd_AVX), SIMDExtension_AVX),
    #endif
  #endif

  #if defined(USE_SSE2) || defined(USE_SSE3) || defined(USE_SSSE3) \
          || defined(USE_SSE4_1) || defined(USE_SSE4_2)
    #if defined(TH_REAL_IS_DOUBLE) || defined(TH_REAL_IS_FLOAT)
      FUNCTION_IMPL(THVector_(cadd_SSE), SIMDExtension_SSE),
    #endif
  #endif

  FUNCTION_IMPL(THVector_(cadd_DEFAULT), SIMDExtension_DEFAULT)
};
void THVector_(cadd)(real *z, const real *x, const real *y, const real c, const ptrdiff_t n) {
  THVector_(cadd_DISPATCHPTR)(z, x, y, c, n);
}

static void (*THVector_(adds_DISPATCHPTR))(real *, const real *, const real, const ptrdiff_t) = &THVector_(adds_DEFAULT);
static FunctionDescription THVector_(adds_DISPATCHTABLE)[] = {
  #if defined(__NEON__)
    #if defined(TH_REAL_IS_FLOAT)
      FUNCTION_IMPL(THVector_(adds_NEON), SIMDExtension_NEON),
    #endif
  #endif

  #if defined(__PPC64__)
    #if defined(TH_REAL_IS_DOUBLE) || defined(TH_REAL_IS_FLOAT)
      FUNCTION_IMPL(THVector_(adds_VSX), SIMDExtension_VSX),
    #endif
  #endif

  #if defined(USE_AVX)
    #if defined(TH_REAL_IS_DOUBLE) || defined(TH_REAL_IS_FLOAT)
      FUNCTION_IMPL(THVector_(adds_AVX), SIMDExtension_AVX),
    #endif
  #endif

  #if defined(USE_SSE2) || defined(USE_SSE3) || defined(USE_SSSE3) \
          || defined(USE_SSE4_1) || defined(USE_SSE4_2)
    #if defined(TH_REAL_IS_DOUBLE) || defined(TH_REAL_IS_FLOAT)
      FUNCTION_IMPL(THVector_(adds_SSE), SIMDExtension_SSE),
    #endif
  #endif

  FUNCTION_IMPL(THVector_(adds_DEFAULT), SIMDExtension_DEFAULT)
};
// Dispatch stubs that just call the pointers
TH_API void THVector_(adds)(real *r_, const real *t, const real value, const ptrdiff_t n) {
  THVector_(adds_DISPATCHPTR)(r_, t, value, n);
}

static void (*THVector_(cmul_DISPATCHPTR))(real *, const real *, const real *, const ptrdiff_t) = &THVector_(cmul_DEFAULT);
static FunctionDescription THVector_(cmul_DISPATCHTABLE)[] = {
  #if defined(__NEON__)
    #if defined(TH_REAL_IS_FLOAT)
      FUNCTION_IMPL(THVector_(cmul_NEON), SIMDExtension_NEON),
    #endif
  #endif

  #if defined(USE_AVX)
    #if defined(TH_REAL_IS_DOUBLE) || defined(TH_REAL_IS_FLOAT)
      FUNCTION_IMPL(THVector_(cmul_AVX), SIMDExtension_AVX),
    #endif
  #endif

  #if defined(USE_SSE2) || defined(USE_SSE3) || defined(USE_SSSE3) \
          || defined(USE_SSE4_1) || defined(USE_SSE4_2)
    #if defined(TH_REAL_IS_DOUBLE) || defined(TH_REAL_IS_FLOAT)
      FUNCTION_IMPL(THVector_(cmul_SSE), SIMDExtension_SSE),
    #endif
  #endif

  FUNCTION_IMPL(THVector_(cmul_DEFAULT), SIMDExtension_DEFAULT)
};
void THVector_(cmul)(real *z, const real *x, const real *y, const ptrdiff_t n) {
  THVector_(cmul_DISPATCHPTR)(z, x, y, n);
}

static void (*THVector_(muls_DISPATCHPTR))(real *, const real *, const real, const ptrdiff_t) = &THVector_(muls_DEFAULT);
static FunctionDescription THVector_(muls_DISPATCHTABLE)[] = {
  #if defined(__NEON__)
    #if defined(TH_REAL_IS_FLOAT)
      FUNCTION_IMPL(THVector_(muls_NEON), SIMDExtension_NEON),
    #endif
  #endif

  #if defined(__PPC64__)
    #if defined(TH_REAL_IS_DOUBLE) || defined(TH_REAL_IS_FLOAT)
      FUNCTION_IMPL(THVector_(muls_VSX), SIMDExtension_VSX),
    #endif
  #endif

  #if defined(USE_AVX)
    #if defined(TH_REAL_IS_DOUBLE) || defined(TH_REAL_IS_FLOAT)
      FUNCTION_IMPL(THVector_(muls_AVX), SIMDExtension_AVX),
    #endif
  #endif

  #if defined(USE_SSE2) || defined(USE_SSE3) || defined(USE_SSSE3) \
          || defined(USE_SSE4_1) || defined(USE_SSE4_2)
    #if defined(TH_REAL_IS_DOUBLE) || defined(TH_REAL_IS_FLOAT)
      FUNCTION_IMPL(THVector_(muls_SSE), SIMDExtension_SSE),
    #endif
  #endif

  FUNCTION_IMPL(THVector_(muls_DEFAULT), SIMDExtension_DEFAULT)
};
void THVector_(muls)(real *y, const real *x, const real c, const ptrdiff_t n) {
  THVector_(muls_DISPATCHPTR)(y, x, c, n);
}

static void (*THVector_(cdiv_DISPATCHPTR))(real *, const real *, const real *, const ptrdiff_t) = &THVector_(cdiv_DEFAULT);
static FunctionDescription THVector_(cdiv_DISPATCHTABLE)[] = {
  #if defined(__NEON__)
    #if defined(TH_REAL_IS_FLOAT)
      FUNCTION_IMPL(THVector_(cdiv_NEON), SIMDExtension_NEON),
    #endif
  #endif

  #if defined(USE_AVX)
    #if defined(TH_REAL_IS_DOUBLE) || defined(TH_REAL_IS_FLOAT)
      FUNCTION_IMPL(THVector_(cdiv_AVX), SIMDExtension_AVX),
    #endif
  #endif

  #if defined(USE_SSE2) || defined(USE_SSE3) || defined(USE_SSSE3) \
          || defined(USE_SSE4_1) || defined(USE_SSE4_2)
    #if defined(TH_REAL_IS_DOUBLE) || defined(TH_REAL_IS_FLOAT)
      FUNCTION_IMPL(THVector_(cdiv_SSE), SIMDExtension_SSE),
    #endif
  #endif

  FUNCTION_IMPL(THVector_(cdiv_DEFAULT), SIMDExtension_DEFAULT)
};
void THVector_(cdiv)(real *z, const real *x, const real *y, const ptrdiff_t n) {
  THVector_(cdiv_DISPATCHPTR)(z, x, y, n);
}

static void (*THVector_(divs_DISPATCHPTR))(real *, const real *, const real, const ptrdiff_t) = &THVector_(divs_DEFAULT);
static FunctionDescription THVector_(divs_DISPATCHTABLE)[] = {
  #if defined(__NEON__)
    #if defined(TH_REAL_IS_FLOAT)
      FUNCTION_IMPL(THVector_(divs_NEON), SIMDExtension_NEON),
    #endif
  #endif

  #if defined(USE_AVX)
    #if defined(TH_REAL_IS_DOUBLE) || defined(TH_REAL_IS_FLOAT)
      FUNCTION_IMPL(THVector_(divs_AVX), SIMDExtension_AVX),
    #endif
  #endif

  #if defined(USE_SSE2) || defined(USE_SSE3) || defined(USE_SSSE3) \
          || defined(USE_SSE4_1) || defined(USE_SSE4_2)
    #if defined(TH_REAL_IS_DOUBLE) || defined(TH_REAL_IS_FLOAT)
      FUNCTION_IMPL(THVector_(divs_SSE), SIMDExtension_SSE),
    #endif
  #endif

  FUNCTION_IMPL(THVector_(divs_DEFAULT), SIMDExtension_DEFAULT)
};
void THVector_(divs)(real *y, const real *x, const real c, const ptrdiff_t n) {
  THVector_(divs_DISPATCHPTR)(y, x, c, n);
}

static void (*THVector_(copy_DISPATCHPTR))(real *, const real *, const ptrdiff_t) = &THVector_(copy_DEFAULT);
static FunctionDescription THVector_(copy_DISPATCHTABLE)[] = {
  #if defined(USE_AVX)
    #if defined(TH_REAL_IS_DOUBLE) || defined(TH_REAL_IS_FLOAT)
      FUNCTION_IMPL(THVector_(copy_AVX), SIMDExtension_AVX),
    #endif
  #endif

  FUNCTION_IMPL(THVector_(copy_DEFAULT), SIMDExtension_DEFAULT)
};
void THVector_(copy)(real *y, const real *x, const ptrdiff_t n) {
  THVector_(copy_DISPATCHPTR)(y, x, n);
}

/* This needs to be called in order to initialize the dispatch pointers at runtime.
 * This function simply checks what SIMD extensions are available, and then walks the dispatch table
 * to choose the best function.
 * NOTE: As implemented, it will initialize the dispatch pointer to the first supported function.
 *       This means that in the dispatch tables, implementations supporting more recent extensions
 *       need to come first
 */
void THVector_(vectorDispatchInit)(void)
{
  uint32_t hostSimdExts = detectHostSIMDExtensions();
  INIT_DISPATCH_PTR(fill);
  INIT_DISPATCH_PTR(cadd);
  INIT_DISPATCH_PTR(adds);
  INIT_DISPATCH_PTR(cmul);
  INIT_DISPATCH_PTR(muls);
  INIT_DISPATCH_PTR(cdiv);
  INIT_DISPATCH_PTR(divs);
  INIT_DISPATCH_PTR(copy);
}

#endif
