#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "TH/generic/THVectorDispatch.cpp"
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

static void (*THVector_(fill_DISPATCHPTR))(scalar_t *, const scalar_t, const ptrdiff_t) = &THVector_(fill_DEFAULT);
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

  FUNCTION_IMPL(THVector_(fill_DEFAULT), SIMDExtension_DEFAULT)
};
void THVector_(fill)(scalar_t *x, const scalar_t c, const ptrdiff_t n) {
  THVector_(fill_DISPATCHPTR)(x, c, n);
}

#if !defined(TH_REAL_IS_BOOL) /* non bool only part */
static void (*THVector_(muls_DISPATCHPTR))(scalar_t *, const scalar_t *, const scalar_t, const ptrdiff_t) = &THVector_(muls_DEFAULT);
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

  FUNCTION_IMPL(THVector_(muls_DEFAULT), SIMDExtension_DEFAULT)
};
void THVector_(muls)(scalar_t *y, const scalar_t *x, const scalar_t c, const ptrdiff_t n) {
  THVector_(muls_DISPATCHPTR)(y, x, c, n);
}

/*
 * This struct's constructor initializes the dispatch tables. It simply checks
 * what SIMD extensions are available, and then walks the dispatch table
 * to choose the best function.
 * NOTE: As implemented, it will initialize the dispatch pointer to the first supported function.
 *       This means that in the dispatch tables, implementations supporting more recent extensions
 *       need to come first
 */
struct THVector_(startup) {
  THVector_(startup)() {
    uint32_t hostSimdExts = detectHostSIMDExtensions();
    INIT_DISPATCH_PTR(fill);
    INIT_DISPATCH_PTR(muls);
  }
};

// Declare a global instance to force static initialization
static THVector_(startup) THVector_(g_startup);

#endif /* non bool only part */

#endif
