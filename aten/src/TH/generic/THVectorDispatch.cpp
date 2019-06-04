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

static void (*THVector_(cadd_DISPATCHPTR))(scalar_t *, const scalar_t *, const scalar_t *, const scalar_t, const ptrdiff_t) = &THVector_(cadd_DEFAULT);
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

  FUNCTION_IMPL(THVector_(cadd_DEFAULT), SIMDExtension_DEFAULT)
};
void THVector_(cadd)(scalar_t *z, const scalar_t *x, const scalar_t *y, const scalar_t c, const ptrdiff_t n) {
  THVector_(cadd_DISPATCHPTR)(z, x, y, c, n);
}

static void (*THVector_(adds_DISPATCHPTR))(scalar_t *, const scalar_t *, const scalar_t, const ptrdiff_t) = &THVector_(adds_DEFAULT);
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

  FUNCTION_IMPL(THVector_(adds_DEFAULT), SIMDExtension_DEFAULT)
};
// Dispatch stubs that just call the pointers
TH_API void THVector_(adds)(scalar_t *r_, const scalar_t *t, const scalar_t value, const ptrdiff_t n) {
  THVector_(adds_DISPATCHPTR)(r_, t, value, n);
}

static void (*THVector_(cmul_DISPATCHPTR))(scalar_t *, const scalar_t *, const scalar_t *, const ptrdiff_t) = &THVector_(cmul_DEFAULT);
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

  FUNCTION_IMPL(THVector_(cmul_DEFAULT), SIMDExtension_DEFAULT)
};
void THVector_(cmul)(scalar_t *z, const scalar_t *x, const scalar_t *y, const ptrdiff_t n) {
  THVector_(cmul_DISPATCHPTR)(z, x, y, n);
}

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

static void (*THVector_(cdiv_DISPATCHPTR))(scalar_t *, const scalar_t *, const scalar_t *, const ptrdiff_t) = &THVector_(cdiv_DEFAULT);
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

  FUNCTION_IMPL(THVector_(cdiv_DEFAULT), SIMDExtension_DEFAULT)
};
void THVector_(cdiv)(scalar_t *z, const scalar_t *x, const scalar_t *y, const ptrdiff_t n) {
  THVector_(cdiv_DISPATCHPTR)(z, x, y, n);
}

static void (*THVector_(divs_DISPATCHPTR))(scalar_t *, const scalar_t *, const scalar_t, const ptrdiff_t) = &THVector_(divs_DEFAULT);
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

  FUNCTION_IMPL(THVector_(divs_DEFAULT), SIMDExtension_DEFAULT)
};
void THVector_(divs)(scalar_t *y, const scalar_t *x, const scalar_t c, const ptrdiff_t n) {
  THVector_(divs_DISPATCHPTR)(y, x, c, n);
}


static void (*THVector_(normal_fill_DISPATCHPTR))(scalar_t *, const int64_t, at::Generator *, const scalar_t, const scalar_t) = &THVector_(normal_fill_DEFAULT);
static FunctionDescription THVector_(normal_fill_DISPATCHTABLE)[] = {
  #if defined(TH_REAL_IS_FLOAT) && defined(USE_AVX2)
      FUNCTION_IMPL(THVector_(normal_fill_AVX2), SIMDExtension_AVX2),
  #endif

  FUNCTION_IMPL(THVector_(normal_fill_DEFAULT), SIMDExtension_DEFAULT)
};
void THVector_(normal_fill)(scalar_t *data,
                            const int64_t size,
                            at::Generator *generator,
                            const scalar_t mean,
                            const scalar_t stddev) {
  THVector_(normal_fill_DISPATCHPTR)(data, size, generator, mean, stddev);
}

#if defined(TH_REAL_IS_FLOAT) || defined(TH_REAL_IS_DOUBLE)
static void (*THVector_(sigmoid_DISPATCHPTR))(scalar_t *, const scalar_t *, const ptrdiff_t) = &THVector_(sigmoid_DEFAULT);
static FunctionDescription THVector_(sigmoid_DISPATCHTABLE)[] = {
  #if defined(TH_REAL_IS_FLOAT) && defined(USE_AVX2)
      FUNCTION_IMPL(THVector_(sigmoid_AVX2), SIMDExtension_AVX2),
  #endif

  FUNCTION_IMPL(THVector_(sigmoid_DEFAULT), SIMDExtension_DEFAULT)
};
void THVector_(sigmoid)(scalar_t *y, const scalar_t *x, const ptrdiff_t n) {
  THVector_(sigmoid_DISPATCHPTR)(y, x, n);
}
#endif

/*
 * This struct's constructor initalizes the dispatch tables. It simply checks
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
    INIT_DISPATCH_PTR(cadd);
    INIT_DISPATCH_PTR(adds);
    INIT_DISPATCH_PTR(cmul);
    INIT_DISPATCH_PTR(muls);
    INIT_DISPATCH_PTR(cdiv);
    INIT_DISPATCH_PTR(divs);
    INIT_DISPATCH_PTR(normal_fill);

#if defined(TH_REAL_IS_FLOAT) || defined(TH_REAL_IS_DOUBLE)
    INIT_DISPATCH_PTR(sigmoid);
#endif
  }
};

// Declare a global instance to force static initialization
static THVector_(startup) THVector_(g_startup);

#endif /* non bool only part */

#endif
