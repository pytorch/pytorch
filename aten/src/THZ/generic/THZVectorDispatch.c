#ifndef THZ_GENERIC_FILE
#define THZ_GENERIC_FILE "generic/THZVectorDispatch.c"
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

static void (*THZVector_(fill_DISPATCHPTR))(ntype *, const ntype, const ptrdiff_t) = &THZVector_(fill_DEFAULT);
static FunctionDescription THZVector_(fill_DISPATCHTABLE)[] = {

  #if defined(USE_AVX)
    #if defined(THZ_NTYPE_IS_COMPLEX)
      FUNCTION_IMPL(THZVector_(fill_AVX), SIMDExtension_AVX),
    #endif
  #endif

  FUNCTION_IMPL(THZVector_(fill_DEFAULT), SIMDExtension_DEFAULT)
};
void THZVector_(fill)(ntype *x, const ntype c, const ptrdiff_t n) {
  THZVector_(fill_DISPATCHPTR)(x, c, n);
}

static void (*THZVector_(cadd_DISPATCHPTR))(ntype *, const ntype *, const ntype *, const ntype, const ptrdiff_t) = &THZVector_(cadd_DEFAULT);
static FunctionDescription THZVector_(cadd_DISPATCHTABLE)[] = {

  #if defined(USE_AVX2)
    #if defined(THZ_NTYPE_IS_COMPLEX)
      FUNCTION_IMPL(THZVector_(cadd_AVX2), SIMDExtension_AVX2),
    #endif
  #endif

  FUNCTION_IMPL(THZVector_(cadd_DEFAULT), SIMDExtension_DEFAULT)
};
void THZVector_(cadd)(ntype *z, const ntype *x, const ntype *y, const ntype c, const ptrdiff_t n) {
  THZVector_(cadd_DISPATCHPTR)(z, x, y, c, n);
}

static void (*THZVector_(adds_DISPATCHPTR))(ntype *, const ntype *, const ntype, const ptrdiff_t) = &THZVector_(adds_DEFAULT);
static FunctionDescription THZVector_(adds_DISPATCHTABLE)[] = {

  #if defined(USE_AVX)
    #if defined(THZ_NTYPE_IS_COMPLEX)
      FUNCTION_IMPL(THZVector_(adds_AVX), SIMDExtension_AVX),
    #endif
  #endif

  FUNCTION_IMPL(THZVector_(adds_DEFAULT), SIMDExtension_DEFAULT)
};
// Dispatch stubs that just call the pointers
TH_API void THZVector_(adds)(ntype *r_, const ntype *t, const ntype value, const ptrdiff_t n) {
  THZVector_(adds_DISPATCHPTR)(r_, t, value, n);
}

static void (*THZVector_(cmul_DISPATCHPTR))(ntype *, const ntype *, const ntype *, const ptrdiff_t) = &THZVector_(cmul_DEFAULT);
static FunctionDescription THZVector_(cmul_DISPATCHTABLE)[] = {

  #if defined(USE_AVX2)
    #if defined(THZ_NTYPE_IS_COMPLEX)
      FUNCTION_IMPL(THZVector_(cmul_AVX2), SIMDExtension_AVX2),
    #endif
  #endif

  FUNCTION_IMPL(THZVector_(cmul_DEFAULT), SIMDExtension_DEFAULT)
};
void THZVector_(cmul)(ntype *z, const ntype *x, const ntype *y, const ptrdiff_t n) {
  THZVector_(cmul_DISPATCHPTR)(z, x, y, n);
}

static void (*THZVector_(muls_DISPATCHPTR))(ntype *, const ntype *, const ntype, const ptrdiff_t) = &THZVector_(muls_DEFAULT);
static FunctionDescription THZVector_(muls_DISPATCHTABLE)[] = {

  #if defined(USE_AVX2)
    #if defined(THZ_NTYPE_IS_COMPLEX)
      FUNCTION_IMPL(THZVector_(muls_AVX2), SIMDExtension_AVX2),
    #endif
  #endif

  FUNCTION_IMPL(THZVector_(muls_DEFAULT), SIMDExtension_DEFAULT)
};
void THZVector_(muls)(ntype *y, const ntype *x, const ntype c, const ptrdiff_t n) {
  THZVector_(muls_DISPATCHPTR)(y, x, c, n);
}

static void (*THZVector_(cdiv_DISPATCHPTR))(ntype *, const ntype *, const ntype *, const ptrdiff_t) = &THZVector_(cdiv_DEFAULT);
static FunctionDescription THZVector_(cdiv_DISPATCHTABLE)[] = {

  #if defined(USE_AVX2)
    #if defined(THZ_NTYPE_IS_COMPLEX)
      FUNCTION_IMPL(THZVector_(cdiv_AVX2), SIMDExtension_AVX2),
    #endif
  #endif

  FUNCTION_IMPL(THZVector_(cdiv_DEFAULT), SIMDExtension_DEFAULT)
};
void THZVector_(cdiv)(ntype *z, const ntype *x, const ntype *y, const ptrdiff_t n) {
  THZVector_(cdiv_DISPATCHPTR)(z, x, y, n);
}

static void (*THZVector_(divs_DISPATCHPTR))(ntype *, const ntype *, const ntype, const ptrdiff_t) = &THZVector_(divs_DEFAULT);
static FunctionDescription THZVector_(divs_DISPATCHTABLE)[] = {

  #if defined(USE_AVX2)
    #if defined(THZ_NTYPE_IS_COMPLEX)
      FUNCTION_IMPL(THZVector_(divs_AVX2), SIMDExtension_AVX2),
    #endif
  #endif

  FUNCTION_IMPL(THZVector_(divs_DEFAULT), SIMDExtension_DEFAULT)
};
void THZVector_(divs)(ntype *y, const ntype *x, const ntype c, const ptrdiff_t n) {
  THZVector_(divs_DISPATCHPTR)(y, x, c, n);
}

static void (*THZVector_(copy_DISPATCHPTR))(ntype *, const ntype *, const ptrdiff_t) = &THZVector_(copy_DEFAULT);
static FunctionDescription THZVector_(copy_DISPATCHTABLE)[] = {
  #if defined(USE_AVX)
    #if defined(THZ_NTYPE_IS_COMPLEX)
      FUNCTION_IMPL(THZVector_(copy_AVX), SIMDExtension_AVX),
    #endif
  #endif

  FUNCTION_IMPL(THZVector_(copy_DEFAULT), SIMDExtension_DEFAULT)
};
void THZVector_(copy)(ntype *y, const ntype *x, const ptrdiff_t n) {
  THZVector_(copy_DISPATCHPTR)(y, x, n);
}

/* This needs to be called in order to initialize the dispatch pointers at runtime.
 * This function simply checks what SIMD extensions are available, and then walks the dispatch table
 * to choose the best function.
 * NOTE: As implemented, it will initialize the dispatch pointer to the first supported function.
 *       This means that in the dispatch tables, implementations supporting more recent extensions
 *       need to come first
 */
void THZVector_(vectorDispatchInit)(void)
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
