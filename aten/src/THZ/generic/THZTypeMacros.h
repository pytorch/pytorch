#ifdef THZ_TYPE_NAME
#undef THZ_TYPE_NAME
#endif

#ifdef THZ_BLAS_NAME
#undef THZ_BLAS_NAME
#endif

#ifdef THZ_MATH_NAME
#undef THZ_MATH_NAME
#endif

#ifdef THZMath_
#undef THZMath_
#endif

#ifdef THZ_LAPACK_NAME
#undef THZ_LAPACK_NAME
#endif

#ifdef THZ_ABS
#undef THZ_ABS
#endif

#ifdef THZ_MULC
#undef THZ_MULC
#endif

#define THZ_TYPE_NAME(NAME) #NAME

#if defined(THZ_NTYPE_IS_ZDOUBLE)
#define THZ_BLAS_NAME(NAME) z##NAME##_
#define THZ_MATH_NAME(NAME) c##NAME
#define THZMath_(NAME) THZ_c##NAME
#define THZ_ABS(Z) cabs(Z)
#define THZ_MULC(Z, C) ((Z) * conj(C))
#elif defined(THZ_NTYPE_IS_ZFLOAT)
#define THZ_BLAS_NAME(NAME) c##NAME##_
#define THZ_MATH_NAME(NAME) c##NAME##f
#define THZMath_(NAME) THZ_c##NAME##f
#define THZ_ABS(Z) cabsf(Z)
#define THZ_MULC(Z, C) ((Z) * conjf(C))
// Int types
#else
#define THZ_BLAS_NAME(NAME)
#define THZ_MATH_NAME(NAME)
#define THZMath_(NAME)
#define THZ_ABS(Z)
#endif

#define THZ_LAPACK_NAME(NAME) THZ_BLAS_NAME(NAME)
