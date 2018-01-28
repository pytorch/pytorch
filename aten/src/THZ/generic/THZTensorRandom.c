#ifndef THZ_GENERIC_FILE
#define THZ_GENERIC_FILE "generic/THZTensorRandom.c"
#else

void THZTensor_(random)(THZTensor *self, THGenerator *_generator)
{

#if defined(THZ_NTYPE_IS_ZFLOAT)
  TH_TENSOR_APPLY(ntype, self, *self_data = (float)(THRandom_random(_generator) % ((1ULL << FLT_MANT_DIG) + 1)) +
    I * (float)(THRandom_random(_generator) % ((1ULL << FLT_MANT_DIG) + 1)););
#elif defined(THZ_NTYPE_IS_ZDOUBLE)
  TH_TENSOR_APPLY(ntype, self, *self_data = (double)(THRandom_random64(_generator) % ((1ULL << DBL_MANT_DIG) + 1)) +
    I * (double)(THRandom_random64(_generator) % ((1ULL << DBL_MANT_DIG) + 1)););
#else
#error "Unknown type"
#endif

}

void THZTensor_(uniform)(THZTensor *self, THGenerator *_generator, double a, double b)
{
  TH_TENSOR_APPLY(ntype, self, *self_data =
    #if defined(THZ_NTYPE_IS_ZFLOAT)
    (ntype)THRandom_uniformFloat(_generator, (ntype)a, (ntype)b) +
      I * (ntype)THRandom_uniformFloat(_generator, (ntype)a, (ntype)b););
    #elif defined(THZ_NTYPE_IS_ZDOUBLE)
    (ntype)THRandom_uniform(_generator, a, b) + I * (ntype)THRandom_uniform(_generator, a, b););
    #else
    #error "Unknown type"
    #endif
}

#endif
