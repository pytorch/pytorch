#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/THTensorRandom.c"
#else

TH_API void THTensor_(random)(THGenerator *_generator, THTensor *self)
{
#if defined(TH_REAL_IS_BYTE)
  TH_TENSOR_APPLY(real, self, *self_data = (unsigned char)(THRandom_random(_generator) % (UCHAR_MAX+1)););
#elif defined(TH_REAL_IS_CHAR)
  TH_TENSOR_APPLY(real, self, *self_data = (char)(THRandom_random(_generator) % (CHAR_MAX+1)););
#elif defined(TH_REAL_IS_SHORT)
  TH_TENSOR_APPLY(real, self, *self_data = (short)(THRandom_random(_generator) % (SHRT_MAX+1)););
#elif defined(TH_REAL_IS_INT)
  TH_TENSOR_APPLY(real, self, *self_data = (int)(THRandom_random(_generator) % (INT_MAX+1UL)););
#elif defined(TH_REAL_IS_LONG)
  TH_TENSOR_APPLY(real, self, *self_data = (long)(THRandom_random(_generator) % (LONG_MAX+1UL)););
#elif defined(TH_REAL_IS_FLOAT)
  TH_TENSOR_APPLY(real, self, *self_data = (float)(THRandom_random(_generator) % ((1UL << FLT_MANT_DIG)+1)););
#elif defined(TH_REAL_IS_DOUBLE)
  TH_TENSOR_APPLY(real, self, *self_data = (float)(THRandom_random(_generator) % ((1UL << DBL_MANT_DIG)+1)););
#else
#error "Unknown type"
#endif
}

TH_API void THTensor_(geometric)(THGenerator *_generator, THTensor *self, double p)
{
  TH_TENSOR_APPLY(real, self, *self_data = (real)THRandom_geometric(_generator, p););
}

TH_API void THTensor_(bernoulli)(THGenerator *_generator, THTensor *self, double p)
{
  TH_TENSOR_APPLY(real, self, *self_data = (real)THRandom_bernoulli(_generator, p););
}

#if defined(TH_REAL_IS_FLOAT) || defined(TH_REAL_IS_DOUBLE)

TH_API void THTensor_(uniform)(THGenerator *_generator, THTensor *self, double a, double b)
{
  TH_TENSOR_APPLY(real, self, *self_data = (real)THRandom_uniform(_generator, a, b););
}

TH_API void THTensor_(normal)(THGenerator *_generator, THTensor *self, double mean, double stdv)
{
  TH_TENSOR_APPLY(real, self, *self_data = (real)THRandom_normal(_generator, mean, stdv););
}

TH_API void THTensor_(exponential)(THGenerator *_generator, THTensor *self, double lambda)
{
  TH_TENSOR_APPLY(real, self, *self_data = (real)THRandom_exponential(_generator, lambda););
}

TH_API void THTensor_(cauchy)(THGenerator *_generator, THTensor *self, double median, double sigma)
{
  TH_TENSOR_APPLY(real, self, *self_data = (real)THRandom_cauchy(_generator, median, sigma););
}

TH_API void THTensor_(logNormal)(THGenerator *_generator, THTensor *self, double mean, double stdv)
{
  TH_TENSOR_APPLY(real, self, *self_data = (real)THRandom_logNormal(_generator, mean, stdv););
}

#endif

#if defined(TH_REAL_IS_LONG)
TH_API void THTensor_(getRNGState)(THGenerator *_generator, THTensor *self)
{
  unsigned long *data;
  long *offset;
  long *left;

  THTensor_(resize1d)(self,626);
  data = (unsigned long *)THTensor_(data)(self);
  offset = (long *)data+624;
  left = (long *)data+625;

  THRandom_getState(_generator, data, offset, left);
}

TH_API void THTensor_(setRNGState)(THGenerator *_generator, THTensor *self)
{
  unsigned long *data;
  long *offset;
  long *left;

  THArgCheck(THTensor_(nElement)(self) == 626, 1, "state should have 626 elements");
  data = (unsigned long *)THTensor_(data)(self);
  offset = (long *)(data+624);
  left = (long *)(data+625);

  THRandom_setState(_generator, data, *offset, *left);
}

#endif

#endif
