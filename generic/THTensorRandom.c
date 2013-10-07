#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/THTensorRandom.c"
#else

TH_API void THTensor_(random)(THTensor *self)
{
#if defined(TH_REAL_IS_BYTE)
  TH_TENSOR_APPLY(real, self, *self_data = (unsigned char)(THRandom_random() % (UCHAR_MAX+1)););
#elif defined(TH_REAL_IS_CHAR)
  TH_TENSOR_APPLY(real, self, *self_data = (char)(THRandom_random() % (CHAR_MAX+1)););
#elif defined(TH_REAL_IS_SHORT)
  TH_TENSOR_APPLY(real, self, *self_data = (short)(THRandom_random() % (SHRT_MAX+1)););
#elif defined(TH_REAL_IS_INT)
  TH_TENSOR_APPLY(real, self, *self_data = (int)(THRandom_random() % (INT_MAX+1UL)););
#elif defined(TH_REAL_IS_LONG)
  TH_TENSOR_APPLY(real, self, *self_data = (long)(THRandom_random() % (LONG_MAX+1UL)););
#elif defined(TH_REAL_IS_FLOAT)
  TH_TENSOR_APPLY(real, self, *self_data = (float)(THRandom_random() % ((1UL << FLT_MANT_DIG)+1)););
#elif defined(TH_REAL_IS_DOUBLE)
  TH_TENSOR_APPLY(real, self, *self_data = (float)(THRandom_random() % ((1UL << DBL_MANT_DIG)+1)););
#else
#error "Unknown type"
#endif
}

TH_API void THTensor_(geometric)(THTensor *self, double p)
{
  TH_TENSOR_APPLY(real, self, *self_data = (real)THRandom_geometric(p););
}

TH_API void THTensor_(bernoulli)(THTensor *self, double p)
{
  TH_TENSOR_APPLY(real, self, *self_data = (real)THRandom_bernoulli(p););
}

#if defined(TH_REAL_IS_FLOAT) || defined(TH_REAL_IS_DOUBLE)

TH_API void THTensor_(uniform)(THTensor *self, double a, double b)
{
  TH_TENSOR_APPLY(real, self, *self_data = (real)THRandom_uniform(a, b););
}

TH_API void THTensor_(normal)(THTensor *self, double mean, double stdv)
{
  TH_TENSOR_APPLY(real, self, *self_data = (real)THRandom_normal(mean, stdv););
}

TH_API void THTensor_(exponential)(THTensor *self, double lambda)
{
  TH_TENSOR_APPLY(real, self, *self_data = (real)THRandom_exponential(lambda););
}

TH_API void THTensor_(cauchy)(THTensor *self, double median, double sigma)
{
  TH_TENSOR_APPLY(real, self, *self_data = (real)THRandom_cauchy(median, sigma););
}

TH_API void THTensor_(logNormal)(THTensor *self, double mean, double stdv)
{
  TH_TENSOR_APPLY(real, self, *self_data = (real)THRandom_logNormal(mean, stdv););
}

#endif

#if defined(TH_REAL_IS_LONG)
TH_API void THTensor_(getRNGState)(THTensor *self)
{
  unsigned long *data;
  long *offset;
  long *left;

  THTensor_(resize1d)(self,626);
  data = (unsigned long *)THTensor_(data)(self);
  offset = (long *)data+624;
  left = (long *)data+625;

  THRandom_getState(data,offset,left);
}

TH_API void THTensor_(setRNGState)(THTensor *self)
{
  unsigned long *data;
  long *offset;
  long *left;

  THArgCheck(THTensor_(nElement)(self) == 626, 1, "state should have 626 elements");
  data = (unsigned long *)THTensor_(data)(self);
  offset = (long *)(data+624);
  left = (long *)(data+625);

  THRandom_setState(data,*offset,*left);
}

#endif

#endif
