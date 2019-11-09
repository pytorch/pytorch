#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "TH/generic/THVectorDefault.cpp"
#else

#include <ATen/Context.h>

void THVector_(fill_DEFAULT)(scalar_t *x, const scalar_t c, const ptrdiff_t n) {
  ptrdiff_t i = 0;

  for(; i <n-4; i+=4)
  {
    x[i] = c;
    x[i+1] = c;
    x[i+2] = c;
    x[i+3] = c;
  }

  for(; i < n; i++)
    x[i] = c;
}

#if !defined(TH_REAL_IS_BOOL) /* non bool only part */

void THVector_(copy_DEFAULT)(scalar_t *x, const scalar_t *y, const ptrdiff_t n) {
  ptrdiff_t i = 0;

  for(; i <n-4; i+=4)
  {
    x[i] = y[i];
    x[i+1] = y[i+1];
    x[i+2] = y[i+2];
    x[i+3] = y[i+3];
  }

  for(; i < n; i++)
    x[i] = y[i];
}

void THVector_(cadd_DEFAULT)(scalar_t *z, const scalar_t *x, const scalar_t *y, const scalar_t c, const ptrdiff_t n)
{
  ptrdiff_t i = 0;

  for(; i<n-4; i+=4)
  {
    z[i] = x[i] + c * y[i];
    z[i+1] = x[i+1] + c * y[i+1];
    z[i+2] = x[i+2] + c * y[i+2];
    z[i+3] = x[i+3] + c * y[i+3];
  }

  for(; i<n; i++)
    z[i] = x[i] + c * y[i];
}

void THVector_(adds_DEFAULT)(scalar_t *y, const scalar_t *x, const scalar_t c, const ptrdiff_t n)
{
  ptrdiff_t i = 0;

  for(; i<n-4; i+=4)
  {
    y[i] = x[i] + c;
    y[i+1] = x[i+1] + c;
    y[i+2] = x[i+2] + c;
    y[i+3] = x[i+3] + c;
  }

  for(; i<n; i++)
    y[i] = x[i] + c;
}

void THVector_(cmul_DEFAULT)(scalar_t *z, const scalar_t *x, const scalar_t *y, const ptrdiff_t n)
{
  ptrdiff_t i = 0;

  for(; i <n-4; i+=4)
  {
    z[i] = x[i] * y[i];
    z[i+1] = x[i+1] * y[i+1];
    z[i+2] = x[i+2] * y[i+2];
    z[i+3] = x[i+3] * y[i+3];
  }

  for(; i < n; i++)
    z[i] = x[i] * y[i];
}

void THVector_(muls_DEFAULT)(scalar_t *y, const scalar_t *x, const scalar_t c, const ptrdiff_t n)
{
  ptrdiff_t i = 0;

  for(; i <n-4; i+=4)
  {
    y[i] = x[i] * c;
    y[i+1] = x[i+1] * c;
    y[i+2] = x[i+2] * c;
    y[i+3] = x[i+3] * c;
  }

  for(; i < n; i++)
    y[i] = x[i] * c;
}

void THVector_(cdiv_DEFAULT)(scalar_t *z, const scalar_t *x, const scalar_t *y, const ptrdiff_t n)
{
  ptrdiff_t i = 0;

  for(; i<n-4; i+=4)
  {
    z[i] = x[i] / y[i];
    z[i+1] = x[i+1] / y[i+1];
    z[i+2] = x[i+2] / y[i+2];
    z[i+3] = x[i+3] / y[i+3];
  }

  for(; i < n; i++)
    z[i] = x[i] / y[i];
}

void THVector_(divs_DEFAULT)(scalar_t *y, const scalar_t *x, const scalar_t c, const ptrdiff_t n)
{
  ptrdiff_t i = 0;

  for(; i<n-4; i+=4)
  {
    y[i] = x[i] / c;
    y[i+1] = x[i+1] / c;
    y[i+2] = x[i+2] / c;
    y[i+3] = x[i+3] / c;
  }

  for(; i < n; i++)
    y[i] = x[i] / c;
}

// Fills 16 normally distributed samples into data, interleaved with a
// stride of 8, i.e. in order of ([0], [8]), ([1], [9]), ...
static void THVector_(interleaved_normal_fill_16)(scalar_t *data,
                                                  const scalar_t mean,
                                                  const scalar_t stddev)
{
  for (int j = 0; j < 8; ++j) {
    const scalar_t u1 = 1 - data[j]; // [0, 1) -> (0, 1] for log.
    const scalar_t u2 = data[j + 8];

    const scalar_t radius = sqrt(-2 * log(u1));
    const scalar_t theta = 2.0f * M_PI * u2;

    data[j] = radius * cos(theta) * stddev + mean;
    data[j + 8] = radius * std::sin(theta) * stddev + mean;
  }
}

void THVector_(normal_fill_DEFAULT)(scalar_t *data,
                                    int64_t size,
                                    at::Generator *generator,
                                    const scalar_t mean,
                                    const scalar_t stddev)
{
  THAssert(size >= 16 && "Size must be >= 16 for normal fill");
  auto gen = at::get_generator_or_default<at::CPUGenerator>(generator, at::detail::getDefaultCPUGenerator());
  // See Note [Acquire lock when using random generators]
  std::lock_guard<std::mutex> lock(gen->mutex_);
  
  for (int64_t i = 0; i < size; ++i) {
#ifdef TH_REAL_IS_FLOAT
    at::uniform_real_distribution<float> uniform(0, 1);
    data[i] = uniform(gen);
#else
    at::uniform_real_distribution<double> uniform(0, 1);
    data[i] = uniform(gen);
#endif
  }

  for (int64_t i = 0; i < size - 15; i += 16) {
    THVector_(interleaved_normal_fill_16)(data + i, mean, stddev);
  }

  if (size % 16 != 0) {
    // Recompute the last 16 values.
    data = data + size - 16;
    for (int64_t i = 0; i < 16; ++i) {
#ifdef TH_REAL_IS_FLOAT
    at::uniform_real_distribution<float> uniform(0, 1);
    data[i] = uniform(gen);
#else
    at::uniform_real_distribution<double> uniform(0, 1);
    data[i] = uniform(gen);
#endif
    }
    THVector_(interleaved_normal_fill_16)(data, mean, stddev);
  }
}

#define VECTOR_IMPLEMENT_FUNCTION(NAME, CFUNC)  \
  void THVector_(NAME)(scalar_t *y, const scalar_t *x, const ptrdiff_t n) \
  { \
    ptrdiff_t i = 0;  \
    for(; i<n-4; i+=4)  \
    { \
      y[i] = CFUNC(x[i]); \
      y[i+1] = CFUNC(x[i+1]); \
      y[i+2] = CFUNC(x[i+2]); \
      y[i+3] = CFUNC(x[i+3]); \
    } \
    for(; i < n; i++) \
      y[i] = CFUNC(x[i]); \
  } \

#define VECTOR_IMPLEMENT_FUNCTION_VALUE(NAME, CFUNC)  \
  void THVector_(NAME)(scalar_t *y, const scalar_t *x, const scalar_t c, const ptrdiff_t n) \
  { \
    ptrdiff_t i = 0;  \
    for(; i<n-4; i+=4)  \
    { \
      y[i] = CFUNC(x[i], c);  \
      y[i+1] = CFUNC(x[i+1], c);  \
      y[i+2] = CFUNC(x[i+2], c);  \
      y[i+3] = CFUNC(x[i+3], c);  \
    } \
    for(; i < n; i++) \
      y[i] = CFUNC(x[i], c);  \
  } \

/* floating point only now */
#if defined(TH_REAL_IS_FLOAT) || defined(TH_REAL_IS_DOUBLE)

#if defined (TH_REAL_IS_FLOAT)
#define TH_MATH_NAME(fn) fn##f
#else
#define TH_MATH_NAME(fn) fn
#endif

VECTOR_IMPLEMENT_FUNCTION(sigmoid_DEFAULT,TH_MATH_NAME(TH_sigmoid))
VECTOR_IMPLEMENT_FUNCTION(erf,TH_MATH_NAME(erf))
VECTOR_IMPLEMENT_FUNCTION(erfc,TH_MATH_NAME(erfc))
VECTOR_IMPLEMENT_FUNCTION(cos,TH_MATH_NAME(cos))
VECTOR_IMPLEMENT_FUNCTION(acos,TH_MATH_NAME(acos))
VECTOR_IMPLEMENT_FUNCTION(cosh,TH_MATH_NAME(cosh))
VECTOR_IMPLEMENT_FUNCTION(tan,TH_MATH_NAME(tan))
VECTOR_IMPLEMENT_FUNCTION(atan,TH_MATH_NAME(atan))
VECTOR_IMPLEMENT_FUNCTION(tanh,TH_MATH_NAME(tanh))
VECTOR_IMPLEMENT_FUNCTION_VALUE(pow,TH_MATH_NAME(pow))
VECTOR_IMPLEMENT_FUNCTION(cinv, TH_MATH_NAME(1.0) / )

#undef TH_MATH_NAME
#endif /* floating point only part */

VECTOR_IMPLEMENT_FUNCTION(neg,-)

#endif /* non bool only part */

#endif
