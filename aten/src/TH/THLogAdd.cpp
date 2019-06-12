#include "THLogAdd.h"

#include <float.h>

#ifdef USE_DOUBLE
#define MINUS_LOG_THRESHOLD -39.14
#else
#define MINUS_LOG_THRESHOLD -18.42
#endif

const double THLog2Pi=1.83787706640934548355;
const double THLogZero=-DBL_MAX;
const double THLogOne=0;

double THLogAdd(double log_a, double log_b)
{
  double minusdif;

  if (log_a < log_b)
  {
    double tmp = log_a;
    log_a = log_b;
    log_b = tmp;
  }

  minusdif = log_b - log_a;
#ifdef DEBUG
  if (isnan(minusdif))
    THError("THLogAdd: minusdif (%f) log_b (%f) or log_a (%f) is nan", minusdif, log_b, log_a);
#endif
  if (minusdif < MINUS_LOG_THRESHOLD)
    return log_a;
  else
    return log_a + log1p(exp(minusdif));
}

double THLogSub(double log_a, double log_b)
{
  double minusdif;

  if (log_a < log_b)
    THError("LogSub: log_a (%f) should be greater than log_b (%f)", log_a, log_b);

  minusdif = log_b - log_a;
#ifdef DEBUG
  if (isnan(minusdif))
    THError("LogSub: minusdif (%f) log_b (%f) or log_a (%f) is nan", minusdif, log_b, log_a);
#endif
  if (log_a == log_b)
    return THLogZero;
  else if (minusdif < MINUS_LOG_THRESHOLD)
    return log_a;
  else
    return log_a + log1p(-exp(minusdif));
}

/* Credits to Leon Bottou */
double THExpMinusApprox(const double x)
{
#define EXACT_EXPONENTIAL 0
#if EXACT_EXPONENTIAL
  return exp(-x);
#else
  /* fast approximation of exp(-x) for x positive */
# define A0   (1.0)
# define A1   (0.125)
# define A2   (0.0078125)
# define A3   (0.00032552083)
# define A4   (1.0172526e-5)
  if (x < 13.0)
  {
/*    assert(x>=0); */
    double y;
    y = A0+x*(A1+x*(A2+x*(A3+x*A4)));
    y *= y;
    y *= y;
    y *= y;
    y = 1/y;
    return y;
  }
  return 0;
# undef A0
# undef A1
# undef A2
# undef A3
# undef A4
#endif
}
