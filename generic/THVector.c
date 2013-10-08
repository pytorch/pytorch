#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/THVector.c"
#else

static TH_INLINE void THVector_(fill)(real *x, const real c, const long n) {
  long i = 0;

  for(; i < n-4; i += 4)
  {
    x[i] = c;
    x[i+1] = c;
    x[i+2] = c;
    x[i+3] = c;
  }

  for(; i < n; i++)
    x[i] = c;
}

static TH_INLINE void THVector_(add)(real *y, const real *x, const real c, const long n)
{
  long i = 0;

  for(;i < n-4; i += 4)
  {
    y[i] += c * x[i];
    y[i+1] += c * x[i+1];
    y[i+2] += c * x[i+2];
    y[i+3] += c * x[i+3];
  }

  for(; i < n; i++)
    y[i] += c * x[i];
}

static TH_INLINE void THVector_(diff)(real *z, const real *x, const real *y, const long n)
{
  long i = 0;

  for(; i < n-4; i += 4)
  {
    z[i] = x[i] - y[i];
    z[i+1] = x[i+1] - y[i+1];
    z[i+2] = x[i+2] - y[i+2];
    z[i+3] = x[i+3] - y[i+3];
  }

  for(; i < n; i++)
    z[i] = x[i] - y[i];
}

static TH_INLINE void THVector_(scale)(real *y, const real c, const long n)
{
  long i = 0;

  for(; i < n-4; i +=4)
  {
    y[i] *= c;
    y[i+1] *= c;
    y[i+2] *= c;
    y[i+3] *= c;
  }

  for(; i < n; i++)
    y[i] *= c;
}

static TH_INLINE void THVector_(mul)(real *y, const real *x, const long n)
{
  long i = 0;

  for(; i < n-4; i += 4)
  {
    y[i] *= x[i];
    y[i+1] *= x[i+1];
    y[i+2] *= x[i+2];
    y[i+3] *= x[i+3];
  }

  for(; i < n; i++)
    y[i] *= x[i];
}

#endif
