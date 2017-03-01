static void THFloatVector_fill_NEON(float *x, const float c, const ptrdiff_t n) {
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

static void THFloatVector_cmul_NEON(float *z, const float *x, const float* y, const ptrdiff_t n) {
  long i = 0;

  for(; i < n-4; i += 4)
  {
    z[i] = x[i] * y[i];
    z[i+1] = x[i+1] * y[i+1];
    z[i+2] = x[i+2] * y[i+2];
    z[i+3] = x[i+3] * y[i+3];
  }

  for(; i < n; i++)
    z[i] = x[i] * y[i];
}

static void THFloatVector_muls_NEON(float *y, const float *x, const float c, const ptrdiff_t n) {
  long i = 0;

  for(; i < n-4; i += 4)
  {
    y[i] = x[i] * c;
    y[i+1] = x[i+1] * c;
    y[i+2] = x[i+2] * c;
    y[i+3] = x[i+3] * c;
  }

  for(; i < n; i++)
    y[i] = x[i] * c;
}

static void THFloatVector_cadd_NEON(float *z, const float *x, const float *y, const float c, const ptrdiff_t n) {
  long i = 0;

  for(;i < n-4; i += 4)
  {
    z[i] = x[i] + c * y[i];
    z[i+1] = x[i+1] + c * y[i+1];
    z[i+2] = x[i+2] + c * y[i+2];
    z[i+3] = x[i+3] + c * y[i+3];
  }

  for(; i < n; i++)
    z[i] = x[i] + c * y[i];
}

static void THFloatVector_adds_NEON(float *y, const float *x, const float c, const ptrdiff_t n) {
  long i = 0;

  for(;i < n-4; i += 4)
  {
    y[i] = x[i] + c;
    y[i+1] = x[i+1] + c;
    y[i+2] = x[i+2] + c;
    y[i+3] = x[i+3] + c;
  }

  for(; i < n; i++)
    y[i] = x[i] + c;
}

static void THFloatVector_cdiv_NEON(float *z, const float *x, const float *y, const ptrdiff_t n) {
  long i = 0;

  for(;i < n-4; i += 4)
  {
    z[i] = x[i] / y[i];
    z[i+1] = x[i+1] / y[i+1];
    z[i+2] = x[i+2] / y[i+2];
    z[i+3] = x[i+3] / y[i+3];
  }

  for(; i < n; i++)
    z[i] = x[i] / y[i];
}

static void THFloatVector_divs_NEON(float *y, const float *x, const float c, const ptrdiff_t n) {
  long i = 0;

  for(;i < n-4; i += 4)
  {
    y[i] = x[i] / c;
    y[i+1] = x[i+1] / c;
    y[i+2] = x[i+2] / c;
    y[i+3] = x[i+3] / c;
  }

  for(; i < n; i++)
    y[i] = x[i] / c;
}
