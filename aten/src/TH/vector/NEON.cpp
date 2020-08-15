static void THFloatVector_fill_NEON(float *x, const float c, const ptrdiff_t n) {
  int64_t i = 0;

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

static void THFloatVector_muls_NEON(float *y, const float *x, const float c, const ptrdiff_t n) {
  int64_t i = 0;

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
