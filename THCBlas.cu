#include "THCBlas.h"
#include "THCGeneral.h"

static cublasHandle_t* handles;
static cublasHandle_t* current_handle;
static int n_devices;

void THCudaBlas_init(THCState *state, int devices, int device)
{
  handles = (cublasHandle_t *)malloc(devices * sizeof(cublasHandle_t));
  for (int i = 0; i < devices; i++) {
    // Create handle on each device:
    cudaSetDevice(i);
    cublasCreate(&handles[i]);
  }

  // Set current handle:
  current_handle = &handles[device];
  n_devices = devices;

  // Restore device:
  cudaSetDevice(device);
}

void THCudaBlas_shutdown(THCState *state)
{
  for (int i = 0; i < n_devices; i++) {
    cublasDestroy(handles[i]);
  }
  free(handles);
}

void THCudaBlas_setHandle(THCState *state, int device)
{
  current_handle = &handles[device];
}

void THCudaBlas_swap(THCState *state, long n, float *x, long incx, float *y, long incy)
{
  if(n == 1)
  {
    incx = 1;
    incy = 1;
  }

  if( (n <= INT_MAX) && (incx <= INT_MAX) && (incy <= INT_MAX) )
  {
    int i_n = (int)n;
    int i_incx = (int)incx;
    int i_incy = (int)incy;
    THCublasCheck(cublasSswap(*current_handle, i_n, x, i_incx, y, i_incy));
    return;
  }
  THError("Cublas_swap only supports n, incx and"
          " incy upto signed integer limits: %d", INT_MAX);
}

void THCudaBlas_scal(THCState *state, long n, float a, float *x, long incx)
{
  if(n == 1)
    incx = 1;

  if( (n <= INT_MAX) && (incx <= INT_MAX) )
  {
    int i_n = (int)n;
    int i_incx = (int)incx;
    THCublasCheck(cublasSscal(*current_handle, i_n, &a, x, i_incx));
    return;
  }
  THError("Cublas_scal only supports n and incx "
          "upto signed integer limits: %d", INT_MAX);
}

void THCudaBlas_copy(THCState *state, long n, float *x, long incx, float *y, long incy)
{
  if(n == 1)
  {
    incx = 1;
    incy = 1;
  }

  if( (n <= INT_MAX) && (incx <= INT_MAX) && (incy <= INT_MAX) )
  {
    int i_n = (int)n;
    int i_incx = (int)incx;
    int i_incy = (int)incy;
    THCublasCheck(cublasScopy(*current_handle, i_n, x, i_incx, y, i_incy));
    return;
  }

  THError("Cublas_copy only supports n, incx and incy "
          "upto signed integer limits: %d", INT_MAX);
}

void THCudaBlas_axpy(THCState *state, long n, float a, float *x, long incx, float *y, long incy)
{
    if(n == 1)
  {
    incx = 1;
    incy = 1;
  }

  if( (n <= INT_MAX) && (incx <= INT_MAX) && (incy <= INT_MAX) )
  {
    int i_n = (int)n;
    int i_incx = (int)incx;
    int i_incy = (int)incy;
    THCublasCheck(cublasSaxpy(*current_handle, i_n, &a, x, i_incx, y, i_incy));
    return;
  }

  THError("Cublas_axpy only supports n, incx and incy "
          "upto signed integer limits: %d", INT_MAX);
}

float THCudaBlas_dot(THCState *state, long n, float *x, long incx, float *y, long incy)
{
  if(n == 1)
  {
    incx = 1;
    incy = 1;
  }

  if( (n <= INT_MAX) && (incx <= INT_MAX) && (incy <= INT_MAX) )
  {
    int i_n = (int)n;
    int i_incx = (int)incx;
    int i_incy = (int)incy;
    float result;
    THCublasCheck(cublasSdot(*current_handle, i_n, x, i_incx, y, i_incy, &result));
    cudaDeviceSynchronize();
    return result;
  }
  THError("Cublas_dot only supports n, incx and incy "
          "upto signed integer limits: %d", INT_MAX);
  return -1;
}

/* Level 2 */
void THCudaBlas_gemv(THCState *state, char trans, long m, long n, float alpha, float *a, long lda, float *x, long incx, float beta, float *y, long incy)
{
  if(n == 1)
    lda = m;

  cublasOperation_t op;
  if (trans == 't') op = CUBLAS_OP_T;
  else if (trans == 'n') op = CUBLAS_OP_N;
  else if (trans == 'c') op = CUBLAS_OP_C;

  if( (m <= INT_MAX) && (n <= INT_MAX) &&
      (lda > 0) && (lda <= INT_MAX) &&
      (incx > 0) && (incx <= INT_MAX) &&
      (incy > 0) && (incy <= INT_MAX) )
  {
    int i_m = (int)m;
    int i_n = (int)n;
    int i_lda = (int)lda;
    int i_incx = (int)incx;
    int i_incy = (int)incy;

    THCublasCheck(cublasSgemv(*current_handle, op, i_m, i_n, &alpha, a, i_lda, x, i_incx, &beta, y, i_incy));
    return;
  }
  THError("Cublas_gemv only supports m, n, lda, incx, incy"
          "in the range 0 < [val] <= %d", INT_MAX);
}

void THCudaBlas_ger(THCState *state, long m, long n, float alpha, float *x, long incx, float *y, long incy, float *a, long lda)
{
  if(n == 1)
    lda = m;

  if( (m <= INT_MAX) && (n <= INT_MAX) && (lda <= INT_MAX)  && (incx <= INT_MAX) && (incy <= INT_MAX) )
    {
      int i_m = (int)m;
      int i_n = (int)n;
      int i_lda = (int)lda;
      int i_incx = (int)incx;
      int i_incy = (int)incy;

      THCublasCheck(cublasSger(*current_handle, i_m, i_n, &alpha, x, i_incx, y, i_incy, a, i_lda));
      return;
    }
  THError("Cublas_ger only supports m, n, lda, incx, incy"
          "with the bound [val] <= %d", INT_MAX);
}

/* Level 3 */
void THCudaBlas_gemm(THCState *state, char transa, char transb, long m, long n, long k, float alpha, float *a, long lda, float *b, long ldb, float beta, float *c, long ldc)
{
  int transa_ = ((transa == 't') || (transa == 'T'));
  int transb_ = ((transb == 't') || (transb == 'T'));

  if(n == 1)
    ldc = m;

  if(transa_)
  {
    if(m == 1)
      lda = k;
  }
  else
  {
    if(k == 1)
      lda = m;
  }

  if(transb_)
  {
    if(k == 1)
      ldb = n;
  }
  else
  {
    if(n == 1)
      ldb = k;
  }

  cublasOperation_t opa;
  if (transa == 't') opa = CUBLAS_OP_T;
  else if (transa == 'n') opa = CUBLAS_OP_N;
  else if (transa == 'c') opa = CUBLAS_OP_C;
  else THError("transa must be one of: t, n, c");

  cublasOperation_t opb;
  if (transb == 't') opb = CUBLAS_OP_T;
  else if (transb == 'n') opb = CUBLAS_OP_N;
  else if (transb == 'c') opb = CUBLAS_OP_C;
  else THError("transb must be one of: t, n, c");

  if( (m <= INT_MAX) && (n <= INT_MAX) && (k <= INT_MAX) && (lda <= INT_MAX)  && (ldb <= INT_MAX) && (ldc <= INT_MAX) )
  {
    int i_m = (int)m;
    int i_n = (int)n;
    int i_k = (int)k;
    int i_lda = (int)lda;
    int i_ldb = (int)ldb;
    int i_ldc = (int)ldc;

    THCublasCheck(cublasSgemm(*current_handle, opa, opb, i_m, i_n, i_k, &alpha, a, i_lda, b, i_ldb, &beta, c, i_ldc));
    return;
  }
  THError("Cublas_gemm only supports m, n, k, lda, ldb, ldc"
          "with the bound [val] <= %d", INT_MAX);
}
