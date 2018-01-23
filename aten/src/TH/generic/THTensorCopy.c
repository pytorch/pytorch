#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/THTensorCopy.c"
#else

#ifndef _WIN32
#define PRAGMA(P) _Pragma(#P)
#else
#define PRAGMA(P) __pragma(P)
#endif

#ifdef _OPENMP
#define TH_OMP_OVERHEAD_THRESHOLD_COPY 20000
#include <omp.h>
#endif

int THTensor_(copyTransposeValid)(THTensor *tensor, THTensor *src) {
  const int MIN_SZ = 60 * 60;
  return THTensor_(isContiguous)(tensor) &&
         THTensor_(nDimension)(src) == 2 &&
         THTensor_(stride)(src, 0) == 1 &&
         THTensor_(stride)(src, 1) == THTensor_(size)(src, 0) &&
         THTensor_(nElement)(tensor) >= MIN_SZ;
}

// special case copy where tensor is contiguous and src is a transposed matrix
// This can be generalized to most copies, but it's tricker
void THTensor_(copyTranspose)(THTensor *tensor, THTensor *src) {
  #define MIN(x, y) (((x) < (y)) ? (x) : (y))
  #define MAX(x, y) (((x) > (y)) ? (x) : (y))

#ifdef TH_REAL_IS_BYTE
  const int BLOCK_SZ = 120;
#else
  const int BLOCK_SZ = 60;
#endif

  THTensor *buf = THTensor_(newWithSize2d)(BLOCK_SZ, BLOCK_SZ);
  real *sp = THTensor_(data)(src);
  real *rp = THTensor_(data)(tensor);
  real *bp = THTensor_(data)(buf);


  int64_t NR = THTensor_(size)(src, 0);
  int64_t NC = THTensor_(size)(src, 1);
  for (int64_t R = 0; R < NR; R += BLOCK_SZ) {
    for (int64_t C = 0; C < NC; C += BLOCK_SZ) {
      real *spo = sp + R + C * NR;
      real *rpo = rp + C + R * NC;

      int nr = MIN(NR - R, BLOCK_SZ);
      int nc = MIN(NC - C, BLOCK_SZ);

      // 1. copy columns from src to buf
      for (int c = 0; c < nc; c++) {
        memcpy(bp + c * BLOCK_SZ, spo + c * NR, nr * sizeof(real));
      }

      // 2. transpose buf in place
      int rc_max = MAX(nr, nc);
      int rc_min = MIN(nr, nc);
      for (int r = 0; r < rc_max; r++) {
        int end = MIN(r, rc_min);
        for (int c = 0; c < end; c++) {
          real tmp = bp[r + BLOCK_SZ * c];
          bp[r + BLOCK_SZ * c] = bp[r * BLOCK_SZ + c];
          bp[r * BLOCK_SZ + c] = tmp;
        }
      }

      // 3. copy rows from buf to dst
      for (int r = 0; r < nr; r++) {
        memcpy(rpo + r * NC, bp + r * BLOCK_SZ, nc * sizeof(real));
      }
    }
  }
  THTensor_(free)(buf);
  #undef MIN
  #undef MAX
}

void THTensor_(copy)(THTensor *tensor, THTensor *src)
{
  if (tensor == src) return;
  ptrdiff_t tensorSize = THTensor_(nElement)(tensor);
  ptrdiff_t srcSize = THTensor_(nElement)(src);
  int tensorContig = THTensor_(isContiguous)(tensor);
  int srcContig = THTensor_(isContiguous)(src);

  int serial_path = 0;
#ifdef _OPENMP
  int inOMP = omp_in_parallel();
#endif
  if (tensorSize == srcSize) {
    if ( tensorContig && srcContig) {
      real *sp = THTensor_(data)(src);
      real *rp = THTensor_(data)(tensor);
#ifndef TH_REAL_IS_HALF
#ifdef _OPENMP
      #pragma omp parallel if ( (tensorSize > TH_OMP_OVERHEAD_THRESHOLD_COPY) && (!inOMP) )
      {
        size_t num_threads = omp_get_num_threads();
        size_t tid = omp_get_thread_num();
        ptrdiff_t offset = tid * (tensorSize / num_threads);
        ptrdiff_t end = (tid == num_threads - 1) ? tensorSize : offset + tensorSize / num_threads;
        ptrdiff_t len = end - offset;
        real *tensorData = rp + offset;
        real *srcData = sp + offset;
        THVector_(copy)(tensorData, srcData, len);
      }
#else
        THVector_(copy)(rp, sp, srcSize);
#endif

#else

#ifdef _OPENMP
      if ((srcSize > TH_OMP_OVERHEAD_THRESHOLD_COPY) && (!inOMP)) {
        ptrdiff_t i;
        #pragma omp parallel for private (i)
        for(i=0; i<srcSize; i++){
          rp[i] = sp[i];
        }
      } else {
        memcpy(rp, sp, srcSize * sizeof(real));
      }
#else
      memcpy(rp, sp, srcSize * sizeof(real));
#endif

#endif

#ifndef TH_REAL_IS_HALF
    } else if (THTensor_(copyTransposeValid)(tensor, src)) {
      THTensor_(copyTranspose)(tensor, src);
#endif
    } else {
#ifdef _OPENMP
      if (inOMP) {
        serial_path = 1;
      } else {
        TH_TENSOR_APPLY2_OMP(srcSize, tensorContig, srcContig, real, tensor, real, src, *tensor_data = *src_data;)
      }
#else
      serial_path = 1;
#endif
    }
  } else {
    serial_path = 1;
  }

  if (serial_path) {
    TH_TENSOR_APPLY2(real, tensor, real, src, *tensor_data = *src_data;)
  }
}

#define IMPLEMENT_THTensor_COPY(TYPENAMESRC, TYPE_SRC) \
void THTensor_(copy##TYPENAMESRC)(THTensor *tensor, TH##TYPENAMESRC##Tensor *src) \
{ \
  TH_TENSOR_APPLY2(real, tensor, TYPE_SRC, src, *tensor_data = (real)(*src_data);) \
}

#define IMPLEMENT_THTensor_COPY_TO_HALF(TYPENAMESRC, TYPE_SRC) \
void THTensor_(copy##TYPENAMESRC)(THTensor *tensor, TH##TYPENAMESRC##Tensor *src) \
{ \
 TH_TENSOR_APPLY2(real, tensor, TYPE_SRC, src, *tensor_data = TH_float2half((float)*src_data);) \
}

#define IMPLEMENT_THTensor_COPY_FROM_HALF(TYPENAMESRC, TYPE_SRC) \
void THTensor_(copy##TYPENAMESRC)(THTensor *tensor, TH##TYPENAMESRC##Tensor *src) \
{ \
 TH_TENSOR_APPLY2(real, tensor, TYPE_SRC, src, *tensor_data = (real)TH_half2float(*src_data);) \
}

#define IMPLEMENT_THTensor_COPY_TO_FROM_HALF(TYPENAMESRC, TYPE_SRC) \
void THTensor_(copy##TYPENAMESRC)(THTensor *tensor, TH##TYPENAMESRC##Tensor *src) \
{ \
 TH_TENSOR_APPLY2(real, tensor, TYPE_SRC, src, *tensor_data = *src_data;) \
}

#ifndef TH_REAL_IS_HALF
IMPLEMENT_THTensor_COPY(Byte, uint8_t)
IMPLEMENT_THTensor_COPY(Char, int8_t)
IMPLEMENT_THTensor_COPY(Short, int16_t)
IMPLEMENT_THTensor_COPY(Int, int32_t)
IMPLEMENT_THTensor_COPY(Long, int64_t)
IMPLEMENT_THTensor_COPY(Float, float)
IMPLEMENT_THTensor_COPY(Double, double)
IMPLEMENT_THTensor_COPY_FROM_HALF(Half, THHalf)
#else
/* only allow pass-through for Half */
IMPLEMENT_THTensor_COPY_TO_FROM_HALF(Half, THHalf)
IMPLEMENT_THTensor_COPY_TO_HALF(Byte, uint8_t)
IMPLEMENT_THTensor_COPY_TO_HALF(Char, int8_t)
IMPLEMENT_THTensor_COPY_TO_HALF(Short, int16_t)
IMPLEMENT_THTensor_COPY_TO_HALF(Int, int32_t)
IMPLEMENT_THTensor_COPY_TO_HALF(Long, int64_t)
IMPLEMENT_THTensor_COPY_TO_HALF(Float, float)
IMPLEMENT_THTensor_COPY_TO_HALF(Double, double)

#endif /* REAL_IS_HALF */

#endif
