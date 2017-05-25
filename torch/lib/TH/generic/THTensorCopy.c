#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/THTensorCopy.c"
#else

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

  long NR = THTensor_(size)(src, 0);
  long NC = THTensor_(size)(src, 1);
  for (long R = 0; R < NR; R += BLOCK_SZ) {
    for (long C = 0; C < NC; C += BLOCK_SZ) {
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
  if (THTensor_(isContiguous)(tensor) && THTensor_(isContiguous)(src) && THTensor_(nElement)(tensor) == THTensor_(nElement)(src)) {
    real *sp = THTensor_(data)(src);
    real *rp = THTensor_(data)(tensor);
    ptrdiff_t sz = THTensor_(nElement)(tensor);
#ifndef TH_REAL_IS_HALF
    THVector_(copy)(rp, sp, sz);
#else
    memcpy(rp, sp, sz * sizeof(real));
#endif
#ifndef TH_REAL_IS_HALF
  } else if (THTensor_(copyTransposeValid)(tensor, src)) {
    THTensor_(copyTranspose)(tensor, src);
#endif
  } else {
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
IMPLEMENT_THTensor_COPY(Byte, unsigned char)
IMPLEMENT_THTensor_COPY(Char, char)
IMPLEMENT_THTensor_COPY(Short, short)
IMPLEMENT_THTensor_COPY(Int, int)
IMPLEMENT_THTensor_COPY(Long, long)
IMPLEMENT_THTensor_COPY(Float, float)
IMPLEMENT_THTensor_COPY(Double, double)
IMPLEMENT_THTensor_COPY_FROM_HALF(Half, THHalf)
#else
/* only allow pass-through for Half */
IMPLEMENT_THTensor_COPY_TO_FROM_HALF(Half, THHalf)
IMPLEMENT_THTensor_COPY_TO_HALF(Byte, unsigned char)
IMPLEMENT_THTensor_COPY_TO_HALF(Char, char)
IMPLEMENT_THTensor_COPY_TO_HALF(Short, short)
IMPLEMENT_THTensor_COPY_TO_HALF(Int, int)
IMPLEMENT_THTensor_COPY_TO_HALF(Long, long)
IMPLEMENT_THTensor_COPY_TO_HALF(Float, float)
IMPLEMENT_THTensor_COPY_TO_HALF(Double, double)

#endif /* REAL_IS_HALF */

#endif
