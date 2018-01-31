#ifndef THZ_GENERIC_FILE
#define THZ_GENERIC_FILE "generic/THZTensorCopy.c"
#else

int THZTensor_(copyTransposeValid)(THZTensor *tensor, THZTensor *src) {
  const int MIN_SZ = 60 * 60;
  return THZTensor_(isContiguous)(tensor) &&
         THZTensor_(nDimension)(src) == 2 &&
         THZTensor_(stride)(src, 0) == 1 &&
         THZTensor_(stride)(src, 1) == THZTensor_(size)(src, 0) &&
         THZTensor_(nElement)(tensor) >= MIN_SZ;
}

// special case copy where tensor is contiguous and src is a transposed matrix
// This can be generalized to most copies, but it's tricker
void THZTensor_(copyTranspose)(THZTensor *tensor, THZTensor *src) {
  #define MIN(x, y) (((x) < (y)) ? (x) : (y))
  #define MAX(x, y) (((x) > (y)) ? (x) : (y))

  const int BLOCK_SZ = 60;

  THZTensor *buf = THZTensor_(newWithSize2d)(BLOCK_SZ, BLOCK_SZ);
  ntype *sp = THZTensor_(data)(src);
  ntype *rp = THZTensor_(data)(tensor);
  ntype *bp = THZTensor_(data)(buf);

  int64_t NR = THZTensor_(size)(src, 0);
  int64_t NC = THZTensor_(size)(src, 1);
  for (int64_t R = 0; R < NR; R += BLOCK_SZ) {
    for (int64_t C = 0; C < NC; C += BLOCK_SZ) {
      ntype *spo = sp + R + C * NR;
      ntype *rpo = rp + C + R * NC;

      int nr = MIN(NR - R, BLOCK_SZ);
      int nc = MIN(NC - C, BLOCK_SZ);

      // 1. copy columns from src to buf
      for (int c = 0; c < nc; c++) {
        memcpy(bp + c * BLOCK_SZ, spo + c * NR, nr * sizeof(ntype));
      }

      // 2. transpose buf in place
      int rc_max = MAX(nr, nc);
      int rc_min = MIN(nr, nc);
      for (int r = 0; r < rc_max; r++) {
        int end = MIN(r, rc_min);
        for (int c = 0; c < end; c++) {
          ntype tmp = bp[r + BLOCK_SZ * c];
          bp[r + BLOCK_SZ * c] = bp[r * BLOCK_SZ + c];
          bp[r * BLOCK_SZ + c] = tmp;
        }
      }

      // 3. copy rows from buf to dst
      for (int r = 0; r < nr; r++) {
        memcpy(rpo + r * NC, bp + r * BLOCK_SZ, nc * sizeof(ntype));
      }
    }
  }
  THZTensor_(free)(buf);
  #undef MIN
  #undef MAX
}

void THZTensor_(copy)(THZTensor *tensor, THZTensor *src)
{
  if (tensor == src) return;
  if (THZTensor_(isContiguous)(tensor) && THZTensor_(isContiguous)(src) && THZTensor_(nElement)(tensor) == THZTensor_(nElement)(src)) {
    ntype *sp = THZTensor_(data)(src);
    ntype *rp = THZTensor_(data)(tensor);
    ptrdiff_t sz = THZTensor_(nElement)(tensor);
#ifndef TH_NTYPE_IS_HALF
    THZVector_(copy)(rp, sp, sz);
#else
    memcpy(rp, sp, sz * sizeof(ntype));
#endif
  } else {
    TH_TENSOR_APPLY2(ntype, tensor, ntype, src, *tensor_data = *src_data;)
  }
}

#define IMPLEMENT_THZTensor_COPY(TYPENAMESRC, TYPE_SRC) \
void THZTensor_(copy##TYPENAMESRC)(THZTensor *tensor, TH##TYPENAMESRC##Tensor *src) \
{ \
  TH_TENSOR_APPLY2(ntype, tensor, TYPE_SRC, src, *tensor_data = (ntype)(*src_data);) \
}

#define IMPLEMENT_THZTensor_COPY_TO_HALF(TYPENAMESRC, TYPE_SRC) \
void THZTensor_(copy##TYPENAMESRC)(THZTensor *tensor, TH##TYPENAMESRC##Tensor *src) \
{ \
 TH_TENSOR_APPLY2(ntype, tensor, TYPE_SRC, src, *tensor_data = TH_float2half((float)*src_data);) \
}

#define IMPLEMENT_THZTensor_COPY_FROM_HALF(TYPENAMESRC, TYPE_SRC) \
void THZTensor_(copy##TYPENAMESRC)(THZTensor *tensor, TH##TYPENAMESRC##Tensor *src) \
{ \
 TH_TENSOR_APPLY2(ntype, tensor, TYPE_SRC, src, *tensor_data = (ntype)TH_half2float(*src_data);) \
}

#define IMPLEMENT_THZTensor_COPY_TO_FROM_HALF(TYPENAMESRC, TYPE_SRC) \
void THZTensor_(copy##TYPENAMESRC)(THZTensor *tensor, TH##TYPENAMESRC##Tensor *src) \
{ \
 TH_TENSOR_APPLY2(ntype, tensor, TYPE_SRC, src, *tensor_data = *src_data;) \
}

#ifndef TH_NTYPE_IS_HALF
IMPLEMENT_THZTensor_COPY(Byte, uint8_t)
IMPLEMENT_THZTensor_COPY(Char, int8_t)
IMPLEMENT_THZTensor_COPY(Short, int16_t)
IMPLEMENT_THZTensor_COPY(Int, int32_t)
IMPLEMENT_THZTensor_COPY(Long, int64_t)
IMPLEMENT_THZTensor_COPY(Float, float)
IMPLEMENT_THZTensor_COPY(Double, double)
IMPLEMENT_THZTensor_COPY(ZFloat, float _Complex)
IMPLEMENT_THZTensor_COPY(ZDouble, double _Complex)
IMPLEMENT_THZTensor_COPY_FROM_HALF(Half, THHalf)
#else
/* only allow pass-through for Half */
IMPLEMENT_THZTensor_COPY_TO_FROM_HALF(Half, THHalf)
IMPLEMENT_THZTensor_COPY_TO_HALF(Byte, uint8_t)
IMPLEMENT_THZTensor_COPY_TO_HALF(Char, int8_t)
IMPLEMENT_THZTensor_COPY_TO_HALF(Short, int16_t)
IMPLEMENT_THZTensor_COPY_TO_HALF(Int, int32_t)
IMPLEMENT_THZTensor_COPY_TO_HALF(Long, int64_t)
IMPLEMENT_THZTensor_COPY_TO_HALF(Float, float)
IMPLEMENT_THZTensor_COPY_TO_HALF(Double, double)
IMPLEMENT_THZTensor_COPY_TO_HALF(ZFloat, float _Complex)
IMPLEMENT_THZTensor_COPY_TO_HALF(ZDouble, double _Complex)

#endif /* NTYPE_IS_HALF */

#endif
