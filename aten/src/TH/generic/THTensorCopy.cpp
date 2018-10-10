#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/THTensorCopy.cpp"
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
         !src->is_empty() &&
         THTensor_(nDimensionLegacyNoScalars)(src) == 2 &&
         THTensor_(stride)(src, 0) == 1 &&
         THTensor_(stride)(src, 1) == THTensor_(size)(src, 0) &&
         THTensor_(nElement)(tensor) >= MIN_SZ;
}

// special case copy where tensor is contiguous and src is a transposed matrix
// This can be generalized to most copies, but it's tricker
void THTensor_(copyTranspose)(THTensor *tensor, THTensor *src) {

#ifdef TH_REAL_IS_BYTE
  const int64_t BLOCK_SZ = 120;
#else
  const int64_t BLOCK_SZ = 60;
#endif

  THTensor *buf = THTensor_(newWithSize2d)(BLOCK_SZ, BLOCK_SZ);
  scalar_t *sp = src->data<scalar_t>();
  scalar_t *rp = tensor->data<scalar_t>();
  scalar_t *bp = buf->data<scalar_t>();


  int64_t NR = THTensor_(size)(src, 0);
  int64_t NC = THTensor_(size)(src, 1);
  for (int64_t R = 0; R < NR; R += BLOCK_SZ) {
    for (int64_t C = 0; C < NC; C += BLOCK_SZ) {
      scalar_t *spo = sp + R + C * NR;
      scalar_t *rpo = rp + C + R * NC;

      int nr = std::min(NR - R, BLOCK_SZ);
      int nc = std::min(NC - C, BLOCK_SZ);

      // 1. copy columns from src to buf
      for (int c = 0; c < nc; c++) {
        memcpy(bp + c * BLOCK_SZ, spo + c * NR, nr * sizeof(scalar_t));
      }

      // 2. transpose buf in place
      int rc_max = std::max(nr, nc);
      int rc_min = std::min(nr, nc);
      for (int r = 0; r < rc_max; r++) {
        int end = std::min(r, rc_min);
        for (int c = 0; c < end; c++) {
          scalar_t tmp = bp[r + BLOCK_SZ * c];
          bp[r + BLOCK_SZ * c] = bp[r * BLOCK_SZ + c];
          bp[r * BLOCK_SZ + c] = tmp;
        }
      }

      // 3. copy rows from buf to dst
      for (int r = 0; r < nr; r++) {
        memcpy(rpo + r * NC, bp + r * BLOCK_SZ, nc * sizeof(scalar_t));
      }
    }
  }
  c10::raw::intrusive_ptr::decref(buf);
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
      scalar_t *sp = src->data<scalar_t>();
      scalar_t *rp = tensor->data<scalar_t>();
#ifndef TH_REAL_IS_HALF
#ifdef _OPENMP
      #pragma omp parallel if ( (tensorSize > TH_OMP_OVERHEAD_THRESHOLD_COPY) && (!inOMP) )
      {
        size_t num_threads = omp_get_num_threads();
        size_t tid = omp_get_thread_num();
        ptrdiff_t offset = tid * (tensorSize / num_threads);
        ptrdiff_t end = (tid == num_threads - 1) ? tensorSize : offset + tensorSize / num_threads;
        ptrdiff_t len = end - offset;
        scalar_t *tensorData = rp + offset;
        scalar_t *srcData = sp + offset;
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
        memcpy(rp, sp, srcSize * sizeof(scalar_t));
      }
#else
      memcpy(rp, sp, srcSize * sizeof(scalar_t));
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
        TH_TENSOR_APPLY2_OMP(srcSize, tensorContig, srcContig, scalar_t, tensor, scalar_t, src, *tensor_data = *src_data;, TH_OMP_OVERHEAD_THRESHOLD_COPY)
      }
#else
      serial_path = 1;
#endif
    }
  } else {
    serial_path = 1;
  }

  if (serial_path) {
    TH_TENSOR_APPLY2(scalar_t, tensor, scalar_t, src, *tensor_data = *src_data;)
  }
}

#ifndef INTER_COPY_TYPE_DEFINED
#define INTER_COPY_TYPE_DEFINED

// C and C++ have a lovely set of implicit conversion rules, where casting
// signed integral values to unsigned integral values is always valid
// (it basically treats the value as if using modulo arithmetic), however
// converting negative floating point values to unsigned integral types
// is UB! This means that: (double)-1 -> (int64_t)-1 -> (uint8_t)255 is
// guaranteed to look like this, but we have (double)-1 -> (uint8_t)<ANYTHING>
// because it's UB. This also makes UBSan really angry.
//
// I think those rules are stupid and we really shouldn't conform to them.
// The structs below ensure that for all unsigned types we use (currently
// only uint8_t), we will do an intermediate convertion via int64_t,
// to ensure that any negative values are wrapped around correctly.
//
// Note that conversions from doubles to signed integral types that can't
// represent a particular value after truncating the fracitonal part are UB as well,
// but fixing them is not as simple as adding an int64_t intermediate, beacuse the
// int64_t -> <smaller signed type> conversion is UB for those large values anyway.
// I guess in that case we just have to live with that, but it's definitely less
// surprising than the thing above.
//
// For the curious:
//   https://en.cppreference.com/w/cpp/language/implicit_conversion
//   The relevant paragraph is "Floating–integral conversions".
template<typename T>
struct inter_copy_type {
  using type = T;
};

template<>
struct inter_copy_type<uint8_t> {
  using type = int64_t;
};

template<typename T>
using inter_copy_type_t = typename inter_copy_type<T>::type;

#endif

#define IMPLEMENT_THTensor_COPY(TYPENAMESRC, TYPE_SRC) \
void THTensor_(copy##TYPENAMESRC)(THTensor *tensor, TH##TYPENAMESRC##Tensor *src) \
{ \
  TH_TENSOR_APPLY2(scalar_t, tensor, TYPE_SRC, src, \
                   *tensor_data = static_cast<scalar_t>( \
                       static_cast<inter_copy_type_t<scalar_t>>(*src_data));) \
}

IMPLEMENT_THTensor_COPY(Byte, uint8_t)
IMPLEMENT_THTensor_COPY(Char, int8_t)
IMPLEMENT_THTensor_COPY(Short, int16_t)
IMPLEMENT_THTensor_COPY(Int, int32_t)
IMPLEMENT_THTensor_COPY(Long, int64_t)
IMPLEMENT_THTensor_COPY(Float, float)
IMPLEMENT_THTensor_COPY(Double, double)
IMPLEMENT_THTensor_COPY(Half, at::Half)

#endif
