#ifndef THS_GENERIC_FILE
#define THS_GENERIC_FILE "generic/THSTensorMath.c"
#else

/* Check header file for some documentation */
#define ROW_PTR2(t, r) (THTensor_(data)(t) + (r) * (t)->stride[0])
#define COL_PTR2(t, c) (THTensor_(data)(t) + (c) * (t)->stride[1])

/*
static real THTensor_(get1d)(THTensor *t, long x0) {
  return THStorage_(get)(t->storage, t->storageOffset + x0*t->stride[0]);
}

static void THTensor_(set1d)(THTensor *t, long x0, real value) {
  THStorage_(set)(t->storage, t->storageOffset + x0*t->stride[0], value);
}

static real THTensor_(get2d)(const THTensor *t, long x0, long x1) {
  return THStorage_(get)(t->storage, t->storageOffset +
      x0*t->stride[0] + x1*t->stride[1]);
}
*/

void THSTensor_(addSmm)(THTensor *r_,
    real beta, THTensor *t, real alpha, THSTensor *sparse, THTensor *dense) {
  long h, i, j, hp0, hp1;
  long dim_i, dim_j, dim_k; // ixj * jxk = ixk
  long nnz;
  THLongTensor *csr, *indicies;
  THTensor *values;
  // batch = i
  // outdim = k
  // indim = j

  if( (sparse->nDimension != 2) || (dense->nDimension != 2))
    THError("matrices expected, got %dD, %dD tensors",
        sparse->nDimension, dense->nDimension);

  if (!sparse->contiguous) THSTensor_(contiguous)(sparse);

  dim_i = THSTensor_(size)(sparse, 0);
  dim_j = THSTensor_(size)(sparse, 1);
  dim_k = THTensor_(size)(dense, 1);

  nnz = THSTensor_(nnz)(sparse);
  indicies = THSTensor_(indicies)(sparse);
  values = THSTensor_(values)(sparse);

  csr = THLongTensor_newWithSize1d(dim_i + 1);
  THLongTensor_zero(csr);

  // Convert the sparse matrix to CSR format
#pragma omp parallel for private(i, h, hp0, hp1) schedule(static) if (nnz > 10000)
  for (i=0; i<nnz; i++) {
    hp0 = THLongTensor_get2d(indicies, 0, i);
    hp1 = (i+1 == nnz) ?  dim_i : THLongTensor_get2d(indicies, 0, i+1);
    if (hp0 != hp1) for (h = hp0; h < hp1; h++) {
      THLongTensor_set1d(csr, h+1, i+1);
    }
  }

  // r_ = alpha * sparse * dense
  THTensor_(resize2d)(r_, dim_i, dim_k);
  THTensor_(zero)(r_);
#pragma omp parallel for private(h, i) schedule(static) if (nnz > 10000)
  for (h = 0; h < dim_i; h++) {
    long i_start = THLongTensor_get1d(csr, h);
    long i_end = THLongTensor_get1d(csr, h+1);
    for (i = i_start; i < i_end; i++) {
      real val = THTensor_(get1d)(values, i);
      if (val == 0) {
        continue;
      }

      long offset = THLongTensor_get2d(indicies, 1, i);
      if (offset >= 0 && offset < dim_j) {
        THBlas_(axpy)(dim_k,
            val * alpha,
            ROW_PTR2(dense, offset), dense->stride[1],
            ROW_PTR2(r_, h), r_->stride[1]);
      } else {
        THError("index out of bound. addSmm: %d not between 1 and %d",
            offset, dim_j);
      }
    }
  }

  // r_ = beta * t + r_
  THTensor_(cadd)(r_, r_, beta, t);
  THFree(csr);
  THFree(indicies);
  THFree(values);
}

void THSTensor_(SgemSm)(THSTensor *r_, THTensor *mat1, THTensor *mat2) {
}

#undef ROW_PTR2
#undef COL_PTR2

#endif

