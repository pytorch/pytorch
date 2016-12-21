#ifndef THS_GENERIC_FILE
#define THS_GENERIC_FILE "generic/THSTensorMath.c"
#else

#define ROW_PTR2(t, r) (THTensor_(data)(t) + (r) * (t)->stride[0])
#define COL_PTR2(t, c) (THTensor_(data)(t) + (c) * (t)->stride[1])

THLongTensor *THSTensor_(toCSR)(long const *indices, long dim, long nnz) {
  long h, i, hp0, hp1;
  THLongTensor *csr = THLongTensor_newWithSize1d(dim + 1);
  THLongTensor_zero(csr);

  // Convert the sparse matrix to CSR format
#pragma omp parallel for private(i, h, hp0, hp1) schedule(static) if (nnz > 10000)
  for (i=0; i<nnz; i++) {
    hp0 = indices[i];
    hp1 = (i+1 == nnz) ?  dim : indices[i+1];
    if (hp0 != hp1) for (h = hp0; h < hp1; h++) {
      THTensor_fastSet1d(csr, h+1, i+1);
    }
  }
  return csr;
}

void THSTensor_(spaddmm)(THTensor *r_,
    real beta, THTensor *t,
    real alpha, THSTensor *sparse, THTensor *dense) {
  long h, i;
  long dim_i, dim_j, dim_k; // ixj * jxk = ixk
  long nnz;
  THLongTensor *csr, *indices;
  THTensor *values;

  THArgCheck(sparse->nDimension == 2, 2,
      "matrices expected, got %dD tensor", sparse->nDimension);
  THArgCheck(dense->nDimension == 2, 2,
      "matrices expected, got %dD tensor", dense->nDimension);

  THSTensor_(contiguous)(sparse);

  dim_i = THSTensor_(size)(sparse, 0);
  dim_j = THSTensor_(size)(sparse, 1);
  dim_k = THTensor_(size)(dense, 1);

  THArgCheck(THTensor_(size)(dense, 0) == dim_j, 3,
      "Expected dim 0 size %d, got %d", dim_j, THTensor_(size)(dense, 0));
  THArgCheck(THTensor_(size)(t, 0) == dim_i, 1,
      "Expected dim 0 size %d, got %d", dim_i, THTensor_(size)(t, 0));
  THArgCheck(THTensor_(size)(t, 1) == dim_k, 1,
      "Expected dim 1 size %d, got %d", dim_k, THTensor_(size)(t, 1));

  nnz     = THSTensor_(nnz)(sparse);
  indices = THSTensor_(indices)(sparse);
  values  = THSTensor_(values)(sparse);

  csr = THSTensor_(toCSR)(THLongTensor_data(indices), dim_i, nnz);

  // r_ = alpha * sparse * dense
  THTensor_(resize2d)(r_, dim_i, dim_k);
  THTensor_(mul)(r_, t, beta);
#pragma omp parallel for private(h, i) schedule(static) if (nnz > 10000)
  for (h = 0; h < dim_i; h++) {
    long i_start = THTensor_fastGet1d(csr, h);
    long i_end = THTensor_fastGet1d(csr, h+1);
    for (i = i_start; i < i_end; i++) {
      real val = THTensor_fastGet1d(values, i);
      long col = THTensor_fastGet2d(indices, 1, i);
      if (col >= 0 && col < dim_j) {
        THBlas_(axpy)(dim_k,
            alpha * val,
            ROW_PTR2(dense, col), dense->stride[1],
            ROW_PTR2(r_, h), r_->stride[1]);
      } else {
        THError("index out of bound. spmm: %d not between 1 and %d",
            col, dim_j);
      }
    }
  }

  THLongTensor_free(csr);
  THLongTensor_free(indices);
  THTensor_(free)(values);
}

void THSTensor_(sspaddmm)(THSTensor *r_,
    real beta, THSTensor *t,
    real alpha, THSTensor *sparse, THTensor *dense) {
  long h, i, p;
  long dim_i, dim_j, dim_k; // ixj * jxk = ixk
  long nnz, r_nnz, t_nnz;
  THLongTensor *csr, *indices, *newi, *narrowi;
  THTensor *values, *newv, *narrowv;

  THArgCheck(sparse->nDimension == 2, 2,
      "matrices expected, got %dD tensor", sparse->nDimension);
  THArgCheck(dense->nDimension == 2, 2,
      "matrices expected, got %dD tensor", dense->nDimension);

  THSTensor_(contiguous)(sparse);

  dim_i = THSTensor_(size)(sparse, 0);
  dim_j = THSTensor_(size)(sparse, 1);
  dim_k = THTensor_(size)(dense, 1);

  THArgCheck(THTensor_(size)(dense, 0) == dim_j, 3,
      "Expected dim 0 size %d, got %d", dim_j, THTensor_(size)(dense, 0));
  THArgCheck(THSTensor_(size)(t, 0) == dim_i, 1,
      "Expected dim 0 size %d, got %d", dim_i, THSTensor_(size)(t, 0));
  THArgCheck(THSTensor_(size)(t, 1) == dim_k, 1,
      "Expected dim 1 size %d, got %d", dim_k, THSTensor_(size)(t, 1));

  nnz     = THSTensor_(nnz)(sparse);
  indices = THSTensor_(indices)(sparse);
  values  = THSTensor_(values)(sparse);

  csr = THSTensor_(toCSR)(THLongTensor_data(indices), dim_i, nnz);

  t_nnz = THSTensor_(nnz)(t);
  r_nnz = nnz * dim_k + t_nnz;
  newi = THLongTensor_newWithSize2d(2, r_nnz);
  newv = THTensor_(newWithSize1d)(r_nnz);
  THTensor_(zero)(newv);

  if (t_nnz != 0) {
    narrowi = THLongTensor_newNarrow(newi, 1, 0, t_nnz);
    narrowv = THTensor_(newNarrow)(newv, 0, 0, t_nnz);

    THLongTensor_copy(narrowi, THSTensor_(indices)(t));
    THTensor_(copy)(narrowv, THSTensor_(values)(t));
    THTensor_(mul)(newv, newv, beta);

    THLongTensor_free(narrowi);
    THTensor_(free)(narrowv);
  }

  // sparse = sparse * dense
  p = t_nnz;

  for (h = 0; h < dim_i; h++) {
    long i_start = THTensor_fastGet1d(csr, h);
    long i_end = THTensor_fastGet1d(csr, h+1);
    for (i = i_start; i < i_end; i++) {
      real val = THTensor_fastGet1d(values, i);
      long col = THTensor_fastGet2d(indices, 1, i);
      if (col >= 0 && col < dim_j) {
        THBlas_(axpy)(dim_k,
            alpha * val,
            ROW_PTR2(dense, col), dense->stride[1],
            ROW_PTR2(newv, p), 1);
      } else {
        THError("index out of bound. sspmm: %d not between 1 and %d",
            col, dim_j);
      }
    }
    // Fill up the indices with the right values
    if (i_start != i_end) {
      for (i = 0; i < dim_k; i++) {
        THTensor_fastSet2d(newi, 0, p + i, h);
        THTensor_fastSet2d(newi, 1, p + i, i);
      }
      p += dim_k;
    }
  }


  THSTensor_(resize2d)(r_, dim_i, dim_k);
  // to avoid a clone
  r_->indices = newi;
  r_-> values = newv;
  r_->    nnz = p;
  THSTensor_(contiguous)(r_);

  THLongTensor_free(csr);
  THLongTensor_free(indices);
  THTensor_(free)(values);
}

void THSTensor_(spcadd)(THTensor *r_, THTensor *dense, real value, THSTensor *sparse) {
  long k;
  THLongTensor  *indices = THSTensor_(indices)(sparse);
  THTensor      *values = THSTensor_(values)(sparse);
  THLongStorage *storage = THSTensor_(newSizeOf)(sparse);
  long          *sizes = storage->data;

  THTensor_(resizeAs)(r_, dense);
  THSTensor_(contiguous)(sparse);

  if (r_ != dense) THTensor_(copy)(r_, dense);

#pragma omp parallel for private(k)
  for (k = 0; k < sparse->nnz; k++) {
    long index = r_->storageOffset;
    for (long d = 0; d < sparse->nDimension; d++)
      index += r_->stride[d] * THTensor_fastGet2d(indices, d, k);
    r_->storage->data[index]  += value * THTensor_fastGet1d(values, k);
  }

  THLongTensor_free(indices);
  THTensor_(free)(values);
  THLongStorage_free(storage);
}

#undef ROW_PTR2
#undef COL_PTR2

#endif

