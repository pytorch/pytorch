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

void THSTensor_(spmm)(THTensor *r_, THSTensor *sparse, THTensor *dense) {
  long h, i;
  long dim_i, dim_j, dim_k; // ixj * jxk = ixk
  long nnz;
  THLongTensor *csr, *indices;
  THTensor *values;

  if( (sparse->nDimension != 2) || (dense->nDimension != 2))
    THError("matrices expected, got %dD, %dD tensors",
        sparse->nDimension, dense->nDimension);

  THSTensor_(contiguous)(sparse);

  dim_i = THSTensor_(size)(sparse, 0);
  dim_j = THSTensor_(size)(sparse, 1);
  dim_k = THTensor_(size)(dense, 1);

  nnz = THSTensor_(nnz)(sparse);
  indices = THSTensor_(indices)(sparse);
  values = THSTensor_(values)(sparse);

  csr = THSTensor_(toCSR)(THLongTensor_data(indices), dim_i, nnz);

  // r_ = sparse * dense
  THTensor_(resize2d)(r_, dim_i, dim_k);
  THTensor_(zero)(r_);
#pragma omp parallel for private(h, i) schedule(static) if (nnz > 10000)
  for (h = 0; h < dim_i; h++) {
    long i_start = THTensor_fastGet1d(csr, h);
    long i_end = THTensor_fastGet1d(csr, h+1);
    for (i = i_start; i < i_end; i++) {
      real val = THTensor_fastGet1d(values, i);
      long col = THTensor_fastGet2d(indices, 1, i);
      if (col >= 0 && col < dim_j) {
        THBlas_(axpy)(dim_k,
            val,
            ROW_PTR2(dense, col), dense->stride[1],
            ROW_PTR2(r_, h), r_->stride[1]);
      } else {
        THError("index out of bound. spmm: %d not between 1 and %d",
            col, dim_j);
      }
    }
  }

  THFree(csr);
  THFree(indices);
  THFree(values);
}

void THSTensor_(sspmm)(THSTensor *r_, THSTensor *sparse, THTensor *dense) {
  long h, i, p;
  long dim_i, dim_j, dim_k; // ixj * jxk = ixk
  long nnz;
  THLongTensor *csr, *indices, *newi;
  THTensor *values, *newv;

  if( (sparse->nDimension != 2) || (dense->nDimension != 2))
    THError("matrices expected, got %dD, %dD tensors",
        sparse->nDimension, dense->nDimension);

  THSTensor_(contiguous)(sparse);

  dim_i = THSTensor_(size)(sparse, 0);
  dim_j = THSTensor_(size)(sparse, 1);
  dim_k = THTensor_(size)(dense, 1);

  nnz = THSTensor_(nnz)(sparse);
  indices = THSTensor_(indices)(sparse);
  values = THSTensor_(values)(sparse);

  csr = THSTensor_(toCSR)(THLongTensor_data(indices), dim_i, nnz);

  newi = THLongTensor_newWithSize2d(2, nnz * dim_k);
  newv = THTensor_(newWithSize1d)(nnz * dim_k);
  THTensor_(zero)(newv);

  // sparse = sparse * dense
  p = 0;

  for (h = 0; h < dim_i; h++) {
    long i_start = THTensor_fastGet1d(csr, h);
    long i_end = THTensor_fastGet1d(csr, h+1);
    for (i = i_start; i < i_end; i++) {
      real val = THTensor_fastGet1d(values, i);
      long col = THTensor_fastGet2d(indices, 1, i);
      if (col >= 0 && col < dim_j) {
        THBlas_(axpy)(dim_k,
            val,
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
  r_->values = newv;
  r_->nnz = p;
  THSTensor_(contiguous)(r_);

  THFree(csr);
  THFree(indices);
  THFree(values);
}

void THSTensor_(spcadd)(THTensor *r_, THTensor *dense, real value, THSTensor *sparse) {
  long k;
  THLongTensor  *indices = THSTensor_(indices)(sparse);
  THTensor      *values = THSTensor_(values)(sparse);
  THLongStorage *storage = THSTensor_(newSizeOf)(sparse);
  long          *sizes = storage->data;

  THTensor_(resizeAs)(r_, dense);
  THSTensor_(contiguous)(sparse);

  if (r_ == dense) {
#pragma omp parallel for private(k)
    for (k = 0; k < sparse->nnz; k++) {
      long index = 0;
      long i2 = r_->storageOffset;
      for (long d = 0; d < sparse->nDimension; d++) {
        index = index + THTensor_fastGet2d(indices, d, k);
        i2 += r_->stride[d] * THTensor_fastGet2d(indices, d, k);
      }
      r_->storage->data[i2]  += THTensor_fastGet1d(values, k);
    }
  } else {
    THTensor_(copy)(r_, dense);
#pragma omp parallel for private(k)
    for (k = 0; k < sparse->nnz; k++) {
      long index = 0;
      long i2 = r_->storageOffset;
      long i3 = dense->storageOffset;
      for (long d = 0; d < sparse->nDimension; d++) {
        index = sizes[d] * index + THTensor_fastGet2d(indices, d, k);
        i2 += r_->stride[d] * THTensor_fastGet2d(indices, d, k);
        i3 += dense->stride[d] * THTensor_fastGet2d(indices, d, k);
      }
      r_->storage->data[i2] = dense->storage->data[i3] + THTensor_fastGet1d(values, k);
    }
  }
}

#undef ROW_PTR2
#undef COL_PTR2

#endif

