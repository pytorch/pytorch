#ifndef THS_GENERIC_FILE
#define THS_GENERIC_FILE "generic/THSTensorMath.c"
#else

#define ROW_PTR2(t, r) (THTensor_(data)(t) + (r) * (t)->stride[0])
#define COL_PTR2(t, c) (THTensor_(data)(t) + (c) * (t)->stride[1])

void THSTensor_(zero)(THSTensor *self) {
    if (self->indices->nDimension) {
      THLongTensor_resizeNd(self->indices, 0, NULL, NULL);
    }
    if (self->values->nDimension) {
      THTensor_(resizeNd)(self->values, 0, NULL, NULL);
    }
  self->nnz = 0;
}

void THSTensor_(mul)(THSTensor *r_, THSTensor *t, real value) {
  if (r_ == t) {
    THTensor *r_values_ = THSTensor_(values)(r_);
    THTensor_(mul)(r_values_, r_values_, value);
    THTensor_(free)(r_values_);
  } else {
    THSTensor_(resizeAs)(r_, t);

    THLongTensor *r_indices_ = THSTensor_(indices)(r_);
    THTensor *r_values_ = THSTensor_(values)(r_);
    THLongTensor *t_indices_ = THSTensor_(indices)(t);
    THTensor *t_values_ = THSTensor_(values)(t);

    THLongTensor_resizeAs(r_indices_, t_indices_);
    THLongTensor_copy(r_indices_, t_indices_);
    THTensor_(mul)(r_values_, t_values_, value);
    r_->nnz = t->nnz;
    r_->contiguous = t->contiguous;

    THLongTensor_free(r_indices_);
    THTensor_(free)(r_values_);
    THLongTensor_free(t_indices_);
    THTensor_(free)(t_values_);
  }
}

void THSTensor_(div)(THSTensor *r_, THSTensor *t, real value) {
  if (r_ == t) {
    THTensor *r_values_ = THSTensor_(values)(r_);
    THTensor_(div)(r_values_, r_values_, value);
    THTensor_(free)(r_values_);
  } else {
    THSTensor_(resizeAs)(r_, t);

    THLongTensor *r_indices_ = THSTensor_(indices)(r_);
    THTensor *r_values_ = THSTensor_(values)(r_);
    THLongTensor *t_indices_ = THSTensor_(indices)(t);
    THTensor *t_values_ = THSTensor_(values)(t);

    THLongTensor_resizeAs(r_indices_, t_indices_);
    THLongTensor_copy(r_indices_, t_indices_);
    THTensor_(div)(r_values_, t_values_, value);
    r_->nnz = t->nnz;
    r_->contiguous = t->contiguous;

    THLongTensor_free(r_indices_);
    THTensor_(free)(r_values_);
    THLongTensor_free(t_indices_);
    THTensor_(free)(t_values_);
  }
}

void THSTensor_(cadd)(THSTensor *r_, THSTensor *t, real value, THSTensor *src) {
  if(!THSTensor_(isSameSizeAs)(t, src)) {
    THError("cadd operands have incompatible sizes or dimension types");
  }
  THSTensor_(contiguous)(t);
  THSTensor_(contiguous)(src);

  if (src->nnz == 0) {
    THSTensor_(copy)(r_, t);
    return;
  }
  if (t->nnz == 0) {
    THSTensor_(mul)(r_, src, value);
    return;
  }

  // saving those because they can be overwritten when doing in-place operations
  ptrdiff_t t_nnz = t->nnz, s_nnz = src->nnz, max_nnz = t_nnz + s_nnz;
  long nDimI = THSTensor_(nDimensionI)(src);
  long nDimV = THSTensor_(nDimensionV)(src);
  THLongTensor *t_indices_ = THSTensor_(indices)(t);
  THTensor *t_values_ = THSTensor_(values)(t);
  THLongTensor *src_indices_ = THSTensor_(indices)(src);
  THTensor *s_values_ = THSTensor_(values)(src);
  THLongTensor *r_indices_ = THLongTensor_newWithSize2d(nDimI, max_nnz);
  THTensor *r_values_ = THSTensor_(newValuesWithSizeOf)(s_values_, max_nnz);
  THTensor_(zero)(r_values_);
  // TODO handle case where src values is empty
  THSTensor_(resizeAs)(r_, src);
  THSTensor_(move)(r_, r_indices_, r_values_);

  THTensor *srcBuffer = THTensor_(new)();
  THTensor *dstBuffer = THTensor_(new)();
  long cmp, d;
  long r_i = 0, t_i = 0, s_i = 0;
  while (t_i < t_nnz || s_i < s_nnz) {
    if (t_i >= t_nnz) {
      cmp = -1;
    } else if (s_i >= s_nnz) {
      cmp = 1;
    } else {
      cmp = 0;
      for (d = 0; d < nDimI; d++) {
        if (THTensor_fastGet2d(t_indices_, d, t_i) < THTensor_fastGet2d(src_indices_, d, s_i)) {
          cmp = 1;
          break;
        }
        if (THTensor_fastGet2d(t_indices_, d, t_i) > THTensor_fastGet2d(src_indices_, d, s_i)) {
          cmp = -1;
          break;
        }
      }
    }
    if (cmp >= 0) {
      for (d = 0; d < nDimI; d++) {
        THTensor_fastSet2d(r_indices_, d, r_i, THTensor_fastGet2d(t_indices_, d, t_i));
      }
      THSTensor_(addSlice)(dstBuffer, dstBuffer, srcBuffer, r_values_, r_values_, 1, t_values_, 0, r_i, r_i, t_i);
      t_i++;
    }
    if (cmp <= 0) {
      for (d = 0; d < nDimI; d++) {
        THTensor_fastSet2d(r_indices_, d, r_i, THTensor_fastGet2d(src_indices_, d, s_i));
      }
      THSTensor_(addSlice)(dstBuffer, dstBuffer, srcBuffer, r_values_, r_values_, value, s_values_, 0, r_i, r_i, s_i);
      s_i++;
    }
    r_i++;
  }

  r_->nnz = r_i;
  r_->contiguous = 1;

  THLongTensor_free(t_indices_);
  THTensor_(free)(t_values_);
  THLongTensor_free(src_indices_);
  THTensor_(free)(s_values_);
  THTensor_(free)(srcBuffer);
  THTensor_(free)(dstBuffer);
}

void THSTensor_(csub)(THSTensor *r_, THSTensor *t, real value, THSTensor *src) {
  THSTensor_(cadd)(r_, t, -value, src);
}

void THSTensor_(cmul)(THSTensor *r_, THSTensor *t, THSTensor *src) {
  if(!THSTensor_(isSameSizeAs)(t, src)) {
    THError("cadd operands have incompatible sizes or dimension types");
  }
  THSTensor_(contiguous)(t);
  THSTensor_(contiguous)(src);

  if (src->nnz == 0 || t->nnz == 0) {
    THSTensor_(zero)(r_);
    return;
  }

  // saving those because they can be overwritten when doing in-place operations
  ptrdiff_t t_nnz = t->nnz, s_nnz = src->nnz;
  ptrdiff_t max_nnz = t_nnz < s_nnz ? t_nnz : s_nnz;
  long nDimI = THSTensor_(nDimensionI)(src);
  long nDimV = THSTensor_(nDimensionV)(src);
  THLongTensor *t_indices_ = THSTensor_(indices)(t);
  THTensor *t_values_ = THSTensor_(values)(t);
  THLongTensor *src_indices_ = THSTensor_(indices)(src);
  THTensor *s_values_ = THSTensor_(values)(src);
  THLongTensor *r_indices_ = THLongTensor_newWithSize2d(nDimI, max_nnz);
  THTensor *r_values_ = THSTensor_(newValuesWithSizeOf)(s_values_, max_nnz);
  THTensor_(zero)(r_values_);
  THSTensor_(resizeAs)(r_, src);
  THSTensor_(move)(r_, r_indices_, r_values_);

  THTensor *src1Buffer = THTensor_(new)();
  THTensor *src2Buffer = THTensor_(new)();
  THTensor *dstBuffer = THTensor_(new)();
  long match, d;
  long r_i = 0, t_i = 0, s_i = 0;
  while (t_i < t_nnz && s_i < s_nnz) {
    match = 1;
    for (d = 0; d < nDimI; d++) {
      if (THTensor_fastGet2d(t_indices_, d, t_i) < THTensor_fastGet2d(src_indices_, d, s_i)) {
        t_i++;
        match = 0;
        break;
      }
      if (THTensor_fastGet2d(t_indices_, d, t_i) > THTensor_fastGet2d(src_indices_, d, s_i)) {
        s_i++;
        match = 0;
        break;
      }
    }
    if (!match) continue;
    for (d = 0; d < nDimI; d++) {
      THTensor_fastSet2d(r_indices_, d, r_i, THTensor_fastGet2d(t_indices_, d, t_i));
    }
    THSTensor_(mulSlice)(dstBuffer, src1Buffer, src2Buffer, r_values_, t_values_, s_values_, 0, r_i, t_i, s_i);
    r_i++;
    t_i++;
    s_i++;
  }

  r_->nnz = r_i;
  r_->contiguous = 1;

  THLongTensor_free(t_indices_);
  THTensor_(free)(t_values_);
  THLongTensor_free(src_indices_);
  THTensor_(free)(s_values_);
  THTensor_(free)(src1Buffer);
  THTensor_(free)(src2Buffer);
  THTensor_(free)(dstBuffer);
}

void THTensor_(spaddcmul)(THTensor *r_, THTensor *t, real value, THSTensor *src1, THSTensor *src2) {
  THSTensor *intermediate = THSTensor_(new)();
  THSTensor_(cmul)(intermediate, src1, src2);
  THSTensor_(spcadd)(r_, t, value, intermediate);
  THSTensor_(free)(intermediate);
}

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

  THArgCheck(sparse->nDimensionI == 2, 2,
      "matrices expected, got %dD tensor", sparse->nDimensionI);
  THArgCheck(sparse->nDimensionV == 0, 2,
      "scalar values expected, got %dD values", sparse->nDimensionV);
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

  THArgCheck(sparse->nDimensionI == 2, 2,
      "matrices expected, got %dD tensor", sparse->nDimensionI);
  THArgCheck(sparse->nDimensionV == 0, 2,
      "scalar values expected, got %dD values", sparse->nDimensionV);
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
  long          nDim = THTensor_(nDimension)(dense);
  long          nDimI = THSTensor_(nDimensionI)(sparse);

  THTensor_(resizeAs)(r_, dense);
  THSTensor_(contiguous)(sparse);

  if (r_ != dense) THTensor_(copy)(r_, dense);


  if (nDim > nDimI) {
    THTensor *srcBuffer = THTensor_(new)();
    THTensor *dstBuffer = THTensor_(new)();
    for (k = 0; k < sparse->nnz; k++) {
      THTensor_(set)(dstBuffer, r_);
      for (long d = 0; d < sparse->nDimensionI; d++) {
        THTensor_(select)(dstBuffer, dstBuffer, 0, THTensor_fastGet2d(indices, d, k));
      }
      THTensor_(select)(srcBuffer, values, 0, k);
      THTensor_(cadd)(dstBuffer, dstBuffer, value, srcBuffer);
    }
    THTensor_(free)(srcBuffer);
    THTensor_(free)(dstBuffer);
  } else {
    #pragma omp parallel for private(k)
    for (k = 0; k < sparse->nnz; k++) {
      long index = r_->storageOffset;
      for (long d = 0; d < sparse->nDimensionI; d++) {
        index += r_->stride[d] * THTensor_fastGet2d(indices, d, k);
      }
      r_->storage->data[index]  += value * THTensor_fastGet1d(values, k);
    }
  }

  THLongTensor_free(indices);
  THTensor_(free)(values);
  THLongStorage_free(storage);
}

#undef ROW_PTR2
#undef COL_PTR2

#endif
