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

void THSTensor_(zeros)(THSTensor *r_, THLongStorage *size)
{
  THSTensor_(resize)(r_, size);
  THSTensor_(zero)(r_);
}

void THSTensor_(zerosLike)(THSTensor *r_, THSTensor *input)
{
  THSTensor_(resizeAs)(r_, input);
  THSTensor_(zero)(r_);
}

void THSTensor_(mul)(THSTensor *r_, THSTensor *t, real value) {
  if (r_ == t) {
    THTensor *r_values_ = THSTensor_(newValues)(r_);
    THTensor_(mul)(r_values_, r_values_, value);
    THTensor_(free)(r_values_);
  } else {
    THSTensor_(resizeAs)(r_, t);

    THLongTensor *r_indices_ = THSTensor_(newIndices)(r_);
    THTensor *r_values_ = THSTensor_(newValues)(r_);
    THLongTensor *t_indices_ = THSTensor_(newIndices)(t);
    THTensor *t_values_ = THSTensor_(newValues)(t);

    THLongTensor_resizeAs(r_indices_, t_indices_);
    THLongTensor_copy(r_indices_, t_indices_);
    THTensor_(mul)(r_values_, t_values_, value);
    r_->nnz = t->nnz;
    r_->coalesced = t->coalesced;

    THLongTensor_free(r_indices_);
    THTensor_(free)(r_values_);
    THLongTensor_free(t_indices_);
    THTensor_(free)(t_values_);
  }
}

/* TODO: add in-place support */
void THSTensor_(pow)(THSTensor *r_, THSTensor *t_, real value) {
  if (value == 0) {
    THError("cannot raise to zeroth power on sparse tensor");
  }

  THSTensor* t = THSTensor_(newCoalesce)(t_);

  THSTensor_(resizeAs)(r_, t);

  THLongTensor *r_indices_ = THSTensor_(newIndices)(r_);
  THTensor *r_values_ = THSTensor_(newValues)(r_);
  THLongTensor *t_indices_ = THSTensor_(newIndices)(t);
  THTensor *t_values_ = THSTensor_(newValues)(t);

  THLongTensor_resizeAs(r_indices_, t_indices_);
  THLongTensor_copy(r_indices_, t_indices_);
  THTensor_(pow)(r_values_, t_values_, value);
  r_->nnz = t->nnz;
  r_->coalesced = t->coalesced;

  THLongTensor_free(r_indices_);
  THTensor_(free)(r_values_);
  THLongTensor_free(t_indices_);
  THTensor_(free)(t_values_);

  THSTensor_(free)(t);
}

#if defined(THS_REAL_IS_FLOAT) || defined(THS_REAL_IS_DOUBLE)
accreal THSTensor_(normall)(THSTensor *self, real value) {
  THSTensor* self_coalesced = THSTensor_(newCoalesce)(self);
  THTensor *self_coalesced_values = THSTensor_(newValues)(self_coalesced);
  accreal result = THTensor_(normall)(self_coalesced_values, value);
  THSTensor_(free)(self_coalesced);
  THTensor_(free)(self_coalesced_values);
  return result;
}

/* floating point only, because that is what TH supports */
#endif

void THSTensor_(div)(THSTensor *r_, THSTensor *t, real value) {
  if (r_ == t) {
    THTensor *r_values_ = THSTensor_(newValues)(r_);
    THTensor_(div)(r_values_, r_values_, value);
    THTensor_(free)(r_values_);
  } else {
    THSTensor_(resizeAs)(r_, t);

    THLongTensor *r_indices_ = THSTensor_(newIndices)(r_);
    THTensor *r_values_ = THSTensor_(newValues)(r_);
    THLongTensor *t_indices_ = THSTensor_(newIndices)(t);
    THTensor *t_values_ = THSTensor_(newValues)(t);

    THLongTensor_resizeAs(r_indices_, t_indices_);
    THLongTensor_copy(r_indices_, t_indices_);
    THTensor_(div)(r_values_, t_values_, value);
    r_->nnz = t->nnz;
    r_->coalesced = t->coalesced;

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
  int t_coalesced = t->coalesced, s_coalesced = src->coalesced;
  int64_t nDimI = THSTensor_(nDimensionI)(src);
  THLongTensor *t_indices_ = THSTensor_(newIndices)(t);
  THTensor *t_values_ = THSTensor_(newValues)(t);
  THLongTensor *src_indices_ = THSTensor_(newIndices)(src);
  THTensor *s_values_ = THSTensor_(newValues)(src);
  THLongTensor *r_indices_ = THLongTensor_newWithSize2d(nDimI, max_nnz);
  THTensor *r_values_ = THSTensor_(newValuesWithSizeOf)(s_values_, max_nnz);
  THTensor_(zero)(r_values_);
  THSTensor_(resizeAs)(r_, src);
  THSTensor_(_move)(r_, r_indices_, r_values_);

  int64_t blockSize = r_values_->stride[0];
  int64_t cmp, d;
  int64_t r_i = 0, t_i = 0, s_i = 0;
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
      THBlas_(axpy)(blockSize, 1,
        THTensor_(data)(t_values_) + t_i * blockSize, 1,
        THTensor_(data)(r_values_) + r_i * blockSize, 1);
      t_i++;
    }
    if (cmp <= 0) {
      for (d = 0; d < nDimI; d++) {
        THTensor_fastSet2d(r_indices_, d, r_i, THTensor_fastGet2d(src_indices_, d, s_i));
      }
      THBlas_(axpy)(blockSize, value,
        THTensor_(data)(s_values_) + s_i * blockSize, 1,
        THTensor_(data)(r_values_) + r_i * blockSize, 1);
      s_i++;
    }
    r_i++;
  }

  r_->nnz = r_i;
  // TODO: I think it may be possible to track inside the loop and
  // detect when we are uncoalesced (e.g., by observing that an
  // index goes backwards) which may be more precise than using the
  // coalesced flag here.  But this is easy.
  r_->coalesced = t_coalesced && s_coalesced;

  THLongTensor_free(t_indices_);
  THTensor_(free)(t_values_);
  THLongTensor_free(src_indices_);
  THTensor_(free)(s_values_);
}

void THSTensor_(csub)(THSTensor *r_, THSTensor *t, real value, THSTensor *src) {
  THSTensor_(cadd)(r_, t, -value, src);
}

void THSTensor_(cmul)(THSTensor *r_, THSTensor *t_, THSTensor *src_) {
  if(!THSTensor_(isSameSizeAs)(t_, src_)) {
    THError("cmul operands have incompatible sizes or dimension types");
  }

  if (src_->nnz == 0 || t_->nnz == 0) {
    THSTensor_(zero)(r_);
    return;
  }

  THSTensor *t = THSTensor_(newCoalesce)(t_);
  THSTensor *src = THSTensor_(newCoalesce)(src_);

  // saving those because they can be overwritten when doing in-place operations
  ptrdiff_t t_nnz = t->nnz, s_nnz = src->nnz;
  ptrdiff_t max_nnz = t_nnz < s_nnz ? t_nnz : s_nnz;
  int64_t nDimI = THSTensor_(nDimensionI)(src);
  THLongTensor *t_indices_ = THSTensor_(newIndices)(t);
  THTensor *t_values_ = THSTensor_(newValues)(t);
  THLongTensor *src_indices_ = THSTensor_(newIndices)(src);
  THTensor *s_values_ = THSTensor_(newValues)(src);
  THLongTensor *r_indices_ = THLongTensor_newWithSize2d(nDimI, max_nnz);
  THTensor *r_values_ = THSTensor_(newValuesWithSizeOf)(s_values_, max_nnz);
  THTensor_(zero)(r_values_);
  THSTensor_(resizeAs)(r_, src);
  THSTensor_(_move)(r_, r_indices_, r_values_);

  THTensor *src1Buffer = THTensor_(new)();
  THTensor *src2Buffer = THTensor_(new)();
  THTensor *dstBuffer = THTensor_(new)();
  int64_t match, d;
  int64_t r_i = 0, t_i = 0, s_i = 0;
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
  r_->coalesced = 1;

  THLongTensor_free(t_indices_);
  THTensor_(free)(t_values_);
  THLongTensor_free(src_indices_);
  THTensor_(free)(s_values_);
  THTensor_(free)(src1Buffer);
  THTensor_(free)(src2Buffer);
  THTensor_(free)(dstBuffer);
  THSTensor_(free)(t);
  THSTensor_(free)(src);
}

void THTensor_(spaddcmul)(THTensor *r_, THTensor *t, real value, THSTensor *src1, THSTensor *src2) {
  THSTensor *intermediate = THSTensor_(new)();
  THSTensor_(cmul)(intermediate, src1, src2);
  THSTensor_(spcadd)(r_, t, value, intermediate);
  THSTensor_(free)(intermediate);
}

THLongTensor *THSTensor_(toCSR)(int64_t const *indices, int64_t dim, int64_t nnz) {
  int64_t h, i, hp0, hp1;
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
    real alpha, THSTensor *sparse_, THTensor *dense) {
  int64_t h, i;
  int64_t dim_i, dim_j, dim_k; // ixj * jxk = ixk
  int64_t nnz;
  THLongTensor *csr, *indices;
  THTensor *values;

  THArgCheck(sparse_->nDimensionI == 2, 2,
      "matrices expected, got %dD tensor", sparse_->nDimensionI);
  THArgCheck(sparse_->nDimensionV == 0, 2,
      "scalar values expected, got %dD values", sparse_->nDimensionV);
  THArgCheck(dense->nDimension == 2, 2,
      "matrices expected, got %dD tensor", dense->nDimension);

  THSTensor *sparse = THSTensor_(newCoalesce)(sparse_);

  dim_i = THSTensor_(size)(sparse, 0);
  dim_j = THSTensor_(size)(sparse, 1);
  dim_k = THTensor_(size)(dense, 1);

  THTensor_(resize2d)(r_, dim_i, dim_k);

  THArgCheck(THTensor_(size)(dense, 0) == dim_j, 3,
      "Expected dim 0 size %d, got %d", dim_j, THTensor_(size)(dense, 0));
  THArgCheck(THTensor_(size)(t, 0) == dim_i, 1,
      "Expected dim 0 size %d, got %d", dim_i, THTensor_(size)(t, 0));
  THArgCheck(THTensor_(size)(t, 1) == dim_k, 1,
      "Expected dim 1 size %d, got %d", dim_k, THTensor_(size)(t, 1));

  nnz     = THSTensor_(nnz)(sparse);
  indices = THSTensor_(newIndices)(sparse);
  values  = THSTensor_(newValues)(sparse);

  csr = THSTensor_(toCSR)(THLongTensor_data(indices), dim_i, nnz);

  // r_ = alpha * sparse * dense
  if (beta == 0) {
    THTensor_(zero)(r_);
  } else if (beta == 1) {
    if (r_ != t) {
      THTensor_(copy)(r_, t);
    }
  } else {
    THTensor_(mul)(r_, t, beta);
  }
#pragma omp parallel for private(h, i) schedule(static) if (nnz > 10000)
  for (h = 0; h < dim_i; h++) {
    int64_t i_start = THTensor_fastGet1d(csr, h);
    int64_t i_end = THTensor_fastGet1d(csr, h+1);
    for (i = i_start; i < i_end; i++) {
      real val = THTensor_fastGet1d(values, i);
      int64_t col = THTensor_fastGet2d(indices, 1, i);
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
  THSTensor_(free)(sparse);
}

void THSTensor_(sspaddmm)(THSTensor *r_,
    real beta, THSTensor *t,
    real alpha, THSTensor *sparse_, THTensor *dense) {

  int64_t h, i, p;
  int64_t dim_i, dim_j, dim_k; // ixj * jxk = ixk
  int64_t nnz, r_nnz, t_nnz;
  THLongTensor *csr, *indices, *newi, *narrowi;
  THTensor *values, *newv, *narrowv;

  THArgCheck(sparse_->nDimensionI == 2, 2,
      "matrices expected, got %dD tensor", sparse_->nDimensionI);
  THArgCheck(sparse_->nDimensionV == 0, 2,
      "scalar values expected, got %dD values", sparse_->nDimensionV);
  THArgCheck(dense->nDimension == 2, 2,
      "matrices expected, got %dD tensor", dense->nDimension);

  THSTensor *sparse = THSTensor_(newCoalesce)(sparse_);

  dim_i = THSTensor_(size)(sparse, 0);
  dim_j = THSTensor_(size)(sparse, 1);
  dim_k = THTensor_(size)(dense, 1);

  THSTensor_(resize2d)(r_, dim_i, dim_k);

  THArgCheck(THTensor_(size)(dense, 0) == dim_j, 3,
      "Expected dim 0 size %d, got %d", dim_j, THTensor_(size)(dense, 0));
  THArgCheck(THSTensor_(size)(t, 0) == dim_i, 1,
      "Expected dim 0 size %d, got %d", dim_i, THSTensor_(size)(t, 0));
  THArgCheck(THSTensor_(size)(t, 1) == dim_k, 1,
      "Expected dim 1 size %d, got %d", dim_k, THSTensor_(size)(t, 1));

  nnz     = THSTensor_(nnz)(sparse);
  indices = THSTensor_(newIndices)(sparse);
  values  = THSTensor_(newValues)(sparse);

  csr = THSTensor_(toCSR)(THLongTensor_data(indices), dim_i, nnz);

  t_nnz = THSTensor_(nnz)(t);
  r_nnz = nnz * dim_k + t_nnz;
  newi = THLongTensor_newWithSize2d(2, r_nnz);
  newv = THTensor_(newWithSize1d)(r_nnz);
  THTensor_(zero)(newv);

  if (t_nnz != 0) {
    narrowi = THLongTensor_newNarrow(newi, 1, 0, t_nnz);
    narrowv = THTensor_(newNarrow)(newv, 0, 0, t_nnz);

    THLongTensor_copy(narrowi, THSTensor_(newIndices)(t));
    THTensor_(copy)(narrowv, THSTensor_(newValues)(t));
    THTensor_(mul)(newv, newv, beta);

    THLongTensor_free(narrowi);
    THTensor_(free)(narrowv);
  }

  // sparse = sparse * dense
  p = t_nnz;

  for (h = 0; h < dim_i; h++) {
    int64_t i_start = THTensor_fastGet1d(csr, h);
    int64_t i_end = THTensor_fastGet1d(csr, h+1);
    for (i = i_start; i < i_end; i++) {
      real val = THTensor_fastGet1d(values, i);
      int64_t col = THTensor_fastGet2d(indices, 1, i);
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


  // to avoid a clone
  r_->indices = newi;
  r_-> values = newv;
  r_->    nnz = p;

  THLongTensor_free(csr);
  THLongTensor_free(indices);
  THTensor_(free)(values);
  THSTensor_(free)(sparse);
}

void THSTensor_(hspmm)(THSTensor *r_, real alpha, THSTensor *sparse_, THTensor *dense) {
  THArgCheck(sparse_->nDimensionI == 2, 2,
      "matrices expected, got %dD tensor", sparse_->nDimensionI);
  THArgCheck(sparse_->nDimensionV == 0, 2,
      "scalar values expected, got %dD values", sparse_->nDimensionV);
  THArgCheck(dense->nDimension == 2, 2,
      "matrices expected, got %dD tensor", dense->nDimension);

  int64_t m = THSTensor_(size)(sparse_, 0);
  int64_t k = THSTensor_(size)(sparse_, 1);
  int64_t n = THTensor_(size)(dense, 1);

  THArgCheck(THTensor_(size)(dense, 0) == k, 3,
      "Expected dim 0 size %d, got %d", k, THTensor_(size)(dense, 0));
  int64_t size[2] = {m, n};
  THSTensor_(rawResize)(r_, 1, 1, size);

  THSTensor *sparse = THSTensor_(newCoalesce)(sparse_);

  int64_t nnz = THSTensor_(nnz)(sparse);
  THLongTensor *indices = THLongTensor_newWithSize2d(1, nnz);

  // Initialize the sparse matrix that will be used with spaddmm to send rows
  // from the dense matrix to rows of the output's value tensor
  THSTensor *newSparse = THSTensor_(newClone)(sparse);
  THLongTensor *spIndices = THSTensor_(newIndices)(newSparse);
  THLongTensor *valueIndices = THLongTensor_new();
  THLongTensor_select(valueIndices, spIndices, 0, 0);

  // Compute output indices
  int64_t i = -1, prevIdx = -1;
  for (int64_t j = 0; j < nnz; j++) {
    int64_t currIdx = THTensor_fastGet1d(valueIndices, j);
    if (currIdx != prevIdx) {
      THTensor_fastSet2d(indices, 0, ++i, currIdx);
      prevIdx = currIdx;
    }
    THTensor_fastSet1d(valueIndices, j, i);
  }
  int64_t outNnz = i + 1;
  THLongTensor_resize2d(indices, 1, outNnz);
  THTensor *values = THTensor_(newWithSize2d)(outNnz, n);
  newSparse->size[0] = outNnz;

  // Compute output values tensor with sparse * dense multiplication
  THSTensor_(spaddmm)(values, 0, values, alpha, newSparse, dense);
  THSTensor_(_move)(r_, indices, values);

  THSTensor_(free)(newSparse);
  THLongTensor_free(spIndices);
  THLongTensor_free(valueIndices);
  THSTensor_(free)(sparse);
}

void THSTensor_(spcadd)(THTensor *r_, THTensor *dense, real value, THSTensor *sparse_) {
  THTensor_(resizeAs)(r_, dense);
  THSTensor *sparse = THSTensor_(newCoalesce)(sparse_);

  int64_t k;
  THLongTensor  *indices = THSTensor_(newIndices)(sparse);
  THTensor      *values = THSTensor_(newValues)(sparse);
  THLongStorage *storage = THSTensor_(newSizeOf)(sparse);
  int64_t       nDim = THTensor_(nDimension)(dense);
  int64_t       nDimI = THSTensor_(nDimensionI)(sparse);

  if (r_ != dense) THTensor_(copy)(r_, dense);

  if (nDim > nDimI) {
    THTensor *srcBuffer = THTensor_(new)();
    THTensor *dstBuffer = THTensor_(new)();
    for (k = 0; k < sparse->nnz; k++) {
      THTensor_(set)(dstBuffer, r_);
      for (int64_t d = 0; d < sparse->nDimensionI; d++) {
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
      int64_t index = r_->storageOffset;
      for (int64_t d = 0; d < sparse->nDimensionI; d++) {
        index += r_->stride[d] * THTensor_fastGet2d(indices, d, k);
      }
      r_->storage->data[index]  += value * THTensor_fastGet1d(values, k);
    }
  }

  THLongTensor_free(indices);
  THTensor_(free)(values);
  THLongStorage_free(storage);
  THSTensor_(free)(sparse);
}

#undef ROW_PTR2
#undef COL_PTR2

#endif
