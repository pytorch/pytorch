#ifndef THZ_GENERIC_FILE
#define THZ_GENERIC_FILE "generic/THZTensorLapack.c"
#else

/*
Check if self is transpose of a contiguous matrix
*/
static int THZTensor_(isTransposedContiguous)(THZTensor *self)
{
  return self->stride[0] == 1 && self->stride[1] == self->size[0];
}
/*
If a matrix is a regular contiguous matrix, make sure it is transposed
because this is what we return from Lapack calls.
*/
static void THZTensor_(checkTransposed)(THZTensor *self)
{
  if(THZTensor_(isContiguous)(self))
    THZTensor_(transpose)(self, NULL, 0, 1);
  return;
}
/*
newContiguous followed by transpose
Similar to (newContiguous), but checks if the transpose of the matrix
is contiguous and also limited to 2D matrices.
*/
static THZTensor *THZTensor_(newTransposedContiguous)(THZTensor *self)
{
  THZTensor *tensor;
  if(THZTensor_(isTransposedContiguous)(self))
  {
    THZTensor_(retain)(self);
    tensor = self;
  }
  else
  {
    tensor = THZTensor_(newContiguous)(self);
    THZTensor_(transpose)(tensor, NULL, 0, 1);
  }

  return tensor;
}

/*
Given the result tensor and src tensor, decide if the lapack call should use the
provided result tensor or should allocate a new space to put the result in.

The returned tensor have to be freed by the calling function.

nrows is required, because some lapack calls, require output space smaller than
input space, like underdetermined gels.
*/
static THZTensor *THZTensor_(checkLapackClone)(THZTensor *result, THZTensor *src, int nrows)
{
  /* check if user wants to reuse src and if it is correct shape/size */
  if (src == result && THZTensor_(isTransposedContiguous)(src) && src->size[1] == nrows)
    THZTensor_(retain)(result);
  else if(src == result || result == NULL) /* in this case, user wants reuse of src, but its structure is not OK */
    result = THZTensor_(new)();
  else
    THZTensor_(retain)(result);
  return result;
}

/*
Same as cloneColumnMajor, but accepts nrows argument, because some lapack calls require
the resulting tensor to be larger than src.
*/
static THZTensor *THZTensor_(cloneColumnMajorNrows)(THZTensor *self, THZTensor *src, int nrows)
{
  THZTensor *result;
  THZTensor *view;

  if (src == NULL)
    src = self;
  result = THZTensor_(checkLapackClone)(self, src, nrows);
  if (src == result)
    return result;

  THZTensor_(resize2d)(result, src->size[1], nrows);
  THZTensor_(checkTransposed)(result);

  if (src->size[0] == nrows)
    THZTensor_(copy)(result, src);
  else
  {
    view = THZTensor_(newNarrow)(result, 0, 0, src->size[0]);
    THZTensor_(copy)(view, src);
    THZTensor_(free)(view);
  }
  return result;
}

/*
Create a clone of src in self column major order for use with Lapack.
If src == self, a new tensor is allocated, in any case, the return tensor should be
freed by calling function.
*/
static THZTensor *THZTensor_(cloneColumnMajor)(THZTensor *self, THZTensor *src)
{
  return THZTensor_(cloneColumnMajorNrows)(self, src, src->size[0]);
}

void THZTensor_(gesv)(THZTensor *rb_, THZTensor *ra_, THZTensor *b, THZTensor *a)
{
  int free_b = 0;
  if (a == NULL) a = ra_;
  if (b == NULL) b = rb_;
  THArgCheck(a->nDimension == 2, 2, "A should have 2 dimensions, but has %d",
      a->nDimension);
  THArgCheck(b->nDimension == 1 || b->nDimension == 2, 1, "B should have 1 or 2 "
      "dimensions, but has %d", b->nDimension);
  THArgCheck(a->size[0] == a->size[1], 2, "A should be square, but is %ldx%ld",
      a->size[0], a->size[1]);
  THArgCheck(a->size[0] == b->size[0], 2, "A,B size incompatible - A has %ld "
      "rows, B has %ld", a->size[0], b->size[0]);

  if (b->nDimension == 1) {
    b = THZTensor_(newWithStorage2d)(b->storage, b->storageOffset, b->size[0],
            b->stride[0], 1, 0);
    free_b = 1;
  }

  int n, nrhs, lda, ldb, info;
  THIntTensor *ipiv;
  THZTensor *ra__;  // working version of A matrix to be passed into lapack GELS
  THZTensor *rb__;  // working version of B matrix to be passed into lapack GELS

  ra__ = THZTensor_(cloneColumnMajor)(ra_, a);
  rb__ = THZTensor_(cloneColumnMajor)(rb_, b);

  n    = (int)ra__->size[0];
  nrhs = (int)rb__->size[1];
  lda  = n;
  ldb  = n;

  ipiv = THIntTensor_newWithSize1d((int64_t)n);
  THZLapack_(gesv)(n, nrhs,
      THZTensor_(data)(ra__), lda, THIntTensor_data(ipiv),
      THZTensor_(data)(rb__), ldb, &info);

  THLapackCheckWithCleanup("Lapack Error in %s : U(%d,%d) is zero, singular U.",
                           THCleanup(
                               THZTensor_(free)(ra__);
                               THZTensor_(free)(rb__);
                               THIntTensor_free(ipiv);
                               if (free_b) THZTensor_(free)(b);),
                           "gesv", info, info);

  THZTensor_(freeCopyTo)(ra__, ra_);
  THZTensor_(freeCopyTo)(rb__, rb_);
  THIntTensor_free(ipiv);
  if (free_b) THZTensor_(free)(b);
}

void THZTensor_(trtrs)(THZTensor *rb_, THZTensor *ra_, THZTensor *b, THZTensor *a,
                      const char *uplo, const char *trans, const char *diag)
{
  int free_b = 0;
  if (a == NULL) a = ra_;
  if (b == NULL) b = rb_;
  THArgCheck(a->nDimension == 2, 2, "A should have 2 dimensions, but has %d",
      a->nDimension);
  THArgCheck(b->nDimension == 1 || b->nDimension == 2, 1, "B should have 1 or 2 "
      "dimensions, but has %d", b->nDimension);
  THArgCheck(a->size[0] == a->size[1], 2, "A should be square, but is %ldx%ld",
      a->size[0], a->size[1]);
  THArgCheck(a->size[0] == b->size[0], 2, "A,B size incompatible - A has %ld "
      "rows, B has %ld", a->size[0], b->size[0]);

  if (b->nDimension == 1) {
    b = THZTensor_(newWithStorage2d)(b->storage, b->storageOffset, b->size[0],
            b->stride[0], 1, 0);
    free_b = 1;
  }

  int n, nrhs, lda, ldb, info;
  THZTensor *ra__; // working version of A matrix to be passed into lapack TRTRS
  THZTensor *rb__; // working version of B matrix to be passed into lapack TRTRS

  ra__ = THZTensor_(cloneColumnMajor)(ra_, a);
  rb__ = THZTensor_(cloneColumnMajor)(rb_, b);

  n    = (int)ra__->size[0];
  nrhs = (int)rb__->size[1];
  lda  = n;
  ldb  = n;

  THZLapack_(trtrs)(uplo[0], trans[0], diag[0], n, nrhs,
                   THZTensor_(data)(ra__), lda,
                   THZTensor_(data)(rb__), ldb, &info);


  THLapackCheckWithCleanup("Lapack Error in %s : A(%d,%d) is zero, singular A",
                           THCleanup(
                              THZTensor_(free)(ra__);
                              THZTensor_(free)(rb__);
                              if (free_b) THZTensor_(free)(b);),
                           "trtrs", info, info);

  THZTensor_(freeCopyTo)(ra__, ra_);
  THZTensor_(freeCopyTo)(rb__, rb_);
  if (free_b) THZTensor_(free)(b);
}

void THZTensor_(gels)(THZTensor *rb_, THZTensor *ra_, THZTensor *b, THZTensor *a)
{
  int free_b = 0;
  // Note that a = NULL is interpreted as a = ra_, and b = NULL as b = rb_.
  if (a == NULL) a = ra_;
  if (b == NULL) b = rb_;
  THArgCheck(a->nDimension == 2, 2, "A should have 2 dimensions, but has %d",
      a->nDimension);
  THArgCheck(b->nDimension == 1 || b->nDimension == 2, 1, "B should have 1 or 2 "
      "dimensions, but has %d", b->nDimension);
  THArgCheck(a->size[0] == b->size[0], 2, "A,B size incompatible - A has %ld "
      "rows, B has %ld", a->size[0], b->size[0]);

  if (b->nDimension == 1) {
    b = THZTensor_(newWithStorage2d)(b->storage, b->storageOffset, b->size[0],
            b->stride[0], 1, 0);
    free_b = 1;
  }

  int m, n, nrhs, lda, ldb, info, lwork;
  THZTensor *work = NULL;
  ntype wkopt = 0;

  THZTensor *ra__ = NULL;  // working version of A matrix to be passed into lapack GELS
  THZTensor *rb__ = NULL;  // working version of B matrix to be passed into lapack GELS

  ra__ = THZTensor_(cloneColumnMajor)(ra_, a);

  m = ra__->size[0];
  n = ra__->size[1];
  lda = m;
  ldb = (m > n) ? m : n;

  rb__ = THZTensor_(cloneColumnMajorNrows)(rb_, b, ldb);

  nrhs = rb__->size[1];
  info = 0;


  /* get optimal workspace size */
  THZLapack_(gels)('N', m, n, nrhs, THZTensor_(data)(ra__), lda,
      THZTensor_(data)(rb__), ldb,
      &wkopt, -1, &info);
  lwork = (int)wkopt;
  work = THZTensor_(newWithSize1d)(lwork);
  THZLapack_(gels)('N', m, n, nrhs, THZTensor_(data)(ra__), lda,
      THZTensor_(data)(rb__), ldb,
      THZTensor_(data)(work), lwork, &info);

  THLapackCheckWithCleanup("Lapack Error in %s : The %d-th diagonal element of the triangular factor of A is zero",
                           THCleanup(THZTensor_(free)(ra__);
                                     THZTensor_(free)(rb__);
                                     THZTensor_(free)(work);
                                     if (free_b) THZTensor_(free)(b);),
                           "gels", info,"");

  /*
   * In the m < n case, if the input b is used as the result (so b == _rb),
   * then rb_ was originally m by nrhs but now should be n by nrhs.
   * This is larger than before, so we need to expose the new rows by resizing.
   */
  if (m < n && b == rb_) {
    THZTensor_(resize2d)(rb_, n, nrhs);
  }

  THZTensor_(freeCopyTo)(ra__, ra_);
  THZTensor_(freeCopyTo)(rb__, rb_);
  THZTensor_(free)(work);
  if (free_b) THZTensor_(free)(b);
}

void THZTensor_(geev)(THZTensor *re_, THZTensor *rv_, THZTensor *a_, const char *jobvr)
{
  int n, lda, lwork, info, ldvr;
  THZTensor *work, *wi, *wr, *a;
  ntype wkopt;
  ntype *rv_data;
  int64_t i;

  THZTensor *re__ = NULL;
  THZTensor *rv__ = NULL;

  THArgCheck(a_->nDimension == 2, 1, "A should be 2 dimensional");
  THArgCheck(a_->size[0] == a_->size[1], 1,"A should be square");

  /* we want to definitely clone a_ for geev*/
  a = THZTensor_(cloneColumnMajor)(NULL, a_);

  n = a->size[0];
  lda = n;

  wi = THZTensor_(newWithSize1d)(n);
  wr = THZTensor_(newWithSize1d)(n);

  rv_data = NULL;
  ldvr = 1;
  if (*jobvr == 'V')
  {
    THZTensor_(resize2d)(rv_,n,n);
    /* guard against someone passing a correct size, but wrong stride */
    rv__ = THZTensor_(newTransposedContiguous)(rv_);
    rv_data = THZTensor_(data)(rv__);
    ldvr = n;
  }
  THZTensor_(resize2d)(re_,n,2);
  re__ = THZTensor_(newContiguous)(re_);

  /* get optimal workspace size */
  THZLapack_(geev)('N', jobvr[0], n, THZTensor_(data)(a), lda, THZTensor_(data)(wr), THZTensor_(data)(wi),
      NULL, 1, rv_data, ldvr, &wkopt, -1, &info);

  lwork = (int)wkopt;
  work = THZTensor_(newWithSize1d)(lwork);

  THZLapack_(geev)('N', jobvr[0], n, THZTensor_(data)(a), lda, THZTensor_(data)(wr), THZTensor_(data)(wi),
      NULL, 1, rv_data, ldvr, THZTensor_(data)(work), lwork, &info);

  THLapackCheckWithCleanup(" Lapack Error in %s : %d off-diagonal elements of an didn't converge to zero",
                           THCleanup(THZTensor_(free)(re__);
                                     THZTensor_(free)(rv__);
                                     THZTensor_(free)(a);
                                     THZTensor_(free)(wi);
                                     THZTensor_(free)(wr);
                                     THZTensor_(free)(work);),
                           "geev", info,"");

  {
    ntype *re_data = THZTensor_(data)(re__);
    ntype *wi_data = THZTensor_(data)(wi);
    ntype *wr_data = THZTensor_(data)(wr);
    for (i=0; i<n; i++)
    {
      re_data[2*i] = wr_data[i];
      re_data[2*i+1] = wi_data[i];
    }
  }

  if (*jobvr == 'V')
  {
    THZTensor_(checkTransposed)(rv_);
    THZTensor_(freeCopyTo)(rv__, rv_);
  }
  THZTensor_(freeCopyTo)(re__, re_);
  THZTensor_(free)(a);
  THZTensor_(free)(wi);
  THZTensor_(free)(wr);
  THZTensor_(free)(work);
}

void THZTensor_(syev)(THZTensor *re_, THZTensor *rv_, THZTensor *a, const char *jobz, const char *uplo)
{
  if (a == NULL) a = rv_;
  THArgCheck(a->nDimension == 2, 1, "A should be 2 dimensional");
  THArgCheck(a->size[0] == a->size[1], 1,"A should be square");

  int n, lda, lwork, info;
  THZTensor *work;
  ntype wkopt;

  THZTensor *rv__ = NULL;
  THZTensor *re__ = NULL;

  rv__ = THZTensor_(cloneColumnMajor)(rv_, a);

  n = rv__->size[0];
  lda = n;

  THZTensor_(resize1d)(re_,n);
  re__ = THZTensor_(newContiguous)(re_);

  /* get optimal workspace size */
  THZLapack_(syev)(jobz[0], uplo[0], n, THZTensor_(data)(rv__), lda,
      THZTensor_(data)(re_), &wkopt, -1, &info);
  lwork = (int)wkopt;
  work = THZTensor_(newWithSize1d)(lwork);
  THZLapack_(syev)(jobz[0], uplo[0], n, THZTensor_(data)(rv__), lda,
      THZTensor_(data)(re_), THZTensor_(data)(work), lwork, &info);

  THLapackCheckWithCleanup("Lapack Error %s : %d off-diagonal elements didn't converge to zero",
                           THCleanup(THZTensor_(free)(rv__);
                                     THZTensor_(free)(re__);
                                     THZTensor_(free)(work);),
                           "syev", info,"");

  // No eigenvectors specified
  if (*jobz == 'N') {
    THZTensor_(fill)(rv_, 0);
  }

  THZTensor_(freeCopyTo)(rv__, rv_);
  THZTensor_(freeCopyTo)(re__, re_);
  THZTensor_(free)(work);
}

void THZTensor_(gesvd)(THZTensor *ru_, THZTensor *rs_, THZTensor *rv_, THZTensor *a, const char* jobu)
{
  THZTensor *ra_ = THZTensor_(new)();
  THZTensor_(gesvd2)(ru_, rs_, rv_,  ra_, a, jobu);
  THZTensor_(free)(ra_);
}

void THZTensor_(gesvd2)(THZTensor *ru_, THZTensor *rs_, THZTensor *rv_, THZTensor *ra_, THZTensor *a, const char* jobu)
{
  if (a == NULL) a = ra_;
  THArgCheck(a->nDimension == 2, 1, "A should be 2 dimensional");

  int k,m, n, lda, ldu, ldvt, lwork, info;
  THZTensor *work;
  THZTensor *rvf_ = THZTensor_(new)();
  ntype wkopt;

  THZTensor *ra__ = NULL;
  THZTensor *ru__ = NULL;
  THZTensor *rs__ = NULL;
  THZTensor *rv__ = NULL;

  ra__ = THZTensor_(cloneColumnMajor)(ra_, a);

  m = ra__->size[0];
  n = ra__->size[1];
  k = (m < n ? m : n);

  lda = m;
  ldu = m;
  ldvt = n;

  THZTensor_(resize1d)(rs_,k);
  THZTensor_(resize2d)(rvf_,ldvt,n);
  if (*jobu == 'A')
    THZTensor_(resize2d)(ru_,m,ldu);
  else
    THZTensor_(resize2d)(ru_,k,ldu);

  THZTensor_(checkTransposed)(ru_);

  /* guard against someone passing a correct size, but wrong stride */
  ru__ = THZTensor_(newTransposedContiguous)(ru_);
  rs__ = THZTensor_(newContiguous)(rs_);
  rv__ = THZTensor_(newContiguous)(rvf_);

  THZLapack_(gesvd)(jobu[0],jobu[0],
       m,n,THZTensor_(data)(ra__),lda,
       THZTensor_(data)(rs__),
       THZTensor_(data)(ru__),
       ldu,
       THZTensor_(data)(rv__), ldvt,
       &wkopt, -1, &info);
  lwork = (int)wkopt;
  work = THZTensor_(newWithSize1d)(lwork);
  THZLapack_(gesvd)(jobu[0],jobu[0],
       m,n,THZTensor_(data)(ra__),lda,
       THZTensor_(data)(rs__),
       THZTensor_(data)(ru__),
       ldu,
       THZTensor_(data)(rv__), ldvt,
       THZTensor_(data)(work),lwork, &info);

  THLapackCheckWithCleanup(" Lapack Error %s : %d superdiagonals failed to converge.",
                           THCleanup(
                               THZTensor_(free)(ru__);
                               THZTensor_(free)(rs__);
                               THZTensor_(free)(rv__);
                               THZTensor_(free)(ra__);
                               THZTensor_(free)(work);),
                           "gesvd", info,"");

  if (*jobu == 'S')
    THZTensor_(narrow)(rv__,NULL,1,0,k);

  THZTensor_(freeCopyTo)(ru__, ru_);
  THZTensor_(freeCopyTo)(rs__, rs_);
  THZTensor_(freeCopyTo)(rv__, rvf_);
  THZTensor_(freeCopyTo)(ra__, ra_);
  THZTensor_(free)(work);

  if (*jobu == 'S') {
    THZTensor_(narrow)(rvf_,NULL,1,0,k);
  }
  THZTensor_(resizeAs)(rv_, rvf_);
  THZTensor_(copy)(rv_, rvf_);
  THZTensor_(free)(rvf_);
}

void THZTensor_(getri)(THZTensor *ra_, THZTensor *a)
{
  if (a == NULL) a = ra_;
  THArgCheck(a->nDimension == 2, 1, "A should be 2 dimensional");
  THArgCheck(a->size[0] == a->size[1], 1, "A should be square");

  int m, n, lda, info, lwork;
  ntype wkopt;
  THIntTensor *ipiv;
  THZTensor *work;
  THZTensor *ra__ = NULL;

  ra__ = THZTensor_(cloneColumnMajor)(ra_, a);

  m = ra__->size[0];
  n = ra__->size[1];
  lda = m;
  ipiv = THIntTensor_newWithSize1d((int64_t)m);

  /* Run LU */
  THZLapack_(getrf)(n, n, THZTensor_(data)(ra__), lda, THIntTensor_data(ipiv), &info);
  THLapackCheckWithCleanup("Lapack Error %s : U(%d,%d) is 0, U is singular",
                           THCleanup(
                               THZTensor_(free)(ra__);
                               THIntTensor_free(ipiv);),
                           "getrf", info, info);

  /* Run inverse */
  THZLapack_(getri)(n, THZTensor_(data)(ra__), lda, THIntTensor_data(ipiv), &wkopt, -1, &info);
  lwork = (int)wkopt;
  work = THZTensor_(newWithSize1d)(lwork);
  THZLapack_(getri)(n, THZTensor_(data)(ra__), lda, THIntTensor_data(ipiv), THZTensor_(data)(work), lwork, &info);
  THLapackCheckWithCleanup("Lapack Error %s : U(%d,%d) is 0, U is singular",
                           THCleanup(
                               THZTensor_(free)(ra__);
                               THZTensor_(free)(work);
                               THIntTensor_free(ipiv);),
                           "getri", info, info);

  THZTensor_(freeCopyTo)(ra__, ra_);
  THZTensor_(free)(work);
  THIntTensor_free(ipiv);
}

void THZTensor_(clearUpLoTriangle)(THZTensor *a, const char *uplo)
{
  THArgCheck(a->nDimension == 2, 1, "A should be 2 dimensional");
  THArgCheck(a->size[0] == a->size[1], 1, "A should be square");

  int n = a->size[0];

  /* Build full matrix */
  ntype *p = THZTensor_(data)(a);
  int64_t i, j;

  /* Upper Triangular Case */
  if (uplo[0] == 'U')
  {
    /* Clear lower triangle (excluding diagonals) */
    for (i=0; i<n; i++) {
     for (j=i+1; j<n; j++) {
        p[n*i + j] = 0;
      }
    }
  }
  /* Lower Triangular Case */
  else if (uplo[0] == 'L')
  {
    /* Clear upper triangle (excluding diagonals) */
    for (i=0; i<n; i++) {
      for (j=0; j<i; j++) {
        p[n*i + j] = 0;
      }
    }
  }
}

void THZTensor_(copyUpLoTriangle)(THZTensor *a, const char *uplo)
{
  THArgCheck(a->nDimension == 2, 1, "A should be 2 dimensional");
  THArgCheck(a->size[0] == a->size[1], 1, "A should be square");

  int n = a->size[0];

  /* Build full matrix */
  ntype *p = THZTensor_(data)(a);
  int64_t i, j;

  /* Upper Triangular Case */
  if (uplo[0] == 'U')
  {
    /* Clear lower triangle (excluding diagonals) */
    for (i=0; i<n; i++) {
     for (j=i+1; j<n; j++) {
        p[n*i + j] = p[n*j+i];
      }
    }
  }
  /* Lower Triangular Case */
  else if (uplo[0] == 'L')
  {
    /* Clear upper triangle (excluding diagonals) */
    for (i=0; i<n; i++) {
      for (j=0; j<i; j++) {
        p[n*i + j] = p[n*j+i];
      }
    }
  }
}

void THZTensor_(potrf)(THZTensor *ra_, THZTensor *a, const char *uplo)
{
  if (a == NULL) a = ra_;
  THArgCheck(a->nDimension == 2, 1, "A should be 2 dimensional");
  THArgCheck(a->size[0] == a->size[1], 1, "A should be square");

  int n, lda, info;
  THZTensor *ra__ = NULL;

  ra__ = THZTensor_(cloneColumnMajor)(ra_, a);

  n = ra__->size[0];
  lda = n;

  /* Run Factorization */
  THZLapack_(potrf)(uplo[0], n, THZTensor_(data)(ra__), lda, &info);
  THLapackCheckWithCleanup("Lapack Error in %s : the leading minor of order %d is not positive definite",
                           THCleanup(THZTensor_(free)(ra__);),
                           "potrf", info, "");

  THZTensor_(clearUpLoTriangle)(ra__, uplo);
  THZTensor_(freeCopyTo)(ra__, ra_);
}

void THZTensor_(potrs)(THZTensor *rb_, THZTensor *b, THZTensor *a, const char *uplo)
{
  int free_b = 0;
  if (b == NULL) b = rb_;

  THArgCheck(a->nDimension == 2, 2, "A should have 2 dimensions, but has %d",
      a->nDimension);
  THArgCheck(b->nDimension == 1 || b->nDimension == 2, 1, "B should have 1 or 2 "
      "dimensions, but has %d", b->nDimension);
  THArgCheck(a->size[0] == a->size[1], 2, "A should be square, but is %ldx%ld",
      a->size[0], a->size[1]);
  THArgCheck(a->size[0] == b->size[0], 2, "A,B size incompatible - A has %ld "
      "rows, B has %ld", a->size[0], b->size[0]);

  if (b->nDimension == 1) {
    b = THZTensor_(newWithStorage2d)(b->storage, b->storageOffset, b->size[0],
            b->stride[0], 1, 0);
    free_b = 1;
  }

  int n, nrhs, lda, ldb, info;
  THZTensor *ra__; // working version of A matrix to be passed into lapack TRTRS
  THZTensor *rb__; // working version of B matrix to be passed into lapack TRTRS

  ra__ = THZTensor_(cloneColumnMajor)(NULL, a);
  rb__ = THZTensor_(cloneColumnMajor)(rb_, b);

  n    = (int)ra__->size[0];
  nrhs = (int)rb__->size[1];
  lda  = n;
  ldb  = n;

  THZLapack_(potrs)(uplo[0], n, nrhs, THZTensor_(data)(ra__),
                   lda, THZTensor_(data)(rb__), ldb, &info);


  THLapackCheckWithCleanup("Lapack Error in %s : A(%d,%d) is zero, singular A",
                           THCleanup(
                               THZTensor_(free)(ra__);
                               THZTensor_(free)(rb__);
                               if (free_b) THZTensor_(free)(b);),
                           "potrs", info, info);

  if (free_b) THZTensor_(free)(b);
  THZTensor_(free)(ra__);
  THZTensor_(freeCopyTo)(rb__, rb_);
}

void THZTensor_(potri)(THZTensor *ra_, THZTensor *a, const char *uplo)
{
  if (a == NULL) a = ra_;
  THArgCheck(a->nDimension == 2, 1, "A should be 2 dimensional");
  THArgCheck(a->size[0] == a->size[1], 1, "A should be square");

  int n, lda, info;
  THZTensor *ra__ = NULL;

  ra__ = THZTensor_(cloneColumnMajor)(ra_, a);

  n = ra__->size[0];
  lda = n;

  /* Run inverse */
  THZLapack_(potri)(uplo[0], n, THZTensor_(data)(ra__), lda, &info);
  THLapackCheckWithCleanup("Lapack Error %s : A(%d,%d) is 0, A cannot be factorized",
                           THCleanup(THZTensor_(free)(ra__);),
                           "potri", info, info);

  THZTensor_(copyUpLoTriangle)(ra__, uplo);
  THZTensor_(freeCopyTo)(ra__, ra_);
}

/*
 Computes the Cholesky factorization with complete pivoting of a ntype symmetric
 positive semidefinite matrix.

 Args:
 * `ra_`    - result Tensor in which to store the factor U or L from the
              Cholesky factorization.
 * `rpiv_`  - result IntTensor containing sparse permutation matrix P, encoded
              as P[rpiv_[k], k] = 1.
 * `a`      - input Tensor; the input matrix to factorize.
 * `uplo`   - string; specifies whether the upper or lower triangular part of
              the symmetric matrix A is stored. "U"/"L" for upper/lower
              triangular.
 * `tol`    - double; user defined tolerance, or < 0 for automatic choice.
              The algorithm terminates when the pivot <= tol.
 */
void THZTensor_(pstrf)(THZTensor *ra_, THIntTensor *rpiv_, THZTensor *a, const char *uplo, ntype tol) {
  THArgCheck(a->nDimension == 2, 1, "A should be 2 dimensional");
  THArgCheck(a->size[0] == a->size[1], 1, "A should be square");

  int n = a->size[0];

  THZTensor *ra__ = THZTensor_(cloneColumnMajor)(ra_, a);
  THIntTensor_resize1d(rpiv_, n);

  // Allocate working tensor
  THZTensor *work = THZTensor_(newWithSize1d)(2 * n);

  // Run Cholesky factorization
  int lda = n;
  int rank, info;

  THZLapack_(pstrf)(uplo[0], n, THZTensor_(data)(ra__), lda,
                   THIntTensor_data(rpiv_), &rank, tol,
                   THZTensor_(data)(work), &info);

  THLapackCheckWithCleanup("Lapack Error %s : matrix is rank deficient or not positive semidefinite",
                           THCleanup(
                               THZTensor_(free)(ra__);
                               THZTensor_(free)(work);),
                           "pstrf", info,"");

  THZTensor_(clearUpLoTriangle)(ra__, uplo);

  THZTensor_(freeCopyTo)(ra__, ra_);
  THZTensor_(free)(work);
}

/*
  Perform a QR decomposition of a matrix.

  In LAPACK, two parts of the QR decomposition are implemented as two separate
  functions: geqrf and orgqr. For flexibility and efficiency, these are wrapped
  directly, below - but to make the common usage convenient, we also provide
  this function, which calls them both and returns the results in a more
  intuitive form.

  Args:
  * `rq_` - result Tensor in which to store the Q part of the decomposition.
  * `rr_` - result Tensor in which to store the R part of the decomposition.
  * `a`   - input Tensor; the matrix to decompose.

*/
void THZTensor_(qr)(THZTensor *rq_, THZTensor *rr_, THZTensor *a)
{
  int m = a->size[0];
  int n = a->size[1];
  int k = (m < n ? m : n);
  THZTensor *ra_ = THZTensor_(new)();
  THZTensor *rtau_ = THZTensor_(new)();
  THZTensor *rr__ = THZTensor_(new)();
  THZTensor_(geqrf)(ra_, rtau_, a);
  THZTensor_(resize2d)(rr__, k, ra_->size[1]);
  THZTensor_(narrow)(rr__, ra_, 0, 0, k);
  THZTensor_(triu)(rr_, rr__, 0);
  THZTensor_(resize2d)(rq_, ra_->size[0], k);
  THZTensor_(orgqr)(rq_, ra_, rtau_);
  THZTensor_(narrow)(rq_, rq_, 1, 0, k);
  THZTensor_(free)(ra_);
  THZTensor_(free)(rtau_);
  THZTensor_(free)(rr__);
}

/*
  The geqrf function does the main work of QR-decomposing a matrix.
  However, rather than producing a Q matrix directly, it produces a sequence of
  elementary reflectors which may later be composed to construct Q - for example
  with the orgqr function, below.

  Args:
  * `ra_`   - Result matrix which will contain:
              i)  The elements of R, on and above the diagonal.
              ii) Directions of the reflectors implicitly defining Q.
  * `rtau_` - Result tensor which will contain the magnitudes of the reflectors
              implicitly defining Q.
  * `a`     - Input matrix, to decompose. If NULL, `ra_` is used as input.

  For further details, please see the LAPACK documentation.

*/
void THZTensor_(geqrf)(THZTensor *ra_, THZTensor *rtau_, THZTensor *a)
{
  if (a == NULL) ra_ = a;
  THArgCheck(a->nDimension == 2, 1, "A should be 2 dimensional");

  THZTensor *ra__ = NULL;

  /* Prepare the input for LAPACK, making a copy if necessary. */
  ra__ = THZTensor_(cloneColumnMajor)(ra_, a);

  int m = ra__->size[0];
  int n = ra__->size[1];
  int k = (m < n ? m : n);
  int lda = m;
  THZTensor_(resize1d)(rtau_, k);

  /* Dry-run to query the suggested size of the workspace. */
  int info = 0;
  ntype wkopt = 0;
  THZLapack_(geqrf)(m, n, THZTensor_(data)(ra__), lda,
                   THZTensor_(data)(rtau_),
                   &wkopt, -1, &info);

  /* Allocate the workspace and call LAPACK to do the ntype work. */
  int lwork = (int)wkopt;
  THZTensor *work = THZTensor_(newWithSize1d)(lwork);
  THZLapack_(geqrf)(m, n, THZTensor_(data)(ra__), lda,
                   THZTensor_(data)(rtau_),
                   THZTensor_(data)(work), lwork, &info);

  THLapackCheckWithCleanup("Lapack Error %s : unknown Lapack error. info = %i",
                           THCleanup(
                               THZTensor_(free)(ra__);
                               THZTensor_(free)(work);),
                           "geqrf", info,"");

  THZTensor_(freeCopyTo)(ra__, ra_);
  THZTensor_(free)(work);
}

/*
  The orgqr function allows reconstruction of a matrix Q with orthogonal
  columns, from a sequence of elementary reflectors, such as is produced by the
  geqrf function.

  Args:
  * `ra_` - result Tensor, which will contain the matrix Q.
  * `a`   - input Tensor, which should be a matrix with the directions of the
            elementary reflectors below the diagonal. If NULL, `ra_` is used as
            input.
  * `tau` - input Tensor, containing the magnitudes of the elementary
            reflectors.

  For further details, please see the LAPACK documentation.

*/
void THZTensor_(orgqr)(THZTensor *ra_, THZTensor *a, THZTensor *tau)
{
  if (a == NULL) a = ra_;
  THArgCheck(a->nDimension == 2, 1, "A should be 2 dimensional");

  THZTensor *ra__ = NULL;
  ra__ = THZTensor_(cloneColumnMajor)(ra_, a);

  int m = ra__->size[0];
  int n = ra__->size[1];
  int k = tau->size[0];
  int lda = m;

  /* Dry-run to query the suggested size of the workspace. */
  int info = 0;
  ntype wkopt = 0;
  THZLapack_(orgqr)(m, k, k, THZTensor_(data)(ra__), lda,
                   THZTensor_(data)(tau),
                   &wkopt, -1, &info);

  /* Allocate the workspace and call LAPACK to do the ntype work. */
  int lwork = (int)wkopt;
  THZTensor *work = THZTensor_(newWithSize1d)(lwork);
  THZLapack_(orgqr)(m, k, k, THZTensor_(data)(ra__), lda,
                   THZTensor_(data)(tau),
                   THZTensor_(data)(work), lwork, &info);

  THLapackCheckWithCleanup(" Lapack Error %s : unknown Lapack error. info = %i",
                           THCleanup(
                               THZTensor_(free)(ra__);
                               THZTensor_(free)(work);),
                           "orgqr", info,"");
  THZTensor_(freeCopyTo)(ra__, ra_);
  THZTensor_(free)(work);
}

/*
  The ormqr function multiplies Q with another matrix from a sequence of
  elementary reflectors, such as is produced by the geqrf function.

  Args:
  * `ra_`   - result Tensor, which will contain the matrix Q' c.
  * `a`     - input Tensor, which should be a matrix with the directions of the
              elementary reflectors below the diagonal. If NULL, `ra_` is used as
              input.
  * `tau`   - input Tensor, containing the magnitudes of the elementary
              reflectors.
  * `c`     - input Tensor, containing the matrix to be multiplied.
  * `side`  - char, determining whether c is left- or right-multiplied with Q.
  * `trans` - char, determining whether to transpose Q before multiplying.

  For further details, please see the LAPACK documentation.

*/
void THZTensor_(ormqr)(THZTensor *ra_, THZTensor *a, THZTensor *tau, THZTensor *c, const char *side, const char *trans)
{
  if (a == NULL) a = ra_;
  THArgCheck(a->nDimension == 2, 1, "A should be 2 dimensional");

  THZTensor *ra__ = NULL;
  ra__ = THZTensor_(cloneColumnMajor)(ra_, c);

  int m = c->size[0];
  int n = c->size[1];
  int k = tau->size[0];
  int lda;
  if (*side == 'L')
  {
    lda = m;
  }
  else
  {
    lda = n;
  }
  int ldc = m;

  /* Dry-run to query the suggested size of the workspace. */
  int info = 0;
  ntype wkopt = 0;
  THZLapack_(ormqr)(side[0], trans[0], m, n, k, THZTensor_(data)(a), lda,
                   THZTensor_(data)(tau), THZTensor_(data)(ra__), ldc,
                   &wkopt, -1, &info);

  /* Allocate the workspace and call LAPACK to do the ntype work. */
  int lwork = (int)wkopt;
  THZTensor *work = THZTensor_(newWithSize1d)(lwork);
  THZLapack_(ormqr)(side[0], trans[0], m, n, k, THZTensor_(data)(a), lda,
                   THZTensor_(data)(tau), THZTensor_(data)(ra__), ldc,
                   THZTensor_(data)(work), lwork, &info);

  THLapackCheckWithCleanup(" Lapack Error %s : unknown Lapack error. info = %i",
                           THCleanup(
                               THZTensor_(free)(ra__);
                               THZTensor_(free)(work);),
                           "ormqr", info,"");
  THZTensor_(freeCopyTo)(ra__, ra_);
  THZTensor_(free)(work);
}

void THZTensor_(btrifact)(THZTensor *ra_, THIntTensor *rpivots_, THIntTensor *rinfo_, int pivot, THZTensor *a)
{
  THArgCheck(THZTensor_(nDimension)(a) == 3, 1, "expected 3D tensor, got %dD", THZTensor_(nDimension)(a));
  if (!pivot) {
    THError("btrifact without pivoting is not implemented on the CPU");
  }

  if (ra_ != a) {
    THZTensor_(resizeAs)(ra_, a);
    THZTensor_(copy)(ra_, a);
  }

  int m = a->size[1];
  int n = a->size[2];
  if (m != n) {
    THError("btrifact is only implemented for square matrices");
  }
  int64_t num_batches = THZTensor_(size)(a, 0);
  THZTensor *ra__;
  int lda;

  if (ra_->stride[1] == 1) {
    // column ordered, what BLAS wants
    lda = ra_->stride[2];
    ra__ = ra_;
  } else {
    // not column ordered, need to make it such (requires copy)
    THZTensor *transp_r_ = THZTensor_(newTranspose)(ra_, 1, 2);
    ra__ = THZTensor_(newClone)(transp_r_);
    THZTensor_(free)(transp_r_);
    THZTensor_(transpose)(ra__, NULL, 1, 2);
    lda = ra__->stride[2];
  }

  THZTensor *ai = THZTensor_(new)();
  THZTensor *rai = THZTensor_(new)();
  THIntTensor *rpivoti = THIntTensor_new();

  int info = 0;
  int *info_ptr = &info;
  if (rinfo_) {
    THIntTensor_resize1d(rinfo_, num_batches);
    info_ptr = THIntTensor_data(rinfo_);
  }

  THIntTensor_resize2d(rpivots_, num_batches, n);

  int64_t batch = 0;
  for (; batch < num_batches; ++batch) {
    THZTensor_(select)(ai, a, 0, batch);
    THZTensor_(select)(rai, ra__, 0, batch);
    THIntTensor_select(rpivoti, rpivots_, 0, batch);

    THZLapack_(getrf)(n, n, THZTensor_(data)(rai), lda,
                     THIntTensor_data(rpivoti), info_ptr);
    if (rinfo_) {
      info_ptr++;
    } else if (info != 0) {
      break;
    }
  }

  THZTensor_(free)(ai);
  THZTensor_(free)(rai);
  THIntTensor_free(rpivoti);

  if (ra__ != ra_) {
    THZTensor_(freeCopyTo)(ra__, ra_);
  }

  if (!rinfo_ && info != 0) {
    THError("failed to factorize batch element %ld (info == %d)", batch, info);
  }
}

void THZTensor_(btrisolve)(THZTensor *rb_, THZTensor *b, THZTensor *atf, THIntTensor *pivots)
{
  THArgCheck(THZTensor_(nDimension)(atf) == 3, 1, "expected 3D tensor, got %dD",
             THZTensor_(nDimension)(atf));
  THArgCheck(THZTensor_(nDimension)(b) == 3 ||
             THZTensor_(nDimension)(b) == 2, 4, "expected 2D or 3D tensor");
  THArgCheck(THZTensor_(size)(atf, 0) ==
             THZTensor_(size)(b, 0), 3, "number of batches must be equal");
  THArgCheck(THZTensor_(size)(atf, 1) ==
             THZTensor_(size)(atf, 2), 3, "A matrices must be square");
  THArgCheck(THZTensor_(size)(atf, 1) ==
             THZTensor_(size)(b, 1), 3, "dimensions of A and b must be equal");

  if (rb_ != b) {
    THZTensor_(resizeAs)(rb_, b);
    THZTensor_(copy)(rb_, b);
  }

  int64_t num_batches = atf->size[0];
  int64_t n = atf->size[1];
  int nrhs = rb_->nDimension > 2 ? rb_->size[2] : 1;

  int lda, ldb;
  THZTensor *atf_;
  THZTensor *rb__;

  // correct ordering of A
  if (atf->stride[1] == 1) {
    // column ordered, what BLAS wants
    lda = atf->stride[2];
    atf_ = atf;
  } else {
    // not column ordered, need to make it such (requires copy)
    // it would be nice if we could use the op(A) flags to automatically
    // transpose A if needed, but this leads to unpredictable behavior if the
    // user clones A_tf later with a different ordering
    THZTensor *transp_r_ = THZTensor_(newTranspose)(atf, 1, 2);
    atf_ = THZTensor_(newClone)(transp_r_);
    THZTensor_(free)(transp_r_);
    THZTensor_(transpose)(atf_, NULL, 1, 2);
    lda = atf_->stride[2];
  }

  // correct ordering of B
  if (rb_->stride[1] == 1) {
    // column ordered
    if (rb_->nDimension == 2 || rb_->size[2] == 1) {
      ldb = n;
    } else {
      ldb = rb_->stride[2];
    }
    rb__ = rb_;
  } else {
    // make column ordered
    if (rb_->nDimension > 2) {
      THZTensor *transp_r_ = THZTensor_(newTranspose)(rb_, 1, 2);
      rb__ = THZTensor_(newClone)(transp_r_);
      THZTensor_(free)(transp_r_);
      THZTensor_(transpose)(rb__, NULL, 1, 2);
      ldb = rb__->stride[2];
    } else {
      rb__ = THZTensor_(newClone)(rb_);
      ldb = n;
    }
  }

  THZTensor *ai = THZTensor_(new)();
  THZTensor *rbi = THZTensor_(new)();
  THIntTensor *pivoti = THIntTensor_new();

  if (!THIntTensor_isContiguous(pivots)) {
      THError("Error: rpivots_ is not contiguous.");
  }

  for (int64_t batch = 0; batch < num_batches; ++batch) {
    THZTensor_(select)(ai, atf_, 0, batch);
    THZTensor_(select)(rbi, rb__, 0, batch);
    THIntTensor_select(pivoti, pivots, 0, batch);

#if defined(THZ_NTYPE_IS_REAL) || defined(THZ_NTYPE_IS_COMPLEX)
    int info;
    THZLapack_(getrs)('N', n, nrhs, THZTensor_(data)(ai), lda,
                     THIntTensor_data(pivoti), THZTensor_(data)(rbi),
                     ldb, &info);
    if (info != 0) {
      THError("Error: Nonzero info.");
    }
#else
    THError("Unimplemented");
#endif
  }

  THZTensor_(free)(ai);
  THZTensor_(free)(rbi);
  THIntTensor_free(pivoti);

  if (atf_ != atf) {
    THZTensor_(free)(atf_);
  }

  if (rb__ != rb_) {
    THZTensor_(freeCopyTo)(rb__, rb_);
  }
}

#endif
