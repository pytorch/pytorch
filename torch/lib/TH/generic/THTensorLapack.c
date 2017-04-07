#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/THTensorLapack.c"
#else

/*
Check if self is transpose of a contiguous matrix
*/
static int THTensor_(isTransposedContiguous)(THTensor *self)
{
  return self->stride[0] == 1 && self->stride[1] == self->size[0];
}
/*
If a matrix is a regular contiguous matrix, make sure it is transposed
because this is what we return from Lapack calls.
*/
static void THTensor_(checkTransposed)(THTensor *self)
{
  if(THTensor_(isContiguous)(self))
    THTensor_(transpose)(self, NULL, 0, 1);
  return;
}
/*
newContiguous followed by transpose
Similar to (newContiguous), but checks if the transpose of the matrix
is contiguous and also limited to 2D matrices.
*/
static THTensor *THTensor_(newTransposedContiguous)(THTensor *self)
{
  THTensor *tensor;
  if(THTensor_(isTransposedContiguous)(self))
  {
    THTensor_(retain)(self);
    tensor = self;
  }
  else
  {
    tensor = THTensor_(newContiguous)(self);
    THTensor_(transpose)(tensor, NULL, 0, 1);
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
static THTensor *THTensor_(checkLapackClone)(THTensor *result, THTensor *src, int nrows)
{
  /* check if user wants to reuse src and if it is correct shape/size */
  if (src == result && THTensor_(isTransposedContiguous)(src) && src->size[1] == nrows)
    THTensor_(retain)(result);
  else if(src == result || result == NULL) /* in this case, user wants reuse of src, but its structure is not OK */
    result = THTensor_(new)();
  else
    THTensor_(retain)(result);
  return result;
}

/*
Same as cloneColumnMajor, but accepts nrows argument, because some lapack calls require
the resulting tensor to be larger than src.
*/
static THTensor *THTensor_(cloneColumnMajorNrows)(THTensor *self, THTensor *src, int nrows)
{
  THTensor *result;
  THTensor *view;

  if (src == NULL)
    src = self;
  result = THTensor_(checkLapackClone)(self, src, nrows);
  if (src == result)
    return result;

  THTensor_(resize2d)(result, src->size[1], nrows);
  THTensor_(checkTransposed)(result);

  if (src->size[0] == nrows)
    THTensor_(copy)(result, src);
  else
  {
    view = THTensor_(newNarrow)(result, 0, 0, src->size[0]);
    THTensor_(copy)(view, src);
    THTensor_(free)(view);
  }
  return result;
}

/*
Create a clone of src in self column major order for use with Lapack.
If src == self, a new tensor is allocated, in any case, the return tensor should be
freed by calling function.
*/
static THTensor *THTensor_(cloneColumnMajor)(THTensor *self, THTensor *src)
{
  return THTensor_(cloneColumnMajorNrows)(self, src, src->size[0]);
}

void THTensor_(gesv)(THTensor *rb_, THTensor *ra_, THTensor *b, THTensor *a)
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
    b = THTensor_(newWithStorage2d)(b->storage, b->storageOffset, b->size[0],
            b->stride[0], 1, 0);
    free_b = 1;
  }

  int n, nrhs, lda, ldb, info;
  THIntTensor *ipiv;
  THTensor *ra__;  // working version of A matrix to be passed into lapack GELS
  THTensor *rb__;  // working version of B matrix to be passed into lapack GELS

  ra__ = THTensor_(cloneColumnMajor)(ra_, a);
  rb__ = THTensor_(cloneColumnMajor)(rb_, b);

  n    = (int)ra__->size[0];
  nrhs = (int)rb__->size[1];
  lda  = n;
  ldb  = n;

  ipiv = THIntTensor_newWithSize1d((long)n);
  THLapack_(gesv)(n, nrhs,
		  THTensor_(data)(ra__), lda, THIntTensor_data(ipiv),
		  THTensor_(data)(rb__), ldb, &info);

  THLapackCheckWithCleanup("Lapack Error in %s : U(%d,%d) is zero, singular U.",
                           THCleanup(
                               THTensor_(free)(ra__);
                               THTensor_(free)(rb__);
                               THIntTensor_free(ipiv);
                               if (free_b) THTensor_(free)(b);),
                           "gesv", info, info);

  THTensor_(freeCopyTo)(ra__, ra_);
  THTensor_(freeCopyTo)(rb__, rb_);
  THIntTensor_free(ipiv);
  if (free_b) THTensor_(free)(b);
}

void THTensor_(trtrs)(THTensor *rb_, THTensor *ra_, THTensor *b, THTensor *a,
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
    b = THTensor_(newWithStorage2d)(b->storage, b->storageOffset, b->size[0],
            b->stride[0], 1, 0);
    free_b = 1;
  }

  int n, nrhs, lda, ldb, info;
  THTensor *ra__; // working version of A matrix to be passed into lapack TRTRS
  THTensor *rb__; // working version of B matrix to be passed into lapack TRTRS

  ra__ = THTensor_(cloneColumnMajor)(ra_, a);
  rb__ = THTensor_(cloneColumnMajor)(rb_, b);

  n    = (int)ra__->size[0];
  nrhs = (int)rb__->size[1];
  lda  = n;
  ldb  = n;

  THLapack_(trtrs)(uplo[0], trans[0], diag[0], n, nrhs,
                   THTensor_(data)(ra__), lda,
                   THTensor_(data)(rb__), ldb, &info);


  THLapackCheckWithCleanup("Lapack Error in %s : A(%d,%d) is zero, singular A",
                           THCleanup(
                              THTensor_(free)(ra__);
                              THTensor_(free)(rb__);
                              if (free_b) THTensor_(free)(b);),
                           "trtrs", info, info);

  THTensor_(freeCopyTo)(ra__, ra_);
  THTensor_(freeCopyTo)(rb__, rb_);
  if (free_b) THTensor_(free)(b);
}

void THTensor_(gels)(THTensor *rb_, THTensor *ra_, THTensor *b, THTensor *a)
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
    b = THTensor_(newWithStorage2d)(b->storage, b->storageOffset, b->size[0],
            b->stride[0], 1, 0);
    free_b = 1;
  }

  int m, n, nrhs, lda, ldb, info, lwork;
  THTensor *work = NULL;
  real wkopt = 0;

  THTensor *ra__ = NULL;  // working version of A matrix to be passed into lapack GELS
  THTensor *rb__ = NULL;  // working version of B matrix to be passed into lapack GELS

  ra__ = THTensor_(cloneColumnMajor)(ra_, a);

  m = ra__->size[0];
  n = ra__->size[1];
  lda = m;
  ldb = (m > n) ? m : n;

  rb__ = THTensor_(cloneColumnMajorNrows)(rb_, b, ldb);

  nrhs = rb__->size[1];
  info = 0;


  /* get optimal workspace size */
  THLapack_(gels)('N', m, n, nrhs, THTensor_(data)(ra__), lda,
		  THTensor_(data)(rb__), ldb,
		  &wkopt, -1, &info);
  lwork = (int)wkopt;
  work = THTensor_(newWithSize1d)(lwork);
  THLapack_(gels)('N', m, n, nrhs, THTensor_(data)(ra__), lda,
		  THTensor_(data)(rb__), ldb,
		  THTensor_(data)(work), lwork, &info);

  THLapackCheckWithCleanup("Lapack Error in %s : The %d-th diagonal element of the triangular factor of A is zero",
                           THCleanup(THTensor_(free)(ra__);
                                     THTensor_(free)(rb__);
                                     THTensor_(free)(work);
                                     if (free_b) THTensor_(free)(b);),
                           "gels", info,"");

  /* rb__ is currently ldb by nrhs; resize it to n by nrhs */
  rb__->size[0] = n;
  if (rb__ != rb_)
    THTensor_(resize2d)(rb_, n, nrhs);

  THTensor_(freeCopyTo)(ra__, ra_);
  THTensor_(freeCopyTo)(rb__, rb_);
  THTensor_(free)(work);
  if (free_b) THTensor_(free)(b);
}

void THTensor_(geev)(THTensor *re_, THTensor *rv_, THTensor *a_, const char *jobvr)
{
  int n, lda, lwork, info, ldvr;
  THTensor *work, *wi, *wr, *a;
  real wkopt;
  real *rv_data;
  long i;

  THTensor *re__ = NULL;
  THTensor *rv__ = NULL;

  THArgCheck(a_->nDimension == 2, 1, "A should be 2 dimensional");
  THArgCheck(a_->size[0] == a_->size[1], 1,"A should be square");

  /* we want to definitely clone a_ for geev*/
  a = THTensor_(cloneColumnMajor)(NULL, a_);

  n = a->size[0];
  lda = n;

  wi = THTensor_(newWithSize1d)(n);
  wr = THTensor_(newWithSize1d)(n);

  rv_data = NULL;
  ldvr = 1;
  if (*jobvr == 'V')
  {
    THTensor_(resize2d)(rv_,n,n);
    /* guard against someone passing a correct size, but wrong stride */
    rv__ = THTensor_(newTransposedContiguous)(rv_);
    rv_data = THTensor_(data)(rv__);
    ldvr = n;
  }
  THTensor_(resize2d)(re_,n,2);
  re__ = THTensor_(newContiguous)(re_);

  /* get optimal workspace size */
  THLapack_(geev)('N', jobvr[0], n, THTensor_(data)(a), lda, THTensor_(data)(wr), THTensor_(data)(wi),
      NULL, 1, rv_data, ldvr, &wkopt, -1, &info);

  lwork = (int)wkopt;
  work = THTensor_(newWithSize1d)(lwork);

  THLapack_(geev)('N', jobvr[0], n, THTensor_(data)(a), lda, THTensor_(data)(wr), THTensor_(data)(wi),
      NULL, 1, rv_data, ldvr, THTensor_(data)(work), lwork, &info);

  THLapackCheckWithCleanup(" Lapack Error in %s : %d off-diagonal elements of an didn't converge to zero",
                           THCleanup(THTensor_(free)(re__);
                                     THTensor_(free)(rv__);
                                     THTensor_(free)(a);
                                     THTensor_(free)(wi);
                                     THTensor_(free)(wr);
                                     THTensor_(free)(work);),
                           "geev", info,"");

  {
    real *re_data = THTensor_(data)(re__);
    real *wi_data = THTensor_(data)(wi);
    real *wr_data = THTensor_(data)(wr);
    for (i=0; i<n; i++)
    {
      re_data[2*i] = wr_data[i];
      re_data[2*i+1] = wi_data[i];
    }
  }

  if (*jobvr == 'V')
  {
    THTensor_(checkTransposed)(rv_);
    THTensor_(freeCopyTo)(rv__, rv_);
  }
  THTensor_(freeCopyTo)(re__, re_);
  THTensor_(free)(a);
  THTensor_(free)(wi);
  THTensor_(free)(wr);
  THTensor_(free)(work);
}

void THTensor_(syev)(THTensor *re_, THTensor *rv_, THTensor *a, const char *jobz, const char *uplo)
{
  if (a == NULL) a = rv_;
  THArgCheck(a->nDimension == 2, 1, "A should be 2 dimensional");
  THArgCheck(a->size[0] == a->size[1], 1,"A should be square");

  int n, lda, lwork, info;
  THTensor *work;
  real wkopt;

  THTensor *rv__ = NULL;
  THTensor *re__ = NULL;

  rv__ = THTensor_(cloneColumnMajor)(rv_, a);

  n = rv__->size[0];
  lda = n;

  THTensor_(resize1d)(re_,n);
  re__ = THTensor_(newContiguous)(re_);

  /* get optimal workspace size */
  THLapack_(syev)(jobz[0], uplo[0], n, THTensor_(data)(rv__), lda,
		  THTensor_(data)(re_), &wkopt, -1, &info);
  lwork = (int)wkopt;
  work = THTensor_(newWithSize1d)(lwork);
  THLapack_(syev)(jobz[0], uplo[0], n, THTensor_(data)(rv__), lda,
		  THTensor_(data)(re_), THTensor_(data)(work), lwork, &info);

  THLapackCheckWithCleanup("Lapack Error %s : %d off-diagonal elements didn't converge to zero",
                           THCleanup(THTensor_(free)(rv__);
                                     THTensor_(free)(re__);
                                     THTensor_(free)(work);),
                           "syev", info,"");

  THTensor_(freeCopyTo)(rv__, rv_);
  THTensor_(freeCopyTo)(re__, re_);
  THTensor_(free)(work);
}

void THTensor_(gesvd)(THTensor *ru_, THTensor *rs_, THTensor *rv_, THTensor *a, const char* jobu)
{
  THTensor *ra_ = THTensor_(new)();
  THTensor_(gesvd2)(ru_, rs_, rv_,  ra_, a, jobu);
  THTensor_(free)(ra_);
}

void THTensor_(gesvd2)(THTensor *ru_, THTensor *rs_, THTensor *rv_, THTensor *ra_, THTensor *a, const char* jobu)
{
  if (a == NULL) a = ra_;
  THArgCheck(a->nDimension == 2, 1, "A should be 2 dimensional");

  int k,m, n, lda, ldu, ldvt, lwork, info;
  THTensor *work;
  THTensor *rvf_ = THTensor_(new)();
  real wkopt;

  THTensor *ra__ = NULL;
  THTensor *ru__ = NULL;
  THTensor *rs__ = NULL;
  THTensor *rv__ = NULL;

  ra__ = THTensor_(cloneColumnMajor)(ra_, a);

  m = ra__->size[0];
  n = ra__->size[1];
  k = (m < n ? m : n);

  lda = m;
  ldu = m;
  ldvt = n;

  THTensor_(resize1d)(rs_,k);
  THTensor_(resize2d)(rvf_,ldvt,n);
  if (*jobu == 'A')
    THTensor_(resize2d)(ru_,m,ldu);
  else
    THTensor_(resize2d)(ru_,k,ldu);

  THTensor_(checkTransposed)(ru_);

  /* guard against someone passing a correct size, but wrong stride */
  ru__ = THTensor_(newTransposedContiguous)(ru_);
  rs__ = THTensor_(newContiguous)(rs_);
  rv__ = THTensor_(newContiguous)(rvf_);

  THLapack_(gesvd)(jobu[0],jobu[0],
		   m,n,THTensor_(data)(ra__),lda,
		   THTensor_(data)(rs__),
		   THTensor_(data)(ru__),
		   ldu,
		   THTensor_(data)(rv__), ldvt,
		   &wkopt, -1, &info);
  lwork = (int)wkopt;
  work = THTensor_(newWithSize1d)(lwork);
  THLapack_(gesvd)(jobu[0],jobu[0],
		   m,n,THTensor_(data)(ra__),lda,
		   THTensor_(data)(rs__),
		   THTensor_(data)(ru__),
		   ldu,
		   THTensor_(data)(rv__), ldvt,
		   THTensor_(data)(work),lwork, &info);

  THLapackCheckWithCleanup(" Lapack Error %s : %d superdiagonals failed to converge.",
                           THCleanup(
                               THTensor_(free)(ru__);
                               THTensor_(free)(rs__);
                               THTensor_(free)(rv__);
                               THTensor_(free)(ra__);
                               THTensor_(free)(work);),
                           "gesvd", info,"");

  if (*jobu == 'S')
    THTensor_(narrow)(rv__,NULL,1,0,k);

  THTensor_(freeCopyTo)(ru__, ru_);
  THTensor_(freeCopyTo)(rs__, rs_);
  THTensor_(freeCopyTo)(rv__, rvf_);
  THTensor_(freeCopyTo)(ra__, ra_);
  THTensor_(free)(work);

  if (*jobu == 'S') {
    THTensor_(narrow)(rvf_,NULL,1,0,k);
  }
  THTensor_(resizeAs)(rv_, rvf_);
  THTensor_(copy)(rv_, rvf_);
  THTensor_(free)(rvf_);
}

void THTensor_(getri)(THTensor *ra_, THTensor *a)
{
  if (a == NULL) a = ra_;
  THArgCheck(a->nDimension == 2, 1, "A should be 2 dimensional");
  THArgCheck(a->size[0] == a->size[1], 1, "A should be square");

  int m, n, lda, info, lwork;
  real wkopt;
  THIntTensor *ipiv;
  THTensor *work;
  THTensor *ra__ = NULL;

  ra__ = THTensor_(cloneColumnMajor)(ra_, a);

  m = ra__->size[0];
  n = ra__->size[1];
  lda = m;
  ipiv = THIntTensor_newWithSize1d((long)m);

  /* Run LU */
  THLapack_(getrf)(n, n, THTensor_(data)(ra__), lda, THIntTensor_data(ipiv), &info);
  THLapackCheckWithCleanup("Lapack Error %s : U(%d,%d) is 0, U is singular",
                           THCleanup(
                               THTensor_(free)(ra__);
                               THIntTensor_free(ipiv);),
                           "getrf", info, info);

  /* Run inverse */
  THLapack_(getri)(n, THTensor_(data)(ra__), lda, THIntTensor_data(ipiv), &wkopt, -1, &info);
  lwork = (int)wkopt;
  work = THTensor_(newWithSize1d)(lwork);
  THLapack_(getri)(n, THTensor_(data)(ra__), lda, THIntTensor_data(ipiv), THTensor_(data)(work), lwork, &info);
  THLapackCheckWithCleanup("Lapack Error %s : U(%d,%d) is 0, U is singular",
                           THCleanup(
                               THTensor_(free)(ra__);
                               THTensor_(free)(work);
                               THIntTensor_free(ipiv);),
                           "getri", info, info);

  THTensor_(freeCopyTo)(ra__, ra_);
  THTensor_(free)(work);
  THIntTensor_free(ipiv);
}

void THTensor_(clearUpLoTriangle)(THTensor *a, const char *uplo)
{
  THArgCheck(a->nDimension == 2, 1, "A should be 2 dimensional");
  THArgCheck(a->size[0] == a->size[1], 1, "A should be square");

  int n = a->size[0];

  /* Build full matrix */
  real *p = THTensor_(data)(a);
  long i, j;

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

void THTensor_(copyUpLoTriangle)(THTensor *a, const char *uplo)
{
  THArgCheck(a->nDimension == 2, 1, "A should be 2 dimensional");
  THArgCheck(a->size[0] == a->size[1], 1, "A should be square");

  int n = a->size[0];

  /* Build full matrix */
  real *p = THTensor_(data)(a);
  long i, j;

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

void THTensor_(potrf)(THTensor *ra_, THTensor *a, const char *uplo)
{
  if (a == NULL) a = ra_;
  THArgCheck(a->nDimension == 2, 1, "A should be 2 dimensional");
  THArgCheck(a->size[0] == a->size[1], 1, "A should be square");

  int n, lda, info;
  THTensor *ra__ = NULL;

  ra__ = THTensor_(cloneColumnMajor)(ra_, a);

  n = ra__->size[0];
  lda = n;

  /* Run Factorization */
  THLapack_(potrf)(uplo[0], n, THTensor_(data)(ra__), lda, &info);
  THLapackCheckWithCleanup("Lapack Error in %s : the leading minor of order %d is not positive definite",
                           THCleanup(THTensor_(free)(ra__);),
                           "potrf", info, "");

  THTensor_(clearUpLoTriangle)(ra__, uplo);
  THTensor_(freeCopyTo)(ra__, ra_);
}

void THTensor_(potrs)(THTensor *rb_, THTensor *b, THTensor *a, const char *uplo)
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
    b = THTensor_(newWithStorage2d)(b->storage, b->storageOffset, b->size[0],
            b->stride[0], 1, 0);
    free_b = 1;
  }

  int n, nrhs, lda, ldb, info;
  THTensor *ra__; // working version of A matrix to be passed into lapack TRTRS
  THTensor *rb__; // working version of B matrix to be passed into lapack TRTRS

  ra__ = THTensor_(cloneColumnMajor)(NULL, a);
  rb__ = THTensor_(cloneColumnMajor)(rb_, b);

  n    = (int)ra__->size[0];
  nrhs = (int)rb__->size[1];
  lda  = n;
  ldb  = n;

  THLapack_(potrs)(uplo[0], n, nrhs, THTensor_(data)(ra__),
                   lda, THTensor_(data)(rb__), ldb, &info);


  THLapackCheckWithCleanup("Lapack Error in %s : A(%d,%d) is zero, singular A",
                           THCleanup(
                               THTensor_(free)(ra__);
                               THTensor_(free)(rb__);
                               if (free_b) THTensor_(free)(b);),
                           "potrs", info, info);

  if (free_b) THTensor_(free)(b);
  THTensor_(free)(ra__);
  THTensor_(freeCopyTo)(rb__, rb_);
}

void THTensor_(potri)(THTensor *ra_, THTensor *a, const char *uplo)
{
  if (a == NULL) a = ra_;
  THArgCheck(a->nDimension == 2, 1, "A should be 2 dimensional");
  THArgCheck(a->size[0] == a->size[1], 1, "A should be square");

  int n, lda, info;
  THTensor *ra__ = NULL;

  ra__ = THTensor_(cloneColumnMajor)(ra_, a);

  n = ra__->size[0];
  lda = n;

  /* Run inverse */
  THLapack_(potri)(uplo[0], n, THTensor_(data)(ra__), lda, &info);
  THLapackCheckWithCleanup("Lapack Error %s : A(%d,%d) is 0, A cannot be factorized",
                           THCleanup(THTensor_(free)(ra__);),
                           "potri", info, info);

  THTensor_(copyUpLoTriangle)(ra__, uplo);
  THTensor_(freeCopyTo)(ra__, ra_);
}

/*
 Computes the Cholesky factorization with complete pivoting of a real symmetric
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
void THTensor_(pstrf)(THTensor *ra_, THIntTensor *rpiv_, THTensor *a, const char *uplo, real tol) {
  THArgCheck(a->nDimension == 2, 1, "A should be 2 dimensional");
  THArgCheck(a->size[0] == a->size[1], 1, "A should be square");

  int n = a->size[0];

  THTensor *ra__ = THTensor_(cloneColumnMajor)(ra_, a);
  THIntTensor_resize1d(rpiv_, n);

  // Allocate working tensor
  THTensor *work = THTensor_(newWithSize1d)(2 * n);

  // Run Cholesky factorization
  int lda = n;
  int rank, info;

  THLapack_(pstrf)(uplo[0], n, THTensor_(data)(ra__), lda,
                   THIntTensor_data(rpiv_), &rank, tol,
                   THTensor_(data)(work), &info);

  THLapackCheckWithCleanup("Lapack Error %s : matrix is rank deficient or not positive semidefinite",
                           THCleanup(
                               THTensor_(free)(ra__);
                               THTensor_(free)(work);),
                           "pstrf", info,"");

  THTensor_(clearUpLoTriangle)(ra__, uplo);

  THTensor_(freeCopyTo)(ra__, ra_);
  THTensor_(free)(work);
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
void THTensor_(qr)(THTensor *rq_, THTensor *rr_, THTensor *a)
{
  int m = a->size[0];
  int n = a->size[1];
  int k = (m < n ? m : n);
  THTensor *ra_ = THTensor_(new)();
  THTensor *rtau_ = THTensor_(new)();
  THTensor *rr__ = THTensor_(new)();
  THTensor_(geqrf)(ra_, rtau_, a);
  THTensor_(resize2d)(rr__, k, ra_->size[1]);
  THTensor_(narrow)(rr__, ra_, 0, 0, k);
  THTensor_(triu)(rr_, rr__, 0);
  THTensor_(resize2d)(rq_, ra_->size[0], k);
  THTensor_(orgqr)(rq_, ra_, rtau_);
  THTensor_(narrow)(rq_, rq_, 1, 0, k);
  THTensor_(free)(ra_);
  THTensor_(free)(rtau_);
  THTensor_(free)(rr__);
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
void THTensor_(geqrf)(THTensor *ra_, THTensor *rtau_, THTensor *a)
{
  if (a == NULL) ra_ = a;
  THArgCheck(a->nDimension == 2, 1, "A should be 2 dimensional");

  THTensor *ra__ = NULL;

  /* Prepare the input for LAPACK, making a copy if necessary. */
  ra__ = THTensor_(cloneColumnMajor)(ra_, a);

  int m = ra__->size[0];
  int n = ra__->size[1];
  int k = (m < n ? m : n);
  int lda = m;
  THTensor_(resize1d)(rtau_, k);

  /* Dry-run to query the suggested size of the workspace. */
  int info = 0;
  real wkopt = 0;
  THLapack_(geqrf)(m, n, THTensor_(data)(ra__), lda,
                   THTensor_(data)(rtau_),
                   &wkopt, -1, &info);

  /* Allocate the workspace and call LAPACK to do the real work. */
  int lwork = (int)wkopt;
  THTensor *work = THTensor_(newWithSize1d)(lwork);
  THLapack_(geqrf)(m, n, THTensor_(data)(ra__), lda,
                   THTensor_(data)(rtau_),
                   THTensor_(data)(work), lwork, &info);

  THLapackCheckWithCleanup("Lapack Error %s : unknown Lapack error. info = %i",
                           THCleanup(
                               THTensor_(free)(ra__);
                               THTensor_(free)(work);),
                           "geqrf", info,"");

  THTensor_(freeCopyTo)(ra__, ra_);
  THTensor_(free)(work);
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
void THTensor_(orgqr)(THTensor *ra_, THTensor *a, THTensor *tau)
{
  if (a == NULL) a = ra_;
  THArgCheck(a->nDimension == 2, 1, "A should be 2 dimensional");

  THTensor *ra__ = NULL;
  ra__ = THTensor_(cloneColumnMajor)(ra_, a);

  int m = ra__->size[0];
  int n = ra__->size[1];
  int k = tau->size[0];
  int lda = m;

  /* Dry-run to query the suggested size of the workspace. */
  int info = 0;
  real wkopt = 0;
  THLapack_(orgqr)(m, k, k, THTensor_(data)(ra__), lda,
                   THTensor_(data)(tau),
                   &wkopt, -1, &info);

  /* Allocate the workspace and call LAPACK to do the real work. */
  int lwork = (int)wkopt;
  THTensor *work = THTensor_(newWithSize1d)(lwork);
  THLapack_(orgqr)(m, k, k, THTensor_(data)(ra__), lda,
                   THTensor_(data)(tau),
                   THTensor_(data)(work), lwork, &info);

  THLapackCheckWithCleanup(" Lapack Error %s : unknown Lapack error. info = %i",
                           THCleanup(
                               THTensor_(free)(ra__);
                               THTensor_(free)(work);),
                           "orgqr", info,"");
  THTensor_(freeCopyTo)(ra__, ra_);
  THTensor_(free)(work);
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
void THTensor_(ormqr)(THTensor *ra_, THTensor *a, THTensor *tau, THTensor *c, const char *side, const char *trans)
{
  if (a == NULL) a = ra_;
  THArgCheck(a->nDimension == 2, 1, "A should be 2 dimensional");

  THTensor *ra__ = NULL;
  ra__ = THTensor_(cloneColumnMajor)(ra_, c);

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
  real wkopt = 0;
  THLapack_(ormqr)(side[0], trans[0], m, n, k, THTensor_(data)(a), lda,
                   THTensor_(data)(tau), THTensor_(data)(ra__), ldc,
                   &wkopt, -1, &info);

  /* Allocate the workspace and call LAPACK to do the real work. */
  int lwork = (int)wkopt;
  THTensor *work = THTensor_(newWithSize1d)(lwork);
  THLapack_(ormqr)(side[0], trans[0], m, n, k, THTensor_(data)(a), lda,
                   THTensor_(data)(tau), THTensor_(data)(ra__), ldc,
                   THTensor_(data)(work), lwork, &info);

  THLapackCheckWithCleanup(" Lapack Error %s : unknown Lapack error. info = %i",
                           THCleanup(
                               THTensor_(free)(ra__);
                               THTensor_(free)(work);),
                           "ormqr", info,"");
  THTensor_(freeCopyTo)(ra__, ra_);
  THTensor_(free)(work);
}

void THTensor_(btrifact)(THTensor *ra_, THIntTensor *rpivots_, THIntTensor *rinfo_, THTensor *a)
{
  THArgCheck(THTensor_(nDimension)(a) == 3, 1, "expected 3D tensor, got %dD", THTensor_(nDimension)(a));

  if (ra_ != a) {
    THTensor_(resizeAs)(ra_, a);
    THTensor_(copy)(ra_, a);
  }

  int m = a->size[1];
  int n = a->size[2];
  if (m != n) {
    THError("btrifact is only implemented for square matrices");
  }
  long num_batches = THTensor_(size)(a, 0);
  THTensor *ra__;
  int lda;

  if (ra_->stride[1] == 1) {
    // column ordered, what BLAS wants
    lda = ra_->stride[2];
    ra__ = ra_;
  } else {
    // not column ordered, need to make it such (requires copy)
    THTensor *transp_r_ = THTensor_(newTranspose)(ra_, 1, 2);
    ra__ = THTensor_(newClone)(transp_r_);
    THTensor_(free)(transp_r_);
    THTensor_(transpose)(ra__, NULL, 1, 2);
    lda = ra__->stride[2];
  }

  THTensor *ai = THTensor_(new)();
  THTensor *rai = THTensor_(new)();
  THIntTensor *rpivoti = THIntTensor_new();

  int info = 0;
  int *info_ptr = &info;
  if (rinfo_) {
    THIntTensor_resize1d(rinfo_, num_batches);
    info_ptr = THIntTensor_data(rinfo_);
  }

  THIntTensor_resize2d(rpivots_, num_batches, n);

  long batch = 0;
  for (; batch < num_batches; ++batch) {
    THTensor_(select)(ai, a, 0, batch);
    THTensor_(select)(rai, ra__, 0, batch);
    THIntTensor_select(rpivoti, rpivots_, 0, batch);

    THLapack_(getrf)(n, n, THTensor_(data)(rai), lda,
                     THIntTensor_data(rpivoti), info_ptr);
    if (rinfo_) {
      info_ptr++;
    } else if (info != 0) {
      break;
    }
  }

  THTensor_(free)(ai);
  THTensor_(free)(rai);
  THIntTensor_free(rpivoti);

  if (ra__ != ra_) {
    THTensor_(freeCopyTo)(ra__, ra_);
  }

  if (!rinfo_ && info != 0) {
    THError("failed to factorize batch element %ld (info == %d)", batch, info);
  }
}

void THTensor_(btrisolve)(THTensor *rb_, THTensor *b, THTensor *atf, THIntTensor *pivots)
{
  THArgCheck(THTensor_(nDimension)(atf) == 3, 1, "expected 3D tensor, got %dD",
             THTensor_(nDimension)(atf));
  THArgCheck(THTensor_(nDimension)(b) == 3 ||
             THTensor_(nDimension)(b) == 2, 4, "expected 2D or 3D tensor");
  THArgCheck(THTensor_(size)(atf, 0) ==
             THTensor_(size)(b, 0), 3, "number of batches must be equal");
  THArgCheck(THTensor_(size)(atf, 1) ==
             THTensor_(size)(atf, 2), 3, "A matrices must be square");
  THArgCheck(THTensor_(size)(atf, 1) ==
             THTensor_(size)(b, 1), 3, "dimensions of A and b must be equal");

  if (rb_ != b) {
    THTensor_(resizeAs)(rb_, b);
    THTensor_(copy)(rb_, b);
  }

  long num_batches = atf->size[0];
  long n = atf->size[1];
  int nrhs = rb_->nDimension > 2 ? rb_->size[2] : 1;

  int lda, ldb;
  THTensor *atf_;
  THTensor *rb__;

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
    THTensor *transp_r_ = THTensor_(newTranspose)(atf, 1, 2);
    atf_ = THTensor_(newClone)(transp_r_);
    THTensor_(free)(transp_r_);
    THTensor_(transpose)(atf_, NULL, 1, 2);
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
      THTensor *transp_r_ = THTensor_(newTranspose)(rb_, 1, 2);
      rb__ = THTensor_(newClone)(transp_r_);
      THTensor_(free)(transp_r_);
      THTensor_(transpose)(rb__, NULL, 1, 2);
      ldb = rb__->stride[2];
    } else {
      rb__ = THTensor_(newClone)(rb_);
      ldb = n;
    }
  }

  THTensor *ai = THTensor_(new)();
  THTensor *rbi = THTensor_(new)();
  THIntTensor *pivoti = THIntTensor_new();

  if (!THIntTensor_isContiguous(pivots)) {
      THError("Error: rpivots_ is not contiguous.");
  }

  for (long batch = 0; batch < num_batches; ++batch) {
    THTensor_(select)(ai, atf_, 0, batch);
    THTensor_(select)(rbi, rb__, 0, batch);
    THIntTensor_select(pivoti, pivots, 0, batch);

#if defined(TH_REAL_IS_FLOAT) || defined(TH_REAL_IS_DOUBLE)
    int info;
    THLapack_(getrs)('N', n, nrhs, THTensor_(data)(ai), lda,
                     THIntTensor_data(pivoti), THTensor_(data)(rbi),
                     ldb, &info);
    if (info != 0) {
      THError("Error: Nonzero info.");
    }
#else
    THError("Unimplemented");
#endif
  }

  THTensor_(free)(ai);
  THTensor_(free)(rbi);
  THIntTensor_free(pivoti);

  if (atf_ != atf) {
    THTensor_(free)(atf_);
  }

  if (rb__ != rb_) {
    THTensor_(freeCopyTo)(rb__, rb_);
  }
}

#endif
