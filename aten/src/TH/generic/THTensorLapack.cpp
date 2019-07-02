#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "TH/generic/THTensorLapack.cpp"
#else

/*
Check if self is transpose of a contiguous matrix
*/
static int THTensor_(isTransposedContiguous)(THTensor *self)
{
  return self->stride(0) == 1 && self->stride(1) == self->size(0);
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
  if (src == result && THTensor_(isTransposedContiguous)(src) && src->size(1) == nrows)
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

  THTensor_(resize2d)(result, src->size(1), nrows);
  THTensor_(checkTransposed)(result);

  if (src->size(0) == nrows) {
    at::Tensor result_wrap = THTensor_wrap(result);
    at::Tensor src_wrap = THTensor_wrap(src);
    at::native::copy_(result_wrap, src_wrap);
  }
  else
  {
    view = THTensor_(newNarrow)(result, 0, 0, src->size(0));
    at::Tensor view_wrap = THTensor_wrap(view);
    at::Tensor src_wrap = THTensor_wrap(src);
    at::native::copy_(view_wrap, src_wrap);
    c10::raw::intrusive_ptr::decref(view);
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
  return THTensor_(cloneColumnMajorNrows)(self, src, src->size(0));
}

void THTensor_(gels)(THTensor *rb_, THTensor *ra_, THTensor *b, THTensor *a)
{
  int free_b = 0;
  // Note that a = NULL is interpreted as a = ra_, and b = NULL as b = rb_.
  if (a == NULL) a = ra_;
  if (b == NULL) b = rb_;
  THArgCheck(a->dim() == 2, 2, "A should have 2 dimensions, but has %d",
      a->dim());
  THArgCheck(!a->is_empty(), 2, "A should not be empty");
  THArgCheck(b->dim() == 1 || b->dim() == 2, 1, "B should have 1 or 2 "
      "dimensions, but has %d", b->dim());
  THArgCheck(!b->is_empty(), 1, "B should not be empty");
  TORCH_CHECK(a->size(0) == b->size(0), "Expected A and b to have same size "
      "at dim 0, but A has ", a->size(0), " rows and B has ", b->size(0), " rows");

  if (THTensor_nDimensionLegacyAll(b) == 1) {
    b = THTensor_(newWithStorage2d)(THTensor_getStoragePtr(b), b->storage_offset(), b->size(0),
            b->stride(0), 1, 0);
    free_b = 1;
  }

  int m, n, nrhs, lda, ldb, info, lwork;
  THTensor *work = NULL;
  scalar_t wkopt = 0;

  THTensor *ra__ = NULL;  // working version of A matrix to be passed into lapack GELS
  THTensor *rb__ = NULL;  // working version of B matrix to be passed into lapack GELS

  ra__ = THTensor_(cloneColumnMajor)(ra_, a);

  m = ra__->size(0);
  n = ra__->size(1);
  lda = m;
  ldb = (m > n) ? m : n;

  rb__ = THTensor_(cloneColumnMajorNrows)(rb_, b, ldb);

  nrhs = rb__->size(1);
  info = 0;


  /* get optimal workspace size */
  THLapack_(gels)('N', m, n, nrhs, ra__->data<scalar_t>(), lda,
                  rb__->data<scalar_t>(), ldb,
                  &wkopt, -1, &info);
  lwork = (int)wkopt;
  work = THTensor_(newWithSize1d)(lwork);
  THLapack_(gels)('N', m, n, nrhs, ra__->data<scalar_t>(), lda,
                  rb__->data<scalar_t>(), ldb,
                  work->data<scalar_t>(), lwork, &info);

  THLapackCheckWithCleanup("Lapack Error in %s : The %d-th diagonal element of the triangular factor of A is zero",
                           THCleanup(c10::raw::intrusive_ptr::decref(ra__);
                                     c10::raw::intrusive_ptr::decref(rb__);
                                     c10::raw::intrusive_ptr::decref(work);
                                     if (free_b) c10::raw::intrusive_ptr::decref(b);),
                           "gels", info,"");

  /*
   * In the m < n case, if the input b is used as the result (so b == _rb),
   * then rb_ was originally m by nrhs but now should be n by nrhs.
   * This is larger than before, so we need to expose the new rows by resizing.
   */
  if (m < n && b == rb_) {
    THTensor_(resize2d)(rb_, n, nrhs);
  }

  THTensor_(freeCopyTo)(ra__, ra_);
  THTensor_(freeCopyTo)(rb__, rb_);
  c10::raw::intrusive_ptr::decref(work);
  if (free_b) c10::raw::intrusive_ptr::decref(b);
}

void THTensor_(geev)(THTensor *re_, THTensor *rv_, THTensor *a_, const char *jobvr)
{
  int n, lda, lwork, info, ldvr;
  THTensor *work=nullptr, *wi, *wr, *a;
  scalar_t wkopt;
  scalar_t *rv_data;
  int64_t i;

  THTensor *re__ = NULL;
  THTensor *rv__ = NULL;

  THArgCheck(a_->dim() == 2, 1, "A should be 2 dimensional");
  THArgCheck(a_->size(0) == a_->size(1), 1,"A should be square");

  /* we want to definitely clone a_ for geev*/
  a = THTensor_(cloneColumnMajor)(NULL, a_);

  n = a->size(0);
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
    rv_data = rv__->data<scalar_t>();
    ldvr = n;
  }
  THTensor_(resize2d)(re_,n,2);
  re__ = THTensor_(newContiguous)(re_);

  if (n > 0) {  // lapack doesn't work with size 0
    /* get optimal workspace size */
    THLapack_(geev)('N', jobvr[0], n, a->data<scalar_t>(), lda, wr->data<scalar_t>(), wi->data<scalar_t>(),
        NULL, 1, rv_data, ldvr, &wkopt, -1, &info);

    lwork = (int)wkopt;
    work = THTensor_(newWithSize1d)(lwork);

    THLapack_(geev)('N', jobvr[0], n, a->data<scalar_t>(), lda, wr->data<scalar_t>(), wi->data<scalar_t>(),
        NULL, 1, rv_data, ldvr, work->data<scalar_t>(), lwork, &info);

    THLapackCheckWithCleanup(" Lapack Error in %s : %d off-diagonal elements of an didn't converge to zero",
                             THCleanup(c10::raw::intrusive_ptr::decref(re__);
                                       c10::raw::intrusive_ptr::decref(rv__);
                                       c10::raw::intrusive_ptr::decref(a);
                                       c10::raw::intrusive_ptr::decref(wi);
                                       c10::raw::intrusive_ptr::decref(wr);
                                       c10::raw::intrusive_ptr::decref(work);),
                             "geev", info,"");
  }

  {
    scalar_t *re_data = re__->data<scalar_t>();
    scalar_t *wi_data = wi->data<scalar_t>();
    scalar_t *wr_data = wr->data<scalar_t>();
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
  c10::raw::intrusive_ptr::decref(a);
  c10::raw::intrusive_ptr::decref(wi);
  c10::raw::intrusive_ptr::decref(wr);
  c10::raw::intrusive_ptr::decref(work);
}

void THTensor_(gesdd)(THTensor *ru_, THTensor *rs_, THTensor *rv_, THTensor *a, const char* some, const char* compute_uv)
{
  THTensor *ra_ = THTensor_(new)();
  THTensor_(gesdd2)(ru_, rs_, rv_,  ra_, a, some, compute_uv);
  c10::raw::intrusive_ptr::decref(ra_);
}

void THTensor_(gesdd2)(THTensor *ru_, THTensor *rs_, THTensor *rv_, THTensor *ra_, THTensor *a,
                       const char* some, const char* compute_uv)
{
  if (a == NULL) a = ra_;
  THArgCheck(a->dim() == 2, 1, "A should be 2 dimensional");
  THArgCheck(!a->is_empty(), 1, "A should not be empty");

  int k, m, n, lda, ldu, ldvt, lwork, info;
  THTensor *work;
  scalar_t wkopt;
  THIntTensor *iwork;

  THTensor *ra__ = NULL;
  THTensor *ru__ = NULL;
  THTensor *rs__ = NULL;
  THTensor *rv__ = NULL;

  ra__ = THTensor_(cloneColumnMajor)(ra_, a);

  m = ra__->size(0);
  n = ra__->size(1);
  k = (m < n ? m : n);

  lda = m;
  ldu = m;
  ldvt = n;

  iwork = k ? THIntTensor_newWithSize1d((int64_t)(8 * m)) : THIntTensor_newWithSize1d((int64_t)(8 * n));

  THTensor_(resize1d)(rs_,k);
  THTensor *rvf_ = NULL;

  if (*compute_uv != 'N') {
    rvf_ = THTensor_(new)();
    THTensor_(resize2d)(rvf_,ldvt,n);
    if (*some == 'A')
      THTensor_(resize2d)(ru_,m,ldu);
    else
      THTensor_(resize2d)(ru_,k,ldu);
  } else {
    THTensor_(resize2d)(rv_,ldvt,n);
    THTensor_(resize2d)(ru_,m,ldu);
  }

  THTensor_(checkTransposed)(ru_);

  char jobz = 'N';
  scalar_t *rs__data = NULL;
  scalar_t *ru__data = NULL;
  scalar_t *rv__data = NULL;

  rs__ = THTensor_(newContiguous)(rs_);
  rs__data = rs__->data<scalar_t>();
  if (*compute_uv != 'N') {
    /* guard against someone passing a correct size, but wrong stride */
    ru__ = THTensor_(newTransposedContiguous)(ru_);
    rv__ = THTensor_(newContiguous)(rvf_);

    ru__data = ru__->data<scalar_t>();
    rv__data = rv__->data<scalar_t>();

    jobz = some[0];
  }

  THLapack_(gesdd)(jobz,
             m,n,ra__->data<scalar_t>(),lda,
             rs__data,
             ru__data,
             ldu,
             rv__data, ldvt,
             &wkopt, -1, THIntTensor_data(iwork), &info);
  lwork = (int)wkopt;
  work = THTensor_(newWithSize1d)(lwork);
  THLapack_(gesdd)(jobz,
             m,n,ra__->data<scalar_t>(),lda,
             rs__data,
             ru__data,
             ldu,
             rv__data, ldvt,
             work->data<scalar_t>(),lwork, THIntTensor_data(iwork), &info);

  if (jobz != 'N') {
    THLapackCheckWithCleanup("Lapack Error %s : %d superdiagonals failed to converge.",
                             THCleanup(
                                 c10::raw::intrusive_ptr::decref(ru__);
                                 c10::raw::intrusive_ptr::decref(rs__);
                                 c10::raw::intrusive_ptr::decref(rv__);
                                 c10::raw::intrusive_ptr::decref(ra__);
                                 c10::raw::intrusive_ptr::decref(work);
                                 c10::raw::intrusive_ptr::decref(iwork);),
                             "gesdd", info, "");
  } else {
    THLapackCheckWithCleanup("Lapack Error %s : %d superdiagonals failed to converge.",
                             THCleanup(
                                 c10::raw::intrusive_ptr::decref(rs__);
                                 c10::raw::intrusive_ptr::decref(ra__);
                                 c10::raw::intrusive_ptr::decref(work);
                                 c10::raw::intrusive_ptr::decref(iwork);),
                             "gesdd", info, "");
  }

  THTensor_(freeCopyTo)(ra__, ra_);
  THTensor_(freeCopyTo)(rs__, rs_);
  c10::raw::intrusive_ptr::decref(work);
  c10::raw::intrusive_ptr::decref(iwork);

  if (jobz != 'N') {
    if (jobz == 'S')
      THTensor_(narrow)(rv__,NULL,1,0,k);

    THTensor_(freeCopyTo)(ru__, ru_);
    THTensor_(freeCopyTo)(rv__, rvf_);

    if (jobz == 'S')
      THTensor_(narrow)(rvf_,NULL,1,0,k);

    THTensor_(resizeAs)(rv_, rvf_);
    at::Tensor rv__wrap = THTensor_wrap(rv_);
    at::Tensor rvf__wrap =  THTensor_wrap(rvf_);
    at::native::copy_(rv__wrap, rvf__wrap);
    c10::raw::intrusive_ptr::decref(rvf_);
  } else {
    THTensor_(zero)(ru_);
    THTensor_(zero)(rv_);
  }
}

void THTensor_(clearUpLoTriangle)(THTensor *a, const char *uplo)
{
  THArgCheck(THTensor_nDimensionLegacyAll(a) == 2, 1, "A should be 2 dimensional");
  THArgCheck(a->size(0) == a->size(1), 1, "A should be square");

  int n = a->size(0);

  /* Build full matrix */
  scalar_t *p = a->data<scalar_t>();
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

void THTensor_(copyUpLoTriangle)(THTensor *a, const char *uplo)
{
  THArgCheck(THTensor_nDimensionLegacyAll(a) == 2, 1, "A should be 2 dimensional");
  THArgCheck(a->size(0) == a->size(1), 1, "A should be square");

  int n = a->size(0);

  /* Build full matrix */
  scalar_t *p = a->data<scalar_t>();
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

void THTensor_(potri)(THTensor *ra_, THTensor *a, const char *uplo)
{
  if (a == NULL) a = ra_;
  THArgCheck(THTensor_nDimensionLegacyAll(a) == 2, 1, "A should be 2 dimensional");
  THArgCheck(a->size(0) == a->size(1), 1, "A should be square");

  int n, lda, info;
  THTensor *ra__ = NULL;

  ra__ = THTensor_(cloneColumnMajor)(ra_, a);

  n = THTensor_sizeLegacyNoScalars(ra__, 0);
  lda = n;

  /* Run inverse */
  THLapack_(potri)(uplo[0], n, ra__->data<scalar_t>(), lda, &info);
  THLapackCheckWithCleanup("Lapack Error %s : A(%d,%d) is 0, A cannot be factorized",
                           THCleanup(c10::raw::intrusive_ptr::decref(ra__);),
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
void THTensor_(pstrf)(THTensor *ra_, THIntTensor *rpiv_, THTensor *a, const char *uplo, scalar_t tol) {
  THArgCheck(THTensor_nDimensionLegacyAll(a) == 2, 1, "A should be 2 dimensional");
  THArgCheck(a->size(0) == a->size(1), 1, "A should be square");

  int n = a->size(0);

  THTensor *ra__ = THTensor_(cloneColumnMajor)(ra_, a);
  THIntTensor_resize1d(rpiv_, n);

  // Allocate working tensor
  THTensor *work = THTensor_(newWithSize1d)(2 * n);

  // Run Cholesky factorization
  int lda = n;
  int rank, info;

  THLapack_(pstrf)(uplo[0], n, ra__->data<scalar_t>(), lda,
                   THIntTensor_data(rpiv_), &rank, tol,
                   work->data<scalar_t>(), &info);

  THLapackCheckWithCleanup("Lapack Error %s : matrix is rank deficient or not positive semidefinite",
                           THCleanup(
                               c10::raw::intrusive_ptr::decref(ra__);
                               c10::raw::intrusive_ptr::decref(work);),
                           "pstrf", info,"");

  THTensor_(clearUpLoTriangle)(ra__, uplo);

  THTensor_(freeCopyTo)(ra__, ra_);
  c10::raw::intrusive_ptr::decref(work);
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
  THArgCheck(a->dim() == 2, 1, "A should be 2 dimensional");
  THArgCheck(!a->is_empty(), 1, "A should not be empty");

  THTensor *ra__ = NULL;

  /* Prepare the input for LAPACK, making a copy if necessary. */
  ra__ = THTensor_(cloneColumnMajor)(ra_, a);

  int m = ra__->size(0);
  int n = ra__->size(1);
  int k = (m < n ? m : n);
  int lda = m;
  THTensor_(resize1d)(rtau_, k);

  /* Dry-run to query the suggested size of the workspace. */
  int info = 0;
  scalar_t wkopt = 0;
  THLapack_(geqrf)(m, n, ra__->data<scalar_t>(), lda,
                   rtau_->data<scalar_t>(),
                   &wkopt, -1, &info);

  /* Allocate the workspace and call LAPACK to do the real work. */
  int lwork = (int)wkopt;
  THTensor *work = THTensor_(newWithSize1d)(lwork);
  THLapack_(geqrf)(m, n, ra__->data<scalar_t>(), lda,
                   rtau_->data<scalar_t>(),
                   work->data<scalar_t>(), lwork, &info);

  THLapackCheckWithCleanup("Lapack Error %s : unknown Lapack error. info = %i",
                           THCleanup(
                               c10::raw::intrusive_ptr::decref(ra__);
                               c10::raw::intrusive_ptr::decref(work);),
                           "geqrf", info,"");

  THTensor_(freeCopyTo)(ra__, ra_);
  c10::raw::intrusive_ptr::decref(work);
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
  THArgCheck(THTensor_nDimensionLegacyAll(a) == 2, 1, "A should be 2 dimensional");

  THTensor *ra__ = NULL;
  ra__ = THTensor_(cloneColumnMajor)(ra_, a);

  int m = THTensor_sizeLegacyNoScalars(ra__, 0);
  int k = THTensor_sizeLegacyNoScalars(tau, 0);
  int lda = m;

  /* Dry-run to query the suggested size of the workspace. */
  int info = 0;
  scalar_t wkopt = 0;
  THLapack_(orgqr)(m, k, k, ra__->data<scalar_t>(), lda,
                   tau->data<scalar_t>(),
                   &wkopt, -1, &info);

  /* Allocate the workspace and call LAPACK to do the real work. */
  int lwork = (int)wkopt;
  THTensor *work = THTensor_(newWithSize1d)(lwork);
  THLapack_(orgqr)(m, k, k, ra__->data<scalar_t>(), lda,
                   tau->data<scalar_t>(),
                   work->data<scalar_t>(), lwork, &info);

  THLapackCheckWithCleanup(" Lapack Error %s : unknown Lapack error. info = %i",
                           THCleanup(
                               c10::raw::intrusive_ptr::decref(ra__);
                               c10::raw::intrusive_ptr::decref(work);),
                           "orgqr", info,"");
  THTensor_(freeCopyTo)(ra__, ra_);
  c10::raw::intrusive_ptr::decref(work);
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
  THArgCheck(THTensor_nDimensionLegacyAll(a) == 2, 1, "A should be 2 dimensional");

  THTensor *ra__ = NULL;
  ra__ = THTensor_(cloneColumnMajor)(ra_, c);

  int m = c->size(0);
  int n = c->size(1);
  int k = THTensor_sizeLegacyNoScalars(tau, 0);
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
  scalar_t wkopt = 0;
  THLapack_(ormqr)(side[0], trans[0], m, n, k, a->data<scalar_t>(), lda,
                   tau->data<scalar_t>(), ra__->data<scalar_t>(), ldc,
                   &wkopt, -1, &info);

  /* Allocate the workspace and call LAPACK to do the real work. */
  int lwork = (int)wkopt;
  THTensor *work = THTensor_(newWithSize1d)(lwork);
  THLapack_(ormqr)(side[0], trans[0], m, n, k, a->data<scalar_t>(), lda,
                   tau->data<scalar_t>(), ra__->data<scalar_t>(), ldc,
                   work->data<scalar_t>(), lwork, &info);

  THLapackCheckWithCleanup(" Lapack Error %s : unknown Lapack error. info = %i",
                           THCleanup(
                               c10::raw::intrusive_ptr::decref(ra__);
                               c10::raw::intrusive_ptr::decref(work);),
                           "ormqr", info,"");
  THTensor_(freeCopyTo)(ra__, ra_);
  c10::raw::intrusive_ptr::decref(work);
}

void THTensor_(btrisolve)(THTensor *rb_, THTensor *b, THTensor *atf, THIntTensor *pivots)
{
  TORCH_CHECK(!atf->is_empty() && THTensor_(nDimensionLegacyNoScalars)(atf) == 3, "expected non-empty 3D tensor, got size: ",
           atf->sizes());
  TORCH_CHECK(!b->is_empty() && (THTensor_(nDimensionLegacyNoScalars)(b) == 3 ||
             THTensor_(nDimensionLegacyNoScalars)(b) == 2), "expected non-empty 2D or 3D tensor, got size: ", b->sizes());
  THArgCheck(THTensor_(size)(atf, 0) ==
             THTensor_(size)(b, 0), 3, "number of batches must be equal");
  THArgCheck(THTensor_(size)(atf, 1) ==
             THTensor_(size)(atf, 2), 3, "A matrices must be square");
  THArgCheck(THTensor_(size)(atf, 1) ==
             THTensor_(size)(b, 1), 3, "dimensions of A and b must be equal");

  if (rb_ != b) {
    THTensor_(resizeAs)(rb_, b);
    at::Tensor rb__wrap = THTensor_wrap(rb_);
    at::Tensor b_wrap = THTensor_wrap(b);
    at::native::copy_(rb__wrap, b_wrap);
  }

  int64_t num_batches = atf->size(0);
  int64_t n = atf->size(1);
  int nrhs = THTensor_nDimensionLegacyAll(rb_) > 2 ? rb_->size(2) : 1;

  int lda, ldb;
  THTensor *atf_;
  THTensor *rb__;

  // correct ordering of A
  if (atf->stride(1) == 1) {
    // column ordered, what BLAS wants
    lda = atf->stride(2);
    atf_ = atf;
  } else {
    // not column ordered, need to make it such (requires copy)
    // it would be nice if we could use the op(A) flags to automatically
    // transpose A if needed, but this leads to unpredictable behavior if the
    // user clones A_tf later with a different ordering
    THTensor *transp_r_ = THTensor_(newTranspose)(atf, 1, 2);
    atf_ = THTensor_(newClone)(transp_r_);
    c10::raw::intrusive_ptr::decref(transp_r_);
    THTensor_(transpose)(atf_, NULL, 1, 2);
    lda = atf_->stride(2);
  }

  // correct ordering of B
  if (rb_->stride(1) == 1) {
    // column ordered
    if (THTensor_nDimensionLegacyAll(rb_) == 2 || rb_->size(2) == 1) {
      ldb = n;
    } else {
      ldb = rb_->stride(2);
    }
    rb__ = rb_;
  } else {
    // make column ordered
    if (THTensor_nDimensionLegacyAll(rb_) > 2) {
      THTensor *transp_r_ = THTensor_(newTranspose)(rb_, 1, 2);
      rb__ = THTensor_(newClone)(transp_r_);
      c10::raw::intrusive_ptr::decref(transp_r_);
      THTensor_(transpose)(rb__, NULL, 1, 2);
      ldb = rb__->stride(2);
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

  for (int64_t batch = 0; batch < num_batches; ++batch) {
    THTensor_(select)(ai, atf_, 0, batch);
    THTensor_(select)(rbi, rb__, 0, batch);
    THIntTensor_select(pivoti, pivots, 0, batch);

#if defined(TH_REAL_IS_FLOAT) || defined(TH_REAL_IS_DOUBLE)
    int info;
    THLapack_(getrs)('N', n, nrhs, ai->data<scalar_t>(), lda,
                     THIntTensor_data(pivoti), rbi->data<scalar_t>(),
                     ldb, &info);
    if (info != 0) {
      THError("Error: Nonzero info.");
    }
#else
    THError("Unimplemented");
#endif
  }

  c10::raw::intrusive_ptr::decref(ai);
  c10::raw::intrusive_ptr::decref(rbi);
  THIntTensor_free(pivoti);

  if (atf_ != atf) {
    c10::raw::intrusive_ptr::decref(atf_);
  }

  if (rb__ != rb_) {
    THTensor_(freeCopyTo)(rb__, rb_);
  }
}

#endif
