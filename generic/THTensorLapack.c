#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/THTensorLapack.c"
#else

/*
Check if self is transpose of a contiguous matrix
*/
static int THTensor_(isTransposed)(THTensor *self)
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
Similar to (newContiguous), but checks if the transpose of the matrix
is contiguous and also limited to 2D matrices
*/
static THTensor *THTensor_(newTransposedContiguous)(THTensor *self)
{
  THTensor *tensor;
  if(THTensor_(isTransposed)(self))
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
  Puts a row-major version of m (suitable as an input for Lapack) with the specified number of rows into the
  storage of r_. If r_ is already row-major and has the correct number of rows, then r_ becomes a tensor
  pointing at the storage of m, and the function returns 0. Otherwise, r_ is resized and filled with a
  row-major copy of m; the function then returns 1.
*/
static int THTensor_(lapackCloneNrows)(THTensor *r_, THTensor *m, int forced, int nrows)
{
  int clone;

  if (!forced && THTensor_(isTransposed)(m) && m->size[1] == nrows)
  {
    clone = 0;
    THTensor_(set)(r_,m);
  }
  else
  {
    clone = 1;
    THTensor_(resize2d)(r_,m->size[1],nrows);
    THTensor_(checkTransposed)(r_);
    /* we need to copy */
    if (m->size[0] == nrows) {
      THTensor_(copy)(r_,m);
    } else {
      THTensor* r_view = THTensor_(newNarrow)(r_,0,0,m->size[0]);
      THTensor_(copy)(r_view,m);
      THTensor_(free)(r_view);
    }
  }
  return clone;
}

static int THTensor_(lapackClone)(THTensor *r_, THTensor *m, int forced)
{
  return THTensor_(lapackCloneNrows)(r_, m, forced, m->size[0]);
}

void THTensor_(gesv)(THTensor *rb_, THTensor *ra_, THTensor *b, THTensor *a)
{
  int n, nrhs, lda, ldb, info;
  THIntTensor *ipiv;
  THTensor *ra__;  // working version of A matrix to be passed into lapack GELS
  THTensor *rb__;  // working version of B matrix to be passed into lapack GELS

  int clonea;    // set to 1 if ra__ should be copied into ra_ at return
  int cloneb;    // set to 1 if rb__ should be copied into rb_ at return
  int destroya;  // set to 1 if ra__ needs to be destroyed at return
  int destroyb;  // set to 1 if rb__ needs to be destroyed at return

  
  if (a == NULL || ra_ == a) /* possibly destroy the inputs  */
  {
    THArgCheck(ra_->nDimension == 2, 1, "A should be 2 dimensional");
    ra__ = THTensor_(new)();
    clonea = THTensor_(lapackClone)(ra__,ra_,0);
    destroya = 1;
  }
  else /*we want to definitely clone and use ra_ as computational space*/
  {
    THArgCheck(a->nDimension == 2, 1, "A should be 2 dimensional");
    clonea = THTensor_(lapackClone)(ra_,a,1);
    ra__ = ra_;
    destroya = 0;
  }
  if (b == NULL || rb_ == b) /* possibly destroy the inputs  */
  {
    THArgCheck(rb_->nDimension == 2, 2, "B should be 2 dimensional");
    rb__ = THTensor_(new)();
    cloneb = THTensor_(lapackClone)(rb__,rb_,0);
    destroyb = 1;
  }
  else /*we want to definitely clone and use rb_ as computational space*/
  {
    THArgCheck(b->nDimension == 2, 2, "B should be 2 dimensional");
    cloneb = THTensor_(lapackClone)(rb_,b,1);
    rb__ = rb_;
    destroyb = 0;
  }

  THArgCheck(ra__->nDimension == 2, 1, "A should be 2 dimensional");
  THArgCheck(rb__->nDimension == 2, 2, "b should be 2 dimensional");
  THArgCheck(ra__->size[0] == ra__->size[1], 1, "A should be square");
  THArgCheck(rb__->size[0] == ra__->size[0], 2, "A,b size incompatible");

  n    = (int)ra__->size[0];
  nrhs = (int)rb__->size[1];
  lda  = n;
  ldb  = n;

  ipiv = THIntTensor_newWithSize1d((long)n);
  THLapack_(gesv)(n, nrhs, 
		  THTensor_(data)(ra__), lda, THIntTensor_data(ipiv),
		  THTensor_(data)(rb__), ldb, &info);

  /* clean up */
  if (destroya)
  {
    if (clonea)
    {
      THTensor_(copy)(ra_,ra__);
    }
    THTensor_(free)(ra__);
  }
  if (destroyb)
  {
    if (cloneb)
    {
      THTensor_(copy)(rb_,rb__);
    }
    THTensor_(free)(rb__);
  }

  if (info < 0)
  {
    THError("Lapack gesv : Argument %d : illegal value", -info);
  }
  else if (info > 0)
  {
    THError("Lapack gesv : U(%d,%d) is zero, singular U.", info,info);
  }

  THIntTensor_free(ipiv);
}

void THTensor_(trtrs)(THTensor *rb_, THTensor *ra_, THTensor *b, THTensor *a,
                      const char *uplo, const char *trans, const char *diag)
{
  int n, nrhs, lda, ldb, info;
  THTensor *ra__; // working version of A matrix to be passed into lapack TRTRS
  THTensor *rb__; // working version of B matrix to be passed into lapack TRTRS

  int clonea;    // set to 1 if ra__ should be copied into ra_ at return
  int cloneb;    // set to 1 if rb__ should be copied into rb_ at return
  int destroya;  // set to 1 if ra__ needs to be destroyed at return
  int destroyb;  // set to 1 if rb__ needs to be destroyed at return


  if (a == NULL || ra_ == a) /* possibly destroy the inputs  */
  {
    THArgCheck(ra_->nDimension == 2, 1, "A should be 2 dimensional");
    ra__ = THTensor_(new)();
    clonea = THTensor_(lapackClone)(ra__,ra_,0);
    destroya = 1;
  }
  else /*we want to clone and use ra_ as computational space*/
  {
    THArgCheck(a->nDimension == 2, 1, "A should be 2 dimensional");
    clonea = THTensor_(lapackClone)(ra_,a,1);
    ra__ = ra_;
    destroya = 0;
  }
  if (b == NULL || rb_ == b) /* possibly destroy the inputs  */
  {
    THArgCheck(rb_->nDimension == 2, 2, "B should be 2 dimensional");
    rb__ = THTensor_(new)();
    cloneb = THTensor_(lapackClone)(rb__,rb_,0);
    destroyb = 1;
  }
  else /*we want to clone and use rb_ as computational space*/
  {
    THArgCheck(b->nDimension == 2, 2, "B should be 2 dimensional");
    cloneb = THTensor_(lapackClone)(rb_,b,1);
    rb__ = rb_;
    destroyb = 0;
  }

  THArgCheck(ra__->nDimension == 2, 1, "A should be 2 dimensional");
  THArgCheck(rb__->nDimension == 2, 2, "b should be 2 dimensional");
  THArgCheck(ra__->size[0] == ra__->size[1], 1, "A should be square");
  THArgCheck(rb__->size[0] == ra__->size[0], 2, "A,b size incompatible");

  n    = (int)ra__->size[0];
  nrhs = (int)rb__->size[1];
  lda  = n;
  ldb  = n;

  THLapack_(trtrs)(uplo[0], trans[0], diag[0], n, nrhs,
                   THTensor_(data)(ra__), lda,
                   THTensor_(data)(rb__), ldb, &info);

  /* clean up */
  if (destroya)
  {
    if (clonea)
    {
      THTensor_(copy)(ra_,ra__);
    }
    THTensor_(free)(ra__);
  }
  if (destroyb)
  {
    if (cloneb)
    {
      THTensor_(copy)(rb_,rb__);
    }
    THTensor_(free)(rb__);
  }

  if (info < 0)
  {
    THError("Lapack trtrs : Argument %d : illegal value", -info);
  }
  else if (info > 0)
  {
    THError("Lapack trtrs : A(%d,%d) is zero, singular A.", info,info);
  }
}

void THTensor_(gels)(THTensor *rb_, THTensor *ra_, THTensor *b, THTensor *a)
{
  // Note that a = NULL is interpreted as a = ra_, and b = NULL as b = rb_.
  int m, n, nrhs, lda, ldb, info, lwork;
  THTensor *work = NULL;
  real wkopt = 0;

  THTensor *ra__;  // working version of A matrix to be passed into lapack GELS
  THTensor *rb__;  // working version of B matrix to be passed into lapack GELS

  int clonea;    // set to 1 if ra__ should be copied into ra_ at return
  int cloneb;    // set to 1 if rb__ should be copied into rb_ at return
  int destroya;  // set to 1 if ra__ needs to be destroyed at return
  int destroyb;  // set to 1 if rb__ needs to be destroyed at return


  if (a == NULL || ra_ == a) /* possibly destroy the inputs  */
  {
    THArgCheck(ra_->nDimension == 2, 1, "A should be 2 dimensional");
    ra__ = THTensor_(new)();
    clonea = THTensor_(lapackClone)(ra__,ra_,0);
    destroya = 1;
  }
  else /*we want to definitely clone and use ra_ as computational space*/
  {
    THArgCheck(a->nDimension == 2, 1, "A should be 2 dimensional");
    clonea = THTensor_(lapackClone)(ra_,a,1);
    ra__ = ra_;
    destroya = 0;
  }

  THArgCheck(ra__->nDimension == 2, 1, "A should be 2 dimensional");

  m = ra__->size[0];
  n = ra__->size[1];
  lda = m;
  ldb = (m > n) ? m : n;

  if (b == NULL || rb_ == b) /* possibly destroy the inputs  */
  {
    THArgCheck(rb_->nDimension == 2, 2, "B should be 2 dimensional");
    THArgCheck(ra_->size[0] == rb_->size[0], 2, "size incompatible A,b");
    rb__ = THTensor_(new)();
    cloneb = THTensor_(lapackCloneNrows)(rb__,rb_,0,ldb);
    destroyb = 1;
  }
  else /*we want to definitely clone and use rb_ as computational space*/
  {
    THArgCheck(ra_->size[0] == b->size[0], 2, "size incompatible A,b");
    THArgCheck(b->nDimension == 2, 2, "B should be 2 dimensional");
    cloneb = THTensor_(lapackCloneNrows)(rb_,b,1,ldb);
    rb__ = rb_;
    destroyb = 0;
  }

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

  /* printf("lwork = %d,%g\n",lwork,THTensor_(data)(work)[0]); */
  if (info != 0)
  {
    THError("Lapack gels : Argument %d : illegal value", -info);
  }

  /* rb__ is currently ldb by nrhs; resize it to n by nrhs */
  rb__->size[0] = n;

  /* clean up */
  if (destroya)
  {
    if (clonea)
    {
      THTensor_(copy)(ra_,ra__);
    }
    THTensor_(free)(ra__);
  }
  if (destroyb)
  {
    if (cloneb)
    {
      THTensor_(resize2d)(rb_, n, nrhs);
      THTensor_(copy)(rb_,rb__);
    }
    THTensor_(free)(rb__);
  }
  THTensor_(free)(work);
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

  THArgCheck(a_->nDimension == 2, 3, "A should be 2 dimensional");
  THArgCheck(a_->size[0] == a_->size[1], 3,"A should be square");

  /* we want to definitely clone */
  a = THTensor_(new)();
  THTensor_(lapackClone)(a,a_,1);
  
  n = a->size[0];
  lda = n;

  wi = THTensor_(new)();
  wr = THTensor_(new)();
  THTensor_(resize2d)(re_,n,2);
  THTensor_(resize1d)(wi,n);
  THTensor_(resize1d)(wr,n);

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
  re__ = THTensor_(newContiguous)(re_);

  /* get optimal workspace size */
  THLapack_(geev)('N', jobvr[0], n, THTensor_(data)(a), lda, THTensor_(data)(wr), THTensor_(data)(wi), 
      NULL, 1, rv_data, ldvr, &wkopt, -1, &info);

  lwork = (int)wkopt;
  work = THTensor_(newWithSize1d)(lwork);

  THLapack_(geev)('N', jobvr[0], n, THTensor_(data)(a), lda, THTensor_(data)(wr), THTensor_(data)(wi), 
      NULL, 1, rv_data, ldvr, THTensor_(data)(work), lwork, &info);

  if (info > 0)
  {
    THError(" Lapack geev : Failed to converge. %d off-diagonal elements of an didn't converge to zero",info);
  }
  else if (info < 0)
  {
    THError("Lapack geev : Argument %d : illegal value", -info);
  }

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
    THTensor_(checkTransposed)(rv_);

  if (*jobvr == 'V' && rv__ != rv_)
    THTensor_(copy)(rv_, rv__);
  if (re__ != re_)
    THTensor_(copy)(re_, re__);

  THTensor_(free)(a);
  THTensor_(free)(wi);
  THTensor_(free)(wr);
  THTensor_(free)(work);
  THTensor_(free)(re__);
  THTensor_(free)(rv__);
}

void THTensor_(syev)(THTensor *re_, THTensor *rv_, THTensor *a, const char *jobz, const char *uplo)
{
  int n, lda, lwork, info;
  THTensor *work;
  real wkopt;

  THTensor *rv__ = NULL;
  THTensor *re__ = NULL;

  int clonev;   // set to 1 if rv__ should be copied into rv_ at return
  int destroyv; // set to 1 if rv__ needs to be destroyed at return
  
  if (a == NULL) /* possibly destroy the inputs  */
  {
    THArgCheck(rv_->nDimension == 2, 1, "A should be 2 dimensional");
    rv__ = THTensor_(new)();
    clonev = THTensor_(lapackClone)(rv__,rv_,0);
    destroyv = 1;
  }
  else /*we want to definitely clone and use ra_ and rb_ as computational space*/
  {
    THArgCheck(a->nDimension == 2, 1, "A should be 2 dimensional");
    clonev = THTensor_(lapackClone)(rv_,a,1);
    rv__ = rv_;
    destroyv = 0;
  }

  THArgCheck(rv__->nDimension == 2, 2, "A should be 2 dimensional");

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

  if (info > 0)
  {
    THError(" Lapack syev : Failed to converge. %d off-diagonal elements of an didn't converge to zero",info);
  }
  else if (info < 0)
  {
    THError("Lapack syev : Argument %d : illegal value", -info);
  }
  /* clean up */
  if (destroyv)
  {
    if (clonev)
    {
      THTensor_(copy)(rv_,rv__);
    }
    THTensor_(free)(rv__);
  }

  if (re__ != re_)
    THTensor_(copy)(re_, re__);
  THTensor_(free)(re__);
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
  int k,m, n, lda, ldu, ldvt, lwork, info;
  THTensor *work;
  real wkopt;

  THTensor *ra__ = NULL;
  THTensor *ru__ = NULL;
  THTensor *rs__ = NULL;
  THTensor *rv__ = NULL;

  int clonea;   // set to 1 if ra__ should be copied into ra_ at return
  int destroya; // set to 1 if ra__ needs to be destroyed at return

  if (a == NULL) /* possibly destroy the inputs  */
  {
    THArgCheck(ra_->nDimension == 2, 1, "A should be 2 dimensional");
    ra__ = THTensor_(new)();
    clonea = THTensor_(lapackClone)(ra__,ra_,0);
    destroya = 1;
  }
  else /*we want to definitely clone */
  {
    THArgCheck(a->nDimension == 2, 1, "A should be 2 dimensional");
    clonea = THTensor_(lapackClone)(ra_,a,1);
    ra__ = ra_;
    destroya = 0;
  }
  
  THArgCheck(ra__->nDimension == 2, 2, "A should be 2 dimensional");

  m = ra__->size[0];
  n = ra__->size[1];
  k = (m < n ? m : n);

  lda = m;
  ldu = m;
  ldvt = n;

  THTensor_(resize1d)(rs_,k);
  THTensor_(resize2d)(rv_,ldvt,n);
  if (*jobu == 'A')
  {
    THTensor_(resize2d)(ru_,m,ldu);
  }
  else
  {
    THTensor_(resize2d)(ru_,k,ldu);
  }
  THTensor_(checkTransposed)(ru_);

  /* guard against someone passing a correct size, but wrong stride */
  ru__ = THTensor_(newTransposedContiguous)(ru_);
  rs__ = THTensor_(newContiguous)(rs_);
  rv__ = THTensor_(newContiguous)(rv_);
  
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
  if (info > 0)
  {
    THError(" Lapack gesvd : %d superdiagonals failed to converge.",info);
  }
  else if (info < 0)
  {
    THError("Lapack gesvd : Argument %d : illegal value", -info);
  }

  /* put the results back */
  if (ru__ != ru_)
    THTensor_(copy)(ru_, ru__);
  if (rs__ != rs_)
    THTensor_(copy)(rs_, rs__);
  if (rv__ != rv_)
    THTensor_(copy)(rv_, rv__);

  /* clean up */
  if (destroya)
  {
    if (clonea)
    {
      THTensor_(copy)(ra_,ra__);
    }
    THTensor_(free)(ra__);
  }
  THTensor_(free)(work);
  THTensor_(free)(ru__);
  THTensor_(free)(rs__);
  THTensor_(free)(rv__);
}

void THTensor_(getri)(THTensor *ra_, THTensor *a)
{
  int m, n, lda, info, lwork;
  real wkopt;
  THIntTensor *ipiv;
  THTensor *work;
  THTensor *ra__;

  int clonea;   // set to 1 if ra__ should be copied into ra_ at return
  int destroya; // set to 1 if ra__ needs to be destroyed at return

  if (a == NULL) /* possibly destroy the inputs  */
  {
    THArgCheck(ra_->nDimension == 2, 1, "A should be 2 dimensional");
    ra__ = THTensor_(new)();
    clonea = THTensor_(lapackClone)(ra__,ra_,0);
    destroya = 1;
  }
  else /*we want to definitely clone */
  {
    THArgCheck(a->nDimension == 2, 1, "A should be 2 dimensional");
    clonea = THTensor_(lapackClone)(ra_,a,1);
    ra__ = ra_;
    destroya = 0;
  }
  
  THArgCheck(ra__->nDimension == 2, 2, "A should be 2 dimensional");
  m = ra__->size[0];
  n = ra__->size[1];
  THArgCheck(m == n, 2, "A should be square");
  lda = m;
  ipiv = THIntTensor_newWithSize1d((long)m);

  /* Run LU */
  THLapack_(getrf)(n, n, THTensor_(data)(ra__), lda, THIntTensor_data(ipiv), &info);
  if (info > 0)
  {
    THError("Lapack getrf : U(%d,%d) is 0, U is singular",info, info);
  }
  else if (info < 0)
  {
    THError("Lapack getrf : Argument %d : illegal value", -info);
  }

  /* Run inverse */
  THLapack_(getri)(n, THTensor_(data)(ra__), lda, THIntTensor_data(ipiv), &wkopt, -1, &info);
  lwork = (int)wkopt;
  work = THTensor_(newWithSize1d)(lwork);
  THLapack_(getri)(n, THTensor_(data)(ra__), lda, THIntTensor_data(ipiv), THTensor_(data)(work), lwork, &info);
  if (info > 0)
  {
    THError("Lapack getri : U(%d,%d) is 0, U is singular",info, info);
  }
  else if (info < 0)
  {
    THError("Lapack getri : Argument %d : illegal value", -info);
  }

  /* clean up */
  if (destroya)
  {
    if (clonea)
    {
      THTensor_(copy)(ra_,ra__);
    }
    THTensor_(free)(ra__);
  }
  THTensor_(free)(work);
  THIntTensor_free(ipiv);
}

void THTensor_(potrf)(THTensor *ra_, THTensor *a)
{
  int n, lda, info;
  char uplo = 'U';
  THTensor *ra__;

  int clonea;   // set to 1 if ra__ should be copied into ra_ at return
  int destroya; // set to 1 if ra__ needs to be destroyed at return

  if (a == NULL) /* possibly destroy the inputs  */
  {
    THArgCheck(ra_->nDimension == 2, 1, "A should be 2 dimensional");
    ra__ = THTensor_(new)();
    clonea = THTensor_(lapackClone)(ra__,ra_,0);
    destroya = 1;
  }
  else /*we want to definitely clone */
  {
    THArgCheck(a->nDimension == 2, 1, "A should be 2 dimensional");
    clonea = THTensor_(lapackClone)(ra_,a,1);
    ra__ = ra_;
    destroya = 0;
  }
  
  THArgCheck(ra__->nDimension == 2, 2, "A should be 2 dimensional");
  THArgCheck(ra__->size[0] == ra__->size[1], 2, "A should be square");
  n = ra__->size[0];
  lda = n;

  /* Run Factorization */
  THLapack_(potrf)(uplo, n, THTensor_(data)(ra__), lda, &info);
  if (info > 0)
  {
    THError("Lapack potrf : A(%d,%d) is 0, A cannot be factorized", info, info);
  }
  else if (info < 0)
  {
    THError("Lapack potrf : Argument %d : illegal value", -info);
  }

  /* Build full upper-triangular matrix */
  {
    real *p = THTensor_(data)(ra__);
    long i,j;
    for (i=0; i<n; i++) {
      for (j=i+1; j<n; j++) {
        p[i*n+j] = 0;
      }
    }
  }

  /* clean up */
  if (destroya)
  {
    if (clonea)
    {
      THTensor_(copy)(ra_,ra__);
    }
    THTensor_(free)(ra__);
  }
}

void THTensor_(potri)(THTensor *ra_, THTensor *a)
{
  int n, lda, info;
  char uplo = 'U';
  THTensor *ra__;

  int clonea;   // set to 1 if ra__ should be copied into ra_ at return
  int destroya; // set to 1 if ra__ needs to be destroyed at return

  if (a == NULL) /* possibly destroy the inputs  */
  {
    THArgCheck(ra_->nDimension == 2, 1, "A should be 2 dimensional");
    ra__ = THTensor_(new)();
    clonea = THTensor_(lapackClone)(ra__,ra_,0);
    destroya = 1;
  }
  else /*we want to definitely clone */
  {
    THArgCheck(a->nDimension == 2, 1, "A should be 2 dimensional");
    clonea = THTensor_(lapackClone)(ra_,a,1);
    ra__ = ra_;
    destroya = 0;
  }
  
  THArgCheck(ra__->nDimension == 2, 2, "A should be 2 dimensional");
  THArgCheck(ra__->size[0] == ra__->size[1], 2, "A should be square");
  n = ra__->size[0];
  lda = n;

  /* Run Factorization */
  THLapack_(potrf)(uplo, n, THTensor_(data)(ra__), lda, &info);
  if (info > 0)
  {
    THError("Lapack potrf : A(%d,%d) is 0, A cannot be factorized", info, info);
  }
  else if (info < 0)
  {
    THError("Lapack potrf : Argument %d : illegal value", -info);
  }

  /* Run inverse */
  THLapack_(potri)(uplo, n, THTensor_(data)(ra__), lda, &info);
  if (info > 0)
  {
    THError("Lapack potri : A(%d,%d) is 0, A cannot be factorized", info, info);
  }
  else if (info < 0)
  {
    THError("Lapack potri : Argument %d : illegal value", -info);
  }

  /* Build full matrix */
  {
    real *p = THTensor_(data)(ra__);
    long i,j;
    for (i=0; i<n; i++) {
      for (j=i+1; j<n; j++) {
        p[i*n+j] = p[j*n+i];
      }
    }
  }

  /* clean up */
  if (destroya)
  {
    if (clonea)
    {
      THTensor_(copy)(ra_,ra__);
    }
    THTensor_(free)(ra__);
  }
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
  /* Prepare the input for LAPACK, making a copy if necessary. */
  THTensor *ra__;

  int clonea;   // set to 1 if ra__ should be copied into ra_ at return
  int destroya; // set to 1 if ra__ needs to be destroyed at return

  if (a == NULL)
  {
    THArgCheck(ra_->nDimension == 2, 1, "A should be 2 dimensional");
    ra__ = THTensor_(new)();
    clonea = THTensor_(lapackClone)(ra__, ra_, 0);
    destroya = 1;
  }
  else
  {
    THArgCheck(a->nDimension == 2, 1, "A should be 2 dimensional");
    clonea = THTensor_(lapackClone)(ra_, a, 1);
    ra__ = ra_;
    destroya = 0;
  }

  /* Check input sizes, and ensure we have space to store the results. */
  THArgCheck(ra__->nDimension == 2, 2, "A should be 2 dimensional");
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

  /* Clean up. */
  if (destroya)
  {
    if (clonea)
    {
      THTensor_(copy)(ra_, ra__);
    }
    THTensor_(free)(ra__);
  }
  THTensor_(free)(work);

  /* Raise a Lua error, if a problem was signalled by LAPACK. */
  if (info < 0)
  {
    THError(" Lapack geqrf : Argument %d : illegal value.", -info);
  }
  else if (info > 0)
  {
    THError(" Lapack geqrf : unknown Lapack error. info = %i", info);
  }
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
  /* Prepare the input for LAPACK, making a copy if necessary. */
  THTensor *ra__;

  int clonea;   // set to 1 if ra__ should be copied into ra_ at return
  int destroya; // set to 1 if ra__ needs to be destroyed at return

  if (a == NULL)
  {
    THArgCheck(ra_->nDimension == 2, 1, "A should be 2 dimensional");
    ra__ = THTensor_(new)();
    clonea = THTensor_(lapackClone)(ra__, ra_, 0);
    destroya = 1;
  }
  else
  {
    THArgCheck(a->nDimension == 2, 1, "A should be 2 dimensional");
    clonea = THTensor_(lapackClone)(ra_, a, 1);
    ra__ = ra_;
    destroya = 0;
  }

  /* Check input sizes. */
  THArgCheck(ra__->nDimension == 2, 2, "A should be 2 dimensional");
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

  /* Clean up. */
  if (destroya)
  {
    if (clonea)
    {
      THTensor_(copy)(ra_, ra__);
    }
    THTensor_(free)(ra__);
  }
  THTensor_(free)(work);

  /* Raise a Lua error, if a problem was signalled by LAPACK. */
  if (info < 0)
  {
    THError(" Lapack orgqr : Argument %d : illegal value.", -info);
  }
  else if (info > 0)
  {
    THError(" Lapack orgqr : unknown Lapack error. info = %i", info);
  }
}

#endif
