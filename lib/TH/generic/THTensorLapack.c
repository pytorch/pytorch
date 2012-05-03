#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/THTensorLapack.c"
#else

static int THTensor_(lapackClone)(THTensor *r_, THTensor *m, int forced)
{
  int clone;

  if (!forced && m->stride[0] == 1 && m->stride[1] == m->size[0])
  {
    clone = 0;
    THTensor_(set)(r_,m);
  }
  else
  {
    clone = 1;
    /* we need to copy */
    THTensor_(resize2d)(r_,m->size[1],m->size[0]);
    THTensor_(transpose)(r_,NULL,0,1);
    THTensor_(copy)(r_,m);
  }
  return clone;
}

TH_API void THTensor_(gesv)(THTensor *rb_, THTensor *ra_, THTensor *b, THTensor *a)
{
  int n, nrhs, lda, ldb, info;
  THIntTensor *ipiv;
  THTensor *ra__;
  THTensor *rb__;

  int clonea;
  int cloneb;
  int destroya;
  int destroyb;

  
  if (a == NULL || ra_ == a) /* possibly destroy the inputs  */
  {
    ra__ = THTensor_(new)();
    clonea = THTensor_(lapackClone)(ra__,ra_,0);
    destroya = 1;
  }
  else /*we want to definitely clone and use ra_ and rb_ as computational space*/
  {
    clonea = THTensor_(lapackClone)(ra_,a,1);
    ra__ = ra_;
    destroya = 0;
  }
  if (b == NULL || rb_ == b) /* possibly destroy the inputs  */
  {
    rb__ = THTensor_(new)();
    cloneb = THTensor_(lapackClone)(rb__,rb_,0);
    destroyb = 1;
  }
  else /*we want to definitely clone and use ra_ and rb_ as computational space*/
  {
    cloneb = THTensor_(lapackClone)(rb_,b,1);
    rb__ = rb_;
    destroyb = 0;
  }

  THArgCheck(ra__->nDimension == 2, 1, "A should be 2 dimensional");
  THArgCheck(rb__->nDimension == 2, 2, "b should be 2 dimensional");
  THArgCheck(ra__->size[0] == ra__->size[1], 1, "A should be square");
  THArgCheck(rb__->size[0] == ra__->size[0], 2, "A,b size incomptable");

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

TH_API void THTensor_(gels)(THTensor *rb_, THTensor *ra_, THTensor *b, THTensor *a)
{
  int m, n, nrhs, lda, ldb, info, lwork;
  THTensor *work = NULL;
  real wkopt = 0;

  THTensor *ra__;
  THTensor *rb__;

  int clonea;
  int cloneb;
  int destroya;
  int destroyb;

  
  if (a == NULL || ra_ == a) /* possibly destroy the inputs  */
  {
    ra__ = THTensor_(new)();
    clonea = THTensor_(lapackClone)(ra__,ra_,0);
    destroya = 1;
  }
  else /*we want to definitely clone and use ra_ and rb_ as computational space*/
  {
    clonea = THTensor_(lapackClone)(ra_,a,1);
    ra__ = ra_;
    destroya = 0;
  }
  if (b == NULL || rb_ == b) /* possibly destroy the inputs  */
  {
    rb__ = THTensor_(new)();
    cloneb = THTensor_(lapackClone)(rb__,rb_,0);
    destroyb = 1;
  }
  else /*we want to definitely clone and use ra_ and rb_ as computational space*/
  {
    cloneb = THTensor_(lapackClone)(rb_,b,1);
    rb__ = rb_;
    destroyb = 0;
  }
  
  THArgCheck(ra__->nDimension == 2, 1, "A should be 2 dimensional");
  THArgCheck(ra_->size[0] == rb__->size[0], 2, "size incompatible A,b");

  m = ra__->size[0];
  n = ra__->size[1];
  nrhs = rb__->size[1];
  lda = m;
  ldb = m;
  info = 0;

  // get optimal workspace size
  THLapack_(gels)('N', m, n, nrhs, THTensor_(data)(ra__), lda, 
		  THTensor_(data)(rb__), ldb, 
		  &wkopt, -1, &info);
  lwork = (int)wkopt;
  work = THTensor_(newWithSize1d)(lwork);
  THLapack_(gels)('N', m, n, nrhs, THTensor_(data)(ra__), lda, 
		  THTensor_(data)(rb__), ldb, 
		  THTensor_(data)(work), lwork, &info);

  //printf("lwork = %d,%g\n",lwork,THTensor_(data)(work)[0]);
  if (info != 0)
  {
    THError("Lapack gels : Argument %d : illegal value", -info);
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
  if (destroyb)
  {
    if (cloneb)
    {
      THTensor_(copy)(rb_,rb__);
    }
    THTensor_(free)(rb__);
  }
  THTensor_(free)(work);
}

TH_API void THTensor_(geev)(THTensor *re_, THTensor *rv_, THTensor *a_, const char *jobvr)
{
  int n, lda, lwork, info, ldvr;
  THTensor *work, *wi, *wr, *a;
  real wkopt;
  real *rv_data;
  long i;

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
    rv_data = THTensor_(data)(rv_);
    ldvr = n;
  }
  // get optimal workspace size
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

  real *re_data = THTensor_(data)(re_);
  real *wi_data = THTensor_(data)(wi);
  real *wr_data = THTensor_(data)(wr);
  for (i=0; i<n; i++)
  {
    re_data[2*i] = wr_data[i];
    re_data[2*i+1] = wi_data[i];
  }
  if (*jobvr == 'V')
  {
    THTensor_(transpose)(rv_,NULL,0,1);
  }
  THTensor_(free)(a);
  THTensor_(free)(wi);
  THTensor_(free)(wr);
  THTensor_(free)(work);
}

TH_API void THTensor_(syev)(THTensor *re_, THTensor *rv_, THTensor *a, const char *jobz, const char *uplo)
{
  int n, lda, lwork, info;
  THTensor *work;
  real wkopt;

  THTensor *rv__;

  int clonea;
  int destroy;
  
  if (a == NULL) /* possibly destroy the inputs  */
  {
    rv__ = THTensor_(new)();
    clonea = THTensor_(lapackClone)(rv__,rv_,0);
    destroy = 1;
  }
  else /*we want to definitely clone and use ra_ and rb_ as computational space*/
  {
    clonea = THTensor_(lapackClone)(rv_,a,1);
    rv__ = rv_;
    destroy = 0;
  }

  THArgCheck(rv__->nDimension == 2, 2, "A should be 2 dimensional");

  n = rv__->size[0];
  lda = n;

  THTensor_(resize1d)(re_,n);

  // get optimal workspace size
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
  if (destroy)
  {
    if (clonea)
    {
      THTensor_(copy)(rv_,rv__);
    }
    THTensor_(free)(rv__);
  }
  THTensor_(free)(work);
}

TH_API void THTensor_(gesvd)(THTensor *ru_, THTensor *rs_, THTensor *rv_, THTensor *a, const char* jobu)
{
  THTensor *ra_ = THTensor_(new)();
  THTensor_(gesvd2)(ru_, rs_, rv_,  ra_, a, jobu);
  THTensor_(free)(ra_);
}

TH_API void THTensor_(gesvd2)(THTensor *ru_, THTensor *rs_, THTensor *rv_, THTensor *ra_, THTensor *a, const char* jobu)
{
  int k,m, n, lda, ldu, ldvt, lwork, info;
  THTensor *work;
  real wkopt;

  THTensor *ra__;

  int clonea;
  int destroy;

  if (a == NULL) /* possibly destroy the inputs  */
  {
    ra__ = THTensor_(new)();
    clonea = THTensor_(lapackClone)(ra__,ra_,0);
    destroy = 1;
  }
  else /*we want to definitely clone */
  {
    clonea = THTensor_(lapackClone)(ra_,a,1);
    ra__ = ra_;
    destroy = 0;
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
  THTensor_(transpose)(ru_,NULL,0,1);
  /* we want to return V not VT*/
  /*THTensor_(transpose)(rv_,NULL,0,1);*/

  THLapack_(gesvd)(jobu[0],jobu[0],
		   m,n,THTensor_(data)(ra__),lda,
		   THTensor_(data)(rs_),
		   THTensor_(data)(ru_),
		   ldu,
		   THTensor_(data)(rv_), ldvt,
		   &wkopt, -1, &info);
  lwork = (int)wkopt;
  work = THTensor_(newWithSize1d)(lwork);
  THLapack_(gesvd)(jobu[0],jobu[0],
		   m,n,THTensor_(data)(ra__),lda,
		   THTensor_(data)(rs_),
		   THTensor_(data)(ru_),
		   ldu,
		   THTensor_(data)(rv_), ldvt,
		   THTensor_(data)(work),lwork, &info);
  if (info > 0)
  {
    THError(" Lapack gesvd : %d superdiagonals failed to converge.",info);
  }
  else if (info < 0)
  {
    THError("Lapack gesvd : Argument %d : illegal value", -info);
  }

  /* clean up */
  if (destroy)
  {
    if (clonea)
    {
      THTensor_(copy)(ra_,ra__);
    }
    THTensor_(free)(ra__);
  }
  THTensor_(free)(work);
}

TH_API void THTensor_(getri)(THTensor *ra_, THTensor *a)
{
  int m, n, lda, info, lwork;
  real wkopt;
  THIntTensor *ipiv;
  THTensor *work;
  THTensor *ra__;

  int clonea;
  int destroy;

  if (a == NULL) /* possibly destroy the inputs  */
  {
    ra__ = THTensor_(new)();
    clonea = THTensor_(lapackClone)(ra__,ra_,0);
    destroy = 1;
  }
  else /*we want to definitely clone */
  {
    clonea = THTensor_(lapackClone)(ra_,a,1);
    ra__ = ra_;
    destroy = 0;
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
  if (destroy)
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

#endif
