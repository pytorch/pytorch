#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "master_worker/master/generic/THDTensorLapack.cpp"
#else

/*
Check if self is transpose of a contiguous matrix
*/
static int THDTensor_(isTransposedContiguous)(THDTensor *self) {
  return self->stride[0] == 1 && self->stride[1] == self->size[0];
}
/*
If a matrix is a regular contiguous matrix, make sure it is transposed
because this is what we return from Lapack calls.
*/
static void THDTensor_(checkTransposed)(THDTensor *self) {
  if (THDTensor_(isContiguous)(self))
    THDTensor_(transpose)(self, NULL, 0, 1);
  return;
}
/*
newContiguous followed by transpose
Similar to (newContiguous), but checks if the transpose of the matrix
is contiguous and also limited to 2D matrices.
*/
static THDTensor *THDTensor_(newTransposedContiguous)(THDTensor *self) {
  THDTensor *tensor;
  if (THDTensor_(isTransposedContiguous)(self)) {
    THDTensor_(retain)(self);
    tensor = self;
  } else {
    tensor = THDTensor_(newContiguous)(self);
    THDTensor_(transpose)(tensor, NULL, 0, 1);
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
static THDTensor *THDTensor_(checkLapackClone)(THDTensor *result, THDTensor *src, int nrows) {
  /* check if user wants to reuse src and if it is correct shape/size */
  if (src == result && THDTensor_(isTransposedContiguous)(src) && src->size[1] == nrows)
    THDTensor_(retain)(result);
  else if (src == result || result == NULL)
    /* in this case, user wants reuse of src, but its structure is not OK */
    result = THDTensor_(new)();
  else
    THDTensor_(retain)(result);
  return result;
}

/*
Same as cloneColumnMajor, but accepts nrows argument, because some lapack calls require
the resulting tensor to be larger than src.
*/
static THDTensor *THDTensor_(cloneColumnMajorNrows)(THDTensor *self, THDTensor *src, int nrows) {
  THDTensor *result;
  THDTensor *view;

  if (src == NULL)
    src = self;
  result = THDTensor_(checkLapackClone)(self, src, nrows);
  if (src == result)
    return result;

  THDTensor_(resize2d)(result, src->size[1], nrows);
  THDTensor_(checkTransposed)(result);

  if (src->size[0] == nrows) {
    THDTensor_(copy)(result, src);
  } else {
    view = THDTensor_(newNarrow)(result, 0, 0, src->size[0]);
    THDTensor_(copy)(view, src);
    THDTensor_(free)(view);
  }
  return result;
}

/*
Create a clone of src in self column major order for use with Lapack.
If src == self, a new tensor is allocated, in any case, the return tensor should be
freed by calling function.
*/
static THDTensor *THDTensor_(cloneColumnMajor)(THDTensor *self, THDTensor *src) {
  return THDTensor_(cloneColumnMajorNrows)(self, src, src->size[0]);
}

/*
 * A verbose set of comments on what the Lapack functions do
 * is contained in their implementation in the TH library
 */

/* TODO this might leak on incorrect data */
void THDTensor_(gesv)(THDTensor *rb, THDTensor *ra, THDTensor *b, THDTensor *a) {
  bool free_b = false;
  if (a == NULL) a = ra;
  if (b == NULL) b = rb;
  THArgCheck(a->nDimension == 2, 2, "A should have 2 dimensions, but has %d",
      a->nDimension);
  THArgCheck(b->nDimension == 1 || b->nDimension == 2, 1, "B should have 1 or 2 "
      "dimensions, but has %d", b->nDimension);
  THArgCheck(a->size[0] == a->size[1], 2, "A should be square, but is %ldx%ld",
      a->size[0], a->size[1]);
  THArgCheck(a->size[0] == b->size[0], 2, "A,B size incompatible - A has %ld "
      "rows, B has %ld", a->size[0], b->size[0]);

  if (b->nDimension == 1) {
    b = THDTensor_(newWithStorage2d)(b->storage, b->storageOffset, b->size[0],
            b->stride[0], 1, 0);
    free_b = true;
  }

  masterCommandChannel->sendMessage(
    packMessage(Functions::tensorGesv, rb, ra, b, a),
    THDState::s_current_worker
  );

  THDTensor_(free)(THDTensor_(cloneColumnMajor)(ra, a));
  THDTensor_(free)(THDTensor_(cloneColumnMajor)(rb, b));

  if (free_b) THDTensor_(free)(b);
}

void THDTensor_(trtrs)(THDTensor *rb, THDTensor *ra, THDTensor *b, THDTensor *a,
                       const char *uplo, const char *trans, const char *diag) {
  bool free_b = false;
  if (a == NULL) a = ra;
  if (b == NULL) b = rb;
  THArgCheck(a->nDimension == 2, 2, "A should have 2 dimensions, but has %d",
      a->nDimension);
  THArgCheck(b->nDimension == 1 || b->nDimension == 2, 1, "B should have 1 or 2 "
      "dimensions, but has %d", b->nDimension);
  THArgCheck(a->size[0] == a->size[1], 2, "A should be square, but is %ldx%ld",
      a->size[0], a->size[1]);
  THArgCheck(a->size[0] == b->size[0], 2, "A,B size incompatible - A has %ld "
      "rows, B has %ld", a->size[0], b->size[0]);

  if (b->nDimension == 1) {
    b = THDTensor_(newWithStorage2d)(b->storage, b->storageOffset, b->size[0],
            b->stride[0], 1, 0);
    free_b = true;
  }

  masterCommandChannel->sendMessage(
    packMessage(Functions::tensorTrtrs, rb, ra, b, a, uplo[0], trans[0], diag[0]),
    THDState::s_current_worker
  );

  THDTensor_(free)(THDTensor_(cloneColumnMajor)(ra, a));
  THDTensor_(free)(THDTensor_(cloneColumnMajor)(rb, b));

  if (free_b) THDTensor_(free)(b);
}

void THDTensor_(gels)(THDTensor *rb, THDTensor *ra, THDTensor *b, THDTensor *a) {
  bool free_b = 0;
  if (a == NULL) a = ra;
  if (b == NULL) b = rb;
  THArgCheck(a->nDimension == 2, 2, "A should have 2 dimensions, but has %d",
      a->nDimension);
  THArgCheck(b->nDimension == 1 || b->nDimension == 2, 1, "B should have 1 or 2 "
      "dimensions, but has %d", b->nDimension);
  THArgCheck(a->size[0] == b->size[0], 2, "A,B size incompatible - A has %ld "
      "rows, B has %ld", a->size[0], b->size[0]);

  if (b->nDimension == 1) {
    b = THDTensor_(newWithStorage2d)(b->storage, b->storageOffset, b->size[0],
            b->stride[0], 1, 0);
    free_b = true;
  }

  masterCommandChannel->sendMessage(
    packMessage(Functions::tensorGels, rb, ra, b, a),
    THDState::s_current_worker
  );

  int m, n, nrhs, ldb;

  THDTensor *ra_ = NULL;
  THDTensor *rb_ = NULL;

  ra_ = THDTensor_(cloneColumnMajor)(ra, a);

  m = ra_->size[0];
  n = ra_->size[1];
  ldb = (m > n) ? m : n;

  rb_ = THDTensor_(cloneColumnMajorNrows)(rb, b, ldb);

  nrhs = rb_->size[1];

  /* rb_ is currently ldb by nrhs; resize it to n by nrhs */
  rb_->size[0] = n;
  if (rb_ != rb)
    THDTensor_(resize2d)(rb, n, nrhs);

  THDTensor_(free)(ra_);
  THDTensor_(free)(rb_);
  if (free_b) THDTensor_(free)(b);
}

void THDTensor_(syev)(THDTensor *re, THDTensor *rv, THDTensor *a,
                      const char *jobz, const char *uplo) {
  if (a == NULL) a = rv;
  THArgCheck(a->nDimension == 2, 1, "A should be 2 dimensional");
  THArgCheck(a->size[0] == a->size[1], 1,"A should be square");

  masterCommandChannel->sendMessage(
    packMessage(Functions::tensorSyev, re, rv, a, jobz[0], uplo[0]),
    THDState::s_current_worker
  );

  THDTensor *rv_ = THDTensor_(cloneColumnMajor)(rv, a);
  THDTensor_(resize1d)(re, rv_->size[0]);
  THDTensor_(free)(rv_);
}

void THDTensor_(geev)(THDTensor *re, THDTensor *rv, THDTensor *a, const char *jobvr) {
  int n;
  THDTensor *a_;

  THDTensor *re_ = NULL;
  THDTensor *rv_ = NULL;

  THArgCheck(a->nDimension == 2, 1, "A should be 2 dimensional");
  THArgCheck(a->size[0] == a->size[1], 1,"A should be square");

  masterCommandChannel->sendMessage(
    packMessage(Functions::tensorGeev, re, rv, a, jobvr[0]),
    THDState::s_current_worker
  );

  /* we want to definitely clone a for geev*/
  a_ = THDTensor_(cloneColumnMajor)(NULL, a);

  n = a_->size[0];

  if (*jobvr == 'V') {
    THDTensor_(resize2d)(rv, n, n);
    /* guard against someone passing a correct size, but wrong stride */
    rv_ = THDTensor_(newTransposedContiguous)(rv);
  }
  THDTensor_(resize2d)(re, n, 2);

  if (*jobvr == 'V') {
    THDTensor_(checkTransposed)(rv);
  }

  THDTensor_(free)(a_);
}

void THDTensor_(gesvd)(THDTensor *ru, THDTensor *rs, THDTensor *rv, THDTensor *a,
                       const char *jobu) {
  THDTensor *ra = THDTensor_(new)();
  THDTensor_(gesvd2)(ru, rs, rv,  ra, a, jobu);
  THDTensor_(free)(ra);
}

void THDTensor_(gesvd2)(THDTensor *ru, THDTensor *rs, THDTensor *rv, THDTensor *ra,
                        THDTensor *a, const char *jobu) {
  if (a == NULL) a = ra;
  THArgCheck(a->nDimension == 2, 1, "A should be 2 dimensional");

  masterCommandChannel->sendMessage(
    packMessage(Functions::tensorGesvd2, ru, rs, rv, ra, a, jobu[0]),
    THDState::s_current_worker
  );

  int k, m, n, ldu, ldvt;
  THDTensor *rvf = THDTensor_(new)();

  THDTensor *ra_ = NULL;
  THDTensor *ru_ = NULL;

  ra_ = THDTensor_(cloneColumnMajor)(ra, a);

  m = ra_->size[0];
  n = ra_->size[1];
  k = (m < n ? m : n);

  ldu = m;
  ldvt = n;

  THDTensor_(resize1d)(rs, k);
  THDTensor_(resize2d)(rvf, ldvt, n);
  if (*jobu == 'A')
    THDTensor_(resize2d)(ru, m, ldu);
  else
    THDTensor_(resize2d)(ru, k, ldu);

  THDTensor_(checkTransposed)(ru);

  /* guard against someone passing a correct size, but wrong stride */
  ru_ = THDTensor_(newTransposedContiguous)(ru);

  if (*jobu == 'S') {
    THDTensor_(narrow)(rvf, NULL, 1, 0, k);
  }
  THDTensor_(resizeAs)(rv, rvf);
  THDTensor_(free)(rvf);
  THDTensor_(free)(ra_);
}

void THDTensor_(getri)(THDTensor *ra, THDTensor *a) {
  if (a == NULL) a = ra;
  THArgCheck(a->nDimension == 2, 1, "A should be 2 dimensional");
  THArgCheck(a->size[0] == a->size[1], 1, "A should be square");

  masterCommandChannel->sendMessage(
    packMessage(Functions::tensorGetri, ra, a),
    THDState::s_current_worker
  );

  THDTensor_(free)(THDTensor_(cloneColumnMajor)(ra, a));
}

void THDTensor_(potrf)(THDTensor *ra, THDTensor *a, const char *uplo) {
  if (a == NULL) a = ra;
  THArgCheck(a->nDimension == 2, 1, "A should be 2 dimensional");
  THArgCheck(a->size[0] == a->size[1], 1, "A should be square");

  masterCommandChannel->sendMessage(
    packMessage(Functions::tensorPotrf, ra, a, uplo[0]),
    THDState::s_current_worker
  );

  THDTensor_(free)(THDTensor_(cloneColumnMajor)(ra, a));
}

void THDTensor_(potrs)(THDTensor *rb, THDTensor *b, THDTensor *a,  const char *uplo) {
  bool free_b = false;
  if (b == NULL) b = rb;

  THArgCheck(a->nDimension == 2, 2, "A should have 2 dimensions, but has %d",
      a->nDimension);
  THArgCheck(b->nDimension == 1 || b->nDimension == 2, 1, "B should have 1 or 2 "
      "dimensions, but has %d", b->nDimension);
  THArgCheck(a->size[0] == a->size[1], 2, "A should be square, but is %ldx%ld",
      a->size[0], a->size[1]);
  THArgCheck(a->size[0] == b->size[0], 2, "A,B size incompatible - A has %ld "
      "rows, B has %ld", a->size[0], b->size[0]);

  if (b->nDimension == 1) {
    b = THDTensor_(newWithStorage2d)(b->storage, b->storageOffset, b->size[0],
            b->stride[0], 1, 0);
    free_b = true;
  }

  masterCommandChannel->sendMessage(
    packMessage(Functions::tensorPotrs, rb, b, a, uplo[0]),
    THDState::s_current_worker
  );

  THDTensor_(free)(THDTensor_(cloneColumnMajor)(NULL, a));
  THDTensor_(free)(THDTensor_(cloneColumnMajor)(rb, b));

  if (free_b) THDTensor_(free)(b);
}

void THDTensor_(potri)(THDTensor *ra, THDTensor *a, const char *uplo) {
  if (a == NULL) a = ra;
  THArgCheck(a->nDimension == 2, 1, "A should be 2 dimensional");
  THArgCheck(a->size[0] == a->size[1], 1, "A should be square");

  masterCommandChannel->sendMessage(
    packMessage(Functions::tensorPotri, ra, a, uplo[0]),
    THDState::s_current_worker
  );

  THDTensor_(free)(THDTensor_(cloneColumnMajor)(ra, a));
}

void THDTensor_(qr)(THDTensor *rq, THDTensor *rr, THDTensor *a) {
  int m = a->size[0];
  int n = a->size[1];
  int k = (m < n ? m : n);
  THDTensor *ra = THDTensor_(new)();
  THDTensor *rtau = THDTensor_(new)();
  THDTensor *rr_ = THDTensor_(new)();
  THDTensor_(geqrf)(ra, rtau, a);
  THDTensor_(resize2d)(rr_, k, ra->size[1]);
  THDTensor_(narrow)(rr_, ra, 0, 0, k);
  THDTensor_(triu)(rr_, rr_, 0);
  THDTensor_(resize2d)(rq, ra->size[0], k);
  THDTensor_(orgqr)(rq, ra, rtau);
  THDTensor_(narrow)(rq, rq, 1, 0, k);
  THDTensor_(free)(ra);
  THDTensor_(free)(rtau);
  THDTensor_(free)(rr_);
}

void THDTensor_(geqrf)(THDTensor *ra, THDTensor *rtau, THDTensor *a) {
  if (a == NULL) ra = a;
  THArgCheck(a->nDimension == 2, 1, "A should be 2 dimensional");

  masterCommandChannel->sendMessage(
    packMessage(Functions::tensorGeqrf, ra, rtau, a),
    THDState::s_current_worker
  );

  THDTensor *ra_ = THDTensor_(cloneColumnMajor)(ra, a);

  int m = ra_->size[0];
  int n = ra_->size[1];
  int k = (m < n ? m : n);
  THDTensor_(resize1d)(rtau, k);
  THDTensor_(free)(ra);
}

void THDTensor_(orgqr)(THDTensor *ra, THDTensor *a, THDTensor *tau) {
  if (a == NULL) a = ra;
  THArgCheck(a->nDimension == 2, 1, "A should be 2 dimensional");

  masterCommandChannel->sendMessage(
    packMessage(Functions::tensorOrgqr, ra, a),
    THDState::s_current_worker
  );

  THDTensor_(free)(THDTensor_(cloneColumnMajor)(ra, a));
}

void THDTensor_(ormqr)(THDTensor *ra, THDTensor *a, THDTensor *tau, THDTensor *c,
                       const char *side, const char *trans) {
  if (a == NULL) a = ra;
  THArgCheck(a->nDimension == 2, 1, "A should be 2 dimensional");
  masterCommandChannel->sendMessage(
    packMessage(Functions::tensorOrmqr, ra, a, tau, c, side[0], trans[0]),
    THDState::s_current_worker
  );
  THDTensor_(free)(THDTensor_(cloneColumnMajor)(ra, c));
}

void THDTensor_(pstrf)(THDTensor *ra, THDIntTensor *rpiv, THDTensor*a,
                       const char* uplo, real tol) {
  THArgCheck(a->nDimension == 2, 1, "A should be 2 dimensional");
  THArgCheck(a->size[0] == a->size[1], 1, "A should be square");

  masterCommandChannel->sendMessage(
    packMessage(Functions::tensorPstrf, ra, rpiv, a, uplo[0], tol),
    THDState::s_current_worker
  );

  int n = a->size[0];

  THDTensor *ra_ = THDTensor_(cloneColumnMajor)(ra, a);
  THDIntTensor_resize1d(rpiv, n);

  THDTensor_(free)(ra_);
}

#endif
