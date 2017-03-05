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

/* TODO implement all those */
void THDTensor_(gesv)(THDTensor *rb, THDTensor *ra, THDTensor *b, THDTensor *a) {}

void THDTensor_(trtrs)(THDTensor *rb, THDTensor *ra, THDTensor *b, THDTensor *a,
                       const char *uplo, const char *trans, const char *diag) {}
void THDTensor_(gels)(THDTensor *rb, THDTensor *ra, THDTensor *b, THDTensor *a) {}

void THDTensor_(syev)(THDTensor *re, THDTensor *rv, THDTensor *a,
                      const char *jobz, const char *uplo) {}
void THDTensor_(geev)(THDTensor *re, THDTensor *rv, THDTensor *a, const char *jobvr) {}
void THDTensor_(gesvd)(THDTensor *ru, THDTensor *rs, THDTensor *rv, THDTensor *a,
                       const char *jobu) {}
void THDTensor_(gesvd2)(THDTensor *ru, THDTensor *rs, THDTensor *rv, THDTensor *ra,
                        THDTensor *a, const char *jobu) {}
void THDTensor_(getri)(THDTensor *ra, THDTensor *a) {}
void THDTensor_(potrf)(THDTensor *ra, THDTensor *a, const char *uplo) {}
void THDTensor_(potrs)(THDTensor *rb, THDTensor *b, THDTensor *a,  const char *uplo) {}
void THDTensor_(potri)(THDTensor *ra, THDTensor *a, const char *uplo) {}
void THDTensor_(qr)(THDTensor *rq, THDTensor *rr, THDTensor *a) {}
void THDTensor_(geqrf)(THDTensor *ra, THDTensor *rtau, THDTensor *a) {}
void THDTensor_(orgqr)(THDTensor *ra, THDTensor *a, THDTensor *tau) {}
void THDTensor_(ormqr)(THDTensor *ra, THDTensor *a, THDTensor *tau, THDTensor *c,
                       const char *side, const char *trans) {}
void THDTensor_(pstrf)(THDTensor *ra, THDIntTensor *rpiv, THDTensor*a,
                       const char* uplo, real tol) {}

#endif
