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
Check if self contains any inf or NaN values
*/
static int THTensor_(isFinite)(THTensor *self)
{
  std::atomic<int> finite{1};
  TH_TENSOR_APPLY(scalar_t, self, if (finite && !std::isfinite(*self_data)) {
                        finite = 0;
                        TH_TENSOR_APPLY_hasFinished = 1; break;
                     });
  return finite;
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

#endif
