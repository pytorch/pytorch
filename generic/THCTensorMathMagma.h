#ifndef THC_GENERIC_FILE
#define THC_GENERIC_FILE "generic/THCTensorMathMagma.h"
#else

#if defined(THC_REAL_IS_FLOAT) || defined(THC_REAL_IS_DOUBLE)

#ifdef USE_MAGMA

static void THCTensor_(copyArray1d)(THCState *state, THCTensor *self, real *src, int k)
{
  long size[1] = { k };
  long stride[1] = { 1 };
  THCTensor_(rawResize)(state, self, 1, size, stride);
  size_t len = k * sizeof(real);
  THCudaCheck(cudaMemcpy(self->storage->data + self->storageOffset, src, len, cudaMemcpyHostToDevice));
}

static void THCTensor_(copyArray2d)(THCState *state, THCTensor *self, real *src, int m, int n)
{
  long size[2] = { m, n };
  long stride[2] = { 1, m };
  THCTensor_(rawResize)(state, self, 2, size, stride);
  size_t len = m * n * sizeof(real);
  THCudaCheck(cudaMemcpy(self->storage->data + self->storageOffset, src, len, cudaMemcpyHostToDevice));
}

static void THCTensor_(copyTensor2d)(THCState *state, real *dst, THCTensor *self)
{
  THAssert(self->nDimension == 2);
  size_t len = THCTensor_(nElement)(state, self)*sizeof(real);
  THCTensor *temp = THCTensor_(newTranspose)(state, self, 0, 1);
  THCTensor *selfc = THCTensor_(newContiguous)(state, temp);
  THCudaCheck(cudaMemcpy(dst, selfc->storage->data + selfc->storageOffset, len, cudaMemcpyDeviceToHost));
  THCTensor_(free)(state, temp);
  THCTensor_(free)(state, selfc);
}

#endif // USE_MAGMA

static THCTensor* THCTensor_(newColumnMajor)(THCState *state, THCTensor *self, THCTensor *src)
{
  THAssert(src->nDimension == 2);
  if (self == src && self->stride[0] == 1 && self->stride[1] == self->size[0])
  {
    THCTensor_(retain)(state, self);
    return self;
  }

  if (self == src)
    self = THCTensor_(new)(state);
  else
    THCTensor_(retain)(state, self);

  long size[2] = { src->size[0], src->size[1] };
  long stride[2] = { 1, src->size[0] };

  THCTensor_(rawResize)(state, self, 2, size, stride);
  THCTensor_(copy)(state, self, src);
  return self;
}

THC_API void THCTensor_(gesv)(THCState *state, THCTensor *rb_, THCTensor *ra_, THCTensor *b_, THCTensor *a_);
THC_API void THCTensor_(gels)(THCState *state, THCTensor *rb_, THCTensor *ra_, THCTensor *b_, THCTensor *a_);
THC_API void THCTensor_(syev)(THCState *state, THCTensor *re_, THCTensor *rv_, THCTensor *a_, const char *jobz, const char *uplo);

#endif // defined(THC_REAL_IS_FLOAT) || defined(THC_REAL_IS_DOUBLE)

#endif
