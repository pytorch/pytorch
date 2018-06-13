#ifndef THS_GENERIC_FILE
#define THS_GENERIC_FILE "generic/THSTensorMath.c"
#else

void THSTensor_(zero)(THSTensor *self) {
  THError("Internal error! This API is deprecated. Shout if you need it.");
}

void THSTensor_(zeros)(THSTensor *r_, THLongStorage *size)
{
  THError("Internal error! This API is deprecated. Shout if you need it.");
}

void THSTensor_(zerosLike)(THSTensor *r_, THSTensor *input)
{
  THError("Internal error! This API is deprecated. Shout if you need it.");
}

void THSTensor_(mul)(THSTensor *r_, THSTensor *t, real value) {
  THError("Internal error! This API is deprecated. Shout if you need it.");
}

/* TODO: add in-place support */
void THSTensor_(pow)(THSTensor *r_, THSTensor *t_, real value) {
  THError("Internal error! This API is deprecated. Shout if you need it.");
}

#if defined(THS_REAL_IS_FLOAT) || defined(THS_REAL_IS_DOUBLE)
accreal THSTensor_(normall)(THSTensor *self, real value) {
  THError("Internal error! THSTensor_(normall)(self, value) shouldn't be called; use self.norm(value) instead");
}

/* floating point only, because that is what TH supports */
#endif

void THSTensor_(div)(THSTensor *r_, THSTensor *t, real value) {
  THError("Internal error! This API is deprecated. Shout if you need it.");
}

int THSTensor_(isSameSizeIgnoringDensity)(const THSTensor *self, const THSTensor* src) {
  THError("Internal error! This API is deprecated. Shout if you need it.");
}

int THSTensor_(isSameDensity)(const THSTensor *self, const THSTensor* src) {
  THError("Internal error! This API is deprecated. Shout if you need it.");
}

void THSTensor_(cadd)(THSTensor *r_, THSTensor *t, real value, THSTensor *src) {
  THError("Internal error! This API is deprecated. Shout if you need it.");
}

void THSTensor_(csub)(THSTensor *r_, THSTensor *t, real value, THSTensor *src) {
  THError("Internal error! This API is deprecated. Shout if you need it.");
}

void THSTensor_(cmul)(THSTensor *r_, THSTensor *t_, THSTensor *src_) {
  THError("Internal error! This API is deprecated. Shout if you need it.");
}

void THTensor_(spaddcmul)(THTensor *r_, THTensor *t, real value, THSTensor *src1, THSTensor *src2) {
  THError("Internal error! This API is deprecated. Shout if you need it.");
}

THLongTensor *THSTensor_(toCSR)(int64_t const *indices, int64_t dim, int64_t nnz) {
  THError("Internal error! This API is deprecated. Shout if you need it.");
}

void THSTensor_(spaddmm)(THTensor *r_,
    real beta, THTensor *t,
    real alpha, THSTensor *sparse_, THTensor *dense) {
  THError("Internal error! This API is deprecated. Shout if you need it.");
}

void THSTensor_(sspaddmm)(THSTensor *r_,
    real beta, THSTensor *t,
    real alpha, THSTensor *sparse_, THTensor *dense) {
  THError("Internal error! This API is deprecated. Shout if you need it.");
}

void THSTensor_(hspmm)(THSTensor *r_, real alpha, THSTensor *sparse_, THTensor *dense) {
  THError("Internal error! This API is deprecated. Shout if you need it.");
}

void THSTensor_(spcadd)(THTensor *r_, THTensor *dense, real value, THSTensor *sparse_) {
  THError("Internal error! This API is deprecated. Shout if you need it.");
}

#endif
