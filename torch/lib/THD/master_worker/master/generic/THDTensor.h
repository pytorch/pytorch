#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "master_worker/master/generic/THDTensor.h"
#else

typedef struct {
  long *size;
  long *stride;
  int nDimension;

  THDStorage *storage;
  ptrdiff_t storageOffset;

  int refcount;
  char flag;

  // Additional fields
  unsigned long long tensor_id;
} THDTensor;

/**** helper functions ****/
THD_API THDDescBuff THDTensor_(sizeDesc)(const THDTensor *tensor);

/**** access methods ****/
THD_API THDStorage* THDTensor_(storage)(const THDTensor *self);
THD_API ptrdiff_t THDTensor_(storageOffset)(const THDTensor *self);
THD_API int THDTensor_(nDimension)(const THDTensor *self);
THD_API long THDTensor_(size)(const THDTensor *self, int dim);
THD_API long THDTensor_(stride)(const THDTensor *self, int dim);
THD_API THLongStorage *THDTensor_(newSizeOf)(THDTensor *self);
THD_API THLongStorage *THDTensor_(newStrideOf)(THDTensor *self);

THD_API void THDTensor_(setFlag)(THDTensor *self, char flag);
THD_API void THDTensor_(clearFlag)(THDTensor *self, char flag);


/**** creation methods ****/
THD_API THDTensor *THDTensor_(new)(void);
THD_API THDTensor *THDTensor_(newWithTensor)(THDTensor *tensor);
/* stride might be NULL */
THD_API THDTensor *THDTensor_(newWithStorage)(THDStorage *storage_,
                                              ptrdiff_t storageOffset_,
                                              THLongStorage *size_,
                                              THLongStorage *stride_);
THD_API THDTensor *THDTensor_(newWithStorage1d)(THDStorage *storage_,
                                                ptrdiff_t storageOffset_,
                                                long size0_, long stride0_);
THD_API THDTensor *THDTensor_(newWithStorage2d)(THDStorage *storage_,
                                                ptrdiff_t storageOffset_,
                                                long size0_, long stride0_,
                                                long size1_, long stride1_);
THD_API THDTensor *THDTensor_(newWithStorage3d)(THDStorage *storage_,
                                                ptrdiff_t storageOffset_,
                                                long size0_, long stride0_,
                                                long size1_, long stride1_,
                                                long size2_, long stride2_);
THD_API THDTensor *THDTensor_(newWithStorage4d)(THDStorage *storage_,
                                                ptrdiff_t storageOffset_,
                                                long size0_, long stride0_,
                                                long size1_, long stride1_,
                                                long size2_, long stride2_,
                                                long size3_, long stride3_);

/* stride might be NULL */
THD_API THDTensor *THDTensor_(newWithSize)(THLongStorage *size_,
                                           THLongStorage *stride_);
THD_API THDTensor *THDTensor_(newWithSize1d)(long size0_);
THD_API THDTensor *THDTensor_(newWithSize2d)(long size0_, long size1_);
THD_API THDTensor *THDTensor_(newWithSize3d)(long size0_, long size1_,
                                             long size2_);
THD_API THDTensor *THDTensor_(newWithSize4d)(long size0_, long size1_,
                                             long size2_, long size3_);

THD_API THDTensor *THDTensor_(newClone)(THDTensor *self);
THD_API THDTensor *THDTensor_(newContiguous)(THDTensor *tensor);
THD_API THDTensor *THDTensor_(newSelect)(THDTensor *tensor, int dimension_,
                                         long sliceIndex_);
THD_API THDTensor *THDTensor_(newNarrow)(THDTensor *tensor, int dimension_,
                                         long firstIndex_, long size_);
THD_API THDTensor *THDTensor_(newTranspose)(THDTensor *tensor, int dimension1_,
                                            int dimension2_);
THD_API THDTensor *THDTensor_(newUnfold)(THDTensor *tensor, int dimension_,
                                         long size_, long step_);
THD_API THDTensor *THDTensor_(newView)(THDTensor *tensor, THLongStorage *size);
THD_API THDTensor *THDTensor_(newExpand)(THDTensor *tensor, THLongStorage *size);

THD_API void THDTensor_(resize)(THDTensor *tensor, THLongStorage *size,
                                THLongStorage *stride);
THD_API void THDTensor_(resizeAs)(THDTensor *tensor, THDTensor *src);
THD_API void THDTensor_(resize1d)(THDTensor *tensor, long size0_);
THD_API void THDTensor_(resize2d)(THDTensor *tensor, long size0_, long size1_);
THD_API void THDTensor_(resize3d)(THDTensor *tensor, long size0_, long size1_,
                                  long size2_);
THD_API void THDTensor_(resize4d)(THDTensor *tensor, long size0_, long size1_,
                                  long size2_, long size3_);
THD_API void THDTensor_(resize5d)(THDTensor *tensor, long size0_, long size1_,
                                  long size2_, long size3_, long size4_);

THD_API void THDTensor_(set)(THDTensor *self, THDTensor *src);
THD_API void THDTensor_(setStorage)(THDTensor *self, THDStorage *storage_,
                                    ptrdiff_t storageOffset_,
                                    THLongStorage *size_,
                                    THLongStorage *stride_);
THD_API void THDTensor_(setStorage1d)(THDTensor *self, THDStorage *storage_,
                                      ptrdiff_t storageOffset_,
                                      long size0_, long stride0_);
THD_API void THDTensor_(setStorage2d)(THDTensor *self, THDStorage *storage_,
                                      ptrdiff_t storageOffset_,
                                      long size0_, long stride0_,
                                      long size1_, long stride1_);
THD_API void THDTensor_(setStorage3d)(THDTensor *self, THDStorage *storage_,
                                      ptrdiff_t storageOffset_,
                                      long size0_, long stride0_,
                                      long size1_, long stride1_,
                                      long size2_, long stride2_);
THD_API void THDTensor_(setStorage4d)(THDTensor *self, THDStorage *storage_,
                                      ptrdiff_t storageOffset_,
                                      long size0_, long stride0_,
                                      long size1_, long stride1_,
                                      long size2_, long stride2_,
                                      long size3_, long stride3_);

THD_API void THDTensor_(narrow)(THDTensor *self, THDTensor *src, int dimension_,
                                long firstIndex_, long size_);
THD_API void THDTensor_(select)(THDTensor *self, THDTensor *src, int dimension_,
                                long sliceIndex_);
THD_API void THDTensor_(transpose)(THDTensor *self, THDTensor *src,
                                   int dimension1_, int dimension2_);
THD_API void THDTensor_(unfold)(THDTensor *self, THDTensor *src, int dimension_,
                                long size_, long step_);

THD_API void THDTensor_(squeeze)(THDTensor *self, THDTensor *src);
THD_API void THDTensor_(squeeze1d)(THDTensor *self, THDTensor *src,
                                   int dimension_);
TH_API void THDTensor_(unsqueeze1d)(THDTensor *self, THDTensor *src, int dimension_);

THD_API int THDTensor_(isContiguous)(const THDTensor *self);
THD_API int THDTensor_(isSameSizeAs)(const THDTensor *self, const THDTensor *src);
THD_API int THDTensor_(isSetTo)(const THDTensor *self, const THDTensor *src);
THD_API int THDTensor_(isSize)(const THDTensor *self, const THLongStorage *dims);
THD_API ptrdiff_t THDTensor_(nElement)(const THDTensor *self);

THD_API void THDTensor_(retain)(THDTensor *self);
THD_API void THDTensor_(free)(THDTensor *self);
THD_API void THDTensor_(freeCopyTo)(THDTensor *self, THDTensor *dst);

/* Slow access methods [check everything] */
THD_API void THDTensor_(set1d)(THDTensor *tensor, long x0, real value);
THD_API void THDTensor_(set2d)(THDTensor *tensor, long x0, long x1, real value);
THD_API void THDTensor_(set3d)(THDTensor *tensor, long x0, long x1,
                               long x2, real value);
THD_API void THDTensor_(set4d)(THDTensor *tensor, long x0, long x1,
                               long x2, long x3, real value);

THD_API real THDTensor_(get1d)(const THDTensor *tensor, long x0);
THD_API real THDTensor_(get2d)(const THDTensor *tensor, long x0, long x1);
THD_API real THDTensor_(get3d)(const THDTensor *tensor, long x0, long x1, long x2);
THD_API real THDTensor_(get4d)(const THDTensor *tensor, long x0, long x1,
                               long x2, long x3);

THD_API accreal THDTensor_(dot)(THDTensor *self, THDTensor *src);
THD_API real THDTensor_(minall)(THDTensor *self);
THD_API real THDTensor_(maxall)(THDTensor *self);
THD_API accreal THDTensor_(sumall)(THDTensor *self);
THD_API accreal THDTensor_(prodall)(THDTensor *self);
THD_API void THDTensor_(neg)(THDTensor *self, THDTensor *src);
THD_API void THDTensor_(cinv)(THDTensor *self, THDTensor *src);
THD_API void THDTensor_(add)(THDTensor *self, THDTensor *src, real value);
THD_API void THDTensor_(sub)(THDTensor *self, THDTensor *src, real value);
THD_API void THDTensor_(mul)(THDTensor *self, THDTensor *src, real value);
THD_API void THDTensor_(div)(THDTensor *self, THDTensor *src, real value);
THD_API void THDTensor_(fmod)(THDTensor *self, THDTensor *src, real value);
THD_API void THDTensor_(remainder)(THDTensor *self, THDTensor *src, real value);
THD_API void THDTensor_(clamp)(THDTensor *self, THDTensor *src, real min_value,
                               real max_value);
THD_API void THDTensor_(cadd)(THDTensor *self, THDTensor *src1, real value,
                              THDTensor *src2);
THD_API void THDTensor_(csub)(THDTensor *self, THDTensor *src1, real value,
                              THDTensor *src2);
THD_API void THDTensor_(cmul)(THDTensor *self, THDTensor *src1, THDTensor *src2);
THD_API void THDTensor_(cpow)(THDTensor *self, THDTensor *src1, THDTensor *src2);
THD_API void THDTensor_(cdiv)(THDTensor *self, THDTensor *src1, THDTensor *src2);
THD_API void THDTensor_(cfmod)(THDTensor *self, THDTensor *src1, THDTensor *src2);
THD_API void THDTensor_(cremainder)(THDTensor *self, THDTensor *src1,
                                    THDTensor *src2);
THD_API void THDTensor_(addcmul)(THDTensor *self, THDTensor *src1, real value,
                                 THDTensor *src2, THDTensor *src3);
THD_API void THDTensor_(addcdiv)(THDTensor *self, THDTensor *src1, real value,
                                 THDTensor *src2, THDTensor *src3);
THD_API void THDTensor_(addmv)(THDTensor *self, real beta, THDTensor *src,
                               real alpha, THDTensor *mat,  THDTensor *vec);
THD_API void THDTensor_(addmm)(THDTensor *self, real beta, THDTensor *src,
                               real alpha, THDTensor *mat1, THDTensor *mat2);
THD_API void THDTensor_(addr)(THDTensor *self,  real beta, THDTensor *src,
                              real alpha, THDTensor *vec1, THDTensor *vec2);
THD_API void THDTensor_(addbmm)(THDTensor *self, real beta, THDTensor *src,
                                real alpha, THDTensor *batch1, THDTensor *batch2);
THD_API void THDTensor_(baddbmm)(THDTensor *self, real beta, THDTensor *src,
                                 real alpha, THDTensor *batch1, THDTensor *batch2);
THD_API void THDTensor_(match)(THDTensor *self, THDTensor *m1,
                               THDTensor *m2, real gain);
THD_API void THDTensor_(sum)(THDTensor *self, THDTensor *src, int dimension);
THD_API void THDTensor_(prod)(THDTensor *self, THDTensor *src, int dimension);
THD_API void THDTensor_(cumsum)(THDTensor *self, THDTensor *src, int dimension);
THD_API void THDTensor_(cumprod)(THDTensor *self, THDTensor *src, int dimension);
THD_API void THDTensor_(sign)(THDTensor *self, THDTensor *src);
THD_API accreal THDTensor_(trace)(THDTensor *self);
THD_API void THDTensor_(cross)(THDTensor *self, THDTensor *src1,
                               THDTensor *src2, int dimension);
THD_API void THDTensor_(cmax)(THDTensor *self, THDTensor *src1, THDTensor *src2);
THD_API void THDTensor_(cmin)(THDTensor *self, THDTensor *src1, THDTensor *src2);
THD_API void THDTensor_(cmaxValue)(THDTensor *self, THDTensor *src, real value);
THD_API void THDTensor_(cminValue)(THDTensor *self, THDTensor *src, real value);

#endif
