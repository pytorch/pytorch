#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "master_worker/master/generic/THDTensor.h"
#else

typedef struct {
  unsigned long long tensor_id;

  long *size;
  long *stride;
  int nDimension;

  THDStorage *storage;
  ptrdiff_t storageOffset;

  int refcount;
  char flag;
} THDTensor;

/**** access methods ****/
THD_API THDStorage* THDTensor_(storage)(const THDTensor *self);
THD_API ptrdiff_t THDTensor_(storageOffset)(const THDTensor *self);
THD_API int THDTensor_(nDimension)(const THDTensor *self);
THD_API long THDTensor_(size)(const THDTensor *self, int dim);
THD_API long THDTensor_(stride)(const THDTensor *self, int dim);
THD_API THLongStorage *THDTensor_(newSizeOf)(THDTensor *self);
THD_API THLongStorage *THDTensor_(newStrideOf)(THDTensor *self);
THD_API real *THDTensor_(data)(const THDTensor *self);

THD_API void THDTensor_(setFlag)(THDTensor *self, const char flag);
THD_API void THDTensor_(clearFlag)(THDTensor *self, const char flag);


/**** creation methods ****/
THD_API THDTensor *THDTensor_(new)(void);
THD_API THDTensor *THDTensor_(newWithTensor)(THDTensor *tensor);
/* stride might be NULL */
THD_API THDTensor *THDTensor_(newWithStorage)(THDStorage *storage_, ptrdiff_t storageOffset_, THLongStorage *size_, THLongStorage *stride_);
THD_API THDTensor *THDTensor_(newWithStorage1d)(THDStorage *storage_, ptrdiff_t storageOffset_,
                                long size0_, long stride0_);
THD_API THDTensor *THDTensor_(newWithStorage2d)(THDStorage *storage_, ptrdiff_t storageOffset_,
                                long size0_, long stride0_,
                                long size1_, long stride1_);
THD_API THDTensor *THDTensor_(newWithStorage3d)(THDStorage *storage_, ptrdiff_t storageOffset_,
                                long size0_, long stride0_,
                                long size1_, long stride1_,
                                long size2_, long stride2_);
THD_API THDTensor *THDTensor_(newWithStorage4d)(THDStorage *storage_, ptrdiff_t storageOffset_,
                                long size0_, long stride0_,
                                long size1_, long stride1_,
                                long size2_, long stride2_,
                                long size3_, long stride3_);

/* stride might be NULL */
THD_API THDTensor *THDTensor_(newWithSize)(THLongStorage *size_, THLongStorage *stride_);
THD_API THDTensor *THDTensor_(newWithSize1d)(long size0_);
THD_API THDTensor *THDTensor_(newWithSize2d)(long size0_, long size1_);
THD_API THDTensor *THDTensor_(newWithSize3d)(long size0_, long size1_, long size2_);
THD_API THDTensor *THDTensor_(newWithSize4d)(long size0_, long size1_, long size2_, long size3_);

THD_API THDTensor *THDTensor_(newClone)(THDTensor *self);
THD_API THDTensor *THDTensor_(newContiguous)(THDTensor *tensor);
THD_API THDTensor *THDTensor_(newSelect)(THDTensor *tensor, int dimension_, long sliceIndex_);
THD_API THDTensor *THDTensor_(newNarrow)(THDTensor *tensor, int dimension_, long firstIndex_, long size_);
THD_API THDTensor *THDTensor_(newTranspose)(THDTensor *tensor, int dimension1_, int dimension2_);
THD_API THDTensor *THDTensor_(newUnfold)(THDTensor *tensor, int dimension_, long size_, long step_);

THD_API void THDTensor_(resize)(THDTensor *tensor, THLongStorage *size, THLongStorage *stride);
THD_API void THDTensor_(resizeAs)(THDTensor *tensor, THDTensor *src);
THD_API void THDTensor_(resize1d)(THDTensor *tensor, long size0_);
THD_API void THDTensor_(resize2d)(THDTensor *tensor, long size0_, long size1_);
THD_API void THDTensor_(resize3d)(THDTensor *tensor, long size0_, long size1_, long size2_);
THD_API void THDTensor_(resize4d)(THDTensor *tensor, long size0_, long size1_, long size2_, long size3_);
THD_API void THDTensor_(resize5d)(THDTensor *tensor, long size0_, long size1_, long size2_, long size3_, long size4_);

THD_API void THDTensor_(set)(THDTensor *self, THDTensor *src);
THD_API void THDTensor_(setStorage)(THDTensor *self, THDStorage *storage_, ptrdiff_t storageOffset_, THLongStorage *size_, THLongStorage *stride_);
THD_API void THDTensor_(setStorage1d)(THDTensor *self, THDStorage *storage_, ptrdiff_t storageOffset_,
                                    long size0_, long stride0_);
THD_API void THDTensor_(setStorage2d)(THDTensor *self, THDStorage *storage_, ptrdiff_t storageOffset_,
                                    long size0_, long stride0_,
                                    long size1_, long stride1_);
THD_API void THDTensor_(setStorage3d)(THDTensor *self, THDStorage *storage_, ptrdiff_t storageOffset_,
                                    long size0_, long stride0_,
                                    long size1_, long stride1_,
                                    long size2_, long stride2_);
THD_API void THDTensor_(setStorage4d)(THDTensor *self, THDStorage *storage_, ptrdiff_t storageOffset_,
                                    long size0_, long stride0_,
                                    long size1_, long stride1_,
                                    long size2_, long stride2_,
                                    long size3_, long stride3_);

THD_API void THDTensor_(narrow)(THDTensor *self, THDTensor *src, int dimension_, long firstIndex_, long size_);
THD_API void THDTensor_(select)(THDTensor *self, THDTensor *src, int dimension_, long sliceIndex_);
THD_API void THDTensor_(transpose)(THDTensor *self, THDTensor *src, int dimension1_, int dimension2_);
THD_API void THDTensor_(unfold)(THDTensor *self, THDTensor *src, int dimension_, long size_, long step_);

THD_API void THDTensor_(squeeze)(THDTensor *self, THDTensor *src);
THD_API void THDTensor_(squeeze1d)(THDTensor *self, THDTensor *src, int dimension_);

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
THD_API void THDTensor_(set3d)(THDTensor *tensor, long x0, long x1, long x2, real value);
THD_API void THDTensor_(set4d)(THDTensor *tensor, long x0, long x1, long x2, long x3, real value);

THD_API real THDTensor_(get1d)(const THDTensor *tensor, long x0);
THD_API real THDTensor_(get2d)(const THDTensor *tensor, long x0, long x1);
THD_API real THDTensor_(get3d)(const THDTensor *tensor, long x0, long x1, long x2);
THD_API real THDTensor_(get4d)(const THDTensor *tensor, long x0, long x1, long x2, long x3);

// TODO: move to THDTensorMath.h
THD_API void THDTensor_(fill)(THDTensor *r_, real value);
THD_API void THDTensor_(zeros)(THDTensor *r_, THLongStorage *size);
THD_API void THDTensor_(ones)(THDTensor *r_, THLongStorage *size);
THD_API ptrdiff_t THDTensor_(numel)(THDTensor *t);


#endif
