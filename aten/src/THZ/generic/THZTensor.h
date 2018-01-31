#ifndef THZ_GENERIC_FILE
#define THZ_GENERIC_FILE "generic/THZTensor.h"
#else

/* a la lua? dim, storageoffset, ...  et les methodes ? */

#define TH_TENSOR_REFCOUNTED 1

typedef struct THZTensor
{
    int64_t *size;
    int64_t *stride;
    int nDimension;

    // Note: storage->size may be greater than the recorded size
    // of a tensor
    THZStorage *storage;
    ptrdiff_t storageOffset;
    int refcount;

    char flag;

} THZTensor;


/**** access methods ****/
TH_API THZStorage* THZTensor_(storage)(const THZTensor *self);
TH_API ptrdiff_t THZTensor_(storageOffset)(const THZTensor *self);
TH_API int THZTensor_(nDimension)(const THZTensor *self);
TH_API int64_t THZTensor_(size)(const THZTensor *self, int dim);
TH_API int64_t THZTensor_(stride)(const THZTensor *self, int dim);
TH_API THLongStorage *THZTensor_(newSizeOf)(THZTensor *self);
TH_API THLongStorage *THZTensor_(newStrideOf)(THZTensor *self);
TH_API ntype *THZTensor_(data)(const THZTensor *self);

TH_API void THZTensor_(setFlag)(THZTensor *self, const char flag);
TH_API void THZTensor_(clearFlag)(THZTensor *self, const char flag);


/**** creation methods ****/
TH_API THZTensor *THZTensor_(new)(void);
TH_API THZTensor *THZTensor_(newWithTensor)(THZTensor *tensor);
/* stride might be NULL */
TH_API THZTensor *THZTensor_(newWithStorage)(THZStorage *storage_, ptrdiff_t storageOffset_, THLongStorage *size_, THLongStorage *stride_);
TH_API THZTensor *THZTensor_(newWithStorage1d)(THZStorage *storage_, ptrdiff_t storageOffset_,
                                int64_t size0_, int64_t stride0_);
TH_API THZTensor *THZTensor_(newWithStorage2d)(THZStorage *storage_, ptrdiff_t storageOffset_,
                                int64_t size0_, int64_t stride0_,
                                int64_t size1_, int64_t stride1_);
TH_API THZTensor *THZTensor_(newWithStorage3d)(THZStorage *storage_, ptrdiff_t storageOffset_,
                                int64_t size0_, int64_t stride0_,
                                int64_t size1_, int64_t stride1_,
                                int64_t size2_, int64_t stride2_);
TH_API THZTensor *THZTensor_(newWithStorage4d)(THZStorage *storage_, ptrdiff_t storageOffset_,
                                int64_t size0_, int64_t stride0_,
                                int64_t size1_, int64_t stride1_,
                                int64_t size2_, int64_t stride2_,
                                int64_t size3_, int64_t stride3_);

/* stride might be NULL */
TH_API THZTensor *THZTensor_(newWithSize)(THLongStorage *size_, THLongStorage *stride_);
TH_API THZTensor *THZTensor_(newWithSize1d)(int64_t size0_);
TH_API THZTensor *THZTensor_(newWithSize2d)(int64_t size0_, int64_t size1_);
TH_API THZTensor *THZTensor_(newWithSize3d)(int64_t size0_, int64_t size1_, int64_t size2_);
TH_API THZTensor *THZTensor_(newWithSize4d)(int64_t size0_, int64_t size1_, int64_t size2_, int64_t size3_);

TH_API THZTensor *THZTensor_(newClone)(THZTensor *self);
TH_API THZTensor *THZTensor_(newContiguous)(THZTensor *tensor);
TH_API THZTensor *THZTensor_(newSelect)(THZTensor *tensor, int dimension_, int64_t sliceIndex_);
TH_API THZTensor *THZTensor_(newNarrow)(THZTensor *tensor, int dimension_, int64_t firstIndex_, int64_t size_);
TH_API THZTensor *THZTensor_(newTranspose)(THZTensor *tensor, int dimension1_, int dimension2_);
TH_API THZTensor *THZTensor_(newUnfold)(THZTensor *tensor, int dimension_, int64_t size_, int64_t step_);
TH_API THZTensor *THZTensor_(newView)(THZTensor *tensor, THLongStorage *size);
TH_API THZTensor *THZTensor_(newExpand)(THZTensor *tensor, THLongStorage *size);

TH_API void THZTensor_(expand)(THZTensor *r, THZTensor *tensor, THLongStorage *size);
TH_API void THZTensor_(expandNd)(THZTensor **rets, THZTensor **ops, int count);

TH_API void THZTensor_(resize)(THZTensor *tensor, THLongStorage *size, THLongStorage *stride);
TH_API void THZTensor_(resizeAs)(THZTensor *tensor, THZTensor *src);
TH_API void THZTensor_(resizeNd)(THZTensor *tensor, int nDimension, int64_t *size, int64_t *stride);
TH_API void THZTensor_(resize1d)(THZTensor *tensor, int64_t size0_);
TH_API void THZTensor_(resize2d)(THZTensor *tensor, int64_t size0_, int64_t size1_);
TH_API void THZTensor_(resize3d)(THZTensor *tensor, int64_t size0_, int64_t size1_, int64_t size2_);
TH_API void THZTensor_(resize4d)(THZTensor *tensor, int64_t size0_, int64_t size1_, int64_t size2_, int64_t size3_);
TH_API void THZTensor_(resize5d)(THZTensor *tensor, int64_t size0_, int64_t size1_, int64_t size2_, int64_t size3_, int64_t size4_);

TH_API void THZTensor_(set)(THZTensor *self, THZTensor *src);
TH_API void THZTensor_(setStorage)(THZTensor *self, THZStorage *storage_, ptrdiff_t storageOffset_, THLongStorage *size_, THLongStorage *stride_);
TH_API void THZTensor_(setStorageNd)(THZTensor *self, THZStorage *storage_, ptrdiff_t storageOffset_, int nDimension, int64_t *size, int64_t *stride);
TH_API void THZTensor_(setStorage1d)(THZTensor *self, THZStorage *storage_, ptrdiff_t storageOffset_,
                                    int64_t size0_, int64_t stride0_);
TH_API void THZTensor_(setStorage2d)(THZTensor *self, THZStorage *storage_, ptrdiff_t storageOffset_,
                                    int64_t size0_, int64_t stride0_,
                                    int64_t size1_, int64_t stride1_);
TH_API void THZTensor_(setStorage3d)(THZTensor *self, THZStorage *storage_, ptrdiff_t storageOffset_,
                                    int64_t size0_, int64_t stride0_,
                                    int64_t size1_, int64_t stride1_,
                                    int64_t size2_, int64_t stride2_);
TH_API void THZTensor_(setStorage4d)(THZTensor *self, THZStorage *storage_, ptrdiff_t storageOffset_,
                                    int64_t size0_, int64_t stride0_,
                                    int64_t size1_, int64_t stride1_,
                                    int64_t size2_, int64_t stride2_,
                                    int64_t size3_, int64_t stride3_);

TH_API void THZTensor_(narrow)(THZTensor *self, THZTensor *src, int dimension_, int64_t firstIndex_, int64_t size_);
TH_API void THZTensor_(select)(THZTensor *self, THZTensor *src, int dimension_, int64_t sliceIndex_);
TH_API void THZTensor_(transpose)(THZTensor *self, THZTensor *src, int dimension1_, int dimension2_);
TH_API void THZTensor_(unfold)(THZTensor *self, THZTensor *src, int dimension_, int64_t size_, int64_t step_);

TH_API void THZTensor_(squeeze)(THZTensor *self, THZTensor *src);
TH_API void THZTensor_(squeeze1d)(THZTensor *self, THZTensor *src, int dimension_);
TH_API void THZTensor_(unsqueeze1d)(THZTensor *self, THZTensor *src, int dimension_);

TH_API int THZTensor_(isContiguous)(const THZTensor *self);
TH_API int THZTensor_(isSameSizeAs)(const THZTensor *self, const THZTensor *src);
TH_API int THZTensor_(isSetTo)(const THZTensor *self, const THZTensor *src);
TH_API int THZTensor_(isSize)(const THZTensor *self, const THLongStorage *dims);
TH_API ptrdiff_t THZTensor_(nElement)(const THZTensor *self);

TH_API void THZTensor_(retain)(THZTensor *self);
TH_API void THZTensor_(free)(THZTensor *self);
TH_API void THZTensor_(freeCopyTo)(THZTensor *self, THZTensor *dst);

/* Slow access methods [check everything] */
TH_API void THZTensor_(set1d)(THZTensor *tensor, int64_t x0, ntype value);
TH_API void THZTensor_(set2d)(THZTensor *tensor, int64_t x0, int64_t x1, ntype value);
TH_API void THZTensor_(set3d)(THZTensor *tensor, int64_t x0, int64_t x1, int64_t x2, ntype value);
TH_API void THZTensor_(set4d)(THZTensor *tensor, int64_t x0, int64_t x1, int64_t x2, int64_t x3, ntype value);

TH_API ntype THZTensor_(get1d)(const THZTensor *tensor, int64_t x0);
TH_API ntype THZTensor_(get2d)(const THZTensor *tensor, int64_t x0, int64_t x1);
TH_API ntype THZTensor_(get3d)(const THZTensor *tensor, int64_t x0, int64_t x1, int64_t x2);
TH_API ntype THZTensor_(get4d)(const THZTensor *tensor, int64_t x0, int64_t x1, int64_t x2, int64_t x3);

/* Debug methods */
TH_API THDescBuff THZTensor_(desc)(const THZTensor *tensor);
TH_API THDescBuff THZTensor_(sizeDesc)(const THZTensor *tensor);

#endif
