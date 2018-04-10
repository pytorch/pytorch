#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/THTensor.h"
#else

/* a la lua? dim, storageoffset, ...  et les methodes ? */

#define TH_TENSOR_REFCOUNTED 1

typedef struct THTensor
{
    int64_t *size;
    int64_t *stride;
    int nDimension;

    // Note: storage->size may be greater than the recorded size
    // of a tensor
    THStorage *storage;
    ptrdiff_t storageOffset;
    int refcount;

    char flag;

} THTensor;


/**** access methods ****/
TH_API THStorage* THTensor_(storage)(const THTensor *self);
TH_API ptrdiff_t THTensor_(storageOffset)(const THTensor *self);
TH_API int THTensor_(nDimension)(const THTensor *self);
TH_API int64_t THTensor_(size)(const THTensor *self, int dim);
TH_API int64_t THTensor_(stride)(const THTensor *self, int dim);
TH_API THLongStorage *THTensor_(newSizeOf)(THTensor *self);
TH_API THLongStorage *THTensor_(newStrideOf)(THTensor *self);
TH_API real *THTensor_(data)(const THTensor *self);

TH_API void THTensor_(setFlag)(THTensor *self, const char flag);
TH_API void THTensor_(clearFlag)(THTensor *self, const char flag);


/**** creation methods ****/
TH_API THTensor *THTensor_(new)(void);
TH_API THTensor *THTensor_(newWithTensor)(THTensor *tensor);
/* stride might be NULL */
TH_API THTensor *THTensor_(newWithStorage)(THStorage *storage_, ptrdiff_t storageOffset_, THLongStorage *size_, THLongStorage *stride_);
TH_API THTensor *THTensor_(newWithStorage1d)(THStorage *storage_, ptrdiff_t storageOffset_,
                                int64_t size0_, int64_t stride0_);
TH_API THTensor *THTensor_(newWithStorage2d)(THStorage *storage_, ptrdiff_t storageOffset_,
                                int64_t size0_, int64_t stride0_,
                                int64_t size1_, int64_t stride1_);
TH_API THTensor *THTensor_(newWithStorage3d)(THStorage *storage_, ptrdiff_t storageOffset_,
                                int64_t size0_, int64_t stride0_,
                                int64_t size1_, int64_t stride1_,
                                int64_t size2_, int64_t stride2_);
TH_API THTensor *THTensor_(newWithStorage4d)(THStorage *storage_, ptrdiff_t storageOffset_,
                                int64_t size0_, int64_t stride0_,
                                int64_t size1_, int64_t stride1_,
                                int64_t size2_, int64_t stride2_,
                                int64_t size3_, int64_t stride3_);

/* stride might be NULL */
TH_API THTensor *THTensor_(newWithSize)(THLongStorage *size_, THLongStorage *stride_);
TH_API THTensor *THTensor_(newWithSize1d)(int64_t size0_);
TH_API THTensor *THTensor_(newWithSize2d)(int64_t size0_, int64_t size1_);
TH_API THTensor *THTensor_(newWithSize3d)(int64_t size0_, int64_t size1_, int64_t size2_);
TH_API THTensor *THTensor_(newWithSize4d)(int64_t size0_, int64_t size1_, int64_t size2_, int64_t size3_);

TH_API THTensor *THTensor_(newClone)(THTensor *self);
TH_API THTensor *THTensor_(newContiguous)(THTensor *tensor);
TH_API THTensor *THTensor_(newSelect)(THTensor *tensor, int dimension_, int64_t sliceIndex_);
TH_API THTensor *THTensor_(newNarrow)(THTensor *tensor, int dimension_, int64_t firstIndex_, int64_t size_);
TH_API THTensor *THTensor_(newTranspose)(THTensor *tensor, int dimension1_, int dimension2_);
TH_API THTensor *THTensor_(newUnfold)(THTensor *tensor, int dimension_, int64_t size_, int64_t step_);
TH_API THTensor *THTensor_(newView)(THTensor *tensor, THLongStorage *size);

// resize* methods simply resize the storage. So they may not retain the current data at current indices.
// This is especially likely to happen when the tensor is not contiguous. In general, if you still need the
// values, unless you are doing some size and stride tricks, do not use resize*.
TH_API void THTensor_(resize)(THTensor *tensor, THLongStorage *size, THLongStorage *stride);
TH_API void THTensor_(resizeAs)(THTensor *tensor, THTensor *src);
TH_API void THTensor_(resizeNd)(THTensor *tensor, int nDimension, int64_t *size, int64_t *stride);
TH_API void THTensor_(resize1d)(THTensor *tensor, int64_t size0_);
TH_API void THTensor_(resize2d)(THTensor *tensor, int64_t size0_, int64_t size1_);
TH_API void THTensor_(resize3d)(THTensor *tensor, int64_t size0_, int64_t size1_, int64_t size2_);
TH_API void THTensor_(resize4d)(THTensor *tensor, int64_t size0_, int64_t size1_, int64_t size2_, int64_t size3_);
TH_API void THTensor_(resize5d)(THTensor *tensor, int64_t size0_, int64_t size1_, int64_t size2_, int64_t size3_, int64_t size4_);

TH_API void THTensor_(set)(THTensor *self, THTensor *src);
TH_API void THTensor_(setStorage)(THTensor *self, THStorage *storage_, ptrdiff_t storageOffset_, THLongStorage *size_, THLongStorage *stride_);
TH_API void THTensor_(setStorageNd)(THTensor *self, THStorage *storage_, ptrdiff_t storageOffset_, int nDimension, int64_t *size, int64_t *stride);
TH_API void THTensor_(setStorage1d)(THTensor *self, THStorage *storage_, ptrdiff_t storageOffset_,
                                    int64_t size0_, int64_t stride0_);
TH_API void THTensor_(setStorage2d)(THTensor *self, THStorage *storage_, ptrdiff_t storageOffset_,
                                    int64_t size0_, int64_t stride0_,
                                    int64_t size1_, int64_t stride1_);
TH_API void THTensor_(setStorage3d)(THTensor *self, THStorage *storage_, ptrdiff_t storageOffset_,
                                    int64_t size0_, int64_t stride0_,
                                    int64_t size1_, int64_t stride1_,
                                    int64_t size2_, int64_t stride2_);
TH_API void THTensor_(setStorage4d)(THTensor *self, THStorage *storage_, ptrdiff_t storageOffset_,
                                    int64_t size0_, int64_t stride0_,
                                    int64_t size1_, int64_t stride1_,
                                    int64_t size2_, int64_t stride2_,
                                    int64_t size3_, int64_t stride3_);

TH_API void THTensor_(narrow)(THTensor *self, THTensor *src, int dimension_, int64_t firstIndex_, int64_t size_);
TH_API void THTensor_(select)(THTensor *self, THTensor *src, int dimension_, int64_t sliceIndex_);
TH_API void THTensor_(transpose)(THTensor *self, THTensor *src, int dimension1_, int dimension2_);
TH_API void THTensor_(unfold)(THTensor *self, THTensor *src, int dimension_, int64_t size_, int64_t step_);

TH_API void THTensor_(squeeze)(THTensor *self, THTensor *src);
TH_API void THTensor_(squeeze1d)(THTensor *self, THTensor *src, int dimension_);
TH_API void THTensor_(unsqueeze1d)(THTensor *self, THTensor *src, int dimension_);

TH_API int THTensor_(isContiguous)(const THTensor *self);
TH_API int THTensor_(isSameSizeAs)(const THTensor *self, const THTensor *src);
TH_API int THTensor_(isSetTo)(const THTensor *self, const THTensor *src);
TH_API int THTensor_(isSize)(const THTensor *self, const THLongStorage *dims);
TH_API ptrdiff_t THTensor_(nElement)(const THTensor *self);

TH_API void THTensor_(retain)(THTensor *self);
TH_API void THTensor_(free)(THTensor *self);
TH_API void THTensor_(freeCopyTo)(THTensor *self, THTensor *dst);

/* Slow access methods [check everything] */
TH_API void THTensor_(set1d)(THTensor *tensor, int64_t x0, real value);
TH_API void THTensor_(set2d)(THTensor *tensor, int64_t x0, int64_t x1, real value);
TH_API void THTensor_(set3d)(THTensor *tensor, int64_t x0, int64_t x1, int64_t x2, real value);
TH_API void THTensor_(set4d)(THTensor *tensor, int64_t x0, int64_t x1, int64_t x2, int64_t x3, real value);

TH_API real THTensor_(get1d)(const THTensor *tensor, int64_t x0);
TH_API real THTensor_(get2d)(const THTensor *tensor, int64_t x0, int64_t x1);
TH_API real THTensor_(get3d)(const THTensor *tensor, int64_t x0, int64_t x1, int64_t x2);
TH_API real THTensor_(get4d)(const THTensor *tensor, int64_t x0, int64_t x1, int64_t x2, int64_t x3);

/* Debug methods */
TH_API THDescBuff THTensor_(desc)(const THTensor *tensor);
TH_API THDescBuff THTensor_(sizeDesc)(const THTensor *tensor);

#endif
