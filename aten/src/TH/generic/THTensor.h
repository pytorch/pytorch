#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "TH/generic/THTensor.h"
#else

/* a la lua? dim, storageoffset, ...  et les methodes ? */

#include <c10/core/TensorImpl.h>

#define THTensor at::TensorImpl

// These used to be distinct types; for some measure of backwards compatibility and documentation
// alias these to the single THTensor type.
#define THFloatTensor THTensor
#define THDoubleTensor THTensor
#define THHalfTensor THTensor
#define THByteTensor THTensor
#define THCharTensor THTensor
#define THShortTensor THTensor
#define THIntTensor THTensor
#define THLongTensor THTensor
#define THBoolTensor THTensor
#define THBFloat16Tensor THTensor

/**** access methods ****/
TH_API THStorage* THTensor_(storage)(const THTensor *self);
TH_API ptrdiff_t THTensor_(storageOffset)(const THTensor *self);

// See [NOTE: nDimension vs nDimensionLegacyNoScalars vs nDimensionLegacyAll]
TH_API int THTensor_(nDimension)(const THTensor *self);
TH_API int THTensor_(nDimensionLegacyNoScalars)(const THTensor *self);
TH_API int THTensor_(nDimensionLegacyAll)(const THTensor *self);
TH_API int64_t THTensor_(size)(const THTensor *self, int dim);
TH_API int64_t THTensor_(stride)(const THTensor *self, int dim);
TH_API scalar_t *THTensor_(data)(const THTensor *self);


/**** creation methods ****/
TH_API THTensor *THTensor_(new)(void);
TH_API THTensor *THTensor_(newWithTensor)(THTensor *tensor);
TH_API THTensor *THTensor_(newWithStorage1d)(THStorage *storage_, ptrdiff_t storageOffset_,
                                int64_t size0_, int64_t stride0_);

/* stride might be NULL */
TH_API THTensor *THTensor_(newWithSize1d)(int64_t size0_);

TH_API THTensor *THTensor_(newClone)(THTensor *self);
TH_API THTensor *THTensor_(newContiguous)(THTensor *tensor);
TH_API THTensor *THTensor_(newSelect)(THTensor *tensor, int dimension_, int64_t sliceIndex_);
TH_API THTensor *THTensor_(newNarrow)(THTensor *tensor, int dimension_, int64_t firstIndex_, int64_t size_);
TH_API THTensor *THTensor_(newTranspose)(THTensor *tensor, int dimension1_, int dimension2_);

// resize* methods simply resize the storage. So they may not retain the current data at current indices.
// This is especially likely to happen when the tensor is not contiguous. In general, if you still need the
// values, unless you are doing some size and stride tricks, do not use resize*.
TH_API void THTensor_(resizeNd)(THTensor *tensor, int nDimension, const int64_t *size, const int64_t *stride);
TH_API void THTensor_(resizeAs)(THTensor *tensor, THTensor *src);
TH_API void THTensor_(resize0d)(THTensor *tensor);
TH_API void THTensor_(resize1d)(THTensor *tensor, int64_t size0_);
TH_API void THTensor_(resize2d)(THTensor *tensor, int64_t size0_, int64_t size1_);
TH_API void THTensor_(resize3d)(THTensor *tensor, int64_t size0_, int64_t size1_, int64_t size2_);
TH_API void THTensor_(resize4d)(THTensor *tensor, int64_t size0_, int64_t size1_, int64_t size2_, int64_t size3_);
TH_API void THTensor_(resize5d)(THTensor *tensor, int64_t size0_, int64_t size1_, int64_t size2_, int64_t size3_, int64_t size4_);
// Note: these are legacy resize functions that treat sizes as size->size() == 0 and size->data<int64_t>() as being 0-terminated.

TH_API void THTensor_(set)(THTensor *self, THTensor *src);

TH_API void THTensor_(narrow)(THTensor *self, THTensor *src, int dimension_, int64_t firstIndex_, int64_t size_);
TH_API void THTensor_(select)(THTensor *self, THTensor *src, int dimension_, int64_t sliceIndex_);
TH_API void THTensor_(transpose)(THTensor *self, THTensor *src, int dimension1_, int dimension2_);
TH_API int THTensor_(isTransposed)(const THTensor *self);

TH_API void THTensor_(squeeze1d)(THTensor *self, THTensor *src, int dimension_);
TH_API void THTensor_(unsqueeze1d)(THTensor *self, THTensor *src, int dimension_);

TH_API int THTensor_(isContiguous)(const THTensor *self);
TH_API int THTensor_(isSameSizeAs)(const THTensor *self, const THTensor *src);
TH_API ptrdiff_t THTensor_(nElement)(const THTensor *self);

TH_API void THTensor_(retain)(THTensor *self);
TH_API void THTensor_(free)(THTensor *self);
TH_API void THTensor_(freeCopyTo)(THTensor *self, THTensor *dst);

/* Slow access methods [check everything] */
TH_API void THTensor_(set0d)(THTensor *tensor, scalar_t value);
TH_API void THTensor_(set1d)(THTensor *tensor, int64_t x0, scalar_t value);
TH_API void THTensor_(set2d)(THTensor *tensor, int64_t x0, int64_t x1, scalar_t value);
TH_API void THTensor_(set3d)(THTensor *tensor, int64_t x0, int64_t x1, int64_t x2, scalar_t value);
TH_API void THTensor_(set4d)(THTensor *tensor, int64_t x0, int64_t x1, int64_t x2, int64_t x3, scalar_t value);

TH_API scalar_t THTensor_(get0d)(const THTensor *tensor);
TH_API scalar_t THTensor_(get1d)(const THTensor *tensor, int64_t x0);
TH_API scalar_t THTensor_(get2d)(const THTensor *tensor, int64_t x0, int64_t x1);
TH_API scalar_t THTensor_(get3d)(const THTensor *tensor, int64_t x0, int64_t x1, int64_t x2);
TH_API scalar_t THTensor_(get4d)(const THTensor *tensor, int64_t x0, int64_t x1, int64_t x2, int64_t x3);

/* Debug methods */
TH_API THDescBuff THTensor_(desc)(const THTensor *tensor);
TH_API THDescBuff THTensor_(sizeDesc)(const THTensor *tensor);

#endif
