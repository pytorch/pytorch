#ifndef THC_GENERIC_FILE
#define THC_GENERIC_FILE "THC/generic/THCTensor.h"
#else

#define THCTensor THTensor

// These used to be distinct types; for some measure of backwards compatibility and documentation
// alias these to the single THCTensor type.
#define THCudaTensor THCTensor
#define THCudaDoubleTensor THCTensor
#define THCudaHalfTensor THCTensor
#define THCudaByteTensor THCTensor
#define THCudaCharTensor THCTensor
#define THCudaShortTensor THCTensor
#define THCudaIntTensor THCTensor
#define THCudaLongTensor THCTensor
#define THCudaBoolTensor THCTensor
#define THCudaBFloat16Tensor THCTensor

/**** access methods ****/
THC_API THCStorage* THCTensor_(storage)(THCState *state, const THCTensor *self);
THC_API ptrdiff_t THCTensor_(storageOffset)(THCState *state, const THCTensor *self);

// See [NOTE: nDimension vs nDimensionLegacyNoScalars vs nDimensionLegacyAll]
THC_API int THCTensor_(nDimension)(THCState *state, const THCTensor *self);
THC_API int THCTensor_(nDimensionLegacyNoScalars)(THCState *state, const THCTensor *self);
THC_API int THCTensor_(nDimensionLegacyAll)(THCState *state, const THCTensor *self);

THC_API int64_t THCTensor_(size)(THCState *state, const THCTensor *self, int dim);
THC_API int64_t THCTensor_(sizeLegacyNoScalars)(THCState *state, const THCTensor *self, int dim);
THC_API int64_t THCTensor_(stride)(THCState *state, const THCTensor *self, int dim);
THC_API int64_t THCTensor_(strideLegacyNoScalars)(THCState *state, const THCTensor *self, int dim);
THC_API scalar_t *THCTensor_(data)(THCState *state, const THCTensor *self);

THC_API void THCTensor_(setFlag)(THCState *state, THCTensor *self, const char flag);
THC_API void THCTensor_(clearFlag)(THCState *state, THCTensor *self, const char flag);


/**** creation methods ****/
THC_API THCTensor *THCTensor_(new)(THCState *state);
THC_API THCTensor *THCTensor_(newWithTensor)(THCState *state, THCTensor *tensor);
THC_API THCTensor *THCTensor_(newWithStorage1d)(THCState *state, THCStorage *storage_, ptrdiff_t storageOffset_,
                                int64_t size0_, int64_t stride0_);
THC_API THCTensor *THCTensor_(newWithStorage2d)(THCState *state, THCStorage *storage_, ptrdiff_t storageOffset_,
                                int64_t size0_, int64_t stride0_,
                                int64_t size1_, int64_t stride1_);
THC_API THCTensor *THCTensor_(newWithStorage3d)(THCState *state, THCStorage *storage_, ptrdiff_t storageOffset_,
                                int64_t size0_, int64_t stride0_,
                                int64_t size1_, int64_t stride1_,
                                int64_t size2_, int64_t stride2_);
THC_API THCTensor *THCTensor_(newWithStorage4d)(THCState *state, THCStorage *storage_, ptrdiff_t storageOffset_,
                                int64_t size0_, int64_t stride0_,
                                int64_t size1_, int64_t stride1_,
                                int64_t size2_, int64_t stride2_,
                                int64_t size3_, int64_t stride3_);

/* stride might be NULL */
THC_API THCTensor *THCTensor_(newWithSize1d)(THCState *state, int64_t size0_);
THC_API THCTensor *THCTensor_(newWithSize2d)(THCState *state, int64_t size0_, int64_t size1_);
THC_API THCTensor *THCTensor_(newWithSize3d)(THCState *state, int64_t size0_, int64_t size1_, int64_t size2_);
THC_API THCTensor *THCTensor_(newWithSize4d)(THCState *state, int64_t size0_, int64_t size1_, int64_t size2_, int64_t size3_);

THC_API THCTensor *THCTensor_(newClone)(THCState *state, THCTensor *self);
THC_API THCTensor *THCTensor_(newContiguous)(THCState *state, THCTensor *tensor);
THC_API THCTensor *THCTensor_(newSelect)(THCState *state, THCTensor *tensor, int dimension_, int64_t sliceIndex_);
THC_API THCTensor *THCTensor_(newNarrow)(THCState *state, THCTensor *tensor, int dimension_, int64_t firstIndex_, int64_t size_);
THC_API THCTensor *THCTensor_(newTranspose)(THCState *state, THCTensor *tensor, int dimension1_, int dimension2_);
THC_API THCTensor *THCTensor_(newFoldBatchDim)(THCState *state, THCTensor *input);

// resize* methods simply resize the storage. So they may not retain the current data at current indices.
// This is especially likely to happen when the tensor is not contiguous. In general, if you still need the
// values, unless you are doing some size and stride tricks, do not use resize*.
THC_API void THCTensor_(resizeNd)(THCState *state, THCTensor *tensor, int nDimension, const int64_t *size, const int64_t *stride);
THC_API void THCTensor_(resizeAs)(THCState *state, THCTensor *tensor, THCTensor *src);
THC_API void THCTensor_(resize0d)(THCState *state, THCTensor *tensor);
THC_API void THCTensor_(resize1d)(THCState *state, THCTensor *tensor, int64_t size0_);
THC_API void THCTensor_(resize2d)(THCState *state, THCTensor *tensor, int64_t size0_, int64_t size1_);
THC_API void THCTensor_(resize3d)(THCState *state, THCTensor *tensor, int64_t size0_, int64_t size1_, int64_t size2_);
THC_API void THCTensor_(resize4d)(THCState *state, THCTensor *tensor, int64_t size0_, int64_t size1_, int64_t size2_, int64_t size3_);
THC_API void THCTensor_(resize5d)(THCState *state, THCTensor *tensor, int64_t size0_, int64_t size1_, int64_t size2_, int64_t size3_, int64_t size4_);

THC_API void THCTensor_(set)(THCState *state, THCTensor *self, THCTensor *src);
THC_API void THCTensor_(setStorageNd)(THCState *state, THCTensor *self, THCStorage *storage, ptrdiff_t storageOffset, int nDimension, const int64_t *size, const int64_t *stride);
THC_API void THCTensor_(setStorage1d)(THCState *state, THCTensor *self, THCStorage *storage_, ptrdiff_t storageOffset_,
                                    int64_t size0_, int64_t stride0_);
THC_API void THCTensor_(setStorage2d)(THCState *state, THCTensor *self, THCStorage *storage_, ptrdiff_t storageOffset_,
                                    int64_t size0_, int64_t stride0_,
                                    int64_t size1_, int64_t stride1_);
THC_API void THCTensor_(setStorage3d)(THCState *state, THCTensor *self, THCStorage *storage_, ptrdiff_t storageOffset_,
                                    int64_t size0_, int64_t stride0_,
                                    int64_t size1_, int64_t stride1_,
                                    int64_t size2_, int64_t stride2_);
THC_API void THCTensor_(setStorage4d)(THCState *state, THCTensor *self, THCStorage *storage_, ptrdiff_t storageOffset_,
                                    int64_t size0_, int64_t stride0_,
                                    int64_t size1_, int64_t stride1_,
                                    int64_t size2_, int64_t stride2_,
                                    int64_t size3_, int64_t stride3_);

THC_API void THCTensor_(narrow)(THCState *state, THCTensor *self, THCTensor *src, int dimension_, int64_t firstIndex_, int64_t size_);
THC_API void THCTensor_(select)(THCState *state, THCTensor *self, THCTensor *src, int dimension_, int64_t sliceIndex_);
THC_API void THCTensor_(transpose)(THCState *state, THCTensor *self, THCTensor *src, int dimension1_, int dimension2_);
THC_API void THCTensor_(unfold)(THCState *state, THCTensor *self, THCTensor *src, int dimension_, int64_t size_, int64_t step_);

THC_API void THCTensor_(squeeze)(THCState *state, THCTensor *self, THCTensor *src);
THC_API void THCTensor_(squeeze1d)(THCState *state, THCTensor *self, THCTensor *src, int dimension_);
THC_API void THCTensor_(unsqueeze1d)(THCState *state, THCTensor *self, THCTensor *src, int dimension_);

THC_API int THCTensor_(isContiguous)(THCState *state, const THCTensor *self);
THC_API int THCTensor_(isSameSizeAs)(THCState *state, const THCTensor *self, const THCTensor *src);
THC_API int THCTensor_(isSetTo)(THCState *state, const THCTensor *self, const THCTensor *src);
THC_API ptrdiff_t THCTensor_(nElement)(THCState *state, const THCTensor *self);

THC_API void THCTensor_(retain)(THCState *state, THCTensor *self);
THC_API void THCTensor_(free)(THCState *state, THCTensor *self);
THC_API void THCTensor_(freeCopyTo)(THCState *state, THCTensor *self, THCTensor *dst);

/* Slow access methods [check everything] */
THC_API void THCTensor_(set0d)(THCState *state, THCTensor *tensor, scalar_t value);
THC_API void THCTensor_(set1d)(THCState *state, THCTensor *tensor, int64_t x0, scalar_t value);
THC_API void THCTensor_(set2d)(THCState *state, THCTensor *tensor, int64_t x0, int64_t x1, scalar_t value);
THC_API void THCTensor_(set3d)(THCState *state, THCTensor *tensor, int64_t x0, int64_t x1, int64_t x2, scalar_t value);
THC_API void THCTensor_(set4d)(THCState *state, THCTensor *tensor, int64_t x0, int64_t x1, int64_t x2, int64_t x3, scalar_t value);

THC_API scalar_t THCTensor_(get0d)(THCState *state, const THCTensor *tensor);
THC_API scalar_t THCTensor_(get1d)(THCState *state, const THCTensor *tensor, int64_t x0);
THC_API scalar_t THCTensor_(get2d)(THCState *state, const THCTensor *tensor, int64_t x0, int64_t x1);
THC_API scalar_t THCTensor_(get3d)(THCState *state, const THCTensor *tensor, int64_t x0, int64_t x1, int64_t x2);
THC_API scalar_t THCTensor_(get4d)(THCState *state, const THCTensor *tensor, int64_t x0, int64_t x1, int64_t x2, int64_t x3);

/* CUDA-specific functions */
THC_API int THCTensor_(getDevice)(THCState *state, const THCTensor *self);
THC_API int THCTensor_(checkGPU)(THCState *state, unsigned int nTensors, ...);

/* debug methods */
THC_API THCDescBuff THCTensor_(sizeDesc)(THCState *state, const THCTensor *tensor);

#endif
