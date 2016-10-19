#include "TH.h"
#include "THNN.h"

#define torch_(NAME) TH_CONCAT_3(torch_, Real, NAME)
#define nn_(NAME) TH_CONCAT_3(nn_, Real, NAME)

#define THNN_CHECK_SHAPE(I1, I2)			\
  if (I1 != NULL && I2 != NULL && !THTensor_(isSameSizeAs)(I1, I2))	\
    {							\
       THDescBuff s1 = THTensor_(sizeDesc)(I1);		\
       THDescBuff s2 = THTensor_(sizeDesc)(I2);		\
       THError(#I1 " and " #I2 " shapes do not match: "	\
	       #I1 " %s, " #I2 " %s", s1.str, s2.str);	\
    }

#define THNN_CHECK_SHAPE_INDICES(I1, I2)             \
  THLongStorage *size2 = THLongTensor_newSizeOf(I2); \
  if (I1 != NULL && I2 != NULL && !THTensor_(isSize)(I1, size2)) \
    {             \
      THDescBuff s1 = THTensor_(sizeDesc)(I1);       \
      THDescBuff s2 = THLongTensor_sizeDesc(I2);     \
      THLongStorage_free(size2);                     \
      THError(#I1 " and " #I2 " shapes do not match: " \
        #I1 " %s, " #I2 " %s", s1.str, s2.str);      \
    } else {      \
      THLongStorage_free(size2);                     \
    }

#define THNN_CHECK_NELEMENT(I1, I2) \
  if (I1 != NULL && I2 != NULL ) {					\
    ptrdiff_t n1 = THTensor_(nElement)(I1);					\
    ptrdiff_t n2 = THTensor_(nElement)(I2);	                                \
    if (n1 != n2)							\
      {									\
	THDescBuff s1 = THTensor_(sizeDesc)(I1);			\
	THDescBuff s2 = THTensor_(sizeDesc)(I2);			\
	THError(#I1 " and " #I2 " have different number of elements: "	\
		#I1 "%s has %ld elements, while "			\
		#I2 "%s has %ld elements", s1.str, n1, s2.str, n2);	\
      }									\
  }

#define THNN_CHECK_DIM_SIZE(T, DIM, DIM_SIZE, SIZE)			\
  if (THTensor_(nDimension)(T) != DIM ||				\
      THTensor_(size)(T, DIM_SIZE) != SIZE) {				\
      THDescBuff s1 = THTensor_(sizeDesc)(T);				\
      THError("Need " #T " of dimension %d and " #T ".size[%d] == %d"	\
	      " but got " #T " to be of shape: %s", DIM, DIM_SIZE, SIZE, s1.str); \
  }

#define THNN_CHECK_DIM_SIZE_INDICES(T, DIM, DIM_SIZE, SIZE)			\
  if (THIndexTensor_(nDimension)(T) != DIM ||				\
      THIndexTensor_(size)(T, DIM_SIZE) != SIZE) {				\
      THDescBuff s1 = THIndexTensor_(sizeDesc)(T);				\
      THError("Need " #T " of dimension %d and " #T ".size[%d] == %d"	\
        " but got " #T " to be of shape: %s", DIM, DIM_SIZE, SIZE, s1.str); \
  }

#define THNN_ARGCHECK(COND, ARG, T, FORMAT)	\
  if (!(COND)) {				\
    THDescBuff s1 = THTensor_(sizeDesc)(T);	\
    THArgCheck(COND, ARG, FORMAT, s1.str);	\
  }

#include "generic/Abs.c"
#include "THGenerateFloatTypes.h"

#include "generic/AbsCriterion.c"
#include "THGenerateFloatTypes.h"

#include "generic/BCECriterion.c"
#include "THGenerateFloatTypes.h"

#include "generic/ClassNLLCriterion.c"
#include "THGenerateFloatTypes.h"

#include "generic/SpatialClassNLLCriterion.c"
#include "THGenerateFloatTypes.h"

#include "generic/DistKLDivCriterion.c"
#include "THGenerateFloatTypes.h"

#include "generic/ELU.c"
#include "THGenerateFloatTypes.h"

#include "generic/HardShrink.c"
#include "THGenerateFloatTypes.h"

#include "generic/HardTanh.c"
#include "THGenerateFloatTypes.h"

#include "generic/L1Cost.c"
#include "THGenerateFloatTypes.h"

#include "generic/LeakyReLU.c"
#include "THGenerateFloatTypes.h"

#include "generic/LogSigmoid.c"
#include "THGenerateFloatTypes.h"

#include "generic/LogSoftMax.c"
#include "THGenerateFloatTypes.h"

#include "generic/LookupTable.c"
#include "THGenerateFloatTypes.h"

#include "generic/MSECriterion.c"
#include "THGenerateFloatTypes.h"

#include "generic/MarginCriterion.c"
#include "THGenerateFloatTypes.h"

#include "generic/SoftMarginCriterion.c"
#include "THGenerateFloatTypes.h"

#include "generic/MultiLabelMarginCriterion.c"
#include "THGenerateFloatTypes.h"

#include "generic/MultiMarginCriterion.c"
#include "THGenerateFloatTypes.h"

#include "generic/Linear.c"
#include "THGenerateFloatTypes.h"

#include "generic/PReLU.c"
#include "THGenerateFloatTypes.h"

#include "generic/RReLU.c"
#include "THGenerateFloatTypes.h"

#include "generic/Sigmoid.c"
#include "THGenerateFloatTypes.h"

#include "generic/SmoothL1Criterion.c"
#include "THGenerateFloatTypes.h"

#include "generic/SoftMax.c"
#include "THGenerateFloatTypes.h"

#include "generic/SoftPlus.c"
#include "THGenerateFloatTypes.h"

#include "generic/SoftShrink.c"
#include "THGenerateFloatTypes.h"

#include "generic/SparseLinear.c"
#include "THGenerateFloatTypes.h"

#include "generic/Sqrt.c"
#include "THGenerateFloatTypes.h"

#include "generic/Square.c"
#include "THGenerateFloatTypes.h"

#include "generic/Tanh.c"
#include "THGenerateFloatTypes.h"

#include "generic/Threshold.c"
#include "THGenerateFloatTypes.h"

#include "generic/TemporalConvolution.c"
#include "THGenerateFloatTypes.h"

#include "generic/TemporalSubSampling.c"
#include "THGenerateFloatTypes.h"

#include "generic/TemporalMaxPooling.c"
#include "THGenerateFloatTypes.h"

#include "generic/BatchNormalization.c"
#include "THGenerateFloatTypes.h"

#include "generic/unfold.c"
#include "THGenerateFloatTypes.h"

#include "generic/SpatialConvolutionMap.c"
#include "THGenerateFloatTypes.h"

#include "generic/SpatialConvolutionMM.c"
#include "THGenerateFloatTypes.h"

#include "generic/SpatialConvolutionLocal.c"
#include "THGenerateFloatTypes.h"

#include "generic/SpatialFullConvolution.c"
#include "THGenerateFloatTypes.h"

#include "generic/SpatialFullConvolutionMap.c"
#include "THGenerateFloatTypes.h"

#include "generic/SpatialDilatedConvolution.c"
#include "THGenerateFloatTypes.h"

#include "generic/SpatialAdaptiveMaxPooling.c"
#include "THGenerateFloatTypes.h"

#include "generic/SpatialAveragePooling.c"
#include "THGenerateFloatTypes.h"

#include "generic/SpatialFractionalMaxPooling.c"
#include "THGenerateFloatTypes.h"

#include "generic/SpatialMaxPooling.c"
#include "THGenerateFloatTypes.h"

#include "generic/SpatialDilatedMaxPooling.c"
#include "THGenerateFloatTypes.h"

#include "generic/SpatialMaxUnpooling.c"
#include "THGenerateFloatTypes.h"

#include "generic/SpatialSubSampling.c"
#include "THGenerateFloatTypes.h"

#include "generic/SpatialUpSamplingNearest.c"
#include "THGenerateFloatTypes.h"

#include "generic/SpatialUpSamplingBilinear.c"
#include "THGenerateFloatTypes.h"

#include "generic/VolumetricAveragePooling.c"
#include "THGenerateFloatTypes.h"

#include "generic/VolumetricConvolution.c"
#include "THGenerateFloatTypes.h"

#include "generic/VolumetricConvolutionMM.c"
#include "THGenerateFloatTypes.h"

#include "generic/VolumetricFullConvolution.c"
#include "THGenerateFloatTypes.h"

#include "generic/VolumetricDilatedConvolution.c"
#include "THGenerateFloatTypes.h"

#include "generic/VolumetricMaxPooling.c"
#include "THGenerateFloatTypes.h"

#include "generic/VolumetricDilatedMaxPooling.c"
#include "THGenerateFloatTypes.h"

#include "generic/VolumetricMaxUnpooling.c"
#include "THGenerateFloatTypes.h"

#include "generic/SpatialReflectionPadding.c"
#include "THGenerateFloatTypes.h"

#include "generic/SpatialReplicationPadding.c"
#include "THGenerateFloatTypes.h"

#include "generic/VolumetricReplicationPadding.c"
#include "THGenerateFloatTypes.h"
