#include "TH.h"
#include "THNN.h"

#include "THTensor.hpp"
#include <cmath>

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
  if (I1 != NULL && I2 != NULL && !I1->sizes().equals(I2->sizes())) \
    {             \
      THDescBuff s1 = THTensor_(sizeDesc)(I1);       \
      THDescBuff s2 = THLongTensor_sizeDesc(I2);     \
      THError(#I1 " and " #I2 " shapes do not match: " \
        #I1 " %s, " #I2 " %s", s1.str, s2.str);      \
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
  if (THTensor_(nDimensionLegacyNoScalars)(T) != DIM ||				\
      THTensor_sizeLegacyNoScalars(T, DIM_SIZE) != SIZE) {				\
      THDescBuff s1 = THTensor_(sizeDesc)(T);				\
      THError("Need " #T " of dimension %d and " #T ".size[%d] == %d"	\
	      " but got " #T " to be of shape: %s", DIM, DIM_SIZE, SIZE, s1.str); \
  }

#define THNN_CHECK_DIM_SIZE_INDICES(T, DIM, DIM_SIZE, SIZE)			\
  if (THIndexTensor_(nDimensionLegacyNoScalars)(T) != DIM ||				\
      THTensor_sizeLegacyNoScalars(T, DIM_SIZE) != SIZE) {				\
      THDescBuff s1 = THIndexTensor_(sizeDesc)(T);				\
      THError("Need " #T " of dimension %d and " #T ".size[%d] == %d"	\
        " but got " #T " to be of shape: %s", DIM, DIM_SIZE, SIZE, s1.str); \
  }

#define THNN_ARGCHECK(COND, ARG, T, FORMAT)	\
  if (!(COND)) {				\
    THDescBuff s1 = THTensor_(sizeDesc)(T);	\
    THArgCheck(COND, ARG, FORMAT, s1.str);	\
  }

#include "generic/AbsCriterion.c"
#include "THGenerateFloatTypes.h"

#include "generic/BCECriterion.c"
#include "THGenerateFloatTypes.h"

#include "generic/ClassNLLCriterion.c"
#include "THGenerateFloatTypes.h"

#include "generic/Col2Im.c"
#include "THGenerateFloatTypes.h"

#include "generic/ELU.c"
#include "THGenerateFloatTypes.h"

#include "generic/HardTanh.c"
#include "THGenerateFloatTypes.h"

#include "generic/Im2Col.c"
#include "THGenerateFloatTypes.h"

#include "generic/GatedLinearUnit.c"
#include "THGenerateFloatTypes.h"

#include "generic/LeakyReLU.c"
#include "THGenerateFloatTypes.h"

#include "generic/LogSigmoid.c"
#include "THGenerateFloatTypes.h"

#include "generic/MSECriterion.c"
#include "THGenerateFloatTypes.h"

#include "generic/SoftMarginCriterion.c"
#include "THGenerateFloatTypes.h"

#include "generic/MultiLabelMarginCriterion.c"
#include "THGenerateFloatTypes.h"

#include "generic/MultiMarginCriterion.c"
#include "THGenerateFloatTypes.h"

#include "generic/RReLU.c"
#include "THGenerateFloatTypes.h"

#include "generic/Sigmoid.c"
#include "THGenerateFloatTypes.h"

#include "generic/SmoothL1Criterion.c"
#include "THGenerateFloatTypes.h"

#include "generic/SoftPlus.c"
#include "THGenerateFloatTypes.h"

#include "generic/SoftShrink.c"
#include "THGenerateFloatTypes.h"

#include "generic/SparseLinear.c"
#include "THGenerateFloatTypes.h"

#include "generic/IndexLinear.c"
#include "THGenerateFloatTypes.h"

#include "generic/Tanh.c"
#include "THGenerateFloatTypes.h"

#include "generic/TemporalRowConvolution.c"
#include "THGenerateFloatTypes.h"

#include "generic/TemporalUpSamplingNearest.c"
#include "THGenerateFloatTypes.h"

#include "generic/TemporalUpSamplingLinear.c"
#include "THGenerateFloatTypes.h"

#include "generic/FeatureLPPooling.c"
#include "THGenerateFloatTypes.h"

#include "generic/unfold.c"
#include "THGenerateFloatTypes.h"

#include "generic/SpatialConvolutionMM.c"
#include "THGenerateFloatTypes.h"

#include "generic/SpatialFullDilatedConvolution.c"
#include "THGenerateFloatTypes.h"

#include "generic/SpatialDilatedConvolution.c"
#include "THGenerateFloatTypes.h"

#include "generic/SpatialAdaptiveMaxPooling.c"
#include "THGenerateFloatTypes.h"

#include "generic/SpatialAdaptiveAveragePooling.c"
#include "THGenerateFloatTypes.h"

#include "generic/SpatialAveragePooling.c"
#include "THGenerateFloatTypes.h"

#include "generic/SpatialFractionalMaxPooling.c"
#include "THGenerateFloatTypes.h"

#include "generic/SpatialDilatedMaxPooling.c"
#include "THGenerateFloatTypes.h"

#include "generic/SpatialMaxUnpooling.c"
#include "THGenerateFloatTypes.h"

#include "generic/SpatialUpSamplingNearest.c"
#include "THGenerateFloatTypes.h"

#include "generic/SpatialUpSamplingBilinear.c"
#include "THGenerateFloatTypes.h"

#include "generic/VolumetricAveragePooling.c"
#include "THGenerateFloatTypes.h"

#include "generic/VolumetricConvolutionMM.c"
#include "THGenerateFloatTypes.h"

#include "generic/VolumetricFullDilatedConvolution.c"
#include "THGenerateFloatTypes.h"

#include "generic/VolumetricDilatedConvolution.c"
#include "THGenerateFloatTypes.h"

#include "generic/VolumetricAdaptiveMaxPooling.c"
#include "THGenerateFloatTypes.h"

#include "generic/VolumetricAdaptiveAveragePooling.c"
#include "THGenerateFloatTypes.h"

#include "generic/VolumetricDilatedMaxPooling.c"
#include "THGenerateFloatTypes.h"

#include "generic/VolumetricFractionalMaxPooling.c"
#include "THGenerateFloatTypes.h"

#include "generic/VolumetricMaxUnpooling.c"
#include "THGenerateFloatTypes.h"

#include "generic/SpatialReflectionPadding.c"
#include "THGenerateFloatTypes.h"

#include "generic/SpatialReplicationPadding.c"
#include "THGenerateFloatTypes.h"

#include "generic/TemporalReflectionPadding.c"
#include "THGenerateFloatTypes.h"

#include "generic/TemporalReplicationPadding.c"
#include "THGenerateFloatTypes.h"

#include "generic/VolumetricReplicationPadding.c"
#include "THGenerateFloatTypes.h"

#include "generic/VolumetricUpSamplingNearest.c"
#include "THGenerateFloatTypes.h"

#include "generic/VolumetricUpSamplingTrilinear.c"
#include "THGenerateFloatTypes.h"

#include "generic/SpatialClassNLLCriterion.c"
#include "THGenerateFloatTypes.h"
