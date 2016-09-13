#include "TH.h"
#include "THNN.h"

#define torch_(NAME) TH_CONCAT_3(torch_, Real, NAME)
#define nn_(NAME) TH_CONCAT_3(nn_, Real, NAME)

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
