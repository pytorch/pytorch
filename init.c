#include "TH.h"
#include "THNN.h"

#define torch_(NAME) TH_CONCAT_3(torch_, Real, NAME)
#define nn_(NAME) TH_CONCAT_3(nn_, Real, NAME)

#include "generic/Abs.c"
#include "THGenerateFloatTypes.h"

#include "generic/AbsCriterion.c"
#include "THGenerateFloatTypes.h"

#include "generic/ClassNLLCriterion.c"
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

#include "generic/SpatialConvolutionMM.c"
#include "THGenerateFloatTypes.h"

#include "generic/SpatialAdaptiveMaxPooling.c"
#include "THGenerateFloatTypes.h"

#include "generic/SpatialAveragePooling.c"
#include "THGenerateFloatTypes.h"

#include "generic/SpatialMaxPooling.c"
#include "THGenerateFloatTypes.h"
