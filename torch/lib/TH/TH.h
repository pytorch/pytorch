#ifndef TH_INC
#define TH_INC

#include "THGeneral.h"

#include "THBlas.h"
#ifdef USE_LAPACK
#include "THLapack.h"
#endif

#include "THAtomic.h"
#include "THVector.h"
#include "THLogAdd.h"
#include "THRandom.h"
#include "THSize.h"
#include "THStorage.h"
#include "THTensor.h"
#include "THTensorApply.h"
#include "THTensorDimApply.h"

#include "THFile.h"
#include "THDiskFile.h"
#include "THMemoryFile.h"

#endif
