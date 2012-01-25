#ifndef TH_INC
#define TH_INC

#include "THBlas.h"

#ifdef __LAPACK__
#include "THLapack.h"
#endif

#include "THVector.h"
#include "THGeneral.h"
#include "THLogAdd.h"
#include "THRandom.h"
#include "THStorage.h"
#include "THTensor.h"
#include "THTensorApply.h"
#include "THTensorDimApply.h"

#include "THFile.h"
#include "THDiskFile.h"
#include "THMemoryFile.h"

#endif
