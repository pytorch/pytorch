#ifndef TH_INC
#define TH_INC

#include <TH/THGeneral.h>

#include <TH/THBlas.h>
#ifdef USE_LAPACK
#include <TH/THLapack.h>
#endif

#include <TH/THVector.h>
#include <TH/THStorageFunctions.h>
#include <TH/THTensor.h>
#include <TH/THTensorApply.h>
#include <TH/THTensorDimApply.h>

#include <TH/THFile.h>
#include <TH/THDiskFile.h>
#include <TH/THMemoryFile.h>

#endif
