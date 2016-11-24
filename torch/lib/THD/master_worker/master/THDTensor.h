#pragma once

#include <TH/TH.h>
#include "../../THD.h"

#define THDTensor         TH_CONCAT_3(THD,Real,Tensor)
#define THDTensor_(NAME)  TH_CONCAT_4(THD,Real,Tensor_,NAME)

#include "generic/THDTensor.h"
#include <TH/THGenerateAllTypes.h>
