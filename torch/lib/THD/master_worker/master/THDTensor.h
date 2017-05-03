#pragma once

#include <TH/TH.h>
#include "../../THD.h"

#define THDTensor         TH_CONCAT_3(THD,Real,Tensor)
#define THDTensor_(NAME)  TH_CONCAT_4(THD,Real,Tensor_,NAME)

#define THD_DESC_BUFF_LEN 64
typedef struct {
  char str[THD_DESC_BUFF_LEN];
} THDDescBuff;

#include "generic/THDTensor.h"
#include <TH/THGenerateAllTypes.h>

#include "generic/THDTensorCopy.h"
#include <TH/THGenerateAllTypes.h>

#include "generic/THDTensorRandom.h"
#include <TH/THGenerateAllTypes.h>

#include "generic/THDTensorMath.h"
#include <TH/THGenerateAllTypes.h>

#include "generic/THDTensorLapack.h"
#include <TH/THGenerateFloatTypes.h>
