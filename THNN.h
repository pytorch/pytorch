#ifndef THNN_H
#define THNN_H

#include <stdbool.h>
#include <TH.h>

#define THNN_(NAME) TH_CONCAT_3(THNN_, Real, NAME)

#define THIndexTensor THLongTensor
#define THIndexTensor_(NAME) THLongTensor_ ## NAME

typedef long TH_index_t;
typedef void THNNState;

#include "generic/THNN.h"
#include <THGenerateFloatTypes.h>

#endif