#ifndef THNN_H
#define THNN_H

#include <stdbool.h>
#include <TH.h>

#define THNN_(NAME) TH_CONCAT_3(THNN_, Real, NAME)

typedef void THNNState;

#include "generic/THNN.h"
#include <THGenerateFloatTypes.h>

#endif