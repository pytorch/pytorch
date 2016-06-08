#ifndef THP_WRAP_UTILS_INC
#define THP_WRAP_UTILS_INC

#define THPUtils_(NAME) TH_CONCAT_4(THP,Real,Utils_,NAME)

#include "generic/utils.h"
#include <TH/THGenerateAllTypes.h>

int THPUtils_getLong(PyObject *index, long *result);

#endif

