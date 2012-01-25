#ifndef TORCH_GENERAL_INC
#define TORCH_GENERAL_INC

#include <stdlib.h>
#include <string.h>

#include "luaT.h"
#include "TH.h"

#ifdef _MSC_VER

#define snprintf _snprintf
#define popen _popen
#define pclose _pclose

#endif

#endif
