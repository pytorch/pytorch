#define __STDC_FORMAT_MACROS

#include <Python.h>
#ifdef _MSC_VER
#include <Windows.h>
#endif
#include <structmember.h>

#define THP_HOST_HALF

#include <stdbool.h>
#include <TH/TH.h>
#include <libshm.h>
#include "THP.h"
#include "copy_utils.h"

#include "generic/Storage.cpp"
#include <TH/THGenerateAllTypes.h>

#include "generic/Storage.cpp"
#include <TH/THGenerateHalfType.h>
