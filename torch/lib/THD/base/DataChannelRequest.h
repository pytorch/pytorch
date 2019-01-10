#pragma once

#include "../THD.h"

#ifndef _THD_CORE
struct _THDRequest;
typedef struct _THDRequest THDRequest;
#endif

THD_API void THDRequest_free(THDRequest* req);
