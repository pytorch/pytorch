#pragma once

#include <THD/THD.h>

#ifndef _THD_CORE
struct _THDRequest;
typedef struct _THDRequest THDRequest;
#endif

THD_API void THDRequest_free(void* req);
