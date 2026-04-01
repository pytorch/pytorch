#pragma once

#include "b.h"

#ifdef C_H
#error "c.h shouldn't be included twice"
#endif

#define C_H

float bar(float x)
{
    return x;
}