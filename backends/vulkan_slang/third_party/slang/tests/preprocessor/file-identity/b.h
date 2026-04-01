#pragma once

#include "c.h"

#ifdef B_H
#error "Shouldn't be included twice"
#endif

#define B_H

float foo(float x)
{
    return x;
}