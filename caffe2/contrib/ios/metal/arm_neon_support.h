// Copyright 2004-present Facebook. All Rights Reserved.

#pragma once

#ifdef __ARM_NEON__
#import <arm_neon.h>
#else
typedef unsigned short uint16_t;
typedef uint16_t       float16_t;
typedef float          float32_t;
#endif
