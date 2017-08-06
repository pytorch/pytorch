// Copyright 2004-present Facebook. All Rights Reserved.

#pragma once

#include "caffe2/core/common.h"

#ifdef __ARM_NEON__
#if CAFFE2_IOS
#include "arm_neon.h"
#elif CAFFE2_ANDROID
#include "caffe2/contrib/opengl/android/arm_neon_support.h"
#endif
#endif
