// Copyright 2004-present Facebook. All Rights Reserved.

#pragma once
#include "caffe2/core/common.h"

#if CAFFE2_IOS
#include <OpenGLES/ES3/gl.h>
#include <OpenGLES/ES3/glext.h>
#elif CAFFE2_ANDROID
#include "caffe2/contrib/opengl/android/gl3stub.h"
#include <EGL/egl.h>
#include <GLES2/gl2.h>
#endif
