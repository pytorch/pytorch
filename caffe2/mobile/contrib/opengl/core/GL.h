
#pragma once
#include "caffe2/core/common.h"

#if CAFFE2_IOS
#include <OpenGLES/ES3/gl.h>
#include <OpenGLES/ES3/glext.h>
#elif CAFFE2_ANDROID
#include <EGL/egl.h>
#include <GLES2/gl2.h>
#include "caffe2/mobile/contrib/opengl/android/gl3stub.h"
#endif
