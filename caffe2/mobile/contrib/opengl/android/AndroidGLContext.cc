#include "AndroidGLContext.h"
#include "caffe2/core/logging.h"
#include "gl3stub.h"
#include <regex>

namespace {

static const std::unordered_map<std::string, GL_Renderer>& renderer_map() {
  static std::unordered_map<std::string, GL_Renderer> m = {
      {"Adreno", Adreno},
      {"Mali", Mali},
      {"NVIDIA", Tegra} /*, {"PowerVR", PowerVR} */};
  return m;
}

} // namespace

EGLContext AndroidGLContext::create_opengl_thread_context() {
  EGLSurface surface = EGL_NO_SURFACE;
  EGLContext context = EGL_NO_CONTEXT;
  EGLDisplay display = eglGetDisplay(EGL_DEFAULT_DISPLAY);
  if (display == EGL_NO_DISPLAY) {
    // We failed to get a display
    CAFFE_THROW("Problem with OpenGL context");
    return context;
  }

  EGLint major;
  EGLint minor;
  eglInitialize(display, &major, &minor);

  const EGLint configAttr[] = {EGL_RENDERABLE_TYPE,
                               EGL_OPENGL_ES2_BIT,
                               EGL_SURFACE_TYPE,
                               EGL_PBUFFER_BIT, // we create a pixelbuffer surface
                               EGL_NONE};

  EGLint numConfig;
  EGLConfig eglConfig;
  if (!eglChooseConfig(display, configAttr, &eglConfig, 1, &numConfig)) {
    // We failed to find a suitable config
    eglMakeCurrent(display, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT);
    eglTerminate(display);
    display = EGL_NO_DISPLAY;
    CAFFE_THROW("Problem with OpenGL context");
    return context;
  }

  const EGLint ctxAttr[] = {EGL_CONTEXT_CLIENT_VERSION,
                            2, // very important!
                            EGL_NONE};

  // Create an EGL context based on the chosen configuration.
  context = eglCreateContext(display, eglConfig, EGL_NO_CONTEXT, ctxAttr);

  // We need a surface. For most mixed JNI/Java based apps it is suggested
  // that we pass a Java surface through JNI and extract the surface
  // Pure NDK apps get passed the android_app structure which includes a surface
  // We want our own OpenGL context for the current thread.
  // Here we create a fake 1x1 'pixel buffer' surface.
  // We don't expecting to run vertex or fragment shaders.

  const EGLint surfaceAttr[] = {EGL_WIDTH, 1, EGL_HEIGHT, 1, EGL_NONE};

  surface = eglCreatePbufferSurface(display, eglConfig, surfaceAttr);

  // Bind context, draw and surface to current thread
  eglMakeCurrent(display, surface, surface, context);

  // Bind the API for this context.  In our case we want to use OpenGL_ES
  eglBindAPI(EGL_OPENGL_ES_API);
  return context;
}

bool AndroidGLContext::opengl_thread_context_exists() {
  return eglGetCurrentContext() != EGL_NO_CONTEXT;
}

bool AndroidGLContext::release_opengl_thread_context() {
  EGLContext display = eglGetCurrentDisplay();
  if (display != EGL_NO_DISPLAY) {
    if (_eglcontext != EGL_NO_CONTEXT) {
      eglDestroyContext(display, _eglcontext);
      _eglcontext = EGL_NO_CONTEXT;
    }
    EGLSurface surface = eglGetCurrentSurface(EGL_DRAW);
    if (surface != EGL_NO_SURFACE) {
      eglDestroySurface(display, surface);
      surface = EGL_NO_SURFACE;
    }
    surface = eglGetCurrentSurface(EGL_READ);
    if (surface != EGL_NO_SURFACE) {
      eglDestroySurface(display, surface);
      surface = EGL_NO_SURFACE;
    }
    eglMakeCurrent(display, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT);
    eglTerminate(display);
    display = EGL_NO_DISPLAY;
  }
  eglReleaseThread();
  return true;
}

void AndroidGLContext::init_gles3() {
  if (!gl3stubInit()) {
    CAFFE_THROW("OpenGL ES 3 not initialized");
  } else {
    LOG(INFO) << "OpenGL ES 3 successfully enabled";
  }
}

GL_Renderer AndroidGLContext::get_platform() {
  std::string rendererStr((const char*)glGetString(GL_RENDERER));
  std::regex regStr("^[A-Za-z]*");
  std::smatch matchs;
  if (std::regex_search(rendererStr, matchs, regStr)) {
    const std::string renderer = *matchs.begin();
    auto found = renderer_map().find(renderer);
    if (found != renderer_map().end()) {
      return found->second;
    }
  }
  CAFFE_THROW("Unsupported GPU renderer");
}

AndroidGLContext::AndroidGLContext() {
  if (!opengl_thread_context_exists()) {
    _eglcontext = create_opengl_thread_context();
    LOG(INFO) << "New EGLContext created";

    if (!supportOpenGLES3(&half_float_supported)) {
      CAFFE_THROW("OpenGL ES 3 not supported");
    }

    if (!isSupportedDevice()) {
      LOG(ERROR) << "Device not fully supported";
    }
  } else {
    _eglcontext = EGL_NO_CONTEXT;
    LOG(INFO) << "Reusing EGLContext, make sure OpenGL ES 3 is supported";
  }
  static std::once_flag once;
  std::call_once(once, [&]() { init_gles3(); });
}

AndroidGLContext::~AndroidGLContext() {
  if (_eglcontext != EGL_NO_CONTEXT) {
    release_opengl_thread_context();
  }
}

void AndroidGLContext::set_context() {}

void AndroidGLContext::reset_context() {}

void AndroidGLContext::flush_context() {}
