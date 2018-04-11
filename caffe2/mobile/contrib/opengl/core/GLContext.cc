
#include "caffe2/core/logging.h"

#include "GL.h"
#include "GLContext.h"
#include "GLLogging.h"

#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>

#if CAFFE2_IOS
#include "sys/utsname.h"
#include <regex>
#endif

void getOpenGLESVersion(int& major, int& minor) {
  glGetIntegerv(GL_MAJOR_VERSION, &major);
  glGetIntegerv(GL_MINOR_VERSION, &minor);
}

bool checkOpenGLExtensions(std::string gl_ext_str) {
  static std::unordered_set<std::string> extensions;
  if (extensions.empty()) {
    const caffe2::string extension_str((const char*)glGetString(GL_EXTENSIONS));
    LOG(INFO) << "GL_EXTENSIONS: " << extension_str;

    std::stringstream ss(extension_str);
    while (!ss.eof()) {
      std::string extension;
      ss >> extension;
      extensions.insert(extension);
    }
  }

  return extensions.count(gl_ext_str) > 0;
}

bool GLContext::GL_EXT_texture_border_clamp_defined() {
  static int major = 0, minor = 0;
  if (major == 0) {
    getOpenGLESVersion(major, minor);
  }

  if (major == 3 && minor == 2) {
    return true;
  }

  return checkOpenGLExtensions("GL_EXT_texture_border_clamp") || // Most common
         checkOpenGLExtensions("GL_OES_texture_border_clamp");
}

bool supportOpenGLES3(bool* half_float_supported) {
  int major = 0, minor = 0;
  getOpenGLESVersion(major, minor);

  LOG(INFO) << "GL_VERSION: OpenGL ES " << major << "." << minor;

  if (major < 3) {
    LOG(ERROR) << "OpenGL ES 3.0 not supported";
    return false;
  }

  if (!checkOpenGLExtensions("GL_EXT_color_buffer_half_float")) {
    LOG(ERROR) << "GL_EXT_color_buffer_half_float is not available";
    if (half_float_supported) {
      *half_float_supported = false;
    }
  }
  return true;
}

#if CAFFE2_IOS
int iPhoneVersion() {
  static int version = 0;
  static std::once_flag once;
  std::call_once(once, [&]() {
    struct utsname systemInfo;
    uname(&systemInfo);
    std::string iphone_ver_str = systemInfo.machine;
    LOG(INFO) << systemInfo.machine;

    if (iphone_ver_str.find("iPhone") != std::string::npos) {
      std::regex regStr("([0-9]+)");
      std::smatch matchs;
      if (std::regex_search(iphone_ver_str, matchs, regStr)) {
        version = stoi(matchs[0]);
      }
    }
  });
  return version;
}
#endif

#if CAFFE2_ANDROID
// whitelist of supported GPUs
bool isSupportedRenderer() {
  static std::unordered_set<std::string> supported_renderers = {
      "Adreno (TM) 540",
      "Adreno (TM) 530",
      "Adreno (TM) 510",
      "Adreno (TM) 430",
      "Adreno (TM) 418",
      "Mali-G71",
      "Mali-T880",
      "NVIDIA Tegra"};
  std::string rendererStr((const char*)glGetString(GL_RENDERER));
  LOG(INFO) << "GL_RENDERER: " << rendererStr;

  int start = rendererStr.find_first_not_of(" ");
  int end = rendererStr.find_last_not_of(" ");
  rendererStr = rendererStr.substr(start, end - start + 1);
  return supported_renderers.count(rendererStr) > 0;
}
#endif

bool isSupportedDevice() {
#if CAFFE2_IOS
  return iPhoneVersion() >= 7; // iPhone 6 and up
#elif CAFFE2_ANDROID
  return isSupportedRenderer();
#else
  return false;
#endif
}
