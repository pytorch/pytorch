#include <android/log.h>
#include <pthread.h>
#include <unistd.h>
#include <cassert>
#include <cmath>
#include <vector>
#define ALOGI(...) \
  __android_log_print(ANDROID_LOG_INFO, "PyTorchTestAppJni", __VA_ARGS__)
#define ALOGE(...) \
  __android_log_print(ANDROID_LOG_ERROR, "PyTorchTestAppJni", __VA_ARGS__)

#include "jni.h"

#include <torch/script.h>

namespace pytorch_testapp_jni {
namespace {

template <typename T>
void log(const char* m, T t) {
  std::ostringstream os;
  os << t << std::endl;
  ALOGI("%s %s", m, os.str().c_str());
}

struct JITCallGuard {
  torch::autograd::AutoGradMode no_autograd_guard{false};
  torch::AutoNonVariableTypeMode non_var_guard{true};
  torch::jit::GraphOptimizerEnabledGuard no_optimizer_guard{false};
};
} // namespace

static void loadAndForwardModel(JNIEnv* env, jclass, jstring jModelPath) {
  const char* modelPath = env->GetStringUTFChars(jModelPath, 0);
  assert(modelPath);

  // To load torchscript model for mobile we need set these guards,
  // because mobile build doesn't support features like autograd for smaller
  // build size which is placed in `struct JITCallGuard` in this example. It may
  // change in future, you can track the latest changes keeping an eye in
  // android/pytorch_android/src/main/cpp/pytorch_jni_jit.cpp
  JITCallGuard guard;
  torch::jit::Module module = torch::jit::load(modelPath);
  module.eval();
  torch::Tensor t = torch::randn({1, 3, 224, 224});
  log("input tensor:", t);
  c10::IValue t_out = module.forward({t});
  log("output tensor:", t_out);
  env->ReleaseStringUTFChars(jModelPath, modelPath);
}
} // namespace pytorch_testapp_jni

JNIEXPORT jint JNI_OnLoad(JavaVM* vm, void*) {
  JNIEnv* env;
  if (vm->GetEnv(reinterpret_cast<void**>(&env), JNI_VERSION_1_6) != JNI_OK) {
    return JNI_ERR;
  }

  jclass c =
      env->FindClass("org/pytorch/testapp/LibtorchNativeClient$NativePeer");
  if (c == nullptr) {
    return JNI_ERR;
  }

  static const JNINativeMethod methods[] = {
      {"loadAndForwardModel",
       "(Ljava/lang/String;)V",
       (void*)pytorch_testapp_jni::loadAndForwardModel},
  };
  int rc = env->RegisterNatives(
      c, methods, sizeof(methods) / sizeof(JNINativeMethod));

  if (rc != JNI_OK) {
    return rc;
  }

  return JNI_VERSION_1_6;
}
