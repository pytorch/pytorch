#include <cassert>
#include <cmath>
#include <vector>

#include <torch/script.h>
#include <ATen/NativeFunctions.h>
#include "jni.h"

namespace pytorch_testapp_jni {
namespace {
  struct JITCallGuard {
    torch::autograd::AutoGradMode no_autograd_guard{false};
    torch::AutoNonVariableTypeMode non_var_guard{true};
    torch::jit::GraphOptimizerEnabledGuard no_optimizer_guard{false};
  };
} // namespace

static void loadAndForwardModel(JNIEnv* env, jclass, jstring jModelPath) {
  const char* modelPath = env->GetStringUTFChars(jModelPath, 0);
  assert(modelPath);

  JITCallGuard guard;
  torch::jit::Module module = torch::jit::load(modelPath);
  module.eval();
  // XXX: Why TypeDefault.randn is not linked?
  //torch::Tensor t = torch::randn({1, 3, 224, 224});
  torch::Tensor t = at::native::randn({1, 3, 224, 224});
  c10::IValue t_out = module.forward({t});
  std::cout << "XXX t:" << t << std::endl;
  std::cout << "XXX t_out:" << t_out << std::endl;

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
