#include <fbjni/fbjni.h>

#include <pytorch_jni.h>

jint JNI_OnLoad(JavaVM* vm, void*) {
  return facebook::jni::initialize(
      vm, [] { pytorch_jni::PytorchJni::registerNatives(); });
}