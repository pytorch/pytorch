/*
 *  Copyright (c) 2018-present, Facebook, Inc.
 *
 *  This source code is licensed under the MIT license found in the LICENSE
 *  file in the root directory of this source tree.
 *
 */
#include <fbjni/fbjni.h>
#include <fbjni/NativeRunnable.h>

using namespace facebook::jni;

JNIEXPORT jint JNI_OnLoad(JavaVM* vm, void* reserved) {
  return facebook::jni::initialize(vm, [] {
    HybridDataOnLoad();
    JNativeRunnable::OnLoad();
    ThreadScope::OnLoad();
  });
}
