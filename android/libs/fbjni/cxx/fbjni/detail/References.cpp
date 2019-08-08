/*
 *  Copyright (c) 2018-present, Facebook, Inc.
 *
 *  This source code is licensed under the MIT license found in the LICENSE
 *  file in the root directory of this source tree.
 *
 */
#include "References.h"

namespace facebook {
namespace jni {

JniLocalScope::JniLocalScope(JNIEnv* env, jint capacity)
    : env_(env) {
  hasFrame_ = false;
  auto pushResult = env->PushLocalFrame(capacity);
  FACEBOOK_JNI_THROW_EXCEPTION_IF(pushResult < 0);
  hasFrame_ = true;
}

JniLocalScope::~JniLocalScope() {
  if (hasFrame_) {
    env_->PopLocalFrame(nullptr);
  }
}

namespace {

#ifdef __ANDROID__

int32_t getAndroidApiLevel() {
  auto cls = findClassLocal("android/os/Build$VERSION");
  auto fld = cls->getStaticField<int32_t>("SDK_INT");
  if (fld) {
    return cls->getStaticFieldValue(fld);
  }
  return 0;
}

bool doesGetObjectRefTypeWork() {
  auto level = getAndroidApiLevel();
  return level >= 14;
}

#else

bool doesGetObjectRefTypeWork() {
  auto jni_version = Environment::current()->GetVersion();
  return jni_version >= JNI_VERSION_1_6;
}

#endif

}

bool isObjectRefType(jobject reference, jobjectRefType refType) {

// TODO Rather than setting it true, use doesGetObjectRefTypeWork(). But it
// causes sample app to freeze
  static bool getObjectRefTypeWorks = true;

  return
    !reference ||
    !getObjectRefTypeWorks ||
    Environment::current()->GetObjectRefType(reference) == refType;
}

}
}
