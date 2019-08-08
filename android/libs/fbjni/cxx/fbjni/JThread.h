/*
 *  Copyright (c) 2018-present, Facebook, Inc.
 *
 *  This source code is licensed under the MIT license found in the LICENSE
 *  file in the root directory of this source tree.
 *
 */
#pragma once

#include <fbjni/fbjni.h>
#include <fbjni/NativeRunnable.h>

namespace facebook {
namespace jni {

class JThread : public JavaClass<JThread> {
 public:
  static constexpr const char* kJavaDescriptor = "Ljava/lang/Thread;";

  void start() {
    static const auto method = javaClassStatic()->getMethod<void()>("start");
    method(self());
  }

  void join() {
    static const auto method = javaClassStatic()->getMethod<void()>("join");
    method(self());
  }

  static local_ref<JThread> create(std::function<void()>&& runnable) {
    auto jrunnable = JNativeRunnable::newObjectCxxArgs(std::move(runnable));
    return newInstance(static_ref_cast<JRunnable::javaobject>(jrunnable));
  }

  static local_ref<JThread> create(std::function<void()>&& runnable, std::string&& name) {
    auto jrunnable = JNativeRunnable::newObjectCxxArgs(std::move(runnable));
    return newInstance(static_ref_cast<JRunnable::javaobject>(jrunnable), make_jstring(std::move(name)));
  }

  static local_ref<JThread> getCurrent() {
    static const auto method = javaClassStatic()->getStaticMethod<local_ref<JThread>()>("currentThread");
    return method(javaClassStatic());
  }

  int getPriority() {
    static const auto method = getClass()->getMethod<jint()>("getPriority");
    return method(self());
  }

  void setPriority(int priority) {
    static const auto method = getClass()->getMethod<void(int)>("setPriority");
    method(self(), priority);
  }
};

}
}
