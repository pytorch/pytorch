package org.pytorch;

public enum Device {
  // Must be in sync with kDeviceCPU, kDeviceVulkan in
  // pytorch_android/src/main/cpp/pytorch_jni_lite.cpp
  CPU(1),
  VULKAN(2),
  ;

  final int jniCode;

  Device(int jniCode) {
    this.jniCode = jniCode;
  }
}
