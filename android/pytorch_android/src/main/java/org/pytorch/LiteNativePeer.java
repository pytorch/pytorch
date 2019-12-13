// (c) Facebook, Inc. and its affiliates. Confidential and proprietary.

package org.pytorch;

import com.facebook.jni.HybridData;
import com.facebook.soloader.nativeloader.NativeLoader;

class LiteNativePeer implements INativePeer {
  static {
    NativeLoader.loadLibrary("pytorch_jni_lite");
  }

  private final HybridData mHybridData;

  private static native HybridData initHybrid(String moduleAbsolutePath);

  LiteNativePeer(String moduleAbsolutePath) {
    mHybridData = initHybrid(moduleAbsolutePath);
  }

  public void resetNative() {
    mHybridData.resetNative();
  }

  public native IValue forward(IValue... inputs);

  public native IValue runMethod(String methodName, IValue... inputs);
}
