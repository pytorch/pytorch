// (c) Facebook, Inc. and its affiliates. Confidential and proprietary.

package org.pytorch;

import com.facebook.jni.HybridData;
import com.facebook.soloader.nativeloader.NativeLoader;

class NativePeer implements INativePeer {
  static {
    NativeLoader.loadLibrary("pytorch_jni");
  }

  private final HybridData mHybridData;

  private static native HybridData initHybrid(String moduleAbsolutePath);

  NativePeer(String moduleAbsolutePath) {
    mHybridData = initHybrid(moduleAbsolutePath);
  }

  public void resetNative() {
    mHybridData.resetNative();
  }

  public native IValue forward(IValue... inputs);

  public native IValue runMethod(String methodName, IValue... inputs);
}
