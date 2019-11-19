// (c) Facebook, Inc. and its affiliates. Confidential and proprietary.

package org.pytorch;

import com.facebook.jni.HybridData;
import com.facebook.soloader.nativeloader.NativeLoader;

class NativePeer implements INativePeer {
  static {
    NativeLoader.loadLibrary("pytorch_jni");
  }

  private final HybridData mHybridData;

  private static native HybridData initHybridFilePath(String moduleAbsolutePath);

  private static native HybridData initHybridReadAdapter(ReadAdapter readAdapter);

  NativePeer(String moduleAbsolutePath) {
    mHybridData = initHybridFilePath(moduleAbsolutePath);
  }

  NativePeer(ReadAdapter readAdapter) {
    mHybridData = initHybridReadAdapter(readAdapter);
  }

  public void resetNative() {
    mHybridData.resetNative();
  }

  public native IValue forward(IValue... inputs);

  public native IValue runMethod(String methodName, IValue... inputs);
}
