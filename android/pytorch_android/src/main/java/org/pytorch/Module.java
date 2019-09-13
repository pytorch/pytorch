// Copyright 2004-present Facebook. All Rights Reserved.

package org.pytorch;

import com.facebook.jni.HybridData;

public class Module {

  private NativePeer mNativePeer;

  public static Module load(final String modelAbsolutePath) {
    return new Module(modelAbsolutePath);
  }

  private Module(final String modelAbsolutePath) {
    this.mNativePeer = new NativePeer(modelAbsolutePath);
  }

  public IValue forward(IValue... inputs) {
    return mNativePeer.forward(inputs);
  }

  public IValue runMethod(String methodName, IValue... inputs) {
    return mNativePeer.runMethod(methodName, inputs);
  }

  private static class NativePeer {
    static {
      System.loadLibrary("pytorch");
    }

    private final HybridData mHybridData;

    private static native HybridData initHybrid(String moduleAbsolutePath);

    NativePeer(String moduleAbsolutePath) {
      mHybridData = initHybrid(moduleAbsolutePath);
    }

    private native IValue forward(IValue... inputs);

    private native IValue runMethod(String methodName, IValue... inputs);
  }
}
