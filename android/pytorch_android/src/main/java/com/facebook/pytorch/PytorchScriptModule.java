package com.facebook.pytorch;

import com.facebook.jni.HybridData;

import java.nio.Buffer;

public class PytorchScriptModule {

  private NativePeer mNativePeer;

  public PytorchScriptModule(final String modelAbsolutePath) {
    this.mNativePeer = new NativePeer(modelAbsolutePath);
  }

  public Tensor run(Tensor input) {
    return mNativePeer.run(input.getRawDataBuffer(), input.dims, input.getTypeCode());
  }

  private static class NativePeer {
    static {
      System.loadLibrary("pytorch_android");
    }

    private final HybridData mHybridData;

    private static native HybridData initHybrid(String moduleAbsolutePath);

    NativePeer(String moduleAbsolutePath) {
      mHybridData = initHybrid(moduleAbsolutePath);
    }

    private native Tensor run(Buffer data, int[] dims, int typeCode);
  }
}
