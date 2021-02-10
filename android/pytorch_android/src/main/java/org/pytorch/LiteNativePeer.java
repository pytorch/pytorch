package org.pytorch;

import com.facebook.jni.HybridData;
import com.facebook.soloader.nativeloader.NativeLoader;
import com.facebook.soloader.nativeloader.SystemDelegate;

class LiteNativePeer implements INativePeer {
  static {
    NativeLoader.init(new SystemDelegate());
    NativeLoader.loadLibrary("pytorch_jni_lite");
    PyTorchCodegenLoader.loadNativeLibs();
  }

  private final HybridData mHybridData;

  private static native HybridData initHybrid(String moduleAbsolutePath, int deviceJniCode);

  LiteNativePeer(String moduleAbsolutePath, Device device) {
    mHybridData = initHybrid(moduleAbsolutePath, device.jniCode);
  }

  public void resetNative() {
    mHybridData.resetNative();
  }

  public native IValue forward(IValue... inputs);

  public native IValue runMethod(String methodName, IValue... inputs);
}
