package org.pytorch;

import com.facebook.jni.HybridData;
import com.facebook.jni.annotations.DoNotStrip;
import com.facebook.soloader.nativeloader.NativeLoader;

class NativePeer implements INativePeer {
  static {
    NativeLoader.loadLibrary("pytorch_jni");
    PyTorchCodegenLoader.loadNativeLibs();
  }

  private final HybridData mHybridData;

  @DoNotStrip
  private static native HybridData initHybrid(String moduleAbsolutePath, int deviceJniCode);

  @DoNotStrip
  private static native HybridData initHybridAndroidAsset(
      String assetName, /* android.content.res.AssetManager */
      Object androidAssetManager,
      int deviceJniCode);

  NativePeer(String moduleAbsolutePath, Device device) {
    mHybridData = initHybrid(moduleAbsolutePath, device.jniCode);
  }

  NativePeer(
      String assetName, /* android.content.res.AssetManager */
      Object androidAssetManager,
      Device device) {
    mHybridData = initHybridAndroidAsset(assetName, androidAssetManager, device.jniCode);
  }

  public void resetNative() {
    mHybridData.resetNative();
  }

  @DoNotStrip
  public native IValue forward(IValue... inputs);

  @DoNotStrip
  public native IValue runMethod(String methodName, IValue... inputs);
}
