// (c) Facebook, Inc. and its affiliates. Confidential and proprietary.

package org.pytorch;

import com.facebook.jni.HybridData;
import com.facebook.jni.annotations.DoNotStrip;
import com.facebook.soloader.nativeloader.NativeLoader;

class NativePeer implements INativePeer {
  static {
    NativeLoader.loadLibrary("pytorch_jni");
  }

  private final HybridData mHybridData;

  @DoNotStrip
  private static native HybridData initHybrid(String moduleAbsolutePath);

  @DoNotStrip
  private static native HybridData initHybridAndroidAsset(
      String assetName, /* android.content.res.AssetManager */ Object androidAssetManager);

  NativePeer(String moduleAbsolutePath) {
    mHybridData = initHybrid(moduleAbsolutePath);
  }

  NativePeer(String assetName, /* android.content.res.AssetManager */ Object androidAssetManager) {
    mHybridData = initHybridAndroidAsset(assetName, androidAssetManager);
  }

  public void resetNative() {
    mHybridData.resetNative();
  }

  @DoNotStrip
  public native IValue forward(IValue... inputs);

  @DoNotStrip
  public native IValue runMethod(String methodName, IValue... inputs);
}
