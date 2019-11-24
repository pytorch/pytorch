package org.pytorch;

import android.content.res.AssetManager;

import com.facebook.soloader.nativeloader.NativeLoader;
import com.facebook.soloader.nativeloader.SystemDelegate;

public final class AndroidUtils {

  public static Module loadModuleFromAsset(final String assetName, final AssetManager assetManager) {
    if (!NativeLoader.isInitialized()) {
      NativeLoader.init(new SystemDelegate());
    }
    return new Module(new NativePeer(assetName, assetManager));
  }
}
