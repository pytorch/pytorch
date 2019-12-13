package org.pytorch;

import android.content.res.AssetManager;

import com.facebook.soloader.nativeloader.NativeLoader;
import com.facebook.soloader.nativeloader.SystemDelegate;

public final class PyTorchAndroid {

  /**
   * Attention:
   * This is not recommended way of loading production modules, as prepackaged assets increase apk size etc.
   * For production usage consider using loading from file on the disk {@link org.pytorch.Module#load(String)}.
   *
   * This method is meant to use in tests and demos.
   */
  public static Module loadModuleFromAsset(final AssetManager assetManager, final String assetName) {
    if (!NativeLoader.isInitialized()) {
      NativeLoader.init(new SystemDelegate());
    }
    return new Module(new NativePeer(assetName, assetManager));
  }
}
