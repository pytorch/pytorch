package org.pytorch;

import android.content.res.AssetManager;
import java.util.Map;

public class LiteModuleLoader {

  /**
   * Loads a serialized TorchScript module from the specified path on the disk to run on specified
   * device. The model should be generated from this api _save_for_lite_interpreter().
   *
   * @param modelPath path to file that contains the serialized TorchScript module.
   * @param extraFiles map with extra files names as keys, content of them will be loaded to values.
   * @param device {@link org.pytorch.Device} to use for running specified module.
   * @return new {@link org.pytorch.Module} object which owns torch::jit::mobile::Module.
   */
  public static Module load(
      final String modelPath, final Map<String, String> extraFiles, final Device device) {
    return new Module(new LiteNativePeer(modelPath, extraFiles, device));
  }

  /**
   * Loads a serialized TorchScript module from the specified path on the disk to run on CPU. The
   * model should be generated from this api _save_for_lite_interpreter().
   *
   * @param modelPath path to file that contains the serialized TorchScript module.
   * @return new {@link org.pytorch.Module} object which owns torch::jit::mobile::Module.
   */
  public static Module load(final String modelPath) {
    return new Module(new LiteNativePeer(modelPath, null, Device.CPU));
  }

  /**
   * Attention: This is not recommended way of loading production modules, as prepackaged assets
   * increase apk size etc. For production usage consider using loading from file on the disk {@link
   * org.pytorch.Module#load(String)}.
   *
   * <p>This method is meant to use in tests and demos.
   */
  public static Module loadModuleFromAsset(
      final AssetManager assetManager, final String assetName, final Device device) {
    return new Module(new LiteNativePeer(assetName, assetManager, device));
  }

  public static Module loadModuleFromAsset(
      final AssetManager assetManager, final String assetName) {
    return new Module(new LiteNativePeer(assetName, assetManager, Device.CPU));
  }
}
