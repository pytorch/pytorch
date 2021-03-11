package org.pytorch;

public class LiteModuleLoader {

  /**
   * Loads a serialized TorchScript module from the specified path on the disk to run on specified
   * device. The model should be generated from this api _save_for_lite_interpreter().
   *
   * @param modelPath path to file that contains the serialized TorchScript module.
   * @param device {@link org.pytorch.Device} to use for running specified module.
   * @return new {@link org.pytorch.Module} object which owns torch::jit::mobile::Module.
   */
  public static Module load(final String modelPath, final Device device) {
    return new Module(new LiteNativePeer(modelPath, device));
  }

  /**
   * Loads a serialized TorchScript module from the specified path on the disk to run on CPU. The
   * model should be generated from this api _save_for_lite_interpreter().
   *
   * @param modelPath path to file that contains the serialized TorchScript module.
   * @return new {@link org.pytorch.Module} object which owns torch::jit::mobile::Module.
   */
  public static Module load(final String modelPath) {
    return new Module(new LiteNativePeer(modelPath, Device.CPU));
  }
}
