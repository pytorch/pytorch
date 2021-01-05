package org.pytorch;

public class LiteModuleLoader {

  public static Module load(final String modelPath, final Device device) {
    return new Module(new LiteNativePeer(modelPath, device));
  }

  public static Module load(final String modelPath) {
    return new Module(new LiteNativePeer(modelPath, Device.CPU));
  }
}
