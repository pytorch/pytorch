package org.pytorch;

public class LiteModuleLoader {

  public static Module load(final String modelPath) {
    return new Module(new LiteNativePeer(modelPath));
  }
}
