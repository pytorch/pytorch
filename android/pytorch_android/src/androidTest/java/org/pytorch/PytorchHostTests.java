package org.pytorch;

import com.facebook.soloader.nativeloader.NativeLoader;
import com.facebook.soloader.nativeloader.SystemDelegate;
import org.junit.BeforeClass;

import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.util.Objects;

public class PytorchHostTests extends PytorchTestBase {
  @BeforeClass
  public static void setUpClass() {
    if (!NativeLoader.isInitialized()) {
      NativeLoader.init(new SystemDelegate());
    }
  }

  @Override
  protected String assetFilePath(String assetName) throws IOException {
    Path tempFile = Files.createTempFile("test", ".pt");
    try (InputStream resource = Objects.requireNonNull(getClass().getClassLoader().getResourceAsStream("test.pt"))) {
      Files.copy(resource, tempFile, StandardCopyOption.REPLACE_EXISTING);
    }
    return tempFile.toAbsolutePath().toString();
  }
}
