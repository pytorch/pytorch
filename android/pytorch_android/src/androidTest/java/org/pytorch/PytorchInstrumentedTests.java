package org.pytorch;

import android.content.Context;
import androidx.test.InstrumentationRegistry;
import androidx.test.runner.AndroidJUnit4;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import org.junit.runner.RunWith;

@RunWith(AndroidJUnit4.class)
public class PytorchInstrumentedTests extends PytorchTestBase {

  @Override
  protected Module loadModel(String path) throws IOException {
    return Module.load(assetFilePath(path));
  }

  private String assetFilePath(String assetName) throws IOException {
    final Context appContext = InstrumentationRegistry.getInstrumentation().getTargetContext();
    File file = new File(appContext.getFilesDir(), assetName);
    if (file.exists() && file.length() > 0) {
      return file.getAbsolutePath();
    }

    try (InputStream is = appContext.getAssets().open(assetName)) {
      try (OutputStream os = new FileOutputStream(file)) {
        byte[] buffer = new byte[4 * 1024];
        int read;
        while ((read = is.read(buffer)) != -1) {
          os.write(buffer, 0, read);
        }
        os.flush();
      }
      return file.getAbsolutePath();
    } catch (IOException e) {
      throw e;
    }
  }
}
