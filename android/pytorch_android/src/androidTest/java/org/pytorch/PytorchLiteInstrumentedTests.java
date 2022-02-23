package org.pytorch;

import android.content.Context;
<<<<<<< HEAD

import androidx.test.InstrumentationRegistry;
import androidx.test.runner.AndroidJUnit4;

import org.junit.runner.RunWith;

=======
import androidx.test.InstrumentationRegistry;
import androidx.test.runner.AndroidJUnit4;
>>>>>>> 5eb5b61221759012909112a6a8de6b3306aa8dea
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
<<<<<<< HEAD
=======
import org.junit.runner.RunWith;
>>>>>>> 5eb5b61221759012909112a6a8de6b3306aa8dea

@RunWith(AndroidJUnit4.class)
public class PytorchLiteInstrumentedTests extends PytorchTestBase {

  @Override
  protected Module loadModel(String path) throws IOException {
    return LiteModuleLoader.load(assetFilePath(path));
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
<<<<<<< HEAD

=======
>>>>>>> 5eb5b61221759012909112a6a8de6b3306aa8dea
}
