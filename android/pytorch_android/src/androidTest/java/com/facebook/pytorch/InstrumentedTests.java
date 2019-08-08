package com.facebook.pytorch;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.util.Log;

import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;

import androidx.test.ext.junit.runners.AndroidJUnit4;
import androidx.test.platform.app.InstrumentationRegistry;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertTrue;

@RunWith(AndroidJUnit4.class)
public class InstrumentedTests {

  private static final String TAG = "PytorchTest";

  @Before
  public void setUp() {
    System.loadLibrary("pytorch_android");
  }

  @Test
  public void testRunSqueezeNet() throws IOException {
    final int[] inputDims = new int[]{1, 3, 224, 224};
    final int numElements = Tensor.numElements(inputDims);
    final float[] data = new float[numElements];
    final Tensor inputTensor = Tensor.newFloatTensor(inputDims, data);
    final String modelFilePath = assetFilePath("squeezenet1_0.pt");
    Log.d(TAG, "smokeTest() modelFilePath:" + modelFilePath);
    final PytorchScriptModule module = new PytorchScriptModule(modelFilePath);
    final Tensor outputTensor = module.run(inputTensor);
    assertNotNull(outputTensor);
    assertArrayEquals(new int[]{1, 1000}, outputTensor.dims);
    float[] outputData = outputTensor.getDataAsFloatArray();
    assertEquals(1000, outputData.length);
  }

  @Test
  public void testBitmapToTensorOnSqueezeNet_husky() throws IOException {
    final Bitmap bitmap = assetAsBitmap("siberian_husky.jpg");
    final Tensor inputTensor = TensorUtils.bitmapToFloatTensorTorchVisionForm(bitmap);
    bitmap.recycle();
    final String modelFilePath = assetFilePath("squeezenet1_0.pt");
    Log.d(TAG, "smokeTest() modelFilePath:" + modelFilePath);
    final PytorchScriptModule module = new PytorchScriptModule(modelFilePath);
    final Tensor outputTensor = module.run(inputTensor);
    assertNotNull(outputTensor);
    assertArrayEquals(new int[]{1, 1000}, outputTensor.dims);
    float[] outputData = outputTensor.getDataAsFloatArray();
    assertEquals(1000, outputData.length);

    int maxI = -1;
    float maxValue = -Float.MAX_VALUE;
    for (int i = 0; i < outputData.length; i++) {
      if (outputData[i] > maxValue) {
        maxValue = outputData[i];
        maxI = i;
      }
    }
    final String s = TestConstants.IMAGENET_CLASSES[maxI];
    assertTrue(s.toLowerCase().contains("husky"));
  }

  private static Bitmap assetAsBitmap(String assetName) throws IOException {
    final Context appContext = InstrumentationRegistry.getInstrumentation().getTargetContext();
    try (InputStream is = appContext.getAssets().open(assetName)) {
      return BitmapFactory.decodeStream(is);
    } catch (IOException e) {
      throw e;
    }
  }

  private static String assetFilePath(String assetName) throws IOException {
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
