package org.pytorch.torchvision;

import android.graphics.Bitmap;

import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.pytorch.Tensor;

import androidx.test.ext.junit.runners.AndroidJUnit4;

import static org.junit.Assert.assertArrayEquals;

@RunWith(AndroidJUnit4.class)
public class TorchVisionInstrumentedTests {

  @Before
  public void setUp() {
    System.loadLibrary("pytorch");
  }

  @Test
  public void smokeTest() {
    Bitmap bitmap = Bitmap.createBitmap(320, 240, Bitmap.Config.ARGB_8888);
    Tensor tensor = TensorImageUtils.bitmapToFloatTensorTorchVisionForm(bitmap);
    assertArrayEquals(new long[] {1l, 3l, 240l, 320l}, tensor.dims);
  }
}
