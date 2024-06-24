package org.pytorch.torchvision;

import static org.junit.Assert.assertArrayEquals;

import android.graphics.Bitmap;
import androidx.test.ext.junit.runners.AndroidJUnit4;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.pytorch.Tensor;

@RunWith(AndroidJUnit4.class)
public class TorchVisionInstrumentedTests {

  @Test
  public void smokeTest() {
    Bitmap bitmap = Bitmap.createBitmap(320, 240, Bitmap.Config.ARGB_8888);
    Tensor tensor =
        TensorImageUtils.bitmapToFloat32Tensor(
            bitmap,
            TensorImageUtils.TORCHVISION_NORM_MEAN_RGB,
            TensorImageUtils.TORCHVISION_NORM_STD_RGB);
    assertArrayEquals(new long[] {1l, 3l, 240l, 320l}, tensor.shape());
  }
}
