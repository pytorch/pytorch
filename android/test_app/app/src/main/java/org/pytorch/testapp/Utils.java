package org.pytorch.testapp;

import java.util.Arrays;

public class Utils {

  public static int[] topK(float[] a, final int topk) {
    float values[] = new float[topk];
    Arrays.fill(values, -Float.MAX_VALUE);
    int ixs[] = new int[topk];
    Arrays.fill(ixs, -1);

    for (int i = 0; i < a.length; i++) {
      for (int j = 0; j < topk; j++) {
        if (a[i] > values[j]) {
          for (int k = topk - 1; k >= j + 1; k--) {
            values[k] = values[k - 1];
            ixs[k] = ixs[k - 1];
          }
          values[j] = a[i];
          ixs[j] = i;
          break;
        }
      }
    }
    return ixs;
  }
}
