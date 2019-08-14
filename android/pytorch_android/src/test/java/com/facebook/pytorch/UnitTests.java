package com.facebook.pytorch;

import org.junit.Test;
import org.junit.Assert;

public class UnitTests {

  @Test
  public void testIValue() {
    IValue v = IValue.int32(5);
    Assert.assertTrue(v.getInt() == 5);
    IValue list = IValue.list(IValue.int32(5), IValue.bool(true));
  }

}
