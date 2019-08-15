package org.pytorch;

import org.junit.Assert;
import org.junit.Test;

public class PytorchUnitTests {

  @Test
  public void testIValue() {
    IValue v = IValue.long64(5l);
    Assert.assertTrue(5l == v.getLong());
    IValue tuple = IValue.tuple(IValue.long64(5l), IValue.bool(true));
  }

}
