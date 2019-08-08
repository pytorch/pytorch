/*
 *  Copyright (c) 2018-present, Facebook, Inc.
 *
 *  This source code is licensed under the MIT license found in the LICENSE
 *  file in the root directory of this source tree.
 *
 */
package com.facebook.jni;

import com.facebook.jni.annotations.DoNotStrip;

@DoNotStrip
public class UnknownCppException extends CppException {
  @DoNotStrip
  public UnknownCppException() {
    super("Unknown");
  }

  @DoNotStrip
  public UnknownCppException(String message) {
    super(message);
  }
}
