package org.pytorch;

/**
 * Codes representing tensor data types.
 */
public enum DType {
  // NOTE: "jniCode" must be kept in sync with pytorch_jni_common.cpp.
  // NOTE: Never serialize "jniCode", because it can change between releases.

  /** Code for dtype torch.uint8. {@link Tensor#dtype()} */
  UINT8(1),
  /** Code for dtype torch.int8. {@link Tensor#dtype()} */
  INT8(2),
  /** Code for dtype torch.int32. {@link Tensor#dtype()} */
  INT32(3),
  /** Code for dtype torch.float32. {@link Tensor#dtype()} */
  FLOAT32(4),
  /** Code for dtype torch.int64. {@link Tensor#dtype()} */
  INT64(5),
  /** Code for dtype torch.float64. {@link Tensor#dtype()} */
  FLOAT64(6),
  ;

  final int jniCode;

  DType(int jniCode) {
    this.jniCode = jniCode;
  }
}
