package org.pytorch;

import java.util.Locale;
import java.util.Map;

public class IValue {
  private static final int TYPE_CODE_NULL = 1;

  private static final int TYPE_CODE_TENSOR = 2;
  private static final int TYPE_CODE_BOOL = 3;
  private static final int TYPE_CODE_LONG = 4;
  private static final int TYPE_CODE_DOUBLE = 5;

  private static final int TYPE_CODE_TUPLE = 6;
  private static final int TYPE_CODE_BOOL_LIST = 7;
  private static final int TYPE_CODE_LONG_LIST = 8;
  private static final int TYPE_CODE_DOUBLE_LIST = 9;
  private static final int TYPE_CODE_TENSOR_LIST = 10;
  private static final int TYPE_CODE_LIST = 11;

  private static final int TYPE_CODE_DICT_STRING_KEY = 12;
  private static final int TYPE_CODE_DICT_LONG_KEY = 13;

  private final int mTypeCode;
  private Object mData;

  private IValue(int typeCode) {
    this.mTypeCode = typeCode;
  }

  public boolean isNull() {
    return TYPE_CODE_NULL == this.mTypeCode;
  }

  public boolean isTensor() {
    return TYPE_CODE_TENSOR == this.mTypeCode;
  }

  public boolean isBool() {
    return TYPE_CODE_BOOL == this.mTypeCode;
  }

  public boolean isLong() {
    return TYPE_CODE_LONG == this.mTypeCode;
  }

  public boolean isDouble() {
    return TYPE_CODE_DOUBLE == this.mTypeCode;
  }

  public boolean isTuple() {
    return TYPE_CODE_TUPLE == this.mTypeCode;
  }

  public boolean isBoolList() {
    return TYPE_CODE_BOOL_LIST == this.mTypeCode;
  }

  public boolean isLongList() {
    return TYPE_CODE_LONG_LIST == this.mTypeCode;
  }

  public boolean isDoubleList() {
    return TYPE_CODE_DOUBLE_LIST == this.mTypeCode;
  }

  public boolean isTensorList() {
    return TYPE_CODE_TENSOR_LIST == this.mTypeCode;
  }

  public boolean isList() {
    return TYPE_CODE_TENSOR_LIST == this.mTypeCode;
  }

  public boolean isDictStringKey() {
    return TYPE_CODE_DICT_STRING_KEY == this.mTypeCode;
  }

  public boolean isDictLongKey() {
    return TYPE_CODE_DICT_LONG_KEY == this.mTypeCode;
  }

  public static IValue optionalNull() {
    return new IValue(TYPE_CODE_NULL);
  }

  public static IValue tensor(Tensor tensor) {
    final IValue iv = new IValue(TYPE_CODE_TENSOR);
    iv.mData = tensor;
    return iv;
  }

  public static IValue bool(boolean value) {
    final IValue iv = new IValue(TYPE_CODE_BOOL);
    iv.mData = value;
    return iv;
  }

  public static IValue long64(long value) {
    final IValue iv = new IValue(TYPE_CODE_LONG);
    iv.mData = value;
    return iv;
  }

  public static IValue double64(double value) {
    final IValue iv = new IValue(TYPE_CODE_DOUBLE);
    iv.mData = value;
    return iv;
  }

  public static IValue boolList(boolean... list) {
    final IValue iv = new IValue(TYPE_CODE_BOOL_LIST);
    iv.mData = list;
    return iv;
  }

  public static IValue longList(long... list) {
    final IValue iv = new IValue(TYPE_CODE_LONG_LIST);
    iv.mData = list;
    return iv;
  }

  public static IValue doubleList(double... list) {
    final IValue iv = new IValue(TYPE_CODE_DOUBLE_LIST);
    iv.mData = list;
    return iv;
  }

  public static IValue tensorList(Tensor... list) {
    final IValue iv = new IValue(TYPE_CODE_TENSOR_LIST);
    iv.mData = list;
    return iv;
  }

  public static IValue list(IValue... array) {
    final int size = array.length;
    if (size > 0) {
      final int typeCode0 = array[0].mTypeCode;
      for (int i = 1; i < size; i++) {
        if (typeCode0 != array[i].mTypeCode) {
          throw new IllegalArgumentException("List must contain items of the same type");
        }
      }
    }

    final IValue iv = new IValue(TYPE_CODE_LIST);
    iv.mData = array;
    return iv;
  }

  public static IValue tuple(IValue... array) {
    final IValue iv = new IValue(TYPE_CODE_TUPLE);
    iv.mData = array;
    return iv;
  }

  public static IValue dictStringKey(Map<String, IValue> map) {
    final IValue iv = new IValue(TYPE_CODE_DICT_STRING_KEY);
    iv.mData = map;
    return iv;
  }

  public static IValue dictLongKey(Map<Long, IValue> map) {
    final IValue iv = new IValue(TYPE_CODE_DICT_LONG_KEY);
    iv.mData = map;
    return iv;
  }

  public Tensor getTensor() {
    preconditionType(TYPE_CODE_TENSOR, mTypeCode);
    return (Tensor) mData;
  }

  public long getLong() {
    preconditionType(TYPE_CODE_LONG, mTypeCode);
    return (long) mData;
  }

  public double getDouble() {
    preconditionType(TYPE_CODE_DOUBLE, mTypeCode);
    return (double) mData;
  }

  public boolean getBool() {
    preconditionType(TYPE_CODE_BOOL, mTypeCode);
    return (boolean) mData;
  }

  public boolean[] getBoolList() {
    preconditionType(TYPE_CODE_BOOL_LIST, mTypeCode);
    return (boolean[]) mData;
  }

  public long[] getLongList() {
    preconditionType(TYPE_CODE_LONG_LIST, mTypeCode);
    return (long[]) mData;
  }

  public double[] getDoubleList() {
    preconditionType(TYPE_CODE_DOUBLE_LIST, mTypeCode);
    return (double[]) mData;
  }

  public Tensor[] getTensorList() {
    preconditionType(TYPE_CODE_TENSOR_LIST, mTypeCode);
    return (Tensor[]) mData;
  }

  public IValue[] getList() {
    preconditionType(TYPE_CODE_LIST, mTypeCode);
    return (IValue[]) mData;
  }

  public IValue[] getTuple() {
    preconditionType(TYPE_CODE_TUPLE, mTypeCode);
    return (IValue[]) mData;
  }

  public Map<String, IValue> getDictStringKey() {
    preconditionType(TYPE_CODE_DICT_STRING_KEY, mTypeCode);
    return (Map<String, IValue>) mData;
  }

  public Map<Long, IValue> getDictLongKey() {
    preconditionType(TYPE_CODE_DICT_LONG_KEY, mTypeCode);
    return (Map<Long, IValue>) mData;
  }

  private void preconditionType(int typeCodeExpected, int typeCode) {
    if (typeCode != typeCodeExpected) {
      throw new IllegalStateException(
          String.format(
              Locale.US, "Expected IValue type %d, actual type %d", typeCodeExpected, typeCode));
    }
  }
}
