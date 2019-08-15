package org.pytorch;

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
  private static final int TYPE_CODE_DICT_DOUBLE_KEY = 13;
  private static final int TYPE_CODE_DICT_LONG_KEY = 14;

  public final int typeCode;

  private Tensor mTensor;
  private boolean mBool;
  private long mLong;
  private double mDouble;
  private IValue[] mTuple;

  private boolean[] mBoolList;
  private long[] mLongList;
  private double[] mDoubleList;
  private Tensor[] mTensorList;

  private IValue[] mList;

  private Map<String, IValue> mMapStringKey;
  private Map<Double, IValue> mMapDoubleKey;
  private Map<Long, IValue> mMapLongKey;

  private IValue(int typeCode) {
    this.typeCode = typeCode;
  }

  public boolean isNull() {
    return TYPE_CODE_NULL == this.typeCode;
  }

  public boolean isTensor() {
    return TYPE_CODE_TENSOR == this.typeCode;
  }

  public boolean isBool() {
    return TYPE_CODE_BOOL == this.typeCode;
  }

  public boolean isLong() {
    return TYPE_CODE_LONG == this.typeCode;
  }

  public boolean isDouble() {
    return TYPE_CODE_DOUBLE == this.typeCode;
  }

  public boolean isTuple() {
    return TYPE_CODE_TUPLE == this.typeCode;
  }

  public boolean isBoolList() {
    return TYPE_CODE_BOOL_LIST == this.typeCode;
  }

  public boolean isLongList() {
    return TYPE_CODE_LONG_LIST == this.typeCode;
  }

  public boolean isDoubleList() {
    return TYPE_CODE_DOUBLE_LIST == this.typeCode;
  }

  public boolean isTensorList() {
    return TYPE_CODE_TENSOR_LIST == this.typeCode;
  }

  public boolean isList() {
    return TYPE_CODE_TENSOR_LIST == this.typeCode;
  }

  public boolean isDictStringKey() {
    return TYPE_CODE_DICT_STRING_KEY == this.typeCode;
  }

  public boolean isDictDoubleKey() {
    return TYPE_CODE_DICT_DOUBLE_KEY == this.typeCode;
  }

  public boolean isDictLongKey() {
    return TYPE_CODE_DICT_LONG_KEY == this.typeCode;
  }

  public static IValue optionalNull() {
    return new IValue(TYPE_CODE_NULL);
  }

  public static IValue tensor(Tensor tensor) {
    final IValue iv = new IValue(TYPE_CODE_TENSOR);
    iv.mTensor = tensor;
    return iv;
  }

  public static IValue bool(boolean value) {
    final IValue iv = new IValue(TYPE_CODE_BOOL);
    iv.mBool = value;
    return iv;
  }

  public static IValue long64(long value) {
    final IValue iv = new IValue(TYPE_CODE_LONG);
    iv.mLong = value;
    return iv;
  }

  public static IValue double64(double value) {
    final IValue iv = new IValue(TYPE_CODE_DOUBLE);
    iv.mDouble = value;
    return iv;
  }

  public static IValue boolList(boolean... list) {
    final IValue iv = new IValue(TYPE_CODE_BOOL_LIST);
    iv.mBoolList = list;
    return iv;
  }

  public static IValue longList(long... list) {
    final IValue iv = new IValue(TYPE_CODE_LONG_LIST);
    iv.mLongList = list;
    return iv;
  }

  public static IValue doubleList(double... list) {
    final IValue iv = new IValue(TYPE_CODE_DOUBLE_LIST);
    iv.mDoubleList = list;
    return iv;
  }

  public static IValue tensorList(Tensor... list) {
    final IValue iv = new IValue(TYPE_CODE_TENSOR_LIST);
    iv.mTensorList = list;
    return iv;
  }

  public static IValue list(IValue... array) {
    final int size = array.length;
    if (size > 0) {
      final int typeCode0 = array[0].typeCode;
      for (int i = 1; i < size; i++) {
        if (typeCode0 != array[i].typeCode) {
          throw new IllegalArgumentException("List must contain items of the same type");
        }
      }
    }

    final IValue iv = new IValue(TYPE_CODE_LIST);
    iv.mList = array;
    return iv;
  }

  public static IValue tuple(IValue... array) {
    final IValue iv = new IValue(TYPE_CODE_TUPLE);
    iv.mTuple = array;
    return iv;
  }

  public static IValue dictStringKey(Map<String, IValue> map) {
    final IValue iv = new IValue(TYPE_CODE_DICT_STRING_KEY);
    iv.mMapStringKey = map;
    return iv;
  }

  public static IValue dictLongKey(Map<Long, IValue> map) {
    final IValue iv = new IValue(TYPE_CODE_DICT_LONG_KEY);
    iv.mMapLongKey = map;
    return iv;
  }

  public static IValue dictDoubleKey(Map<Double, IValue> map) {
    final IValue iv = new IValue(TYPE_CODE_DICT_DOUBLE_KEY);
    iv.mMapDoubleKey = map;
    return iv;
  }

  public Tensor getTensor() {
    preconditionType(TYPE_CODE_TENSOR, typeCode);
    return mTensor;
  }

  public long getLong() {
    preconditionType(TYPE_CODE_LONG, typeCode);
    return mLong;
  }

  public double getDouble() {
    preconditionType(TYPE_CODE_DOUBLE, typeCode);
    return mDouble;
  }

  public boolean getBoolean() {
    preconditionType(TYPE_CODE_BOOL, typeCode);
    return mBool;
  }

  public boolean[] getBoolList() {
    preconditionType(TYPE_CODE_BOOL_LIST, typeCode);
    return mBoolList;
  }

  public long[] getLongList() {
    preconditionType(TYPE_CODE_LONG_LIST, typeCode);
    return mLongList;
  }

  public double[] getDoubleList() {
    preconditionType(TYPE_CODE_DOUBLE_LIST, typeCode);
    return mDoubleList;
  }

  public Tensor[] getTensorList() {
    preconditionType(TYPE_CODE_TENSOR_LIST, typeCode);
    return mTensorList;
  }

  public IValue[] getList() {
    preconditionType(TYPE_CODE_LIST, typeCode);
    return mList;
  }

  public IValue[] getTuple() {
    preconditionType(TYPE_CODE_TUPLE, typeCode);
    return mTuple;
  }

  public Map<String, IValue> getDictStringKey() {
    preconditionType(TYPE_CODE_DICT_STRING_KEY, typeCode);
    return mMapStringKey;
  }

  public Map<Long, IValue> getDictLongKey() {
    preconditionType(TYPE_CODE_DICT_LONG_KEY, typeCode);
    return mMapLongKey;
  }

  public Map<Double, IValue> getDictDoubleKey() {
    preconditionType(TYPE_CODE_DICT_DOUBLE_KEY, typeCode);
    return mMapDoubleKey;
  }

  private void preconditionType(int typeCodeExpected, int typeCode) {
    if (typeCode != typeCodeExpected) {
      throw new IllegalStateException("Expected IValue type " + typeCodeExpected);
    }
  }
}
