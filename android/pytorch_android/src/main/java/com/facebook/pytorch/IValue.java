package com.facebook.pytorch;

import android.support.annotation.Nullable;

import java.util.List;
import java.util.Map;

public class IValue {
  public static final int TYPE_CODE_TENSOR = 1;
  public static final int TYPE_CODE_BOOL = 2;
  public static final int TYPE_CODE_LONG64 = 3;
  public static final int TYPE_CODE_DOUBLE64 = 4;
  public static final int TYPE_CODE_TUPLE = 5;
  public static final int TYPE_CODE_LIST = 6;
  public static final int TYPE_CODE_OPTIONAL = 7;
  public static final int TYPE_CODE_DICT_STRING_KEY = 8;
  public static final int TYPE_CODE_DICT_DOUBLE_KEY = 9;
  public static final int TYPE_CODE_DICT_LONG_KEY = 10;

  public final int typeCode;

  private Tensor mTensor;
  private boolean mBool;
  private long mLong;
  private double mDouble;
  private IValue[] mTuple;
  private IValue[] mList;
  private @Nullable
  IValue mOptionalValue;
  private Map<String, IValue> mMapStringKey;
  private Map<Double, IValue> mMapDoubleKey;
  private Map<Long, IValue> mMapLongKey;

  private IValue(int typeCode) {
    this.typeCode = typeCode;
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
    final IValue iv = new IValue(TYPE_CODE_LONG64);
    iv.mLong = value;
    return iv;
  }

  public static IValue double64(double value) {
    final IValue iv = new IValue(TYPE_CODE_DOUBLE64);
    iv.mDouble = value;
    return iv;
  }

  public static IValue optional(@Nullable IValue ivalue) {
    final IValue iv = new IValue(TYPE_CODE_OPTIONAL);
    iv.mOptionalValue = ivalue;
    return iv;
  }

  public static IValue optionalNull() {
    final IValue iv = new IValue(TYPE_CODE_OPTIONAL);
    iv.mOptionalValue = null;
    return iv;
  }

  public static IValue list(List<IValue> list) {
    final int size = list.size();
    if (size > 0) {
      final int typeCode0 = list.get(0).typeCode;
      for (int i = 1; i < size; i++) {
        if (typeCode0 != list.get(i).typeCode) {
          throw new IllegalArgumentException("List must contain items of the same type");
        }
      }
    }

    final IValue iv = new IValue(TYPE_CODE_LIST);
    IValue[] a = new IValue[size];
    list.toArray(a);
    iv.mList = a;
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

  public static IValue tuple(List<IValue> list) {
    final IValue iv = new IValue(TYPE_CODE_TUPLE);
    IValue[] a = new IValue[list.size()];
    list.toArray(a);
    iv.mTuple = a;
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
    preconditionType(TYPE_CODE_LONG64, typeCode);
    return mLong;
  }

  public double getDouble() {
    preconditionType(TYPE_CODE_DOUBLE64, typeCode);
    return mDouble;
  }

  public boolean getBoolean() {
    preconditionType(TYPE_CODE_BOOL, typeCode);
    return mBool;
  }

  public IValue[] getList() {
    preconditionType(TYPE_CODE_LIST, typeCode);
    return mList;
  }

  public IValue[] getTuple() {
    preconditionType(TYPE_CODE_TUPLE, typeCode);
    return mTuple;
  }

  public @Nullable
  IValue getOptional() {
    preconditionType(TYPE_CODE_OPTIONAL, typeCode);
    return mOptionalValue;
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
