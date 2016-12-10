#pragma once

namespace thd {

/*
 * The following notation comes from:
 * docs.python.org/3.5/library/struct.html#module-struct
 * except from 'T', which stands for Tensor
 */

enum class Type : char {
  CHAR = 'c',
  UCHAR = 'B',
  FLOAT = 'f',
  DOUBLE = 'd',
  SHORT = 'h',
  USHORT = 'H',
  INT = 'i',
  UINT = 'I',
  LONG = 'l',
  ULONG = 'L',
  LONG_LONG = 'q',
  ULONG_LONG = 'Q',
  LONG_STORAGE = 'X',
  TENSOR = 'T',
  STORAGE = 'S',
};

inline bool isFloat(Type t) {
  return (t == Type::FLOAT || t == Type::DOUBLE);
}

inline bool isObject(Type t) {
  return (t == Type::TENSOR || t == Type::STORAGE);
}

inline bool isInteger(Type t) {
  return (t == Type::CHAR || t == Type::UCHAR ||
          t == Type::SHORT || t == Type:: USHORT ||
          t == Type::INT || t == Type::UINT ||
          t == Type::LONG || t == Type::ULONG ||
          t == Type::LONG_LONG || t == Type::ULONG_LONG);
}

} // namespace thd
