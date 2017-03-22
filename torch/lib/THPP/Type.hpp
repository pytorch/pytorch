#pragma once

namespace thpp {

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
  HALF = 'a',
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
  return (t == Type::FLOAT || t == Type::DOUBLE || t == Type::HALF);
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

inline const char* toString(Type t) {
  switch (t) {
    case Type::CHAR: return "Char";
    case Type::UCHAR: return "Byte";
    case Type::FLOAT: return "Float";
    case Type::DOUBLE: return "Double";
    case Type::HALF: return "Half";
    case Type::SHORT: return "Short";
    case Type::USHORT: return "UShort";
    case Type::INT: return "Int";
    case Type::UINT: return "UInt";
    case Type::LONG: return "Long";
    case Type::ULONG: return "ULong";
    case Type::LONG_LONG: return "LongLong";
    case Type::ULONG_LONG: return "ULongLong";
    case Type::LONG_STORAGE: return "LongStorage";
    case Type::TENSOR: return "Tensor";
    case Type::STORAGE: return "Storage";
    default: return "<unknown>";
  }
}

} // namespace thpp
