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

} // namespace thd
