#pragma once

// Fixture header for tools/test/test_header_only_linter.py. The symbols here
// are the ones good.txt and bad.txt file under this header.

inline int a() {
  return 0;
}

inline int symC(int x, int y) {
  return x + y;
}

inline int symD() {
  return 1;
}
