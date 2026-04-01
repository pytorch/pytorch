//TEST:SIMPLE:
// #include support

int foo() { return 0; }

#include "include-a.slang.h"

int baz() { return bar(); }