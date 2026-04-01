//TEST:SIMPLE: -Itests/preprocessor/include
// #include support

int foo() { return 0; }

#include "pragma-once-c.h"

// If include worked this will be defined
#ifndef ONLY_DEFINED_ONCE_C
// And so hitting this indicates and error (and will fail as bar isn't defined)
int baz() { return bar(); }
#endif

