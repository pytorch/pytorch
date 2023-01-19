#include "examples/test_cc_shared_library/bar.h"
#include "examples/test_cc_shared_library/baz.h"
#include "examples/test_cc_shared_library/preloaded_dep.h"
#include "examples/test_cc_shared_library/qux.h"

int foo() {
  bar();
  baz();
  qux();
  preloaded_dep();
  return 42;
}
