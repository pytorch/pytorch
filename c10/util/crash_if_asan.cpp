#include <c10/util/crash_if_asan.h>

namespace c10 {
int crash_if_asan(int arg) {
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays)
  volatile char x[3];
  x[arg] = 0;
  return x[0];
}

int crash_if_ubsan() {
  // This code should work perfectly fine, as vtables are identical for Foo and
  // Baz unless rtti and ubsan are enabled
  struct Foo {
    virtual int bar() = 0;
    virtual ~Foo() = default;
  };
  struct Baz {
    virtual int bar() {
      return 17;
    }
    virtual ~Baz() = default;
  };
  Baz x{};
  // Purposely cast through `void*` so there's no fixups applied.
  // NOLINTNEXTLINE(bugprone-casting-through-void,-warnings-as-errors)
  auto y = static_cast<Foo*>(static_cast<void*>(&x));
  auto rc = y->bar();
  return rc;
}
} // namespace c10
