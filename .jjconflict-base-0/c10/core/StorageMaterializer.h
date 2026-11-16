#pragma once

#include <c10/macros/Export.h>

namespace c10 {

struct StorageImpl;

// Pluggable one-shot materialization hook for StorageImpl.
// Called on write-path access before returning a mutable pointer.
// The hook is cleared only after successful invocation; if it throws, the hook
// remains installed and the next write will retry. Hooks must not trigger
// re-entrant materialization on the same storage.
using MaterializeFn = void (*)(StorageImpl*);

} // namespace c10
