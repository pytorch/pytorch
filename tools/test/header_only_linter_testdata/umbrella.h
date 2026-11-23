#pragma once

// Fixture header for tools/test/test_header_only_linter.py. Defines nothing
// itself: it exists so the linter's include-closure walk can be tested, the
// way ATen/cpu/vec/vec.h only pulls in the headers that define its symbols.
// Includes are resolved from the repo root, hence the full path.
#include <tools/test/header_only_linter_testdata/bbb.h>
