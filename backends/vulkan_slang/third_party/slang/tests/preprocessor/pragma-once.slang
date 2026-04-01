//TEST(smoke):SIMPLE:
//TEST(smoke):SIMPLE: -file-system load-file
//TEST(smoke):SIMPLE: -file-system os

// Test support for `#pragma once`

// We will include two header files:
// one that uses `#pragma once`, and
// one that doesn't.
//
// The first file defines a function `foo()`,
// and the second defines a macro `BAR`
//
#include "pragma-once-a.h"
#include "pragma-once-b.h"

// We will include the files again, and
// before we do so we need to undefine
// the macro from the second file so
// that it doesn't get a redefinition diagnostic.
//
#undef BAR
//
// We don't do anything about the function
// in the first file, because we expect
// the `#pragma once` to cause it to be
// ignored on this second time.
//
#include "pragma-once-a.h"
#include "pragma-once-b.h"

// Make sure relative paths are handled
#include "./pragma-once-a.h"
#include "./pragma-once-a.h"

#include ".\pragma-once-a.h"
#include "./include\../pragma-once-a.h"

#include "../preprocessor/./pragma-once-a.h"

#include "include/pragma-once-c.h"
#include "./include\pragma-once-c.h"

#ifndef ONLY_DEFINED_ONCE_C
#undef BAR
#endif

// Now let's use both the function and the
// macro, to confirm that they are both
// defined as expected.
//
// Note: if we accidentally include file
// `a.h` more than once, we'd expect to
// get an error here, because the two
// function definitions conflict.
//
float test(float x)
{
	return foo(x) + BAR(x);
}
