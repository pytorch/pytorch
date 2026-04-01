// macro-expansion.slang
//
//TEST:SIMPLE:

// Test a bug where macro expansion isn't being
// triggered for back-to-back uses of a function-like
// macro.

#define IGNORE(THING) /* empty */

void test()
{
	IGNORE(badStuff)
	IGNORE(moreBad)
	;
}
