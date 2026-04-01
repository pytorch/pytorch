//TEST:SIMPLE:

// Test support for escaped newlines in macro definitions.
//
// A complete lexer would handle backslash-escaped newlines
// in every possible context (including, e.g., in the middle
// of an identifier), but we are not going to go to such
// lengths right now.

#define FOO(x, y) 	\
	x				\
	y				\
	/* */

FOO(float, bar)(float a)
{
	FOO(return, a);
}

float foo(float x)
{
	return bar(x);
}
