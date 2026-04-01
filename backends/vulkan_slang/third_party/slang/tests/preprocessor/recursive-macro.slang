//TEST(compute):SIMPLE:

// Test for correct recursive macro behavior. In particular: 

// The letter of the spec is that we should macro expand
// each argument *before* substitution, and then go and
// macro-expand the substituted body. This means that we
// can invoke a macro as part of an argument to an
// invocation of the same macro:
//
//     FOO( 1, FOO(22, 2, 2), 333 );

// Also in the case on NO_EXPAND, it will not be expanded in the substitution. 

#define ARG_EXPAND(x) ( x )

// This macro should expand to NO_EXPAND(int 'a'), and not multiply invoke NO_EXPAND, because
// one the args have been expanded.
#define NO_EXPAND(a) NO_EXPAND(int a)

// Should expand to int NO_EXPAND(int b)
int NO_EXPAND(b)
{
    return b + ARG_EXPAND(ARG_EXPAND(1));
}

// The same issue can arise for object-like macros and not
// just function-like, so we will test that case here as well.

int objectLikeMacroTest( int REC )
{
#define REC REC

	return REC;	
}

// We also need to check the case where macros are
// mutually, rather than directly, recursive

int mutuallyRecursiveMacroTest( int XYZ )
{
	#define ABC XYZ
	#define XYZ ABC
	return XYZ;
}