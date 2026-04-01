// variadic-macro.slang
//DIAGNOSTIC_TEST:SIMPLE:-E

// First we text a variadic macro with the default variadic
// parameter name (`__VA_ARGS__`) and other non-variadic parameter.

#define A(X, ...) (X ; __VA_ARGS__)

A(1, 2, 3)

// Next we test a varadic macro where the variadic parameter
// has been given an explicit name.

#define B(B_ARGS...) (B_ARGS)

B(4, 5)

// Finally, we test the case where the variadic parameter
// might have zero arguments passed for it, so that
// the result produces a trailing comma.
//
// TODO: If we add support for deleting trailing commas
// in these cases to Slang, we can update this case to
// test that support.

#define C(Y, ARGS...) (Y , ARGS)

C(1)
C(2, 3)
C(4, 5, 6)
