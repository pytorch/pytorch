// macro-parens-from-expansion.slang
//DIAGNOSTIC_TEST:SIMPLE:-E

// This test covers a case where the parentheses for
// a function-like macro invocation argument list are
// themselves produced by a macro invocation.
//
// In the below code, it is important that the macros
// `LPAREN` and `RPAREN` are not considered busy for
// the invocation of `INNER`, despite the fact that
// both of those macros were expanded as part of
// producing the arguments to the `INNER` invocation.

#define LPAREN (
#define RPAREN )
#define INNER(X) LPAREN X RPAREN
#define M(X) X
#define OUTER(X) M( INNER LPAREN X RPAREN )

// output: ( 3 )
OUTER(3)
