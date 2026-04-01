// keyword-undeclared-identifier.slang

//DIAGNOSTIC_TEST:SIMPLE:

// Test that using a contextual keyword in
// a context where it is an underfined
// identifier produces a reasonable error
// message instead of an internal compiler error
//
// Note that HLSL has keywords with very
// common names like `triangle` and `sample`,
// so it is easy for those to collide with
// local variable names.
//
// Slang decides to make almost all keywords
// contextual, so that they are looked up
// in lexical scope and can be shadowed by
// user-defined variables or functions.
//
// The problem in this case is that code
// could easily be refactored so that it
// uses one of the contextual keywrods in
// a place where it is no longer shadowed,
// but contextually needs to be treated
// as an expression.

int instanceTest()
{
	return instance;
}

int triangleTest()
{
	return triangle;
}
