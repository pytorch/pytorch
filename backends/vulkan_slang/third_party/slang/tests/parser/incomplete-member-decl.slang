// incomplete-member-decl.slang

//DIAGNOSTIC_TEST:SIMPLE:

// Regresion test to ensure parser doesn't go into infinite loop
// on incomplete/malformed member decalration

struct Outer
{
    // Programmer was *trying* to declarae a field, but somehow
    // ended up with two type specifiers.
    //
    // Parser sees the second (generic) type specifier and assumes
    // it must represent a method declaration, at which point it
    // fails to find a parameter list (no opening `(`), and then
    // fails to find a body (no opening `{`), but finds a bare identifier
    // instead.
    //
    int MyType<X> inner;
}