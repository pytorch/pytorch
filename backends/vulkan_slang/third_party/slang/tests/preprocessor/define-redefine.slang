//TEST:SIMPLE:

// Confirm that we handle redefinition
// of an existing define (with a diagnostic)

#define FOO 1.0f

float foo() { return FOO + 2.0; }

#define FOO 2.0f
