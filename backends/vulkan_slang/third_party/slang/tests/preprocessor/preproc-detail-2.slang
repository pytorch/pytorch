//DIAGNOSTIC_TEST:SIMPLE:-E

// Macro parameters must have unique names

#define A(x, x) a x b x c

A(,)

