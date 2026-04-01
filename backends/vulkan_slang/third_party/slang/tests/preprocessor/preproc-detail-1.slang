//DIAGNOSTIC_TEST:SIMPLE:-E

// If a macro can take a single parameter, it is valid to pass in 'nothing'.
// Old Slang outputs an error about the wrong amount of parameters
// Correct output: a b

#define A(x) a x b

A()

