//DIAGNOSTIC_TEST:SIMPLE:-E

#define CONCAT(a, b) a ## b

// Correct output AB;
// Old Slang output
// ab;

#define A a
#define B b

CONCAT(A, B);
