//DIAGNOSTIC_TEST:SIMPLE:-E

#define CONCAT(a, b) a ## b

#define A a
#define B b

#define A2 A
#define B2 B

// Correct output: a A2B2 b;
// Old Slang output 
// a ab b ;

CONCAT(A2 A2, B2 B2);
