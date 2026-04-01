//DIAGNOSTIC_TEST:SIMPLE:-E

#define A a
#define B b

// Correct output 
// "A B"
// Old Slang output 
// # a b ;

#define STRINGIFY(x) #x
STRINGIFY(A B);