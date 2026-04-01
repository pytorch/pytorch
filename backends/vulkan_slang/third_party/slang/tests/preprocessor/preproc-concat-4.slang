//DIAGNOSTIC_TEST:SIMPLE:-E

#define CONCAT(a, b) a ## b

#define A a
#define B b

#define A2 A
#define B2 B

#define STRINGIFY(x) #x

// Should be 
// CONCAT(a, b) A2B2 CONCAT(a, b)
// CONCAT is disabled, A and B are expanded on next pass
// A2 B2 are first and last tokens pre expansion args
// 
// Old Slang outputs 
// ab ab ab 

CONCAT(CONCAT(A, B) A2, B2 CONCAT(A, B))
