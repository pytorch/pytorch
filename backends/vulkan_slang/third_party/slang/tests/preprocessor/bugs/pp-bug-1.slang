// pp-bug-1.slang
//DIAGNOSTIC_TEST:SIMPLE:-E

// The bug in this case was related to a use-after-free
// where the list of "busy" macros for a function-like
// macro invocation was being set based on what was
// busy when reading the macro name, which could include
// streams that had been popped by the time arguments
// had been read.

#define NUM_CASES 2
#define X1(M) M(0)
#define X2(M) M(0) M(1)
#define CONCAT2(a, b) a##b
#define CONCAT(a, b) CONCAT2(a, b)
#define FOREACH(M) CONCAT(X, NUM_CASES)(M)
#define CASE(i) i
// Should output: 0 1
// Outputs: 0 CASE ( 1 )
FOREACH(CASE)