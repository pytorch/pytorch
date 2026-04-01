//TEST:SIMPLE:
// #ifdef support

#define A

#ifdef A
int foo() { return 0; }
#else
BadThing thatWontCompile;
#endif

#ifdef BadThing
AnotherError onThisLine;
#else
int bar() { return foo(); }
#endif