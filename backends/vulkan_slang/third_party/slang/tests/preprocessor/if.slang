//TEST:SIMPLE:
// #ifdef support


#if (1 - 1*2) < 0
int foo() { return 0; }
#else
BadThing thatWontCompile;
#endif

#if (1 >> 1) && ~999
AnotherError onThisLine;
#else
int bar() { return foo(); }
#endif