//TEST:SIMPLE:
// #define support

#define FOO 1.0f

float foo() { return FOO + 2.0; }

#define BAR 99

#if BAR > 10
int bar() { return 0; }
#else
BadThing shouldntCompile;
#endif
