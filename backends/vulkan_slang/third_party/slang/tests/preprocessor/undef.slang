//TEST:SIMPLE:
// #undef support

// warning: undef of something not defined
#undef FOO

#define BAR 1.0f

float foo() { return BAR + 2.0; }

#ifdef BAR
// okay
#else
#error not okay
#endif

#undef BAR

typedef float BAR;

BAR bar() { return 2.0; }

#if !defined(BAR)
// okay
#else
#error not okay
#endif


#define FOO

#undef FOO
