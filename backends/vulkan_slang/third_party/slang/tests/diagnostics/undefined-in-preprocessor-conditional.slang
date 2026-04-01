//DIAGNOSTIC_TEST(smoke):SIMPLE:

// Use an undefined identifier in a preprocessor conditional

#define FOO  1
#define BORT 1

#if FOO && BART
#error Should not get here
#endif
