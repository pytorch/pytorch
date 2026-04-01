// if-macro.slang

//TEST:SIMPLE:

// Test that a `#if` test can invoke a function-like macro.
//
// Note: this test is a reproducer for a bug reported by a user.

#define is_valid_TEST 1
#define isValid(name) (is_valid_##name != 0)
int test() {
#if isValid(TEST)
    return 1;
#else
    return NO_SUCH_SYMBOL;
#endif
}