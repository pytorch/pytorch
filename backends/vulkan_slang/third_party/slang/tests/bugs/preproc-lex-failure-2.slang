//DIAGNOSTIC_TEST:SIMPLE:-E

// Test that diagnostics appear for invalid tokens.

#if 1
// Octal warning
000
/// Unclosed string literal
"Hello!
#else
Hello
#endif
