//DIAGNOSTIC_TEST:SIMPLE:-E

#if 0
// Octal warning
000
/// Unclosed string literal
"Hello!
/// A non 7 bit ascii code
Š
#else
Hello
#endif