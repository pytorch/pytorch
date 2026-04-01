//DIAGNOSTIC_TEST:SIMPLE:-E

// GCC: <source>:1:9: error: '##' cannot appear at either end of a macro expansion. 
// Clang: <source>:1:21: error: '##' cannot appear at start of macro expansion
// Old Slang outputs Hello ## There;
#define POUND_POUND ##

Hello POUND_POUND There;
