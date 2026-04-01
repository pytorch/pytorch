//DIAGNOSTIC_TEST:SIMPLE:-E

// This os a regression test for a bug in Slang preprocessor macro expansion.

#define CONCAT2(x, y) x ## y
#define CONCAT(x, y) CONCAT2(x, y)

#define SOMETHING someThing

// Should be someThingElse
CONCAT(SOMETHING, Else)
// Should be SOMETHINGAnother, but old Slang expands to produce someThingAnother
CONCAT2(SOMETHING, Another)