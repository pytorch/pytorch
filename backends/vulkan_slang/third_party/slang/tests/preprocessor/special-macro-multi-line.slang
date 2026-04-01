//DIAGNOSTIC_TEST:SIMPLE:-E

// Output here varies by compiler. Gcc makes the line the one with M. Clang makes it the end bracket.

#define M(ARG) __LINE__


int x = M
(
ignored
)
;
