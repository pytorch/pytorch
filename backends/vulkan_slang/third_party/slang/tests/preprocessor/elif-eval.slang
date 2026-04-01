//TEST:SIMPLE:

#ifndef SOME_MACRO
#   error Not this one!
// The expression after #elif shouldn't be evaluated, because #ifndef evaluated as true
#elif SOME_OTHER_MACRO
#else
#error This isn't valid!
#endif
