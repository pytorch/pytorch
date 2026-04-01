//TEST:SIMPLE:
// Check #if expression is ignored if outer #if/#ifndef means it is skipped

#if 0
BadThing thatWontCompile;
#if a + b == 27
BadThing thatWontCompile;
#endif
BadThing thatWontCompile;
#endif

#ifdef SOMETHING_SILLY
BadThing thatWontCompile;
#if SOMETHING_SILLY
BadThing thatWontCompile;
#endif
#endif
