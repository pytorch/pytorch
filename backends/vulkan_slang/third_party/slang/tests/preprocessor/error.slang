//TEST:SIMPLE:

// #error support

#define FIRST 0

#if FIRST
#error Not this one!
#else

#error This isn't valid!

#endif
