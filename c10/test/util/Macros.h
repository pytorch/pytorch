#ifndef C10_TEST_CORE_MACROS_MACROS_H_

#ifdef _WIN32
#define DISABLED_ON_WINDOWS(x) DISABLED_##x
#else
#define DISABLED_ON_WINDOWS(x) x
#endif

#endif // C10_MACROS_MACROS_H_
