#pragma once
#include <stdexcept>
#include <stdarg.h>

static inline void barf(const char *fmt, ...) {
  char msg[2048];
  va_list args;
  va_start(args, fmt);
  vsnprintf(msg, 2048, fmt, args);
  va_end(args);
  throw std::runtime_error(msg);
}

#if defined(_MSC_VER) && _MSC_VER <= 1900
#define __func__ __FUNCTION__
#endif

#if defined(__GNUC__) || defined(__ICL) || defined(__clang__)
#define AT_EXPECT(x, y) (__builtin_expect((x),(y)))
#else
#define AT_EXPECT(x, y) (x)
#endif

#define ASSERT(cond) \
  if (AT_EXPECT(!(cond), 0)) { \
    barf("%s:%u: %s: Assertion `%s` failed.", __FILE__, __LINE__, __func__, #cond); \
  }

//note: msg must be a string literal
//node: In, ##__VA_ARGS '##' supresses the comma if __VA_ARGS__ is empty
#define ASSERTM(cond, msg, ...) \
  if (AT_EXPECT(!(cond), 0)) { \
    barf("%s:%u: %s: Assertion `%s` failed: " msg , __FILE__, __LINE__, __func__, #cond,##__VA_ARGS__); \
  }

#define TRY_CATCH_ELSE(fn, catc, els)                           \
  {                                                             \
    /* avoid mistakenly passing if els code throws exception*/  \
    bool _passed = false;                                       \
    try {                                                       \
      fn;                                                       \
      _passed = true;                                           \
      els;                                                      \
    } catch (std::runtime_error &e) {                           \
      ASSERT(!_passed);                                         \
      catc;                                                     \
    }                                                           \
  }

#define ASSERT_THROWSM(fn, message)     \
  TRY_CATCH_ELSE(fn, ASSERT(std::string(e.what()).find(message) != std::string::npos), ASSERT(false))

#define ASSERT_THROWS(fn)  \
  ASSERT_THROWSM(fn, "");

#define ASSERT_EQUAL(t1, t2) \
  ASSERT(t1.equal(t2));

// allclose broadcasts, so check same size before allclose.
#define ASSERT_ALLCLOSE(t1, t2)   \
  ASSERT(t1.is_same_size(t2));    \
  ASSERT(t1.allclose(t2));

// allclose broadcasts, so check same size before allclose.
#define ASSERT_ALLCLOSE_TOLERANCES(t1, t2, atol, rtol)   \
  ASSERT(t1.is_same_size(t2));    \
  ASSERT(t1.allclose(t2, atol, rtol));
